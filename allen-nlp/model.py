import torch
from torch.nn.functional import nll_loss, binary_cross_entropy,binary_cross_entropy_with_logits

from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder,Seq2VecEncoder, MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from typing import Any, Dict, List, Optional

from allennlp.modules import InputVariationalDropout
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper, StackedBidirectionalLstm
from allennlp.modules.matrix_attention import LinearMatrixAttention
from allennlp.training.metrics import BooleanAccuracy, Auc, CategoricalAccuracy

from self_attentive import SelfAttentive
from qangaroo import MyQangarooReader
from decoder import Decoder, SANDecoder

@Model.register("msprm")
class MultiStepParaRankModel(Model):
    
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 pq_attention: MatrixAttention,
                 p_selfattention: MatrixAttention,
                 supports_pooling: Seq2VecEncoder,
                 query_pooling: Seq2VecEncoder,
                 candidates_pooling: Seq2VecEncoder,
                 decoder: Decoder,
                 dropout: float = 0.2,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MultiStepParaRankModel, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        
        self.phrase_layer = phrase_layer
        
        self.pq_attention = pq_attention
        self.p_selfattention = p_selfattention
        
        self.supports_pooling = supports_pooling
        self.query_pooling = query_pooling
        self.candidates_pooling = candidates_pooling
        
        self.decoder = decoder
        
        self.dropout = InputVariationalDropout(p=dropout)
        
        self._support_accuracy = Auc()
        self._candidate_accuracy = CategoricalAccuracy()
        
        initializer(self)
        
    def forward(self, 
                query: Dict[str, torch.LongTensor],
                supports: Dict[str, torch.LongTensor],                
                candidates: Dict[str, torch.LongTensor],
                answer: Dict[str, torch.LongTensor] = None,
                answer_index: torch.IntTensor=None,                
                metadata: List[Dict[str, Any]] = None,
                supports_labels: torch.Tensor = None,
                
               ) -> Dict[str, torch.Tensor]:
        embedded_supports = self.text_field_embedder(supports)
        embedded_query = self.text_field_embedder(query)
        embedded_candidates = self.text_field_embedder(candidates)
        
        batch_size, support_num, support_length, embed_dim = embedded_supports.size()
        _, candidate_num, candidate_length, _ = embedded_candidates.size()
        _, query_length, _ = embedded_query.size()
        
        supports_mask_para = util.get_text_field_mask(supports)
        supports_mask_seq = util.get_text_field_mask(supports,num_wrapping_dims=1)
        supports_mask_seq_expand = supports_mask_seq.view(-1,supports_mask_seq.size(-1))

        candidates_mask_para = util.get_text_field_mask(candidates)
        candidates_mask_seq = util.get_text_field_mask(candidates, num_wrapping_dims=1)
        candidates_mask_seq_expand = candidates_mask_seq.view(-1, candidates_mask_seq.size(-1))

        query_mask = util.get_text_field_mask(query)        
        query_mask_expand = query_mask.unsqueeze(1).expand(query_mask.size(0),support_num, query_mask.size(1))
        query_mask_expand = query_mask_expand.contiguous().view(-1, query_mask_expand.size(-1))

        embedded_supports_expand = embedded_supports.view(-1, support_length, embed_dim)
        embedded_candidates_expand = embedded_candidates.view(-1, candidate_length, embed_dim)
        
        encoded_query = self.phrase_layer(self.dropout(embedded_query), query_mask)
        encoded_supports = self.phrase_layer(self.dropout(embedded_supports_expand), supports_mask_seq_expand)
        encoded_candidates = self.phrase_layer(self.dropout(embedded_candidates_expand), candidates_mask_seq_expand)

        encoded_query_expand = encoded_query.unsqueeze(1).expand(batch_size, support_num, encoded_query.size(1), encoded_query.size(2))
        encoded_query_expand = encoded_query_expand.contiguous().view(-1,encoded_query.size(1), encoded_query.size(2))  
        
        # Co-attention

        # shape: (batch_size*passage_num, passage_length, question_length )
        supports_query_similarity = self.pq_attention(encoded_supports, encoded_query_expand)

        # shape: (batch_size*passage_num, passage_length, question_length )
        supports_query_attention = util.masked_softmax(supports_query_similarity, query_mask_expand,memory_efficient=True)
        # shape: (batch_size*passage_num, passage_length, encoding_dim)
        supports_query_vectors = util.weighted_sum(encoded_query_expand, supports_query_attention) 

        # shape: (batch_size*passage_num, query_length, passage_length)
        query_passage_attention = util.masked_softmax(supports_query_similarity.transpose(1,2), supports_mask_seq_expand,memory_efficient=True)
        # shape: (batch_size*passage_num, query_length, encoding_dim)
        query_supports_vectors = util.weighted_sum(encoded_supports, query_passage_attention)

        # shape: (batch_size*passage_num, passage_length, encoding_dim)
        supports_query_vectors_2 = torch.bmm(supports_query_attention, query_supports_vectors)
        # shape: (batch_size*passage_num, passage_length, encoding_dim*2)
        supports_coattention_vectors = torch.cat([supports_query_vectors, supports_query_vectors_2], dim=-1)

        # Fusion, 暂时用简单的fusion函数
        #supports_coattention_vectors = co_attention_fusion(torch.cat([encoded_supports,supports_query_vectors_final], dim=-1))        
        suppports_self_similarity = self.p_selfattention(supports_coattention_vectors, supports_coattention_vectors)
        supports_selfattention = util.masked_softmax(suppports_self_similarity, supports_mask_seq_expand, memory_efficient=True)
        supports_selfatt_vectors =util.weighted_sum(supports_coattention_vectors, supports_selfattention) 
        #support_selfatt_fusion = self_attention_fusion(util.combine_tensors('1,2,1-2,1*2',[supports_coattention_vectors, supports_selfatt_vectors]))

        supports_pooling_vectors = self.supports_pooling(supports_selfatt_vectors, supports_mask_seq_expand)
        supports_pooling_vectors = supports_pooling_vectors.view(batch_size, support_num,-1)
        question_pooling_vectors = self.query_pooling(encoded_query,query_mask)

        candidates_pooling_vectors = self.candidates_pooling(encoded_candidates, candidates_mask_seq_expand)
        candidates_pooling_vectors = candidates_pooling_vectors.view(batch_size, -1, candidates_pooling_vectors.size(-1))
        
        # supports porb normalized, candidates_score: unnormalized
        supports_prob, candidates_score= self.decoder(supports_pooling_vectors, question_pooling_vectors, candidates_pooling_vectors, supports_mask_para)
        candidates_score = util.replace_masked_values(candidates_score, candidates_mask_para, -1e7)
        
        output_dict = {
            "supports_prob": supports_prob,
            "candidates_score": candidates_score
        }
        if supports_labels is not None:
            supports_prob = util.replace_masked_values(supports_prob, supports_mask_para,-1e32)
            s_loss = binary_cross_entropy_with_logits(supports_prob, supports_labels)
            c_loss = nll_loss(util.masked_log_softmax(candidates_score, candidates_mask_para),answer_index.squeeze(-1))
            loss = s_loss + c_loss
            self._support_accuracy(supports_prob.view(-1,1).squeeze(), supports_labels.view(-1,1).squeeze(), supports_mask_para.view(-1).squeeze().detach().cpu())
            self._candidate_accuracy(candidates_score, answer_index.squeeze(-1))
            
            output_dict['loss'] = loss
            output_dict['s_loss'] = s_loss
            output_dict['c_loss'] = c_loss            
        
        return output_dict
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'support_auc': self._support_accuracy.get_metric(reset),
            'candidate_acc': self._candidate_accuracy.get_metric(reset)
        }
        
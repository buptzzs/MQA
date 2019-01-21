import torch
import torch.nn as nn

from typing import List, Union, Dict
import numpy as np

import flair
from flair.data import Sentence
from flair.embeddings import Embeddings, TokenEmbeddings, StackedEmbeddings
from flair.nn import LockedDropout, WordDropout
from flair.training_utils import clear_embeddings
from torchtext.vocab import CharNGram

from model import BiAttention, EncoderRNN, SelfAttention

class CharNgramEmbeddings(TokenEmbeddings):
    
    def __init__(self):
        super().__init__()
        self.embeddings = CharNGram()
        self.name = 'CharNGram'
        
    @property
    def embedding_length(self) -> int:
        return 100
    
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            
            for token in sentence:
                word = token.text
                word_embedding = self.embeddings[word].squeeze()
                token.set_embedding(self.name, word_embedding)
                
        return sentences

class MultiSentenceEmbeddings(Embeddings):
    def __init__(self,  
                 embeddings: List[TokenEmbeddings], 
                 reproject_words_dimension: int = None,
                 word_dropout: float = 0.0):
        
        super().__init__()
        
        self.embeddings = StackedEmbeddings(embeddings=embeddings)
        self.staic_embeddings = False
        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length
            
        self.use_word_dropout: bool = word_dropout > 0.0
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)
        
        self._embedding_length: int = self.length_of_all_token_embeddings
        self.reproject = reproject_words_dimension is not None
        
        if self.reproject:
            self.word_reprojection_map = torch.nn.Linear(self.length_of_all_token_embeddings, 
                                                         reproject_words_dimension)
            self._embedding_length = reproject_words_dimension
            
            
        self.to(flair.device)
        
    @property
    def embedding_length(self) -> int:
        return self._embedding_length    
        
    def embed(self, sentences: Union[List[Sentence], Sentence]):
        
        if type(sentences) is Sentence:
            sentences = [sentences]
            
        sentences.sort(key=lambda x: len(x), reverse=True)
        
        self.embeddings.embed(sentences)

        longest_token_sequence_in_batch: int = len(sentences[0])

        all_sentence_tensors = []
        lengths: List[int] = []

        # go through each sentence in batch
        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token: Token = token
                word_embeddings.append(token.get_embedding().unsqueeze(0))

            # PADDING: pad shorter sentences out
            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.FloatTensor(np.zeros(self.length_of_all_token_embeddings, dtype='float')).unsqueeze(0))

            word_embeddings_tensor = torch.cat(word_embeddings, 0)

            sentence_states = word_embeddings_tensor

            # ADD TO SENTENCE LIST: add the representation
            all_sentence_tensors.append(sentence_states.unsqueeze(1))

        # --------------------------------------------------------------------
        # GET REPRESENTATION FOR ENTIRE BATCH
        # --------------------------------------------------------------------
        sentence_tensor = torch.cat(all_sentence_tensors, 1)
        sentence_tensor = sentence_tensor.permute(1,0,2)
        sentence_tensor = sentence_tensor.to(flair.device)
        if self.reproject:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)
        
        clear_embeddings(sentences, also_clear_word_embeddings=True)
        return sentence_tensor
    
    
class SimpleQANet(nn.Module):
    
    def __init__(self, config, embeddings: MultiSentenceEmbeddings):
        super().__init__()
        self.config = config
        
        self.embedding_layer = embeddings
        self.rnn = EncoderRNN(embeddings.embedding_length, config.hidden, 1, True, True, 
                              config.rnn_dropout, False)
        
        self.qc_att = BiAttention(config.hidden*2, 0.2)
        self.linear_1 = nn.Sequential(
                nn.Linear(config.hidden*8, config.hidden),
                nn.ReLU()
        )    
        
        self.rnn_2 = EncoderRNN(config.hidden, config.hidden, 1, False, True, config.rnn_dropout, False)
        
        self.self_att = SelfAttention(config.hidden*2, config.hidden*2, config.att_dropout)       
        self.self_att_2 = SelfAttention(config.hidden*2, config.hidden*2, config.att_dropout)        
        
        self.self_att_c = SelfAttention(config.hidden*2, config.hidden*2, config.att_dropout)        
        self.to(flair.device)
        
    def forward(self, item):

        supports = item['supports']
        candidates = item['candidates']
        query = item['query']
        label = item['label']

        # embeddings.embed(supports)
        c_embedding_tensor = self.embedding_layer.embed(candidates)
        s_embedding_tensor = self.embedding_layer.embed(supports)
        q_embedding_tensor = self.embedding_layer.embed(query)            
        # Embedding 

        

        q_out = self.rnn(q_embedding_tensor)
        c_out = self.rnn(c_embedding_tensor)
        
        s_out = self.rnn(s_embedding_tensor)

        
        support_len = s_out.size(0)
        q_out = q_out.expand(support_len, q_out.size(1), q_out.size(2))
        
        # s_out:[supports_len, seq_len, hidden*2], q_out: [support_len, seq_len, hidden*2]
        output = self.qc_att(s_out, q_out)
        output = self.linear_1(output)
        output = self.rnn_2(output)
        
        # self-attention pooling 
        # [support_len, hidden*2]
        output = self.self_att(output)
        # [1, hidden*2]
        output = self.self_att_2(output.unsqueeze(0))

        # [candidate_len, hidden*2]
        c_out = self.self_att_c(c_out)
        
        # score layer
        score = torch.mm(c_out, torch.tanh(output.transpose(0, 1)))
        return score    
import torch
import torch.nn as nn
from torch.nn import Parameter

from overrides import overrides

from allennlp.modules.attention import BilinearAttention
from allennlp.nn import util
from decoder import Decoder
import random

@Decoder.register("san_decoder")
class SANDecoder(Decoder):
    
    def __init__(self,
                 support_dim: int,
                 query_dim: int,
                 candidates_dim: int,
                 num_step: int = 1,
                 reason_type: int = 0,
                 reason_dropout_p: float = 0.2,
                 dropout_p: float = 0.4
                 ) -> None:
        """
        Parameters
        ----------
        
        reason_type: 0: random
                     1: only last
                     2: avg
        """
        super().__init__()
        
        assert num_step > 0
        assert reason_type < 3 and reason_type >=0
        
        self.num_step = num_step
        self.reason_type = reason_type
        self.dropout_p = dropout_p
        self.reason_dropout_p = reason_dropout_p
        
        self.supports_predictor = BilinearAttention(query_dim, support_dim, normalize=False)
        self.candidates_predictor = BilinearAttention(support_dim, candidates_dim, normalize=False)
        
        self.rnn = nn.GRUCell(support_dim, query_dim)
        self.alpha = Parameter(torch.zeros(1,1))

    @overrides
    def forward(self,
               supports_vectors: torch.FloatTensor,
               query_vectors: torch.FloatTensor,
               candidates_vectors: torch.FloatTensor,
               supports_mask: torch.LongTensor = None):
        """
        Parameters
        ----------
        supports_vectors: (batch_size, supports_length, supports_dim)
        query_vectors: (batch_size, query_dim)
        candidates_vectors: (batch_size, candidates_lenght, candidates_dim)
        
        Returns
        -------
        supports_probability: (batch_size, supports_length) | normalized
        candidates_score: (batch_size, candidates_length) | unnormalized
        """
        
        h0 = query_vectors
        memory = supports_vectors
        memory_mask = supports_mask
        
        supports_probabilities_list = []
        candidates_scores_list = []
        
        for i in range(self.num_step):
            supports_prob = self.supports_predictor(h0, memory, memory_mask)
            
            x_i = util.weighted_sum(memory, supports_prob)
            candidates_score = self.candidates_predictor(x_i, candidates_vectors)
            
            h0 = self.rnn(x_i, h0)
            
            supports_probabilities_list.append(supports_prob)
            candidates_scores_list.append(candidates_score)
            
        # stochastic dropout    
        if self.reason_type == 0:
            supports_probabilities = torch.stack(supports_probabilities_list,2)
            candidates_scores = torch.stack(candidates_scores_list, 2)      
            
            batch_size = h0.size(0)
            mask = self.generate_mask(batch_size)
            mask = mask.unsqueeze(1)
            
            supports_probabilities = supports_probabilities * mask.expand_as(supports_probabilities)
            candidates_scores = candidates_scores * mask.expand_as(candidates_scores)
            final_supports_prob = torch.mean(supports_probabilities, 2)
            final_candidates_score = torch.mean(candidates_scores, 2)  
        # prediction from the final step
        elif self.reason_type == 1:
            final_supports_prob = supports_probabilities_list[-1]
            final_candidates_score = candidates_scores_list[-1]
        # prediction averaged from all the steps     
        elif self.reason_type == 2:
            supports_probabilities = torch.stack(supports_probabilities_list,2)
            candidates_scores = torch.stack(candidates_scores_list, 2)
            
            final_supports_prob = torch.mean(supports_probabilities, 2)
            final_candidates_score = torch.mean(candidates_scores, 2)
        return final_supports_prob, final_candidates_score
            
    def generate_mask(self, batch_size: int) -> torch.Tensor:
        if self.training:
            dropout_p = self.reason_dropout_p
        else:
            dropout_p = 0.0

        new_data = self.alpha.data.new_zeros(batch_size, self.num_step)
        new_data = (1-dropout_p) * (new_data.zero_() + 1)
        for i in range(new_data.size(0)):
            one = random.randint(0, new_data.size(1)-1)
            new_data[i][one] = 1
        mask = 1.0/(1 - dropout_p) * torch.bernoulli(new_data)
        mask.requires_grad = False
        return mask            
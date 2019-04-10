import torch
from torch.nn import Parameter
from allennlp.nn import util
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

@Seq2VecEncoder.register('self_attentive')
class SelfAttentive(Seq2VecEncoder):
    
    def __init__(self,
                 dim: int,
                ) -> None:
        super().__init__()
        self.weight = Parameter(torch.Tensor(dim,1))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, matrix: torch.Tensor, matrix_mask: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, dim) -> (batch_size, seq_len)
        similarity = torch.matmul(matrix, self.weight).squeeze(-1) 
        similarity =  util.masked_softmax(similarity, matrix_mask)
        return util.weighted_sum(matrix, similarity)
        
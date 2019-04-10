import torch
import torch.nn as nn
from allennlp.common.registrable import Registrable


class Decoder(nn.Module, Registrable):
    
    def forward(self,
               supports_vectors: torch.FloatTensor,
               query_vectors: torch.FloatTensor,
               candidates_vectors: torch.FloatTensor,
               supports_mask: torch.LongTensor = None):
        raise NotImplementedError
        

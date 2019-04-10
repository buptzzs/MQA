import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn


class EmbeddingLayer(nn.Module):
    """ Embedding Layer 
    
    Attributes:
        word_embedding: use Glove pretrained embedding vectors
        nGram_embeding: use CharNGram pretrained embedding vectors
    TODO:
        add Char CNN embedding
        add Bert Embedding
    """
    
    def __init__(self, word_vectors, charNGram_vectors):
        super(EmbeddingLayer, self).__init__()
        
        charNGram_vectors[1] = torch.zeros(100)
        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        self.nGram_embedding = nn.Embedding.from_pretrained(charNGram_vectors, freeze=True)
        
    def forward(self, glove_input, charNGram_input=None):
        '''
        Arguments:
            glove_input: shape of [n_sentence, n_word]
            charNgram_input: shape of [n_sentence, n_word]
            
        return:
            embedding: shape of [n_sentence, n_word, glove_dim + charNgram_dim]
        '''
        g_emb = self.word_embedding(glove_input)
        if charNGram_input is not None
            c_emb = self.nGram_embedding(charNGram_input)
            return torch.cat([g_emb, c_emb], dim=-1)
        return g_emb

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)
            output, hidden = self.rnns[i](output, hidden)
            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask=None):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        
        if mask is not None:
            att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GateLayer(nn.Module):
    def __init__(self, d_input, d_output):
        super(GateLayer, self).__init__()
        self.linear = nn.Linear(d_input, d_output)
        self.gate = nn.Linear(d_input, d_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.linear(input) * self.sigmoid(self.gate(input))
    
    
class SelfAttention(nn.Module):

    def __init__(self, n_input: int, attn_dimension: int = 64, dropout: float = 0.4) -> None:
        super().__init__()

        self.dropout = LockedDropout(dropout)
        self.n_input = n_input
        self.n_attn = attn_dimension
        self.ws1 = nn.Linear(n_input, self.n_attn, bias=False)
        self.ws2 = nn.Linear(self.n_attn, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self, init_range: float = 0.1) -> None:
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        # size: (bsz, sent_len, rep_dim)
        size = inputs.size()
        inputs = self.dropout(inputs)
        compressed_emb = inputs.contiguous().view(-1,size[-1])
        hbar = self.tanh(
            self.ws1(compressed_emb)
        )  # (bsz * sent_len, attention_dim)
        alphas = self.ws2(hbar)  # (bsz * sent_len, 1)
        alphas = alphas.view(size[:2]) # (bsz, sent_len)
        alphas = self.softmax(alphas)  # (bsz, sent_len)
        # (bsz, rep_dim)
        return torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)    
    
    
class CoAttention(nn.Module):
    
    def __init__(self, hidden_dims, att_type=0, dropout=0.2):
        super(CoAttention, self).__init__()
        self.dropout = LockedDropout(dropout)        
        self.G = nn.Linear(hidden_dims, hidden_dims, bias=True)
        self.att_type = att_type
    def forward(self, a1, a2):
        '''
            a1: n * L * d
            a2: n * K * d
        return:
            M1: n* L * d
            M2: n * K * d
            
        '''
        a1 = self.dropout(a1)
        a2 = self.dropout(a2)
        
        if self.att_type == 0:
            a2_ = self.G(a2)
            L = torch.bmm(a1, a2_.permute(0, 2, 1))
        elif self.att_type == 1:
            a1_ = F.relu(self.G(a1))
            a2_ = F.relu(self.G(a2))
            L = torch.bmm(a1_, a2_.permute(0, 2, 1))            
        else:
            L = torch.bmm(a1, a2.permute(0,2,1))
            
        A1 = torch.softmax(L, 2) # N, L , K
        A2 = torch.softmax(L, 1) 
        A2 = A2.permute(0,2,1) # N, K, L
        
        M_1 = torch.bmm(A1, a2) # N, L, d
        M_2 = torch.bmm(A2, a1) # N, K, d
        
        if self.att_type == 2:
            M_3 = torch.bmm(A1, M_2)
            M_1 = torch.cat([M_1,M_3], dim=-1)

        return M_1, M_2
        
class FusionLayer(nn.Module):
    
    def __init__(self, dim=100, dropout=0.2):
        super().__init__()
        self.dropout = LockedDropout(dropout)        
        self.linear = nn.Linear(dim*2, dim, bias=True)
        self.act = nn.ReLU()
        
    def forward(self, a1, a2):
        assert a1.size() == a2.size()
        
        mid = torch.cat([a1 - a2, a1 * a2], -1)
        return self.act(self.linear(self.dropout(mid)))
    
class PoolingLayer(nn.Module):
    '''
    pooling operation: max pooling and attentive pooling 
    '''
    
    def __init__(self, max_pooling=True, dim=None, dropout=0.2):            
        super(PoolingLayer, self).__init__()
        
        self.max_pooling = max_pooling
        if not max_pooling:
            self.dropout = LockedDropout(dropout)
            self.linear = nn.Linear(dim, 1, bias=True)
        
    def forward(self, x):
        '''
        x: n * L * d
        '''
        if self.max_pooling:
            out, _ = torch.max(x, 1)
            return out
        else:
            score = self.linear(self.dropout(x)) # n * L * 1
            alpha = torch.softmax(score, -1) 
            out = torch.bmm(x.permute(0,2,1), alpha) # n * d * 1
            out = out.squeeze()
            return out    
        
        
class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """
    def __init__(self, x_size, y_size, identity=False, dropout=0.2):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None          

    def forward(self, x, y):
        """
        x = batch * len * h1
        y = batch * h2
        """


        Wy = self.linear(y) if self.linear is not None else y  # batch * h1
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)  # batch * len
        return xWy        
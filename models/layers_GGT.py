import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# build a decoder layer with a multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model).cuda()
    
    def forward(self, x, mask=None):
        x2 = self.norm_1(x)
        t, avg_scores = self.attn_1(x2, x2, x2, mask)
        x = x + self.dropout_1(t)  # add and norm
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, avg_scores
    

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # calculate attention weights
        att_output, att_weights = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # # return attention weights for all the heads
        # att_w = []
        # for i in range(self.h):
        #     att_w.append(att_weights[:, i].detach())
        # att_weights = att_w
        
        # return single attention weights, averaged over the last two heads (in our case, they learned structures)
        att_weights = att_weights.detach()[:, -2:].sum(dim=1)/2  # average over some heads

        # # return single attention weights, averaged over all heads
        # att_weights = att_weights.detach().sum(dim=1)/self.h  # average over all heads
        
        # concatenate heads and put through final linear layer
        concat = att_output.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        
        output = self.out(concat)
        
        return output, att_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model).to(device="cuda:0")
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
    

def attention(q, k, v, d_k, mask=None, dropout=None):
    """
    :param q: queries, B x N_HEADS x seq_len x d_k
    :param k: keys, same dim as q
    :param v: values, same dim as q
    :param d_k: d_model/n_heads = 128/8 = 16
    :param mask: mask for padding and future steps in the scores!
    :param dropout: dropout layer if any
    :return: attention vector of shape B x N_HEADS x seq_len x d_k
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)  # Add N_HEADS dimension
        scores = scores.masked_fill(mask == 0, -1e9)  # zero out unwanted elements
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        print("Dropout in attention layer is note None!")
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output, scores


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def generate_mask_sequence(size, device="cuda:0"):
    """
    :param size: seq_len
    :param device: cuda or cpu
    :return: mask with future timesteps zero-valued. shape 1 x size x size
    """
    x = torch.ones((size, size), device=device)
    x = torch.triu(x, diagonal=1)
    return x.unsqueeze(0) == 0  # invert and convert to byte


def generate_mask_pad(length, shape, device="cuda:0"):
    batch_size = shape[0]
    seq_len = shape[1]
    x = torch.zeros((batch_size, seq_len), device=device)
    for i in range(batch_size):
        x[i, :length[i]] = 1
    
    return x.unsqueeze(-1) == 1  # convert to byte


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from models.layers_GGT import *
from utils.utils import sample_sigmoid


def init_weights(m):
    r"""
    Apply xavier initialization on Linear layers
    
    :param m: PyTorch layer
    """
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class DecoderMLP(nn.Module):
    r"""
    Simple MLP decoder that generates in one shot the adjacency and node feature matrices (A, X)
    
    Two MLP heads with one hidden layer output A and X.
    The soft adjacency matrix is made simmetric by summing the transpose.
    A simpler and more effective version of this decoder would only model the lower triangular adjacency matrix.
    """
    
    def __init__(self, max_n_nodes=10, bottleneck_encoder=900):
        super().__init__()
        
        self.max_n_nodes = max_n_nodes
        
        self.output_adj_1 = nn.Linear(bottleneck_encoder, 1600)  # b, 400
        self.output_adj_2 = nn.Linear(1600, max_n_nodes ** 2)  # b, 100
        
        self.output_coord_1 = nn.Linear(bottleneck_encoder, 1600)  # b, 400
        self.output_coord_2 = nn.Linear(1600, max_n_nodes * 2)  # b, 20
        
        self.apply(init_weights)
    
    def forward(self, x):
        output_adj = F.relu_(self.output_adj_1(x))
        output_adj = torch.sigmoid(self.output_adj_2(output_adj))
        output_adj = output_adj.reshape(x.shape[0], self.max_n_nodes, self.max_n_nodes)  # A
        output_adj = (output_adj + output_adj.permute(0, 2, 1)) / 2.  # make simmetric by averaging with transposed
        
        output_coord = F.relu_(self.output_coord_1(x))
        output_coord = torch.tanh(self.output_coord_2(output_coord))
        output_coord = output_coord.reshape(x.shape[0], self.max_n_nodes, 2)  # X
        
        return output_adj, output_coord


class DecoderGRU(nn.Module):
    r"""
    Simple RNN decoder that generates recurrently the adjacency and node feature matrices (A, X)
    
    First, a linear layer compresses the input into a vector of size d_model.
    A single-layer GRU recurrently outputs hidden representations h_t of the graph at different timesteps
    Two MLP heads with one hidden layer take as input the representation h_t and output a_t and x_t.
    """
    
    def __init__(self, d_model, n_hidden=2, dropout=0., adj_size=30, coord_size=2, visual_size=256):
        super().__init__()
        self.input_linear = nn.Linear(coord_size + adj_size + visual_size, d_model)
        self.lstm = torch.nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_hidden, batch_first=True,
                                 dropout=dropout)
        
        self.output_adj_1 = nn.Linear(d_model, d_model // 2)
        self.output_adj_2 = nn.Linear(d_model // 2, adj_size)
        
        self.output_coord_1 = nn.Linear(d_model, d_model // 2)
        self.output_coord_2 = nn.Linear(d_model // 2, coord_size)
        
        self.h = None  # hidden state of the GRU cell
        self.apply(init_weights)
    
    def forward(self, x, input_len):
        seq_len = x.shape[1]
        x = self.input_linear(x)
        x = pack_padded_sequence(x, input_len, batch_first=True)
        output_packed, self.h = self.lstm(x, self.h)
        output, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=seq_len)  # h_t == output
        
        output_adj = F.relu_(self.output_adj_1(output))
        output_adj = torch.sigmoid(self.output_adj_2(output_adj))  # a_t == output_adj
        
        output_coord = F.relu_(self.output_coord_1(output))
        output_coord = torch.tanh(self.output_coord_2(output_coord))  # x_t == output_coord
        return output_adj, output_coord
    
    def reset_hidden(self):
        r"""
        Reset hidden states
        """
        self.h = None


class DecoderGRUAtt(nn.Module):
    r"""
    Simple RNN decoder with single-head self-attention that generates recurrently the adjacency and node feature
    matrices (A, X)

    First, a linear layer compresses the input into a vector of size d_model.
    A single-layer GRU recurrently outputs hidden representations h_t of the graph at different timesteps
    A single-head self-attention layer is applied on top of the recurrent component
    Two MLP heads with one hidden layer take as input the representation h_t and output a_t and x_t.
    """
    
    def __init__(self, d_model, n_hidden=2, dropout=0., adj_size=30, coord_size=2, visual_size=256, heads=1,
                 concat=False):
        super().__init__()
        self.concat = concat  # if true, concatenate instead of adding in the skip connection after the self-attention
        self.input_linear = nn.Linear(coord_size + adj_size + visual_size, d_model)
        self.lstm = torch.nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_hidden, batch_first=True,
                                 dropout=dropout)
        
        self.attention = MultiHeadAttention(heads, d_model, dropout=0.)
        
        if self.concat:
            self.output_adj_1 = nn.Linear(d_model * 2, d_model // 2)
            self.output_coord_1 = nn.Linear(d_model * 2, d_model // 2)
        else:
            self.output_adj_1 = nn.Linear(d_model, d_model // 2)
            self.output_coord_1 = nn.Linear(d_model, d_model // 2)
        self.output_adj_2 = nn.Linear(d_model // 2, adj_size)
        self.output_coord_2 = nn.Linear(d_model // 2, coord_size)
        
        self.h = None  # hidden state of the GRU cell
        self.apply(init_weights)
    
    def forward(self, x, input_len, get_att_weights=False):
        seq_len = x.shape[1]
        x = self.input_linear(x)
        x = pack_padded_sequence(x, input_len, batch_first=True)
        output_packed, self.h = self.lstm(x, self.h)
        output, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=seq_len)  # h_t == output
        
        mask_sequence = generate_mask_sequence(size=output.shape[1])  # mask the future timesteps in the sequences
        mask_pad = generate_mask_pad(input_len, output.shape)  # mask the padding elements in the end of the sequences
        mask = mask_sequence & mask_pad  # mask to be applied in the self-attention
        
        output_att, att_weights = self.attention(output, output, output, mask=mask)  # self-attention
        
        if self.concat:
            output = torch.cat([output, output_att], dim=2)
        else:
            output = output + output_att
        
        output_adj = F.relu_(self.output_adj_1(output))
        output_adj = torch.sigmoid(self.output_adj_2(output_adj))  # a_t == output_adj
        
        output_coord = F.relu_(self.output_coord_1(output))
        output_coord = torch.tanh(self.output_coord_2(output_coord))  # x_t == output_coord
        
        # optionally, return the attention weights for further analysis
        if get_att_weights:
            return output_adj, output_coord, att_weights
        return output_adj, output_coord
    
    def reset_hidden(self):
        r"""
        Reset hidden states
        """
        self.h = None


class DecoderGraphRNN(nn.Module):
    r"""
    A GraphRNN decoder extended for labeled graph generation, that generates recurrently the adjacency and node feature
    matrices (A, X).

    See original implementation from You et al. (2018). On top of it, we add the node features head taking as input
    the representation produced by the node-level (or graph-level) RNN.
    """
    def __init__(self, d_model, n_hidden=2, dropout=0., adj_size=30, coord_size=2, visual_size=256, hidden_size=16):
        super().__init__()
        self.input_linear = nn.Linear(coord_size + adj_size + visual_size, d_model)
        self.lstm = torch.nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_hidden, batch_first=True,
                                 dropout=dropout)
        self.output_hidden = nn.Linear(d_model, hidden_size)
        
        self.input_adj = nn.Linear(1, hidden_size // 2)
        self.adj_lstm = torch.nn.GRU(input_size=hidden_size // 2, hidden_size=hidden_size, num_layers=4,
                                     batch_first=True,
                                     dropout=dropout)
        self.output_adj_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_adj_2 = nn.Linear(hidden_size // 2, 1)
        
        self.output_coord_1 = nn.Linear(d_model, d_model // 2)
        self.output_coord_2 = nn.Linear(d_model // 2, coord_size)
        
        self.h = None  # hidden state of the GRU cell node-wise
        self.h_adj = None  # hidden state of the GRU cell edge-wise
        
        self.apply(init_weights)
    
    def forward(self, x, y_adj, input_len):
        seq_len = x.shape[1]
        x = self.input_linear(x)
        x = pack_padded_sequence(x, input_len, batch_first=True)
        output_packed, self.h = self.lstm(x, self.h)
        output, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=seq_len)
        # generate node-level representation
        
        h_adj = self.output_hidden(output)
        h_adj = pack_padded_sequence(h_adj, input_len, batch_first=True).data
        h_adj = h_adj.unsqueeze(0)
        hidden_null = torch.zeros(4 - 1, h_adj.size(1), h_adj.size(2)).to("cuda:0")
        # the node-level representation initializes the first hidden state in the edge-wise RNN
        self.h_adj = torch.cat((h_adj, hidden_null), dim=0)
        
        # creates the input of the edge-wise RNN with teacher-forcing, prepending a zero-vector
        x_adj = torch.cat((torch.zeros(y_adj.size(0), y_adj.size(1), 1).to("cuda:0"), y_adj[:, :, 0:-1]), dim=2)
        x_adj = pack_padded_sequence(x_adj, input_len, batch_first=True)
        x_adj, pad_container = x_adj[0], x_adj[1]
        x_adj = x_adj.unsqueeze(-1)
        
        # generate edge-wise representation
        x_adj = self.input_adj(x_adj)
        output_adj, self.h_adj = self.adj_lstm(x_adj, self.h_adj)
        output_adj = PackedSequence(output_adj, pad_container, None, None)
        output_adj = pad_packed_sequence(output_adj, batch_first=True)[0]  # edge-wise representation
        
        output_adj = F.relu_(self.output_adj_1(output_adj))
        output_adj = torch.sigmoid(self.output_adj_2(output_adj))
        output_adj = output_adj.squeeze(-1)  # a_t == output_adj
        
        output_coord = F.relu_(self.output_coord_1(output))
        output_coord = torch.tanh(self.output_coord_2(output_coord))  # x_t == output_coord
        
        return output_adj, output_coord
    
    def generate(self, x, args=None):
        r"""
        Generates the node features x_t and adjacency vector a_t for the next timestep t using GraphRNN
        
        :param x: input sequence from the encoder
        :param args: parsed arguments
        :return: a_t, x_t
        """
        max_prev_node = 4
        
        x = self.input_linear(x)
        output, self.h = self.lstm(x, self.h)  # generates node-wise representation h_t
        
        h_adj = self.output_hidden(output)
        hidden_null = torch.zeros(4 - 1, h_adj.size(1), h_adj.size(2)).to("cuda:0")
        self.h_adj = torch.cat((h_adj, hidden_null), dim=0)  # initializes the edge-wise hidden vectors
        
        output_adj = torch.zeros(1, 1, max_prev_node).to("cuda:0")  # a_(t,0)
        x_adj = torch.zeros(1, 1, 1).to("cuda:0")  # x_(t,0)
        
        for i in range(max_prev_node):
            x_adj = self.input_adj(x_adj)
            x_adj, self.h_adj = self.adj_lstm(x_adj, self.h_adj)
            
            x_adj = F.relu_(self.output_adj_1(x_adj))
            x_adj = torch.sigmoid(self.output_adj_2(x_adj))
            x_adj = sample_sigmoid(args, x_adj)
            output_adj[:, :, i:i + 1] = x_adj[:]  # a_(t,i)
        
        output_coord = F.relu_(self.output_coord_1(output))
        output_coord = torch.tanh(self.output_coord_2(output_coord))  # x_t
        
        return output_adj, output_coord
    
    def reset_hidden(self):
        r"""
        Reset hidden states
        """
        self.h = None
        self.h_adj = None


class DecoderGraphRNNAtt(nn.Module):
    r"""
    A GraphRNN decoder extended for labeled graph generation with single-head self-attention on top, that generates
    recurrently the adjacency and node feature matrices (A, X).

    See original implementation from You et al. (2018). On top of it, we add the node features head taking as input
    the representation produced by the node-level (or graph-level) RNN, and a self-attention layer.
    """
    def __init__(self, d_model, n_hidden=2, dropout=0., adj_size=30, coord_size=2, visual_size=256, concat=False,
                 heads=1, hidden_size=16):
        super().__init__()
        self.input_linear = nn.Linear(coord_size + adj_size + visual_size, d_model)
        self.lstm = torch.nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_hidden, batch_first=True,
                                 dropout=dropout)
        
        self.concat = concat
        self.attention = MultiHeadAttention(heads, d_model, dropout=0.)
        
        if self.concat:
            self.output_hidden = nn.Linear(d_model * 2, hidden_size)
            self.output_coord_1 = nn.Linear(d_model * 2, d_model // 2)
        else:
            self.output_hidden = nn.Linear(d_model, hidden_size)
            self.output_coord_1 = nn.Linear(d_model, d_model // 2)
        
        self.input_adj = nn.Linear(1, hidden_size // 2)
        self.adj_lstm = torch.nn.GRU(input_size=hidden_size // 2, hidden_size=hidden_size, num_layers=4,
                                     batch_first=True,
                                     dropout=dropout)
        self.output_adj_1 = nn.Linear(hidden_size, hidden_size // 2)
        self.output_adj_2 = nn.Linear(hidden_size // 2, 1)
        
        self.output_coord_2 = nn.Linear(d_model // 2, coord_size)
        
        self.h = None  # hidden state of the GRU cell node-wise
        self.h_adj = None  # hidden state of the GRU cell edge-wise
        self.history = None  # history of generate node-wise vector representations, used for self-attention
        
        self.apply(init_weights)
    
    def forward(self, x, y_adj, input_len):
        seq_len = x.shape[1]
        x = self.input_linear(x)
        # generates node-wise representation h_t
        x = pack_padded_sequence(x, input_len, batch_first=True)
        output_packed, self.h = self.lstm(x, self.h)
        output, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=seq_len)

        # generate self-attention masks
        mask_sequence = generate_mask_sequence(size=output.shape[1])
        mask_pad = generate_mask_pad(input_len, output.shape)
        mask = mask_sequence & mask_pad
        
        # apply self-attention
        output_att, _ = self.attention(output, output, output, mask=mask)
        
        # skip-connection
        if self.concat:
            output = torch.cat([output, output_att], dim=2)
        else:
            output = output + output_att
        
        h_adj = self.output_hidden(output)
        h_adj = pack_padded_sequence(h_adj, input_len, batch_first=True).data
        h_adj = h_adj.unsqueeze(0)
        hidden_null = torch.zeros(4 - 1, h_adj.size(1), h_adj.size(2)).to("cuda:0")
        self.h_adj = torch.cat((h_adj, hidden_null), dim=0)  # initializes the edge-wise hidden vectors
        
        x_adj = torch.cat((torch.zeros(y_adj.size(0), y_adj.size(1), 1).to("cuda:0"), y_adj[:, :, 0:-1]), dim=2)
        x_adj = pack_padded_sequence(x_adj, input_len, batch_first=True)
        x_adj, pad_container = x_adj[0], x_adj[1]
        x_adj = x_adj.unsqueeze(-1)
        
        x_adj = self.input_adj(x_adj)
        output_adj, self.h_adj = self.adj_lstm(x_adj, self.h_adj)  # get edge-wise representation
        output_adj = PackedSequence(output_adj, pad_container, None, None)
        output_adj = pad_packed_sequence(output_adj, batch_first=True)[0]
        
        output_adj = F.relu_(self.output_adj_1(output_adj))
        output_adj = torch.sigmoid(self.output_adj_2(output_adj))
        output_adj = output_adj.squeeze(-1)  # a_t
        
        output_coord = F.relu_(self.output_coord_1(output))
        output_coord = torch.tanh(self.output_coord_2(output_coord))  # x_t
        
        return output_adj, output_coord
    
    def generate(self, x, args=None):
        r"""
        Generates the node features x_t and adjacency vector a_t for the next timestep t using GraphRNN
        
        :param x: input sequence from the encoder
        :param args: parsed arguments
        :return: a_t, x_t
        """
        max_prev_node = 4
        
        x = self.input_linear(x)
        output, self.h = self.lstm(x, self.h)  # generates node-wise representation h_t
        
        # add the current node-wise representation to the history for this datapoint, for usage in the self-attention
        self.update_history(output)
        input_att = self.history.clone()
        
        # generate self-attention masks
        mask_sequence = generate_mask_sequence(size=input_att.shape[1])
        mask_pad = generate_mask_pad([self.history.shape[1]], input_att.shape)
        mask = mask_sequence & mask_pad
        
        # apply self-attention
        output_att, _ = self.attention(input_att, input_att, input_att, mask=mask)
        output_att = output_att[:, -1:, :]
        
        # skip-connection
        if self.concat:
            output = torch.cat([output, output_att], dim=2)
        else:
            output = output + output_att
        
        h_adj = self.output_hidden(output)
        hidden_null = torch.zeros(4 - 1, h_adj.size(1), h_adj.size(2)).to("cuda:0")
        self.h_adj = torch.cat((h_adj, hidden_null), dim=0)  # initializes the edge-wise hidden vectors
        
        output_adj = torch.zeros(1, 1, max_prev_node).to("cuda:0")  # a_(t,0)
        x_adj = torch.zeros(1, 1, 1).to("cuda:0")  # x_(t,0)
        
        for i in range(max_prev_node):
            x_adj = self.input_adj(x_adj)
            x_adj, self.h_adj = self.adj_lstm(x_adj, self.h_adj)
            
            x_adj = F.relu_(self.output_adj_1(x_adj))
            x_adj = torch.sigmoid(self.output_adj_2(x_adj))
            x_adj = sample_sigmoid(args, x_adj)
            output_adj[:, :, i:i + 1] = x_adj[:]
        
        output_coord = F.relu_(self.output_coord_1(output))
        output_coord = torch.tanh(self.output_coord_2(output_coord))
        return output_adj, output_coord
    
    def update_history(self, output):
        r"""
        Update the history of node-wise representations for the current datapoint, h_(0:t)
        
        :param output: node-wise representation at the current timestep h_t
        """
        if self.history is None:
            self.history = output.clone()
        else:
            self.history = torch.cat([self.history, output], dim=1)
    
    def reset_hidden(self):
        r"""
        Reset hidden states
        """
        self.h = None
        self.h_adj = None
        self.history = None


class DecoderGGT(nn.Module):
    r"""
    The proposed Generative Graph Transformer (GGT) decoder based on multi-head self-attention layers like in
    Transformer decoders (Vaswani et al. 2017). Generates recurrently the adjacency and node feature matrices (A, X).

    First, transform the input from the encoder to a vector of size d_model
    Positionally encode the input
    Pass through N decoding blocks with Multi-Head self-attention, MLPs, skip-connections and layer normalizations.
    Finally, pass to two heads to emit the soft-adjacency vector a_t and node features x_t
    """
    def __init__(self, d_model, N=12, dropout=0., adj_size=30, coord_size=2, visual_size=256, heads=8, max_n_nodes=21,
                 use_pe=True):
        super().__init__()
        self.N = N
        self.use_pe = use_pe
        if self.use_pe:
            self.pe = PositionalEncoder(d_model, max_seq_len=max_n_nodes)
        self.input_linear = nn.Linear(coord_size + adj_size + visual_size, d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout=dropout), N)
        self.norm = nn.LayerNorm(d_model)
        
        self.output_adj_1 = nn.Linear(d_model, d_model // 2)
        self.output_adj_2 = nn.Linear(d_model // 2, adj_size)
        self.output_coord_1 = nn.Linear(d_model, d_model // 2)
        self.output_coord_2 = nn.Linear(d_model // 2, coord_size)
    
    def forward(self, x, input_len, get_att_weights=False):
        att_weights = []  # to store self-attention weights for studying behavior and plotting
        
        # mask for self-attention
        mask_sequence = generate_mask_sequence(size=x.shape[1])
        mask_pad = generate_mask_pad(input_len, x.shape)
        mask = mask_sequence & mask_pad
        
        # transform to d_model size
        x = F.relu_(self.input_linear(x))
        
        # positional encoding
        if self.use_pe:
            x = self.pe(x)
        
        # pass through the N decoding layers, optionally store self-attention weight matrices
        for i in range(self.N):
            x, avg_scores = self.layers[i](x, mask)
            if get_att_weights and i == 8:  # i in {0,1,2,3,4,5,6,7,8,9,10,11,12}:
                att_weights.append(avg_scores)
        
        output = x
        output_adj = F.relu(self.output_adj_1(output))
        output_adj = torch.sigmoid(self.output_adj_2(output_adj))  # a_t
        
        output_coord = F.relu_(self.output_coord_1(output))
        output_coord = torch.tanh(self.output_coord_2(output_coord))  # x_t
        
        if get_att_weights:
            return output_adj, output_coord, att_weights
        return output_adj, output_coord
    
    def reset_hidden(self):
        r"""
        There are no hidden states in GGT, we leave it for compatibility with the general training function
        """
        pass

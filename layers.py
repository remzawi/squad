"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop = nn.Dropout(drop_prob)
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = self.drop(emb)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb
    
class CharEmbedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        char_vec (torch.Tensor): charracted embedingd file.
        word_len (torch.Tensor): Max word len
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, char_vec, word_len, hidden_size, drop_prob):
        super(CharEmbedding, self).__init__()
        self.drop = nn.Dropout(drop_prob)
        self.embed = nn.Embedding.from_pretrained(char_vec, freeze=False)
        nn.init.uniform_(self.embed.weight, -0.001, 0.001)
        self.char_cnn = nn.Conv2d(word_len, hidden_size, (1, 5))
        

    def forward(self, x):
        B, S, W = x.size()
        emb = self.embed(x)   # (batch_size, seq_len, word_len, embed_size)
        emb = self.drop(emb)
        emb = self.char_cnn(emb.permute(0, 2, 1, 3)) # (batch_size , hidden_size, seq_len, conv_size)
        emb = F.relu(emb)
        emb = torch.max(emb.permute(0, 2, 1, 3), -1)[0] # (batch_size, seq_len, hidden_size)

        return emb
    
class EmbeddingWithChar(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, char_vec, word_len, drop_prob, char_prop=0.2):
        super(EmbeddingWithChar, self).__init__()
        self.drop = nn.Dropout(drop_prob)
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        char_size = int(hidden_size*char_prop)
        word_size = hidden_size - char_size
        self.char_embed = CharEmbedding(char_vec, word_len, char_size, drop_prob)
        self.proj = nn.Linear(word_vectors.size(1), word_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x, char_x):
        word_emb = self.word_embed(x)   # (batch_size, seq_len, embed_size)
        word_emb = self.drop(word_emb)
        word_emb = self.proj(word_emb)  # (batch_size, seq_len, word_size)
        char_emb = self.char_embed(char_x) # (batch_size, seq_len, char_size)
        emb = torch.cat([word_emb, char_emb], dim = -1) # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop = nn.Dropout(drop_prob)
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = self.drop(c)  # (bs, c_len, hid_size)
        q = self.drop(q)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    
    
class PositionalEncoding(nn.Module):
    """
    Implements a non-trainable positional encoder based on sin and cos functions
    Based on the implementation originaly proposed
    Adapted to Pytorch based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, hidden_size, drop_prob, para_limit):
        super(PositionalEncoding, self).__init__()
        self.drop = nn.Dropout(drop_prob)
        pe = torch.zeros(para_limit, hidden_size)
        position = torch.arange(0, para_limit, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.drop(x)
        return x
    
    
class DWConv(nn.Module):
    def __init__(self, nin, nout, kernel_size):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size, padding=kernel_size//2, groups=nin)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x.permute(0,2,1))
        out = self.pointwise(out)
        return F.relu(out.permute(0,2,1))
    
class ConvBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, drop_prob):
        super(ConvBlock, self).__init__()
        self.conv = DWConv(input_size, hidden_size, kernel_size)
        self.drop = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(input_size)
    def forward(self, x):
        norm = self.norm(x)
        conv = self.conv(norm)
        return self.drop(x + conv)
    
class SelfAttention(nn.Module):
    """
    Self attention adapted from homework 5
    """
    def __init__(self, hidden_size, n_head, drop_prob):
        super(SelfAttention, self).__init__()
        assert hidden_size % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # regularization
        self.attn_drop = nn.Dropout(drop_prob)
        #self.resid_drop = nn.Dropout(drop_prob)
        # output projection
        #self.proj = nn.Linear(hidden_size, hidden_size)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # no need for masking
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        #y = self.resid_drop(self.proj(y))
        return y
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, n_head, drop_prob, att_drop_prob = None):
        super(SelfAttentionBlock, self).__init__()
        if att_drop_prob is None:
            att_drop_prob = drop_prob
        self.att = SelfAttention(hidden_size, n_head, att_drop_prob)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(drop_prob)
        
    def forward(self, x):
        norm = self.norm(x)
        att = self.att(x)
        return self.drop(x+att)
    
class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_size, drop_prob):
        super(FeedForwardBlock, self).__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(drop_prob)
    def forward(self, x):
        norm = self.norm(x)
        proj = F.relu(self.proj(norm))
        return self.drop(x + proj)
    
class Resizer(nn.Module):
    def __init__(self, input_size, output_size, drop_prob):
        super(Resizer, self).__init__()
        self.conv = DWConv(input_size, output_size, 1)
        self.drop = nn.Dropout(drop_prob)
    def forward(self, x):
        out = self.conv(x)
        return self.drop(out)
    
class EncoderBlock(nn.Module):
    def __init__(self, enc_size, para_limit, n_conv, kernel_size, drop_prob, n_head = 8, att_drop_prob = None, final_prob = 0.9):
        super(EncoderBlock, self).__init__()
        self.pos = PositionalEncoding(enc_size, drop_prob, para_limit)
        self.convs = nn.ModuleList([ConvBlock(enc_size, enc_size, kernel_size, drop_prob) for i in range(n_conv)])
        self.att = SelfAttentionBlock(enc_size, n_head, drop_prob, att_drop_prob)
        self.ff = FeedForwardBlock(enc_size, drop_prob)
        self.n_layers = n_conv + 2
        self.final_prob = final_prob
        self.drop = nn.Dropout(drop_prob)
        self.do_depth = final_prob < 1
    def forward(self, x):
        out = self.pos(x) 
        if self.do_depth:
            for i, conv in enumerate(self.convs):
                out = self.drop_layer(out, conv, 1 - (i+1)/self.n_layers*(1-self.final_prob))
            #out = self.convs(out)
            out = self.drop_layer(out, self.att, 1 - (self.n_layers - 1)/self.n_layers*(1-self.final_prob))
            out = self.drop_layer(out, self.ff, self.final_prob)
            return out
        else:
            for i, conv in enumerate(self.convs):
                out = conv(out)

            #out = self.convs(out)
            out = self.att(out)
            out = self.ff(out)
            #out = self.drop_layer(out, self.att, 1 - (self.n_layers - 1)/self.n_layers*(1-self.final_prob))
            #out = self.drop_layer(out, self.ff, self.final_prob)
            return out
            
    def drop_layer(self, x, layer, prob):
        if self.training:
            if torch.rand(1) < prob:
                return layer(x)
            else:
                return x
        else:
            return layer(x)
            
    
class StackedEncoderBlocks(nn.Module):
    def __init__(self, n_blocks, hidden_size, para_limit, n_conv, kernel_size, drop_prob, n_head = 8, att_drop_prob = None):
        super(StackedEncoderBlocks, self).__init__()
        self.encoders = nn.ModuleList([EncoderBlock(hidden_size, para_limit, n_conv, kernel_size, drop_prob, n_head = 8, att_drop_prob = None)
                                       for i in range(n_blocks)])
    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x
    
class OutputBlock(nn.Module):
    def __init__(self, hidden_size):
        super(OutputBlock, self).__init__() 
        self.proj = nn.Linear(2 * hidden_size, 1, bias=False)
        
    def forward(self, x1, x2, mask):
        x = torch.cat([x1, x2], dim = -1)
        proj = self.proj(x)
        log_p = masked_softmax(proj.squeeze(), mask, log_softmax=True)
        return log_p
        
           

    
    

        



        
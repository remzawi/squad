"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
    
class BiDAFChar(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vec, word_len, hidden_size, drop_prob=0.):
        super(BiDAFChar, self).__init__()
        self.emb = layers.EmbeddingWithChar(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    char_vec = char_vec,
                                    word_len = word_len,
                                    drop_prob = drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)
        

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANet(nn.Module):
    def __init__(self, word_vectors, char_vec, word_len,  emb_size, enc_size=128, 
                 drop_prob=0.1, n_head=8, LN_train=True, DP_residual=False,
                 mask_pos=False,two_pos=False,total_prob=True,final_prob=0.9):
        super(QANet, self).__init__()
        self.emb = layers.EmbeddingWithChar(word_vectors=word_vectors,
                                    hidden_size=emb_size,
                                    char_vec = char_vec,
                                    word_len = word_len,
                                    drop_prob = drop_prob,
                                    char_prop=0.4,
                                    hwy_drop=drop_prob,
                                    char_dim=200)
        
        self.emb_resize = layers.Resizer(input_size=emb_size,
                                         output_size=enc_size,
                                         kernel_size=1,
                                         drop_prob=0,
                                         bias=True)
        
        self.emb_enc = layers.EncoderBlock(enc_size=enc_size,
                                           para_limit=1000,
                                           n_conv=4,
                                           kernel_size=7,
                                           drop_prob=drop_prob,
                                           n_head=n_head,
                                           att_drop_prob=drop_prob,
                                           final_prob=final_prob, 
                                           LN_train=LN_train,
                                           DP_residual=DP_residual,
                                           mask_pos=mask_pos,
                                           two_pos=two_pos)
        
        self.att = layers.BiDAFAttention(hidden_size=enc_size,
                                         drop_prob=0)
        
        self.att_resize = layers.Resizer(input_size=4*enc_size,
                                         output_size=enc_size,
                                         kernel_size=1,
                                         drop_prob=0,
                                         bias=True)
        
        self.model_enc = layers.StackedEncoderBlocks(n_blocks=7,
                                                     hidden_size=enc_size,
                                                     para_limit=1000,
                                                     n_conv=2,
                                                     kernel_size=5,
                                                     drop_prob=drop_prob,
                                                     n_head=n_head,
                                                     att_drop_prob=drop_prob,
                                                     final_prob=final_prob, 
                                                     LN_train=LN_train,
                                                     DP_residual=DP_residual,
                                                     mask_pos=mask_pos,
                                                     two_pos=two_pos,
                                                     total_prob=total_prob)
        
        self.out_beg = layers.OutputBlock(enc_size)
        
        self.out_end = layers.OutputBlock(enc_size)
        
        self.drop = nn.Dropout(drop_prob)

        
        
        
    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, emb_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, emb_size)
        
        c_res_emb = self.emb_resize(c_emb)  # (batch_size, c_len, enc_size)
        q_res_emb = self.emb_resize(q_emb)  # (batch_size, q_len, enc_size)
        
        c_enc = self.emb_enc(c_res_emb, c_mask)    # (batch_size, c_len, enc_size)
        q_enc = self.emb_enc(q_res_emb, q_mask)    # (batch_size, q_len, enc_size)
        
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * enc_size)
        
        att_res = self.att_resize(att)  # (batch_size, c_len, enc_size)
        
        out1 = self.model_enc(att_res, c_mask)  # (batch_size, c_len, enc_size)
        out2 = self.model_enc(out1, c_mask) # (batch_size, c_len, enc_size)
        out3 = self.model_enc(out2, c_mask) # (batch_size, c_len, enc_size)
        
        log_p1 = self.out_beg(out1, out2, c_mask) # (batch_size, c_len)
        log_p2 = self.out_end(out1, out3, c_mask) # (batch_size, c_len)
        
        return log_p1, log_p2


        
class QANet2(nn.Module):
    def __init__(self, word_vectors, char_vec, word_len,  emb_size, enc_size=128, 
                 drop_prob=0.1, n_head=8, LN_train=True, DP_residual=False,
                 mask_pos=False,two_pos=False,rel=False,total_prob=True,final_prob=0.9,freeze=True):
        super(QANet2, self).__init__()
        self.emb = layers.EmbeddingWithChar(word_vectors=word_vectors,
                                    hidden_size=emb_size,
                                    char_vec = char_vec,
                                    word_len = word_len,
                                    drop_prob = drop_prob,
                                    char_prop=0.4,
                                    hwy_drop=drop_prob,
                                    char_dim=200,
                                    bias = False,
                                    freeze=freeze,
                                    act='relu')
        
        self.emb_resize = layers.Resizer(input_size=emb_size,
                                         output_size=enc_size,
                                         kernel_size=1,
                                         drop_prob=0,
                                         bias=True)
        
        
        
        self.emb_enc = layers.EncoderBlock(enc_size=enc_size,
                                           para_limit=1000,
                                           n_conv=4,
                                           kernel_size=7,
                                           drop_prob=drop_prob,
                                           n_head=n_head,
                                           att_drop_prob=drop_prob,
                                           final_prob=final_prob, 
                                           LN_train=LN_train,
                                           DP_residual=DP_residual,
                                           mask_pos=mask_pos,
                                           two_pos=two_pos,
                                           rel=rel,
                                           act='gelu',                                           
                                           pos_emb=True,
                                           from_pretrained=False,
                                           freeze_pos=False)
        
        self.att = layers.BiDAFAttention(hidden_size=enc_size,
                                         drop_prob=drop_prob)
        
        self.gelu = layers.GeluBlock(hidden_size=4*enc_size,
                                     drop_prob=drop_prob)
        
        self.att_resize = layers.Resizer(input_size=4*enc_size,
                                         output_size=enc_size,
                                         kernel_size=1,
                                         drop_prob=0,
                                         bias=True)
        
        self.model_enc = layers.StackedEncoderBlocks(n_blocks=7,
                                                     hidden_size=enc_size,
                                                     para_limit=1000,
                                                     n_conv=2,
                                                     kernel_size=5,
                                                     drop_prob=drop_prob,
                                                     n_head=n_head,
                                                     att_drop_prob=drop_prob,
                                                     final_prob=final_prob, 
                                                     LN_train=LN_train,
                                                     DP_residual=DP_residual,
                                                     mask_pos=mask_pos,
                                                     two_pos=two_pos,
                                                     rel=rel,
                                                     total_prob=total_prob,
                                                     act='gelu')
        
        self.out_beg = layers.OutputBlock(enc_size)
        
        self.out_end = layers.OutputBlock(enc_size)
        
        self.drop = nn.Dropout(drop_prob)

        
        
        
    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, emb_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, emb_size)
        
        c_res_emb = self.emb_resize(c_emb)  # (batch_size, c_len, enc_size)
        q_res_emb = self.emb_resize(q_emb)  # (batch_size, q_len, enc_size)
        
        c_enc = self.emb_enc(c_res_emb, c_mask)    # (batch_size, c_len, enc_size)
        q_enc = self.emb_enc(q_res_emb, q_mask)    # (batch_size, q_len, enc_size)
        
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * enc_size)
        
        att = self.gelu(att)
        
        att_res = self.att_resize(att)  # (batch_size, c_len, enc_size)
        
        out1 = self.model_enc(att_res, c_mask)  # (batch_size, c_len, enc_size)
        out2 = self.model_enc(out1, c_mask) # (batch_size, c_len, enc_size)
        out3 = self.model_enc(out2, c_mask) # (batch_size, c_len, enc_size)
        
        log_p1 = self.out_beg(out1, out2, c_mask) # (batch_size, c_len)
        log_p2 = self.out_end(out1, out3, c_mask) # (batch_size, c_len)
        
        return log_p1, log_p2
    
class QANet3(nn.Module):
    def __init__(self, word_vectors, char_vec, word_len,  emb_size, enc_size=128, 
                 drop_prob=0.1, n_head=8, LN_train=True, DP_residual=False,
                 mask_pos=False,two_pos=False,rel=False,total_prob=True,final_prob=1.0,freeze=True):
        super(QANet3, self).__init__()
        self.emb = layers.EmbeddingWithChar(word_vectors=word_vectors,
                                    hidden_size=emb_size,
                                    char_vec = char_vec,
                                    word_len = word_len,
                                    drop_prob = drop_prob,
                                    char_prop=0.4,
                                    hwy_drop=drop_prob,
                                    char_dim=200,
                                    bias = True,
                                    freeze=freeze,
                                    act='gelu')
        
        self.emb_resize = layers.Resizer(input_size=emb_size,
                                         output_size=enc_size,
                                         kernel_size=1,
                                         drop_prob=0,
                                         bias=True)
        
        self.pos_emb = layers.PosEmbeddings(hidden_size=enc_size,
                                            drop_prob=0,
                                            para_limit=1000,
                                            scale=False,
                                            from_pretrained=True,
                                            freeze = True)
        
        self.emb_enc = layers.EncoderBlock3(enc_size=enc_size,
                                           para_limit=1000,
                                           n_conv=4,
                                           kernel_size=7,
                                           drop_prob=drop_prob,
                                           n_head=n_head,
                                           att_drop_prob=drop_prob,
                                           final_prob=final_prob, 
                                           LN_train=LN_train,
                                           DP_residual=DP_residual,
                                           mask_pos=mask_pos,
                                           two_pos=two_pos,
                                           rel=rel,
                                           act='gelu')
        
        self.att = layers.BiDAFAttention(hidden_size=enc_size,
                                         drop_prob=drop_prob)
        
        self.gelu = layers.HighwayEncoder(num_layers=1,
                                          hidden_size=4 * enc_size,
                                          drop_prob=drop_prob,
                                          act='gelu')
        
        self.att_resize = layers.Resizer(input_size=4*enc_size,
                                         output_size=enc_size,
                                         kernel_size=1,
                                         drop_prob=0,
                                         bias=True)
        
        self.model_enc = layers.StackedEncoderBlocks(n_blocks=7,
                                                     hidden_size=enc_size,
                                                     para_limit=1000,
                                                     n_conv=2,
                                                     kernel_size=5,
                                                     drop_prob=drop_prob,
                                                     n_head=n_head,
                                                     att_drop_prob=drop_prob,
                                                     final_prob=final_prob, 
                                                     LN_train=LN_train,
                                                     DP_residual=DP_residual,
                                                     mask_pos=mask_pos,
                                                     two_pos=two_pos,
                                                     rel=rel,
                                                     total_prob=total_prob,
                                                     act='gelu')
        
        self.out_beg = layers.OutputBlock(enc_size)
        
        self.out_end = layers.OutputBlock(enc_size)
        
        self.drop = nn.Dropout(drop_prob)

        
        
        
    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, emb_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, emb_size)
        
        c_res_emb = self.emb_resize(c_emb)  # (batch_size, c_len, enc_size)
        q_res_emb = self.emb_resize(q_emb)  # (batch_size, q_len, enc_size)
        
        c_enc = self.emb_enc(c_res_emb, c_mask, embeddings = self.pos_emb)    # (batch_size, c_len, enc_size)
        q_enc = self.emb_enc(q_res_emb, q_mask, embeddings = self.pos_emb)    # (batch_size, q_len, enc_size)
        
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * enc_size)
        
        att = self.gelu(att)
        
        att_res = self.att_resize(att)  # (batch_size, c_len, enc_size)
        
        out1 = self.model_enc(att_res, c_mask, embeddings = self.pos_emb)  # (batch_size, c_len, enc_size)
        out2 = self.model_enc(out1, c_mask, embeddings = self.pos_emb) # (batch_size, c_len, enc_size)
        out3 = self.model_enc(out2, c_mask, embeddings = self.pos_emb) # (batch_size, c_len, enc_size)
        
        log_p1 = self.out_beg(out1, out2, c_mask) # (batch_size, c_len)
        log_p2 = self.out_end(out1, out3, c_mask) # (batch_size, c_len)
        
        return log_p1, log_p2
    
    
class QANet4(nn.Module):
    def __init__(self, word_vectors, char_vec, word_len,  emb_size, enc_size=128, 
                 drop_prob=0.1, n_head=8, LN_train=True, DP_residual=False,
                 mask_pos=False,two_pos=False,rel=False,total_prob=True,final_prob=1.0,freeze=True):
        super(QANet4, self).__init__()
        self.emb = layers.EmbeddingWithChar(word_vectors=word_vectors,
                                    hidden_size=emb_size,
                                    char_vec = char_vec,
                                    word_len = word_len,
                                    drop_prob = drop_prob,
                                    char_prop=0.4,
                                    hwy_drop=drop_prob,
                                    char_dim=200,
                                    bias = True,
                                    freeze=freeze,
                                    act='gelu')
        
        self.emb_resize = layers.Resizer(input_size=emb_size,
                                         output_size=enc_size,
                                         kernel_size=1,
                                         drop_prob=0,
                                         bias=True)
        
        self.pos_emb = layers.PosEmbeddings(hidden_size=enc_size,
                                            drop_prob=0,
                                            para_limit=1000,
                                            scale=False,
                                            from_pretrained=True,
                                            freeze = True)
        
        self.emb_enc = layers.EncoderBlock3(enc_size=enc_size,
                                           para_limit=1000,
                                           n_conv=4,
                                           kernel_size=7,
                                           drop_prob=drop_prob,
                                           n_head=n_head,
                                           att_drop_prob=drop_prob,
                                           final_prob=final_prob, 
                                           LN_train=LN_train,
                                           DP_residual=DP_residual,
                                           mask_pos=mask_pos,
                                           two_pos=two_pos,
                                           rel=rel,
                                           act='gelu')
        
        self.att = layers.BiDAFAttention(hidden_size=enc_size,
                                         drop_prob=drop_prob)
        
        self.gelu = layers.HighwayEncoder(num_layers=1,
                                          hidden_size=4 * enc_size,
                                          drop_prob=drop_prob,
                                          act='gelu')
        
        self.att_resize = layers.Resizer(input_size=4*enc_size,
                                         output_size=enc_size,
                                         kernel_size=1,
                                         drop_prob=0,
                                         bias=True)
        
        self.model_enc = layers.StackedEncoderBlocks(n_blocks=7,
                                                     hidden_size=enc_size,
                                                     para_limit=1000,
                                                     n_conv=2,
                                                     kernel_size=5,
                                                     drop_prob=drop_prob,
                                                     n_head=n_head,
                                                     att_drop_prob=drop_prob,
                                                     final_prob=final_prob, 
                                                     LN_train=LN_train,
                                                     DP_residual=DP_residual,
                                                     mask_pos=mask_pos,
                                                     two_pos=two_pos,
                                                     rel=rel,
                                                     total_prob=total_prob,
                                                     act='gelu')
        
        self.out_beg = layers.OutputBlock(enc_size)
        self.out_end = layers.OutputBlock(enc_size)
        
        self.drop = nn.Dropout(drop_prob)

        
        
        
    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, emb_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, emb_size)
        
        c_res_emb = self.emb_resize(c_emb)  # (batch_size, c_len, enc_size)
        q_res_emb = self.emb_resize(q_emb)  # (batch_size, q_len, enc_size)
        
        c_enc = self.emb_enc(c_res_emb, c_mask, embeddings = self.pos_emb)    # (batch_size, c_len, enc_size)
        q_enc = self.emb_enc(q_res_emb, q_mask, embeddings = self.pos_emb)    # (batch_size, q_len, enc_size)
        
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * enc_size)
        
        att = self.gelu(att)
        
        att_res = self.att_resize(att)  # (batch_size, c_len, enc_size)
        
        out1 = self.model_enc(att_res, c_mask, embeddings = self.pos_emb)  # (batch_size, c_len, enc_size)
        out2 = self.model_enc(out1, c_mask, embeddings = self.pos_emb) # (batch_size, c_len, enc_size)
        out3 = self.model_enc(out2, c_mask, embeddings = self.pos_emb) # (batch_size, c_len, enc_size)
        out4 = self.model_enc(out3, c_mask, embeddings = self.pos_emb) # (batch_size, c_len, enc_size)
        
        log_p1 = self.out_beg(out1, out3, c_mask) # (batch_size, c_len)
        log_p2 = self.out_end(out2, out4, c_mask) # (batch_size, c_len)
        
        return log_p1, log_p2
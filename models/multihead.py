import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from scipy import io as scio
import os
from torch.utils.data import DataLoader

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        d_k = 63
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context,attn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, attn_dropout, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        # self.hid_dim = int(config['hid_dim'])
        self.attn_dropout = attn_dropout
        # self.out_dim = int(config['out_dim'])
        # self.output_dim = int(config['output_dim'])
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(self.embed_size, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.embed_size, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.embed_size, bias=False)
        # self.pre_layer = PreNet(self.embed_size, self.hid_dim, self.out_dim, self.dropout)
        # self.out_layer = nn.Linear(self.out_dim, self.output_dim)
        self.layernorm = nn.LayerNorm(self.embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, embed_size]
        input_K: [batch_size, len_k, embed_size]
        input_V: [batch_size, len_v(=len_k), embed_size]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # residual, batch_size = input_Q, input_Q.size(0)
        input_Q = torch.tensor(input_Q)
        # input_Q = input_Q.unsqueeze(dim=1)
        residual, batch_size = input_Q, input_Q.size(0)
        #(23,1,2048)

        input_K = torch.tensor(input_K)
        # input_K = input_K.unsqueeze(dim=1)

        input_V = torch.tensor(input_V)
        # input_V = input_V.unsqueeze(dim=1)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        # K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        # V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2) # V: [batch_size, n_heads, len_v(=len_k), d_v]
        #(23,4,1,128)
        # if attn_mask is not None:
        #
        #    attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(input_Q, input_K, input_V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)# context: [batch_size, len_q, n_heads * d_v]
        #(23,1,512)
        output = self.fc(context) # [batch_size, len_q, d_model]
        #(23,1,2048)
        # output = nn.LayerNorm(self.embed_size)(output + residual)
        # output = self.layernorm(output + residual)
        nn.BatchNorm1d(2048)
        # output= output.permute(1,0,2)
        # output = output[0]
        # output = self.pre_layer(output)
        # output = self.out_layer(output)

        return output, attn

class MultiHeadAttention_vat(nn.Module):
    def __init__(self, embed_size, attn_dropout, d_k, d_v, n_heads):
        super(MultiHeadAttention_vat, self).__init__()
        self.embed_size = embed_size
        # self.hid_dim = int(config['hid_dim'])
        self.attn_dropout = attn_dropout
        # self.out_dim = int(config['out_dim'])
        # self.output_dim = int(config['output_dim'])
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(self.embed_size, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.embed_size, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.embed_size, bias=False)
        # self.pre_layer = PreNet(self.embed_size, self.hid_dim, self.out_dim, self.dropout)
        # self.out_layer = nn.Linear(self.out_dim, self.output_dim)
        self.layernorm = nn.LayerNorm(self.embed_size)

    def forward(self, input_Q1, input_K1, input_V1, input_Q2, input_K2, input_V2):
        '''
        input_Q: [batch_size, len_q, embed_size]
        input_K: [batch_size, len_k, embed_size]
        input_V: [batch_size, len_v(=len_k), embed_size]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        # residual, batch_size = input_Q, input_Q.size(0)
        input_Q1 = torch.tensor(input_Q1)
        input_Q2 = torch.tensor(input_Q2)
        # input_Q = input_Q.unsqueeze(dim=1)
        residual, batch_size = input_Q1, input_Q1.size(0)
        #(23,1,2048)

        input_K1= torch.tensor(input_K1)
        input_K2 = torch.tensor(input_K2)
        # input_K = input_K.unsqueeze(dim=1)

        input_V1 = torch.tensor(input_V1)
        input_V2 = torch.tensor(input_V2)
        # input_V = input_V.unsqueeze(dim=1)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q1 = self.W_Q(input_Q1).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        # K1 = self.W_K(input_K1).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        # V1 = self.W_V(input_V1).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # Q2 = self.W_Q(input_Q2).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        # K2 = self.W_K(input_K2).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        # V2 = self.W_V(input_V2).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        #(23,4,1,128)
        # if attn_mask is not None:
        #
        #    attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context1, attn1 = ScaledDotProductAttention()(input_Q1, input_K1, input_V1)
        context2, attn2 = ScaledDotProductAttention()(input_Q2, input_K2, input_V2)
        context = context2 + context1
        attn = attn1 + attn2
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)# context: [batch_size, len_q, n_heads * d_v]
        #(23,1,512)
        output = self.fc(context) # [batch_size, len_q, d_model]
        #(23,1,2048)
        # output = nn.LayerNorm(self.embed_size)(output + residual)
        # output = self.layernorm(output + residual)
        nn.BatchNorm1d(2048)
        # output= output.permute(1,0,2)
        # output = output[0]
        # output = self.pre_layer(output)
        # output = self.out_layer(output)

        return output, attn


if __name__ == '__main__':
    D1 = torch.rand(23, 1, 2048)
    B1 = torch.rand(23, 1,  2048)
    C1 = torch.rand(23, 1, 2048)
    D2 = torch.rand(23, 1, 2048)
    B2 = torch.rand(23, 1, 2048)
    C2 = torch.rand(23, 1, 2048)

    config = dict()
    config['embed_size'] = '2048'
    config['d_i'] = '128,128'
    config['n_heads'] = '4'
    config['hid_dim'] = '512'
    config['out_dim'] = '128'
    config['output_dim'] = '63'
    config['dropout'] = '0.1'

    net = MultiHeadAttention_vat()
    output = net(D1, B1, C1, D2, B2, C2)
    print(output)
    print('over')
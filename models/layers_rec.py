import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import init
import torch.nn.functional as F
from .SelfAttention import ScaledDotProductAttention
import matplotlib.pyplot as plt
from .Multihead import MultiHeadAttention,MultiHeadAttention_vat
from mpl_toolkits.axes_grid1 import make_axes_locatable

class FinalAddLayer(nn.Module):
    def __init__(self, in_dims, num_classes):
        super(FinalAddLayer, self).__init__()
        self.in_dims = in_dims
        self.num_classes = num_classes

        self.W1 = nn.Parameter(torch.FloatTensor(torch.ones(self.in_dims[1], self.num_classes)), requires_grad=True)
        self.W2 = nn.Parameter(torch.FloatTensor(torch.ones(self.in_dims[1], self.num_classes)), requires_grad=True)
        self.W3 = nn.Parameter(torch.FloatTensor(torch.ones(self.in_dims[1], self.num_classes)), requires_grad=True)
        self.W4 = nn.Parameter(torch.FloatTensor(torch.ones(self.in_dims[1], self.num_classes)), requires_grad=True)

    def forward(self, x1, x2, x3, x4):
    # def forward(self, x1, x2, x3): #v+t
        out = torch.mul(self.W1, x1) + torch.mul(self.W2, x2) + torch.mul(self.W3, x3) + torch.mul(self.W4, x4)
        # out = torch.mul(self.W1, x1) + torch.mul(self.W2, x2) + torch.mul(self.W3, x3) #v+t
        return out


class AELayer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(AELayer, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.hid_dims = (in_dims+out_dims)//2

        self.Encoder = nn.Sequential(
            # nn.Linear(self.in_dims, self.out_dims),
            # nn.Sigmoid(),
            nn.Linear(self.in_dims, self.hid_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hid_dims, self.hid_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hid_dims, self.out_dims),
        )

        self.weights = self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                m.bias.data.zero_()

    def forward(self, x):
        encode = self.Encoder(x)
        return encode


class DecoderLayer(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(DecoderLayer, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.hid_dims = (in_dims+out_dims)//2

        self.Decoder = nn.Sequential(
            nn.Linear(self.in_dims, self.hid_dims),
            nn.Tanh(),
            nn.Linear(self.hid_dims, self.hid_dims),
            nn.Tanh(),
            nn.Linear(self.hid_dims, self.out_dims),
            nn.ReLU(),
        )

        self.weights = self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                m.bias.data.zero_()

    def forward(self, x):
        decode = self.Decoder(x)
        return decode


class DiscriminatorLayer(nn.Module):
    def __init__(self, in_dims):
        super(DiscriminatorLayer, self).__init__()
        self.in_dims = in_dims
        self.hid_dims = (in_dims)//2

        self.Discriminator = nn.Sequential(
            nn.Linear(self.in_dims, self.hid_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hid_dims, self.hid_dims),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hid_dims, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.Discriminator(x)
        return logits


class DynamicGraphConvolution_t(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, DEVICE):
        super(DynamicGraphConvolution_t, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))
        self.relu = nn.LeakyReLU(0.2)

        self.gap = nn.AdaptiveAvgPool1d(1) # unused
        self.conv_global = nn.Conv1d(in_features, in_features, 1) # unused
        self.bn_global = nn.BatchNorm1d(in_features) # unused
        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1) # unused

        self.FAt = ScaledDotProductAttention(d_model=num_nodes, d_k=num_nodes, d_v=num_nodes, h=1)
        self.CAt = ScaledDotProductAttention(d_model=in_features, d_k=in_features, d_v=in_features, h=1)

        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_v_with_t = self.get_network(self_type='vt')
        self.trans_v_with_v = self.get_network(self_type='vv')
        self.trans_v_with_at = self.encoder_vat(self_type='vat')

        self.embed_size = int(2048)
        self.v = float(0.1)
        self.a = float(1)
        self.t = float(1)
        # self.D = float(1)
        self.hid_dim = int(512)
        self.attn_dropout_t = float(0.2)
        self.attn_dropout_a = float(0.1)
        self.attn_dropout_v = float(0.1)
        self.attn_dropout_vat = float(0.1)
        self.attn_dropout = float(0.2)
        self.out_dim = int(128)
        self.output_dim = int(63)
        self.d_k = 32
        self.d_v = 32
        self.n_heads = int(4)
        self.dropout = float(0.1)

    def get_network(self, self_type='v'):
        if self_type in ['va']:
            attn_dropout = 0.1

        elif self_type in ['vt']:
            attn_dropout = 0.2

        elif self_type in ['vv']:
            attn_dropout = 0.1

        return MultiHeadAttention(embed_size=63,
                                  attn_dropout=attn_dropout,
                                  d_k=63,
                                  d_v=63,
                                  n_heads=1)

    def encoder_vat(self,self_type='vat'):
         if self_type in ['vat']:
             attn_dropout = 0.1

         return MultiHeadAttention_vat(embed_size=63,
                                   attn_dropout=attn_dropout,
                                   d_k=63,
                                   d_v=63,
                                   n_heads=1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x, x_1):
        """ D-GCN module

        Shape:
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """

        out_static = self.forward_static_gcn(x)
        x = x + out_static  # residual
        out = x

        out_static_1 = self.forward_static_gcn(x_1)
        x_1 = x_1 + out_static_1  # residual

        FAt, x_FAt = self.FAt(x, x, x)
        x = x + x_FAt
        #(64,128,63)
        x = x.transpose(1, 2)

        FAt_1, x_FAt_1 = self.FAt(x_1, x_1, x_1)
        x_1 = x_1 + x_FAt_1
        #(64,128,63)
        x_1 = x_1.transpose(1, 2)

        # CAt, x_CAt = self.CAt(x, x, x) #(64,1,63,63)
        _, CAt = self.trans_v_with_t(x_1, x, x) #Q,K,V
        CAt = CAt.view(CAt.size(0), CAt.size(2), -1)
        x = x.transpose(1, 2)
        dynamic_adj = CAt

        dynamic_adj1 = dynamic_adj.cpu().numpy()
        dynamic_adj1 = dynamic_adj1[0]

        out_dynamic = self.forward_dynamic_gcn(x, dynamic_adj)
        x = x + out_dynamic

        return x, dynamic_adj1
    
    

if __name__ == '__main__':
    data = torch.rand(128, 256)
    Input_X = torch.rand(16, 1024, 63)
    Input_Y = torch.rand(16, 2048)
    in_dim = 1024
    out_dim = 256
    num_classes = 63
    rank = 2
    use_softmax = False
    device = None

    model = DynamicGraphConvolution(in_features=in_dim, out_features=in_dim, num_nodes=num_classes, device=device)
    print(model)
    out = model(Input_X)
    print(out.shape)
    print('over')




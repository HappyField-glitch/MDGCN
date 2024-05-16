import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .loss import RecLoss
# from .layers import AELayer, DynamicGraphConvolution, DiscriminatorLayer
from .layers_rec import AELayer, DynamicGraphConvolution_t, DiscriminatorLayer,DecoderLayer
from .functions import SIMSE, DiffLoss, MSE
from .functions import ReverseLayerF
# from .SelfAttention import ScaledDotProductAttention
from .Multihead import MultiHeadAttention,MultiHeadAttention_vat,ScaledDotProductAttention

import scipy.io
from datetime import datetime

class PreNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, dropout):

        super(PreNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        # self.linear_1 = nn.Sequential(
        #     nn.Linear(self.in_size, self.hidden_size),
        #     nn.BatchNorm1d(self.hidden_size),
        # )
        #
        # self.linear_2 = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.BatchNorm1d(self.hidden_size),
        # )

        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_size)


    def forward(self, data):
        """
        Args:
            data:  tensor of shape (batch_size, in_size)
        """
        normed = self.norm(data)
        # dropped = self.drop(normed)
        y1 = self.linear_1(data)
        nn.BatchNorm1d(512)
        y_1 = F.relu(y1)
        y2 = self.linear_2(y_1)
        nn.BatchNorm1d(512)
        y_2 = F.relu(y2)
        y3 = self.linear_3(y_2)
        nn.BatchNorm1d(128)
        y_3 = F.relu(y3)

        return y_3

class MMDLNetV0(nn.Module):
    def __init__(self, config):
        super(MMDLNetV0, self).__init__()
        self.num_classes = config.num_classes
        self.in_dims = config.in_dims
        self.hid_dims = config.hid_dims
        self.out_dims = config.out_dims
        self.lr = config.learning_rate
        self.device = torch.device('cuda:0')
        self.alpha_weight = config.alpha_weight
        self.beta_weight = config.beta_weight
        self.gamma_weight = config.gamma_weight

        self.ClsLoss = nn.BCEWithLogitsLoss()
        self.loss_recon1 = MSE()
        self.loss_recon2 = SIMSE()
        self.loss_diff = DiffLoss()
        self.loss_similarity = torch.nn.CrossEntropyLoss()

        # self.PrePreNet_B = AELayer(in_dims=2206, out_dims=2048) #
        self.PreNet_V = AELayer(in_dims=self.in_dims[0], out_dims=self.hid_dims[0])
        self.PreNet_T = AELayer(in_dims=self.in_dims[0], out_dims=self.hid_dims[0])
        self.PreNet_A = AELayer(in_dims=self.in_dims[0], out_dims=self.hid_dims[0])
        self.PreNet_S = AELayer(in_dims=self.in_dims[1], out_dims=self.hid_dims[1])
        self.Decoder = DecoderLayer(in_dims=self.hid_dims[2], out_dims=self.in_dims[1])
        self.D = DiscriminatorLayer(in_dims=self.hid_dims[1])

        self.fc = nn.Conv2d(self.hid_dims[1],  self.num_classes, (1, 1), bias=False)

        self.gcn_t = DynamicGraphConvolution_t(self.hid_dims[0], self.hid_dims[0], self.num_classes, self.device)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())

        self.gen_guidance = nn.Conv1d(self.hid_dims[1], 32, 1)
        # self.SSA = ScaledDotProductAttention(d_model=self.num_classes, d_k=self.num_classes, d_v=self.num_classes, h=1)
        self.layernorm = nn.LayerNorm(self.num_classes, eps=1e-5)
        
        self.linear = nn.Conv1d(384, self.hid_dims[1], 1)
        self.last_linear = nn.Conv1d(self.hid_dims[1], self.num_classes, 1)

        """
        Construct a MulT model.
        """
        self.embed_size = int(128)
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
        # self.pre_layer = PreNet(self.embed_size, self.hid_dim, self.out_dim, self.dropout)
        # self.out_layer = nn.Linear(self.out_dim, self.output_dim)
        # self.gan_layer = G_D_loss(self.embed_size, self.embed_size)
        # self.criterionCycle = torch.nn.L1Loss()
        self.pre_layer = PreNet(self.embed_size, self.hid_dim, self.out_dim, self.dropout)

        """
        Crossmodal Attentions
        """
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_v_with_t = self.get_network(self_type='vt')
        self.trans_v_with_v = self.get_network(self_type='vv')
        self.trans_v_with_at = self.encoder_vat(self_type='vat')

    def get_network(self, self_type='v'):
        if self_type in ['va']:
            attn_dropout = self.attn_dropout_a

        elif self_type in ['vt']:
            attn_dropout = self.attn_dropout_t

        elif self_type in ['vv']:
            attn_dropout = self.attn_dropout_v

        return MultiHeadAttention(embed_size=self.embed_size,
                                  attn_dropout=attn_dropout,
                                  d_k=self.d_k,
                                  d_v=self.d_v,
                                  n_heads=self.n_heads)

    def encoder_vat(self,self_type='vat'):
         if self_type in ['vat']:
             attn_dropout = self.attn_dropout_vat

         return MultiHeadAttention_vat(embed_size=self.embed_size,
                                   attn_dropout=attn_dropout,
                                   d_k=self.d_k,
                                   d_v=self.d_v,
                                   n_heads=self.n_heads)

    def get_optimizer(self):
        optim = [{'params': self.parameters(), 'lr': self.lr}]
        return optim

    def forward(self, V, T, A, Y_true, p=0.0):
        """
        :param A: [batch_size, dims]
        :param B: [batch_size, dims]
        :param Y_true: [batch_size, num_classes]
        :return:
        """
        """
        audio, and vision should have dimension [batch_size, seq_len, n_features]
        """

        # Modal encoder & decoder
        encode_V = self.PreNet_V(V)
        encode_T = self.PreNet_T(T)
        encode_A = self.PreNet_A(A)


        # share encoder & decoder
        encode_Vs = self.PreNet_S(V)
        encode_Ts = self.PreNet_S(T)
        encode_As = self.PreNet_S(A)


        # encode_Vs1 = encode_Vs.unsqueeze(dim=1)
        # encode_Ts1 = encode_Ts.unsqueeze(dim=1)
        # encode_As1 = encode_As.unsqueeze(dim=1)

        # shared_code, _ = self.trans_v_with_at(encode_As1, encode_Vs1, encode_Vs1, encode_Ts1, encode_Vs1, encode_Vs1)
        # shared_code, _ = self.trans_v_with_t(encode_Ts1, encode_Vs1, encode_Vs1) #去掉a
        # shared_code, _ = self.trans_v_with_a(encode_As1, encode_Vs1, encode_Vs1) #去掉t
        # shared_code = shared_code.permute(1, 0, 2)
        # shared_code = shared_code[0]

        # self.W = nn.Parameter(torch.FloatTensor(torch.ones(self.out_dim, self.output_dim)), requires_grad=True).to(self.device)

        # encode_Ts = 0.1 * encode_Ts
        encode_As = 1.5 * encode_As
        # encode_Ts = 0.95 * encode_Ts

        shared_code = torch.cat((encode_Vs, encode_Ts, encode_As), dim=1)
        # shared_code = torch.cat((encode_Vs, encode_Ts), dim=1) #消融a
        # shared_code = torch.cat((encode_Vs, encode_As), dim=1) #消融t
        # shared_code = torch.cat((encode_Ts, encode_As), dim=1) #消融v
        # shared_code = encode_Vs #消融a，t
        # shared_code = encode_As #消融v，t
        # shared_code = encode_Ts #消融v，a
        shared_code = self.linear(shared_code.unsqueeze(2)).squeeze(2)

        decode_V = self.Decoder(torch.cat((shared_code, encode_V), dim=1)) #井老师：用shared_code与单独编码后的concat，再去解码算重构损失
        decode_T = self.Decoder(torch.cat((shared_code, encode_T), dim=1))
        decode_A = self.Decoder(torch.cat((shared_code, encode_A), dim=1))

        # Reconstruction Constraint
        loss_mse_V = self.loss_recon1(decode_V, V) + self.loss_recon2(decode_V, V)
        loss_mse_T = self.loss_recon1(decode_T, T) + self.loss_recon2(decode_T, T)
        loss_mse_A = self.loss_recon1(decode_A, A) + self.loss_recon2(decode_A, A)
        loss_mse = self.alpha_weight * (loss_mse_V + loss_mse_T + loss_mse_A)
        # loss_mse = self.alpha_weight * (loss_mse_V + loss_mse_T) #去掉a
        # loss_mse = self.alpha_weight * (loss_mse_V + loss_mse_A)  # 去掉t
        # loss_mse = self.alpha_weight * (loss_mse_T + loss_mse_A)  # 去掉v
        # loss_mse = self.alpha_weight * (loss_mse_V) #消融a，t
        # loss_mse = self.alpha_weight * (loss_mse_A) #消融v，t
        # loss_mse = self.alpha_weight * (loss_mse_T) #消融v，a
        
        # pre modal
        reversed_shared_codeV = ReverseLayerF.apply(encode_Vs, p)
        share_V_label = self.D(reversed_shared_codeV)
        # print(share_V_label)
        # print(share_V_label.shape)
        reversed_shared_codeT = ReverseLayerF.apply(encode_Ts, p)
        share_T_label = self.D(reversed_shared_codeT)
        # print(share_T_label)
        reversed_shared_codeA = ReverseLayerF.apply(encode_As, p)
        share_A_label = self.D(reversed_shared_codeA)
        
        # Adversarial Similarity Constraint
        modal_V_label = Variable(torch.zeros(share_V_label.size(0)).long()).to(self.device)
        # print(modal_V_label)
        # print(modal_V_label.shape)
        modal_T_label = Variable(torch.ones(share_T_label.size(0)).long()).to(self.device)
        # print(modal_T_label)
        modal_A_label = Variable(torch.ones(share_A_label.size(0)).long()*2).to(self.device) #原代码是long()*2
        loss_simi_V = self.loss_similarity(share_V_label, modal_V_label)
        loss_simi_T = self.loss_similarity(share_T_label, modal_T_label)
        loss_simi_A = self.loss_similarity(share_A_label, modal_A_label)
        loss_simi = self.gamma_weight * (loss_simi_V + loss_simi_T + loss_simi_A)
        # loss_simi = self.gamma_weight * (loss_simi_V + loss_simi_T) #去掉a
        # loss_simi = self.gamma_weight * (loss_simi_V + loss_simi_A)  # 去掉t
        # loss_simi = self.gamma_weight * (loss_simi_T + loss_simi_A)  # 去掉v
        # loss_simi = self.gamma_weight * (loss_simi_V) #消融a，t
        # loss_simi = self.gamma_weight * (loss_simi_A) #消融v，t
        # loss_simi = self.gamma_weight * (loss_simi_T) #消融v，a
        
        # Orthogonality Constraint
        # loss_diff_V = self.loss_diff(encode_V, encode_Vs)  # previous
        # loss_diff_T = self.loss_diff(encode_T, encode_Ts)  # previous
        # loss_diff_A = self.loss_diff(encode_A, encode_As)  # previous
        loss_diff_V = self.loss_diff(encode_V, encode_V) + self.loss_diff(encode_Vs, encode_Vs)
        loss_diff_T = self.loss_diff(encode_T, encode_T) + self.loss_diff(encode_Ts, encode_Ts)
        loss_diff_A = self.loss_diff(encode_A, encode_A) + self.loss_diff(encode_As, encode_As)
        loss_diff = 0.1 * self.beta_weight * (loss_diff_V + loss_diff_T + loss_diff_A)
        # loss_diff = self.beta_weight * (loss_diff_V + loss_diff_T) #去掉a
        # loss_diff = self.beta_weight * (loss_diff_V + loss_diff_A)  # 去掉t
        # loss_diff = self.beta_weight * (loss_diff_T + loss_diff_A)  # 去掉v
        # loss_diff = self.beta_weight * (loss_diff_V) #消融a，t
        # loss_diff = self.beta_weight * (loss_diff_A) #消融v，t
        # loss_diff = self.beta_weight * (loss_diff_T) #消融v，a

        vv = encode_V.view(encode_V.size(0), encode_V.size(1), 1)
        vv = vv.repeat(1, 1, self.output_dim)
        vt = encode_T.view(encode_T.size(0), encode_T.size(1), 1)
        vt = vt.repeat(1, 1, self.output_dim)
        va = encode_A.view(encode_A.size(0), encode_A.size(1), 1)
        va = va.repeat(1, 1, self.output_dim)
        vat = shared_code.view(shared_code.size(0), shared_code.size(1), 1)
        vat = vat.repeat(1, 1, self.output_dim)

        # vv = encode_V1.view(encode_V1.size(0), encode_V1.size(1), 1)
        # vv = vv.repeat(1, 1, self.output_dim)
        # vt = encode_T1.view(encode_T1.size(0), encode_T1.size(1), 1)
        # vt = vt.repeat(1, 1, self.output_dim)
        # va = encode_A1.view(encode_A1.size(0), encode_A1.size(1), 1)
        # va = va.repeat(1, 1, self.output_dim)
        # vat = shared_code1.view(shared_code1.size(0), shared_code1.size(1), 1)
        # vat = vat.repeat(1, 1, self.output_dim)
        #(64,128,63)

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # GCN

        #ZT
        # vat0, static_v = self.gcn_vat(vv,vat,vt)
        vat0, adj_t = self.gcn_t(vat,vt)
        vat = vat + vat0
        # scipy.io.savemat(f'adj/adj_t_{current_time}.mat', {'adj_t': adj_t})

        #ZA
        # vat1, static_t = self.gcn_vat(zv,vat,va)
        vat1, adj_a = self.gcn_t(vat,va)
        vat = vat + vat1
        # scipy.io.savemat(f'adj/adj_a_{current_time}.mat', {'adj_a': adj_a})

        #ZV
        # vat2, static_a = self.gcn_vat(zv,vat,vv)
        vat2, adj_v = self.gcn_t(vat,vv)
        vat = vat + vat2
        # scipy.io.savemat(f'adj/adj_v_{current_time}.mat', {'adj_v': adj_v})

        # vat = vv + vt+ va + vat

        # final cls
        # z_hat = torch.mul(self.W, vv) + torch.mul(self.W, vt) + torch.mul(self.W, va)
        score = self.last_linear(vat)  # B*num_classes*num_classes
        mask_mat = self.mask_mat.detach()
        score = (score * mask_mat).sum(-1)

        loss_cls = self.ClsLoss(score, Y_true)

        
        return score, loss_cls, loss_diff, loss_simi, loss_mse
        # return score, loss_cls, loss_simi, loss_mse

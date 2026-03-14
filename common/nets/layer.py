import torch
import torch.nn as nn
from torch.nn import functional as F

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x

class SelfAttn(nn.Module):
    def __init__(self, f_dim, hid_dim=None, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads
        if hid_dim is None:
            hid_dim = f_dim

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)

        self.layer_norm = nn.LayerNorm(f_dim, eps=1e-6)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.ff = MLP_res_block(f_dim, hid_dim, dropout)

    def self_attn(self, x):
        BS, V, f = x.shape

        q = self.w_qs(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        k = self.w_ks(x).view(BS, -1, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        v = self.w_vs(x).view(BS, -1, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn = torch.matmul(q, k.transpose(-1, -2)) / self.norm  # bs, h, V, V
        attn = F.softmax(attn, dim=-1)  # bs, h, V, V
        attn = self.dropout1(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(BS, V, -1)
        out = self.dropout2(self.fc(out))
        return out

    def forward(self, x):
        BS, V, f = x.shape
        assert f == self.f_dim

        x = x + self.self_attn(self.layer_norm(x))
        x = self.ff(x)

        return x

class CrossAttn(nn.Module):
    def __init__(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        self.L_self_attn_layer = SelfAttn(f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        self.R_self_attn_layer = SelfAttn(f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        self.build_cross_attn(f_dim, n_heads, d_q, d_v, dropout)

    def build_cross_attn(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.layer_norm1 = nn.LayerNorm(f_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(f_dim, eps=1e-6)
        self.ffL = MLP_res_block(f_dim, f_dim, dropout)
        self.ffR = MLP_res_block(f_dim, f_dim, dropout)

    def cross_attn(self, Lf, Rf, mask_L2R=None, mask_R2L=None):
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf2 = self.layer_norm1(Lf)
        Rf2 = self.layer_norm2(Rf)

        Lq = self.w_qs(Lf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lk = self.w_ks(Lf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lv = self.w_vs(Lf2).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        Rq = self.w_qs(Rf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Rk = self.w_ks(Rf2).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Rv = self.w_vs(Rf2).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        attn_R2L = torch.matmul(Lq, Rk.transpose(-1, -2)) / self.norm  # bs, h, V, V
        attn_L2R = torch.matmul(Rq, Lk.transpose(-1, -2)) / self.norm  # bs, h, V, V

        if mask_L2R is not None:
            attn_L2R = attn_L2R.masked_fill(mask_L2R == 0, -1e9)
        if mask_R2L is not None:
            attn_R2L = attn_R2L.masked_fill(mask_R2L == 0, -1e9)

        attn_R2L = F.softmax(attn_R2L, dim=-1)  # bs, h, V, V
        attn_L2R = F.softmax(attn_L2R, dim=-1)  # bs, h, V, V

        attn_R2L = self.dropout1(attn_R2L)
        attn_L2R = self.dropout1(attn_L2R)

        feat_L2R = torch.matmul(attn_L2R, Lv).transpose(1, 2).contiguous().view(BS, V, -1)
        feat_R2L = torch.matmul(attn_R2L, Rv).transpose(1, 2).contiguous().view(BS, V, -1)

        feat_L2R = self.dropout2(self.fc(feat_L2R))
        feat_R2L = self.dropout2(self.fc(feat_R2L))

        Lf = self.ffL(Lf + feat_R2L)
        Rf = self.ffR(Rf + feat_L2R)

        return Lf, Rf

    def forward(self, Lf, Rf, mask_L2R=None, mask_R2L=None):
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf = self.L_self_attn_layer(Lf)
        Rf = self.R_self_attn_layer(Rf)
        Lf, Rf = self.cross_attn(Lf, Rf, mask_L2R, mask_R2L)

        return Lf, Rf

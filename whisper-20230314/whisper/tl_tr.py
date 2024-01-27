# -*- coding: utf-8 -*-
# @Time    : 1/26/23 11:13 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : high_mdls.py

# high models

import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from whisper.model import ResidualAttentionBlock, Linear

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class TLTR(nn.Module):
    def __init__(self, label_dim=527, n_layer=33, rep_dim=1280, mode='basic'):
        super().__init__()
        self.mode = mode
        self.n_layer = n_layer
        self.rep_dim = rep_dim
        self.label_dim = label_dim

        # (baseline) mean pool over time and layer, and mlp head
        if mode == 'mean_mlp' or mode == 'last_mlp':
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.label_dim))

        # (baseline) mean pool over time, and weight average over layers, and mlp head
        if mode == 'wa_mlp':
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.label_dim))
            self.layer_weight = torch.nn.Parameter(torch.tensor([1 / self.n_layer] * self.n_layer))

        # (baseline) mean pool over layer, and apply a original rep_dim transformer
        if 'mean_tr' in mode or 'last_tr' in mode:
            self.num_att_head = int(mode.split('_')[-1])
            self.time_tr = ResidualAttentionBlock(self.rep_dim, self.num_att_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.label_dim))

        # (baseline) weight average over layers, and apply a original rep_dim transformer
        if 'wa_tr' in mode:
            self.num_att_head = int(mode.split('_')[-1])
            self.layer_weight = torch.nn.Parameter(torch.tensor([1 / self.n_layer] * self.n_layer))
            self.time_tr = ResidualAttentionBlock(self.rep_dim, self.num_att_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.label_dim))

        # (baseline) weight average over layers, and apply a low-dimensional transformer
        if 'wa_down_tr' in mode: # 512_1
            self.inter_rep_dim = int(mode.split('_')[-2])
            self.num_att_head = int(mode.split('_')[-1])

            self.down_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.inter_rep_dim))
            self.layer_weight = torch.nn.Parameter(torch.tensor([1 / self.n_layer] * self.n_layer))
            self.time_tr = ResidualAttentionBlock(self.inter_rep_dim, self.num_att_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.inter_rep_dim), nn.Linear(self.inter_rep_dim, self.label_dim))

        # (proposed), tl-tr, weight average over layers, and apply a original rep_dim transformer
        if 'lw_tr' in mode:
            self.num_tatt_head = int(mode.split('_')[-2])
            self.num_latt_head = int(mode.split('_')[-1])
            self.time_tr = ResidualAttentionBlock(self.rep_dim, self.num_tatt_head)
            self.layer_tr = ResidualAttentionBlock(self.rep_dim, self.num_latt_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.label_dim))

        # (proposed), tl-tr with low-dimension projection, lower the dimension of the transformer # lw_down_tr_512_1_8
        if 'lw_down_tr' in mode:
            self.inter_rep_dim = int(mode.split('_')[-3])
            self.num_tatt_head = int(mode.split('_')[-2])
            self.num_latt_head = int(mode.split('_')[-1])

            self.down_layer = nn.Sequential(nn.LayerNorm(self.rep_dim), nn.Linear(self.rep_dim, self.inter_rep_dim))
            self.time_tr = ResidualAttentionBlock(self.inter_rep_dim, self.num_tatt_head)
            self.layer_tr = ResidualAttentionBlock(self.inter_rep_dim, self.num_latt_head)
            self.mlp_layer = nn.Sequential(nn.LayerNorm(self.inter_rep_dim), nn.Linear(self.inter_rep_dim, self.label_dim))

    def forward(self, audio_rep):
        # audio_rep in shape (# batch size, #whisper_enc_layer, time length after (20x) pooling, whisper_enc_dim)
        # e.g., (B, 32, 25, 1280) for whisper large-v1

        # (baseline)
        if self.mode == 'mean_mlp':
            audio_rep = torch.mean(audio_rep, dim=1)
            audio_rep = torch.mean(audio_rep, dim=1)
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

        # (baseline)
        elif self.mode == 'last_mlp':
            audio_rep = audio_rep[:, -1, :, :] # get the last layer
            audio_rep = torch.mean(audio_rep, dim=1)
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

        # (baseline)
        elif self.mode == 'wa_mlp':
            audio_rep = torch.mean(audio_rep, dim=2) # [B, 32 1280]
            audio_rep = torch.permute(audio_rep, (0, 2, 1)) # (B, 1280, 32)
            audio_rep = (audio_rep @ self.layer_weight) / self.layer_weight.sum()
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

        # (baseline)
        elif 'mean_tr' in self.mode:
            audio_rep = torch.mean(audio_rep, dim=1) # [B, 25, 1280]
            audio_rep = self.time_tr(audio_rep) # [B, 25, 1280]
            audio_rep = torch.mean(audio_rep, dim=1)  # [B*32, 1280]
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

        # (baseline) time transformer on the last layer representation
        elif 'last_tr' in self.mode:
            audio_rep = audio_rep[:, -1, :, :]  # [B, 25, 1280]
            audio_rep = self.time_tr(audio_rep) # [B, 25, 1280]
            audio_rep = torch.mean(audio_rep, dim=1)  # [B*32, 1280]
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

        # (baseline) time transformer on the layer-wise weight-average representation
        elif 'wa_tr' in self.mode:
            audio_rep = torch.permute(audio_rep, (0, 2, 3, 1)) # (B, 25, 1280, 32)
            audio_rep = (audio_rep @ self.layer_weight) / self.layer_weight.sum() # [B, 25, 1280]
            audio_rep = self.time_tr(audio_rep) # [B, 25, 1280]
            audio_rep = torch.mean(audio_rep, dim=1)  # [B*25, 1280]
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

        # (baseline) weight average with low-dimension projection
        elif 'wa_down_tr' in self.mode:
            audio_rep = torch.permute(audio_rep, (0, 2, 3, 1)) # (B, 25, 1280, 32)
            audio_rep = (audio_rep @ self.layer_weight) / self.layer_weight.sum() # [B, 25, 1280]
            audio_rep = self.down_layer(audio_rep)
            audio_rep = self.time_tr(audio_rep) # [B, 25, 1280]
            audio_rep = torch.mean(audio_rep, dim=1)  # [B*32, 1280]
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

        # (proposed) tl-tr
        elif 'lw_tr' in self.mode:
            B = audio_rep.shape[0]
            audio_rep = audio_rep.reshape(B*self.n_layer, audio_rep.shape[2], audio_rep.shape[3]) # [B*32, 25, 1280]
            audio_rep = self.time_tr(audio_rep) # [B*32, 25, 1280]
            audio_rep = torch.mean(audio_rep, dim=1) # [B*32, 1280]
            audio_rep = audio_rep.reshape(B, self.n_layer, audio_rep.shape[1]) # [B, 32, 1280]
            audio_rep = self.layer_tr(audio_rep) # [B, 32, 1280]
            audio_rep = torch.mean(audio_rep, dim=1)  # [B, 1280]
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

        #(proposed)  tl-tr with low-dimensional projection
        elif 'lw_down_tr' in self.mode:
            B = audio_rep.shape[0]
            audio_rep = self.down_layer(audio_rep)
            audio_rep = audio_rep.reshape(B*self.n_layer, audio_rep.shape[2], audio_rep.shape[3]) # [B*32, 25, 1280]
            audio_rep = self.time_tr(audio_rep) # [B*32, 25, 1280]
            audio_rep = torch.mean(audio_rep, dim=1) # [B*32, 1280]
            audio_rep = audio_rep.reshape(B, self.n_layer, audio_rep.shape[1]) # [B, 32, 1280]
            audio_rep = self.layer_tr(audio_rep) # [B, 32, 1280]
            audio_rep = torch.mean(audio_rep, dim=1)  # [B, 1280]
            audio_rep = self.mlp_layer(audio_rep)
            return audio_rep

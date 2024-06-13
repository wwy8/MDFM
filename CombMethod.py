#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numbers

from einops import rearrange
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F



class Comb(nn.Module):
    def __init__(self, dim, bias=False):
        super(Comb, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 7
        self.meta_to_chan = nn.Sequential(
                            nn.Conv2d(81, dim*2, 1),
                            nn.ReLU(),
                            nn.Conv2d(dim*2, dim, 1),
                            nn.Sigmoid()
                        )
        self.meta_to_hiddem = nn.Linear(81, self.patch_size ** 2)
    def forward(self, x, meta):
        B, C, H, W = x.size()
        hidden = self.to_hidden(x)


        meta_logit = self.meta_to_chan(meta.unsqueeze(-1).unsqueeze(-1))
        m_fft = torch.fft.rfft2(meta_logit.float())
        x_fft = torch.fft.rfft2(x.float())
        out_chan = x_fft * m_fft
        out_chan = torch.fft.irfft2(out_chan, s=(H, W))


        meta = self.meta_to_hiddem(meta)
        k = meta.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        q, _, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        k_patch = k.view(B, 1, 1, 1, self.patch_size, self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())




        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output) + out_chan
        return output

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# metadata = torch.rand([32, 81]).cuda()
# feat = torch.rand([32, 32, 112, 112]).cuda()
# feat1 = torch.rand([32, 32, 112, 112]).cuda()
# net = Comb(32).cuda()
# out = net(feat, metadata)
# print(out.size())
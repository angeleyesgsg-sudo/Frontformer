# -*- coding: utf-8 -*-
"""FrontFormer modularized from the original monolithic training script."""

import torch
import torch.nn as nn

class VarAggregator(nn.Module):
    """[PERF 7] Fuse per-variable features with SE channel attention.
    Lets the model learn which meteorological variables are most informative.
    """
    def __init__(self, num_vars=8, d_model=256, se_ratio=16):
        super().__init__()
        self.fuse = nn.Conv2d(num_vars * d_model, d_model, kernel_size=1)
        nn.init.xavier_uniform_(self.fuse.weight)
        nn.init.constant_(self.fuse.bias, 0.)
        # SE block: global avg pool -> FC -> ReLU -> FC -> Sigmoid
        se_hidden = max(d_model // se_ratio, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, se_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(se_hidden, d_model),
            nn.Sigmoid(),
        )

    def forward(self, feats_list):
        x = self.fuse(torch.cat(feats_list, dim=1))
        w = self.se(x).view(x.shape[0], -1, 1, 1)  # [PERF 7] channel recalibration
        return x * w


class ShallowStem(nn.Module):
    """[PERF 16] Residual two-layer conv stem for better gradient flow."""
    def __init__(self, d_model=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(32, d_model),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + x)  # [PERF 16] residual


class SimpleMSPyramidD17(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1)
        self.down1 = nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1)
        for m in [self.proj, self.down1, self.down2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # [BUG FIX 2] Corrected order: fine -> enc -> coarse (was [enc, high, low]).
        p_high = self.proj(x)       # finest resolution (stride x1)
        p_enc  = self.down1(p_high) # stride x2
        p_low  = self.down2(p_enc)  # coarsest (stride x4)
        return [p_high, p_enc, p_low]  # consistent with pixel decoder

# -*- coding: utf-8 -*-
"""FrontFormer modularized from the original monolithic training script."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PixelDecoder(nn.Module):
    def __init__(self, in_dim=256, mask_dim=256):
        super().__init__()
        self.lateral_shallow = nn.Conv2d(in_dim, mask_dim, 1)
        self.lateral_high = nn.Conv2d(in_dim, mask_dim, 1)
        self.lateral_enc = nn.Conv2d(in_dim, mask_dim, 1)
        self.lateral_low = nn.Conv2d(in_dim, mask_dim, 1)

        self.refine_low  = ConvGNAct(mask_dim, mask_dim)
        self.refine_enc  = ConvGNAct(mask_dim, mask_dim)
        self.refine_high = ConvGNAct(mask_dim, mask_dim)
        # [PERF 8] concat(high, shallow) -> 2*mask_dim -> mask_dim
        self.merge_conv = ConvGNAct(mask_dim * 2, mask_dim)
        self.refine_out = nn.Sequential(
            ConvGNAct(mask_dim, mask_dim),
            ConvGNAct(mask_dim, mask_dim),
            nn.Conv2d(mask_dim, mask_dim, 1),
        )

    def forward(self, shallow, p_high, p_enc, p_low):
        low = self.refine_low(self.lateral_low(p_low))
        enc = self.lateral_enc(p_enc) + F.interpolate(low, size=p_enc.shape[-2:], mode="bilinear", align_corners=False)
        enc = self.refine_enc(enc)
        high = self.lateral_high(p_high) + F.interpolate(enc, size=p_high.shape[-2:], mode="bilinear", align_corners=False)
        high = self.refine_high(high)
        shallow = self.lateral_shallow(shallow)
        # [PERF 8] Concat instead of add — preserves both feature streams
        out = self.merge_conv(torch.cat([high, shallow], dim=1))
        out = self.refine_out(out)
        return out

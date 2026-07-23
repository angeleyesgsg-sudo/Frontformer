# -*- coding: utf-8 -*-
"""FrontFormer modularized from the original monolithic training script."""

import os
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set DEFORMABLE_DETR_PATH to the cloned Deformable-DETR repository when it is
# not already available on PYTHONPATH. This replaces the original hard-coded path.
_deformable_detr_path = os.environ.get("DEFORMABLE_DETR_PATH", "")
if _deformable_detr_path and _deformable_detr_path not in sys.path:
    sys.path.insert(0, _deformable_detr_path)

try:
    from models.ops.modules import MSDeformAttn
except ImportError as exc:
    raise ImportError(
        "Cannot import MSDeformAttn. Clone Deformable-DETR, compile its ops, "
        "and set the DEFORMABLE_DETR_PATH environment variable or PYTHONPATH."
    ) from exc

from .backbone import SimpleMSPyramidD17

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, num_levels=3, num_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, num_levels, nhead, num_points)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # [PERF 17] mid-FFN dropout
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.ffn(src)
        src = self.norm2(src + self.dropout2(src2))
        return src


class DeformableEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class DeformableDecoderLayerD17(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, num_levels=3, num_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MSDeformAttn(d_model, num_levels, nhead, num_points)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # [PERF 17] mid-FFN dropout
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, src, spatial_shapes, level_start_index,
                src_padding_mask=None, query_pos=None, query_attn_mask=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = tgt.permute(1, 0, 2)
        # [CRITICAL FIX B] Apply DN group isolation mask to self-attention
        tgt2, _ = self.self_attn(q, k, value=v, attn_mask=query_attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2.permute(1, 0, 2)))

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, spatial_shapes, level_start_index, src_padding_mask)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class DeformableDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, reference_points, src, spatial_shapes, level_start_index,
                src_padding_mask=None, query_pos=None, bbox_embed=None, query_attn_mask=None):
        output = tgt
        intermediate = []
        ref_pts = reference_points  # (B, Q, num_levels, 2)
        for layer in self.layers:
            output = layer(output, ref_pts, src, spatial_shapes, level_start_index,
                           src_padding_mask, query_pos, query_attn_mask=query_attn_mask)
            if self.return_intermediate:
                intermediate.append(output)
            # [PERF 9] Iterative box refinement: update reference points after each layer
            if bbox_embed is not None:
                delta = bbox_embed(output).sigmoid()  # (B, Q, 4)
                new_xy = delta[..., :2]               # updated (cx, cy)
                ref_pts = ref_pts.clone()
                ref_pts[..., :2] = new_xy.unsqueeze(2).expand_as(ref_pts[..., :2])
        if self.return_intermediate:
            return torch.stack(intermediate, dim=0)  # (L, B, Q, C)
        return output.unsqueeze(0)


class DeformableTransformerD17(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 num_levels=3, num_points=8, return_intermediate_dec=True):
        super().__init__()
        self.num_levels = num_levels
        self.d_model = d_model
        self.pyramid = SimpleMSPyramidD17(in_dim=d_model, out_dim=d_model)
        self.encoder = DeformableEncoder(
            DeformableEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024,
                                   dropout=0.1, num_levels=num_levels, num_points=num_points),
            num_encoder_layers
        )
        self.decoder = DeformableDecoder(
            DeformableDecoderLayerD17(d_model=d_model, nhead=nhead, dim_feedforward=1024,
                                      dropout=0.1, num_levels=num_levels, num_points=num_points),
            num_decoder_layers,
            return_intermediate=return_intermediate_dec
        )
        self.level_embed = nn.Parameter(torch.Tensor(num_levels, d_model))
        self.ref_point_proj = nn.Linear(d_model, num_levels * 2)
        nn.init.normal_(self.level_embed)
        # [PERF 9] Lightweight bbox head for iterative reference point refinement
        self.bbox_embed_ref = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    @staticmethod
    def _get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl in range(spatial_shapes.shape[0]):
            H, W = spatial_shapes[lvl]
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij"
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), dim=-1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, dim=1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, query_embed, pos, dn_boxes=None, dn_pad=0, query_attn_mask=None):
        """dn_boxes: (B, dn_pad, 4) noisy GT boxes in cxcywh [0,1] for DN queries.
        dn_pad: number of DN queries prepended to query_embed.
        """
        B, C, H, W = src.shape
        device = src.device

        multi_feats = self.pyramid(src)
        multi_pos = []
        for lvl, feat in enumerate(multi_feats):
            _, _, Hi, Wi = feat.shape
            pos_lvl = F.interpolate(pos, size=(Hi, Wi), mode="bilinear", align_corners=False)
            pos_lvl = pos_lvl + self.level_embed[lvl].view(1, C, 1, 1)
            multi_pos.append(pos_lvl)

        masks = [torch.zeros((B, f.shape[2], f.shape[3]), dtype=torch.bool, device=device) for f in multi_feats]

        src_flatten, mask_flatten, pos_flatten, spatial_shapes = [], [], [], []
        for feat, mask, pos_lvl in zip(multi_feats, masks, multi_pos):
            _, _, Hi, Wi = feat.shape
            spatial_shapes.append((Hi, Wi))
            src_flatten.append(feat.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_flatten.append(pos_lvl.flatten(2).transpose(1, 2))

        src_flatten = torch.cat(src_flatten, dim=1)
        mask_flatten = torch.cat(mask_flatten, dim=1)
        pos_flatten = torch.cat(pos_flatten, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = spatial_shapes.prod(1).cumsum(0)
        level_start_index = F.pad(level_start_index[:-1], (1, 0), value=0)
        valid_ratios = torch.ones((B, self.num_levels, 2), device=device)

        reference_points_enc = self._get_reference_points(spatial_shapes, valid_ratios, device)
        memory = self.encoder(src_flatten, pos_flatten, reference_points_enc, spatial_shapes, level_start_index, mask_flatten)

        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)
        # [CRITICAL FIX A] DN queries use noisy GT boxes as initial reference points.
        # Normal queries use learned ref_point_proj as before.
        query_ref = self.ref_point_proj(query_embed).view(B, -1, self.num_levels, 2).sigmoid()
        if dn_boxes is not None and dn_pad > 0:
            # dn_boxes is (B, dn_pad, 4) in cxcywh format; use cx,cy as reference
            dn_ref = dn_boxes[:, :, :2].unsqueeze(2).expand(-1, -1, self.num_levels, -1)  # (B,dn_pad,L,2)
            dn_ref = dn_ref.clamp(0.0, 1.0)
            query_ref = torch.cat([dn_ref, query_ref[:, dn_pad:]], dim=1)
        reference_points_dec = query_ref * valid_ratios[:, None]

        # [PERF 9] Pass bbox_embed_ref for iterative refinement
        # [CRITICAL FIX B] Pass query_attn_mask to isolate DN groups in self-attention
        hs = self.decoder(tgt, reference_points_dec, memory, spatial_shapes, level_start_index,
                          src_padding_mask=mask_flatten, query_pos=query_embed,
                          bbox_embed=self.bbox_embed_ref,
                          query_attn_mask=query_attn_mask)

        # [BUG FIX 1] hs is (L, B, Q, C) from decoder — no transpose needed.
        return {
            'hs': hs,  # (L, B, Q, C)
            'memory': memory,
            'multi_feats': multi_feats,
            'multi_pos': multi_pos,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
        }

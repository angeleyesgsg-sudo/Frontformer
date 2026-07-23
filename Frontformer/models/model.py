# -*- coding: utf-8 -*-
"""FrontFormer modularized from the original monolithic training script."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from .backbone import ShallowStem, VarAggregator
from .common import MLP, NestedTensor, PositionEmbeddingSine
from .dn import build_dn_queries
from .pixel_decoder import PixelDecoder
from .transformer import DeformableTransformerD17

class DETRSeg(nn.Module):
    def __init__(self,
                 num_vars=8,
                 num_classes=2,
                 num_queries=100,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 use_aux_loss=True,
                 mask_dim=256,
                 var_names=None):
        super().__init__()
        self.num_vars = num_vars
        if var_names is None:
            var_names = [f"var{i}" for i in range(num_vars)]
        self.var_names = list(var_names)
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_aux_loss = use_aux_loss
        self.mask_dim = mask_dim

        self.var_embeds = nn.ModuleList([nn.Conv2d(1, d_model, kernel_size=1) for _ in range(num_vars)])
        for conv in self.var_embeds:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.)

        self.var_agg = VarAggregator(num_vars=num_vars, d_model=d_model)
        self.shallow_stem = ShallowStem(d_model)

        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
        )

        self.position_embedding = PositionEmbeddingSine(num_pos_feats=d_model // 2)
        self.input_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.label_enc = nn.Embedding(num_classes, d_model)
        nn.init.normal_(self.label_enc.weight, std=0.02)

        self.transformer = DeformableTransformerD17(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_levels=3,
            num_points=12,
            return_intermediate_dec=True,
        )

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        self.mask_embed_head = MLP(d_model, d_model, mask_dim, 3)
        self.pixel_decoder = PixelDecoder(in_dim=d_model, mask_dim=mask_dim)

        nn.init.constant_(self.bbox_embed[-1].bias, 0.)

    def freeze_for_seg(self):
        for p in self.class_embed.parameters():
            p.requires_grad = False
        for p in self.bbox_embed.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x, dn_args=None):
        B, V, H, W = x.shape
        assert V == self.num_vars, f"输入通道数 {V} != {self.num_vars}"

        feats = []
        for i in range(self.num_vars):
            feats.append(self.var_embeds[i](x[:, i:i+1]))

        src_var = self.var_agg(feats)
        shallow = self.shallow_stem(src_var)
        src_half = self.downsample(src_var)

        nested = NestedTensor(src_half, mask=None)
        pos = self.position_embedding(nested)
        src_proj = self.input_proj(src_half)

        normal_query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        dn_meta = None
        dn_pad = 0

        if self.training and dn_args is not None:
            dn_query_embed, dn_boxes, dn_mask, dn_meta = build_dn_queries(
                dn_args['targets'],
                num_classes=self.num_classes,
                hidden_dim=self.d_model,
                label_enc=self.label_enc,
                dn_number=dn_args.get('dn_number', 5),
                label_noise_ratio=dn_args.get('label_noise_ratio', 0.2),
                box_noise_scale=dn_args.get('box_noise_scale', 0.4),
                device=x.device,
            )
            if dn_query_embed is not None and dn_meta['pad_size'] > 0:
                query_embed = torch.cat([dn_query_embed, normal_query_embed], dim=1)
                dn_pad = dn_meta['pad_size']
                # [FIX 2] Build DN self-attention mask.
                # Rules:
                #   - Normal queries attend freely to each other.
                #   - DN queries are fully blocked from normal queries (and vice versa).
                #   - Each DN group is isolated: group g can only attend within itself.
                # Padding slots (queries beyond known_num[b] for sample b) differ per
                # sample, but nn.MultiheadAttention takes a shared 2D mask. We use the
                # *minimum* valid group size across the batch as the open window, so
                # padding slots are never accidentally unmasked.
                total_q = dn_pad + self.num_queries
                attn_mask = torch.ones(total_q, total_q, dtype=torch.bool, device=x.device)
                # Normal queries attend to each other freely
                attn_mask[dn_pad:, dn_pad:] = False
                # Each DN group window: use min known_num for conservative masking
                known_nums = dn_meta['known_num']  # list of per-sample GT counts
                known_num_min = min(n for n in known_nums if n > 0) if any(known_nums) else 0
                known_num_max = max(known_nums) if known_nums else 0
                dn_number = dn_args.get('dn_number', 5)
                if known_num_max > 0:
                    for g in range(dn_number):
                        s = g * known_num_max        # start of group g (uses max for stride)
                        e = s + known_num_min        # open only the guaranteed-valid window
                        if e <= dn_pad:
                            attn_mask[s:e, s:e] = False
                dn_meta['attn_mask'] = attn_mask
            else:
                query_embed = normal_query_embed
                dn_meta['attn_mask'] = None
        else:
            query_embed = normal_query_embed

        with autocast('cuda', enabled=False):
            # [CRITICAL FIX A] Pass dn_boxes so DN queries start from noisy GT reference points
            _pass_dn_boxes = dn_boxes.float() if (dn_pad > 0 and dn_boxes is not None) else None
            _attn_mask = dn_meta['attn_mask'].float() if (dn_meta is not None
                          and dn_meta.get('attn_mask') is not None) else None
            trans_out = self.transformer(
                src_proj.float(), query_embed.float(), pos.float(),
                dn_boxes=_pass_dn_boxes,
                dn_pad=dn_pad,
                query_attn_mask=_attn_mask,
            )

        # [BUG FIX 1] hs is already (L, B, Q, C) — no permute needed.
        hs = trans_out['hs'].to(src_proj.dtype)  # (L, B, Q, C)
        # [BUG FIX 12] Unpack matching pyramid's corrected return order [p_high, p_enc, p_low]
        p_high, p_enc, p_low = trans_out['multi_feats']
        p_high = p_high.to(src_proj.dtype)
        p_enc  = p_enc.to(src_proj.dtype)
        p_low  = p_low.to(src_proj.dtype)
        shallow_half = F.interpolate(shallow, size=p_high.shape[-2:], mode="bilinear", align_corners=False)

        mask_features = self.pixel_decoder(shallow_half, p_high, p_enc, p_low)

        outputs_class = self.class_embed(hs)
        outputs_boxes = self.bbox_embed(hs).sigmoid()
        # [FIX 1] Only skip masks during *training* det stage.
        # Eval always runs the full forward so mask metrics and match_hungarian are valid,
        # and det->seg checkpoint transfer captures a consistent feature distribution.
        compute_masks = (not self.training) or getattr(self, 'training_masks', True)
        if compute_masks:
            mask_embed = self.mask_embed_head(hs)
            outputs_masks = torch.einsum('lbqc,bchw->lbqhw', mask_embed, mask_features)
        else:
            outputs_masks = torch.zeros(
                (*outputs_boxes.shape[:3], mask_features.shape[-2], mask_features.shape[-1]),
                device=hs.device, dtype=hs.dtype)

        if dn_pad > 0:
            out = {
                'pred_logits': outputs_class[-1][:, dn_pad:],
                'pred_boxes': outputs_boxes[-1][:, dn_pad:],
                'pred_masks': outputs_masks[-1][:, dn_pad:],
                'dn_pred_logits': outputs_class[-1][:, :dn_pad],
                'dn_pred_boxes': outputs_boxes[-1][:, :dn_pad],
                'dn_pred_masks': outputs_masks[-1][:, :dn_pad],
                'dn_meta': dn_meta,
                'mask_features': mask_features,
            }
        else:
            out = {
                'pred_logits': outputs_class[-1],
                'pred_boxes': outputs_boxes[-1],
                'pred_masks': outputs_masks[-1],
                'dn_meta': None,
                'mask_features': mask_features,
            }

        if self.use_aux_loss:
            aux_list = []
            for c, b, m in zip(outputs_class[:-1], outputs_boxes[:-1], outputs_masks[:-1]):
                if dn_pad > 0:
                    aux_list.append({
                        'pred_logits': c[:, dn_pad:],
                        'pred_boxes': b[:, dn_pad:],
                        'pred_masks': m[:, dn_pad:],
                    })
                else:
                    aux_list.append({
                        'pred_logits': c,
                        'pred_boxes': b,
                        'pred_masks': m,
                    })
            out['aux_outputs'] = aux_list

        return out

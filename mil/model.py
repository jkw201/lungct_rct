# -*- coding: utf-8 -*-
"""Side-Separated Attention MIL model."""

from __future__ import annotations

import torch
import torch.nn as nn


class Simple3DEncoder(nn.Module):
    """Small 3D CNN encoder for CT patches."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class SideSeparatedAttnMIL(nn.Module):
    """
    Side-separated attention MIL.

    The max-pooling patient head assumes the patient-level label means
    "at least one shoulder is positive." If labels are side-specific, use a
    side-specific supervised objective instead of this patient-level MIL head.
    """

    def __init__(self, feat_dim: int = 256, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = Simple3DEncoder(out_dim=feat_dim)
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.classifier = nn.Linear(feat_dim, 1)

    def forward_single_side(self, patches):
        feats = self.encoder(patches)
        scores = self.attn(feats).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        bag_feat = (feats * weights[:, None]).sum(dim=0)
        logit = self.classifier(bag_feat).squeeze(-1)
        return logit, weights.detach()

    def forward(self, left_list, right_list, valid_mask=None):
        device = next(self.parameters()).device
        logits = []
        attn_weights = []
        for i in range(len(left_list)):
            z_left, w_left = self.forward_single_side(left_list[i].to(device))
            z_right, w_right = self.forward_single_side(right_list[i].to(device))
            if valid_mask is not None:
                v_left, v_right = valid_mask[i].to(device)
                if v_left < 0.5:
                    z_left = torch.tensor(-1e5, device=device)
                if v_right < 0.5:
                    z_right = torch.tensor(-1e5, device=device)
            logits.append(torch.max(z_left, z_right))
            attn_weights.append((w_left, w_right))
        return torch.stack(logits, dim=0), attn_weights

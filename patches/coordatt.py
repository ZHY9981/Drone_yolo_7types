"""
Coordinate Attention (CoordAtt) Module
======================================
Paper: Hou et al., "Coordinate Attention for Efficient Mobile Network Design," CVPR 2021
https://arxiv.org/abs/2103.02907

This implementation is injected into the YOLO26s neck via the model config YAML
(e.g., `yolo26s-v16-p34-coordatt.yaml`). Registered as `CoordAtt` in Ultralytics
task.py so it can be referenced by name in model architecture definitions.

How it works:
1. Decompose 2D global pooling into two 1D encodings (X-direction + Y-direction)
2. Concatenate, pass through shared 1×1 conv for cross-channel interaction
3. Split back into separate X and Y attention maps
4. Multiply both onto the input feature map

Why it helps aerial detection:
Standard SE attention loses all spatial information via global average pooling.
CoordAtt preserves position cues along each axis — critical for small aerial
targets where "where" is as important as "what."
"""

import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """Coordinate Attention — CVPR 2021.

    Captures position information along X and Y directions separately,
    unlike SE which loses all spatial info via global pooling.

    Args:
        inp (int): Input channels.
        oup (int): Output channels (same as inp for residual-style usage).
        reduction (int): Channel reduction ratio (default 32).
    """

    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super().__init__()
        # 1D adaptive pooling along each spatial axis
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Shared 1×1 conv for cross-channel interaction
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        # Separate 1×1 convs for X and Y attention maps
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # Step 1: 1D pooling along each axis
        x_h = self.pool_h(x)               # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, 1, W] → [B, C, W, 1]

        # Step 2: Shared feature encoding
        y = torch.cat([x_h, x_w], dim=2)   # [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Step 3: Split back into H and W branches
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)      # [B, C, W, 1] → [B, C, 1, W]

        # Step 4: Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()    # [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()    # [B, C, 1, W]

        # Step 5: Apply attention (element-wise multiplication)
        return identity * a_w * a_h

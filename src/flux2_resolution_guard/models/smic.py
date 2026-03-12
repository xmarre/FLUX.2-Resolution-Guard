from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..utils.image_ops import build_feature_mask, ensure_batch, ensure_mask_batch, make_full_mask
from ..utils.warp import clamp_flow, upsample_flow, warp_image
from .blocks import ConvGNAct, DownBlock, ResidualBlock, UpBlock


@dataclass(slots=True)
class SMICConfig:
    input_channels: int = 16
    base_channels: int = 32
    flow_grid_size: int = 16
    residual_scale: float = 0.15
    max_flow: float = 0.08


class SMICCorrectionModel(nn.Module):
    """
    Stable-Manifold Inward Compander.

    Image-domain correction model intended for FLUX.2 high-resolution drift correction.
    It supports both whole-image and masked correction. The same model can be trained
    with anchor images from stable lower-resolution FLUX.2 outputs.
    """

    def __init__(self, config: Optional[SMICConfig] = None) -> None:
        super().__init__()
        self.config = config or SMICConfig()

        c = self.config.base_channels
        self.stem = nn.Sequential(
            ConvGNAct(self.config.input_channels, c),
            ResidualBlock(c),
        )
        self.down1 = DownBlock(c, c * 2)
        self.down2 = DownBlock(c * 2, c * 4)
        self.down3 = DownBlock(c * 4, c * 8)

        self.mid = nn.Sequential(
            ResidualBlock(c * 8),
            ResidualBlock(c * 8),
        )

        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)

        self.flow_head = nn.Sequential(
            ConvGNAct(c * 8, c * 4),
            nn.AdaptiveAvgPool2d((self.config.flow_grid_size, self.config.flow_grid_size)),
            nn.Conv2d(c * 4, 2, kernel_size=1),
        )
        self.residual_head = nn.Sequential(
            ConvGNAct(c, c),
            nn.Conv2d(c, 3, kernel_size=3, padding=1),
        )
        self.gate_head = nn.Sequential(
            ConvGNAct(c, c // 2),
            nn.Conv2d(c // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def build_inputs(
        self,
        edited: torch.Tensor,
        original: Optional[torch.Tensor] = None,
        anchor: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mp_ratio: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edited = ensure_batch(edited)
        b, _, h, w = edited.shape
        device = edited.device
        dtype = edited.dtype

        original = ensure_batch(original) if original is not None else edited
        anchor = ensure_batch(anchor) if anchor is not None else edited

        if original.shape[-2:] != (h, w):
            original = torch.nn.functional.interpolate(original, size=(h, w), mode="bilinear", align_corners=False)
        if anchor.shape[-2:] != (h, w):
            anchor = torch.nn.functional.interpolate(anchor, size=(h, w), mode="bilinear", align_corners=False)

        if mask is None:
            mask = make_full_mask(b, h, w, device, dtype)
        else:
            mask = ensure_mask_batch(mask, h, w, device=device).to(dtype)

        soft_mask, dist = build_feature_mask(mask)

        if not torch.is_tensor(mp_ratio):
            mp_ratio = torch.tensor([float(mp_ratio)], device=device, dtype=dtype).repeat(b)
        mp_ratio = mp_ratio.to(device=device, dtype=dtype).view(b, 1, 1, 1).expand(b, 1, h, w)

        delta = edited - original
        anchor_delta = edited - anchor

        x = torch.cat(
            [
                edited,
                original,
                anchor,
                delta,
                mask,
                soft_mask,
                dist,
                mp_ratio,
            ],
            dim=1,
        )
        return x, mask

    def forward(
        self,
        edited: torch.Tensor,
        original: Optional[torch.Tensor] = None,
        anchor: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mp_ratio: float | torch.Tensor = 1.0,
        strength: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        x, mask = self.build_inputs(
            edited=edited,
            original=original,
            anchor=anchor,
            mask=mask,
            mp_ratio=mp_ratio,
        )

        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        mid = self.mid(s3)

        u2 = self.up3(mid, s2)
        u1 = self.up2(u2, s1)
        u0 = self.up1(u1, s0)

        flow_lowres = self.flow_head(mid)
        flow = upsample_flow(flow_lowres, edited.shape[-2], edited.shape[-1])
        flow = clamp_flow(flow, max_magnitude=self.config.max_flow * float(strength))

        warped = warp_image(edited, flow)
        residual = torch.tanh(self.residual_head(u0)) * self.config.residual_scale * float(strength)
        gate = self.gate_head(u0)

        corrected = warped + gate * residual
        corrected = edited * (1.0 - mask) + corrected * mask
        corrected = corrected.clamp(0.0, 1.0)

        return {
            "corrected": corrected,
            "warped": warped,
            "flow": flow,
            "residual": residual,
            "gate": gate,
            "mask": mask,
        }

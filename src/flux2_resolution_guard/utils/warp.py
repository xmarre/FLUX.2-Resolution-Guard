from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .image_ops import build_feature_mask, ensure_batch, ensure_mask_batch, image_center, make_coord_grid


def upsample_flow(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if flow.dim() != 4 or flow.size(1) != 2:
        raise ValueError(f"Expected Bx2xHc xWc flow tensor, got shape {tuple(flow.shape)}")
    return F.interpolate(flow, size=(height, width), mode="bicubic", align_corners=False)


def flow_to_sampling_grid(flow: torch.Tensor) -> torch.Tensor:
    b, _, h, w = flow.shape
    base = make_coord_grid(b, h, w, flow.device, flow.dtype)
    # base in [-1, 1], flow already expected in normalized units
    grid = (base + flow).permute(0, 2, 3, 1).contiguous()
    return grid


def warp_image(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    image = ensure_batch(image)
    if flow.shape[-2:] != image.shape[-2:]:
        flow = upsample_flow(flow, image.shape[-2], image.shape[-1])
    grid = flow_to_sampling_grid(flow)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)


def radial_inward_flow(
    mask: torch.Tensor,
    strength: float,
    mp_ratio: float,
    anisotropy_x: float = 1.0,
    anisotropy_y: float = 1.0,
) -> torch.Tensor:
    mask = ensure_batch(mask)
    b, _, h, w = mask.shape
    soft_mask, dist = build_feature_mask(mask)
    x0, y0 = image_center(mask)

    coords = make_coord_grid(b, h, w, mask.device, mask.dtype)
    dx = coords[:, 0] - x0[:, None, None]
    dy = coords[:, 1] - y0[:, None, None]

    radius = torch.sqrt((dx * anisotropy_x) ** 2 + (dy * anisotropy_y) ** 2 + 1e-8)
    inward = torch.stack([-dx, -dy], dim=1) / torch.clamp(radius[:, None], min=1e-4)
    falloff = torch.clamp(soft_mask * (1.0 - radius[:, None] * 0.5), min=0.0, max=1.0)
    scale = float(max(mp_ratio - 1.0, 0.0)) * float(strength) * 0.015
    flow = inward * falloff * scale
    return flow


def clamp_flow(flow: torch.Tensor, max_magnitude: float = 0.08) -> torch.Tensor:
    mag = torch.sqrt((flow**2).sum(dim=1, keepdim=True) + 1e-8)
    factor = torch.clamp(max_magnitude / mag, max=1.0)
    return flow * factor

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..utils.image_ops import gaussian_blur, rgb_to_oklab


@dataclass(slots=True)
class LossWeights:
    pix: float = 1.0
    lowfreq: float = 0.75
    seam: float = 0.75
    oklab: float = 0.5
    identity: float = 0.2
    warp: float = 0.1


def charbonnier(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps**2).mean()


def seam_ring(mask: torch.Tensor, sigma_small: float = 1.5, sigma_large: float = 5.0) -> torch.Tensor:
    small = gaussian_blur(mask, sigma=sigma_small)
    large = gaussian_blur(mask, sigma=sigma_large)
    ring = (large - small).abs()
    return ring / torch.clamp(ring.amax(dim=(-2, -1), keepdim=True), min=1e-6)


def lowfreq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return charbonnier(gaussian_blur(pred, sigma=5.0), gaussian_blur(target, sigma=5.0))


def seam_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ring = seam_ring(mask)
    return charbonnier(pred * ring, target * ring)


def oklab_lowfreq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_ok = gaussian_blur(rgb_to_oklab(pred), sigma=5.0)
    tgt_ok = gaussian_blur(rgb_to_oklab(target), sigma=5.0)
    return charbonnier(pred_ok, tgt_ok)


def identity_loss(pred: torch.Tensor, edited: torch.Tensor, mp_ratio: torch.Tensor) -> torch.Tensor:
    # Encourage no-op behavior near ratio 1.0
    strength = torch.clamp(1.15 - mp_ratio.view(-1, 1, 1, 1), min=0.0, max=1.0)
    return charbonnier(pred * strength, edited * strength)


def warp_regularization(flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    smooth = dx.abs().mean() + dy.abs().mean()
    outside = (1.0 - mask) * flow.abs().sum(dim=1, keepdim=True)
    return smooth + outside.mean()


def total_loss(
    corrected: torch.Tensor,
    target: torch.Tensor,
    edited: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
    mp_ratio: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_pix = charbonnier(corrected * mask, target * mask)
    l_low = lowfreq_loss(corrected, target)
    l_seam = seam_loss(corrected, target, mask)
    l_ok = oklab_lowfreq_loss(corrected, target)
    l_id = identity_loss(corrected, edited, mp_ratio)
    l_warp = warp_regularization(flow, mask)

    total = (
        weights.pix * l_pix
        + weights.lowfreq * l_low
        + weights.seam * l_seam
        + weights.oklab * l_ok
        + weights.identity * l_id
        + weights.warp * l_warp
    )
    metrics = {
        "loss_total": float(total.detach().cpu()),
        "loss_pix": float(l_pix.detach().cpu()),
        "loss_lowfreq": float(l_low.detach().cpu()),
        "loss_seam": float(l_seam.detach().cpu()),
        "loss_oklab": float(l_ok.detach().cpu()),
        "loss_identity": float(l_id.detach().cpu()),
        "loss_warp": float(l_warp.detach().cpu()),
    }
    return total, metrics

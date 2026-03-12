from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
import torch

from ..models.smic import SMICConfig, SMICCorrectionModel
from ..utils.image_ops import (
    gaussian_blur,
    make_full_mask,
    mask_pil_to_tensor,
    pil_to_tensor,
    split_frequencies,
    tensor_to_pil,
)
from ..utils.warp import radial_inward_flow, warp_image


def _prepare_pil_inputs(
    image: Image.Image,
    anchor_image: Optional[Image.Image] = None,
    mask_image: Optional[Image.Image] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edited = pil_to_tensor(image).unsqueeze(0)
    anchor = pil_to_tensor(anchor_image).unsqueeze(0) if anchor_image is not None else edited.clone()
    if mask_image is not None:
        mask = mask_pil_to_tensor(mask_image).unsqueeze(0)
    else:
        _, _, h, w = edited.shape
        mask = make_full_mask(1, h, w, edited.device, edited.dtype)
    return edited, anchor, mask


@torch.no_grad()
def analytic_compand_correction_tensor(
    image: torch.Tensor,
    anchor_image: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    mp_ratio: float = 1.0,
    strength: float = 0.7,
    blur_sigma: float = 5.0,
) -> torch.Tensor:
    """
    Training-free correction.

    Strategy:
    - apply a small inward radial companding warp
    - restore low-frequency color/contrast from the anchor image if present
    - preserve high-frequency detail from the high-resolution image
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if anchor_image is not None and anchor_image.dim() == 3:
        anchor_image = anchor_image.unsqueeze(0)

    b, _, h, w = image.shape
    if mask is None:
        mask = make_full_mask(b, h, w, image.device, image.dtype)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.to(device=image.device, dtype=image.dtype)

    flow = radial_inward_flow(mask, strength=strength, mp_ratio=mp_ratio)
    warped = warp_image(image, flow)

    split_high = split_frequencies(warped, sigma=blur_sigma)
    if anchor_image is not None:
        if anchor_image.shape[-2:] != image.shape[-2:]:
            anchor_image = torch.nn.functional.interpolate(anchor_image, size=(h, w), mode="bilinear", align_corners=False)
        split_anchor = split_frequencies(anchor_image, sigma=blur_sigma)
        factor = min(max((mp_ratio - 1.0) / 1.25, 0.0), 1.0) * strength
        base = split_high.base * (1.0 - 0.65 * factor) + split_anchor.base * (0.65 * factor)
    else:
        base = split_high.base * (1.0 + 0.04 * strength)
    corrected = (base + split_high.detail).clamp(0.0, 1.0)
    corrected = image * (1.0 - mask) + corrected * mask
    return corrected


@torch.no_grad()
def load_model_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> SMICCorrectionModel:
    payload = torch.load(checkpoint_path, map_location=device)
    cfg_dict = payload.get("model_config", {})
    config = SMICConfig(**cfg_dict) if cfg_dict else SMICConfig()
    model = SMICCorrectionModel(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def correct_tensors(
    model: SMICCorrectionModel,
    image: torch.Tensor,
    anchor_image: Optional[torch.Tensor] = None,
    original_image: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    mp_ratio: float = 1.0,
    strength: float = 1.0,
) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if anchor_image is not None and anchor_image.dim() == 3:
        anchor_image = anchor_image.unsqueeze(0)
    if original_image is not None and original_image.dim() == 3:
        original_image = original_image.unsqueeze(0)

    image = image.to(next(model.parameters()).device)
    if anchor_image is not None:
        anchor_image = anchor_image.to(image.device)
    if original_image is not None:
        original_image = original_image.to(image.device)
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(image.device)

    outputs = model(
        edited=image,
        original=original_image if original_image is not None else image,
        anchor=anchor_image if anchor_image is not None else image,
        mask=mask,
        mp_ratio=mp_ratio,
        strength=strength,
    )
    return outputs["corrected"]


@torch.no_grad()
def correct_image_with_checkpoint(
    image: Image.Image,
    checkpoint_path: str | Path,
    anchor_image: Optional[Image.Image] = None,
    original_image: Optional[Image.Image] = None,
    mask_image: Optional[Image.Image] = None,
    mp_ratio: float = 1.0,
    strength: float = 1.0,
    device: str = "cpu",
) -> Image.Image:
    model = load_model_checkpoint(checkpoint_path, device=device)
    edited = pil_to_tensor(image).unsqueeze(0).to(device)
    anchor = pil_to_tensor(anchor_image).unsqueeze(0).to(device) if anchor_image is not None else edited
    original = pil_to_tensor(original_image).unsqueeze(0).to(device) if original_image is not None else edited
    if mask_image is not None:
        mask = mask_pil_to_tensor(mask_image).unsqueeze(0).to(device)
    else:
        _, _, h, w = edited.shape
        mask = make_full_mask(1, h, w, edited.device, edited.dtype)
    corrected = correct_tensors(
        model=model,
        image=edited,
        anchor_image=anchor,
        original_image=original,
        mask=mask,
        mp_ratio=mp_ratio,
        strength=strength,
    )
    return tensor_to_pil(corrected.cpu())


@torch.no_grad()
def analytic_compand_correction(
    image: Image.Image,
    anchor_image: Optional[Image.Image] = None,
    mask_image: Optional[Image.Image] = None,
    mp_ratio: float = 1.0,
    strength: float = 0.7,
    blur_sigma: float = 5.0,
) -> Image.Image:
    edited = pil_to_tensor(image).unsqueeze(0)
    anchor = pil_to_tensor(anchor_image).unsqueeze(0) if anchor_image is not None else None
    mask = mask_pil_to_tensor(mask_image).unsqueeze(0) if mask_image is not None else None
    corrected = analytic_compand_correction_tensor(
        image=edited,
        anchor_image=anchor,
        mask=mask,
        mp_ratio=mp_ratio,
        strength=strength,
        blur_sigma=blur_sigma,
    )
    return tensor_to_pil(corrected)

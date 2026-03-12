from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flux2_resolution_guard.inference.image import (
    analytic_compand_correction_tensor,
    load_model_checkpoint,
)


def _comfy_image_to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    # Comfy IMAGE is usually BHWC
    if image.shape[-1] == 3:
        return image.permute(0, 3, 1, 2).contiguous()
    if image.shape[1] == 3:
        return image.contiguous()
    raise ValueError(f"Unsupported IMAGE tensor shape: {tuple(image.shape)}")


def _bchw_to_comfy_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    return image.permute(0, 2, 3, 1).contiguous()


def _normalize_mask(mask: Optional[torch.Tensor], height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if mask is None:
        return torch.ones((1, 1, height, width), device=device, dtype=dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        if mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        else:
            mask = mask.unsqueeze(1)
    elif mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask.permute(0, 3, 1, 2)
    mask = mask.to(device=device, dtype=dtype)
    if mask.shape[-2:] != (height, width):
        mask = torch.nn.functional.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    return mask.clamp(0.0, 1.0)


@dataclass
class RGModelHandle:
    model: torch.nn.Module
    device: str
    checkpoint_path: str


class Flux2RGLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {"default": "checkpoints/smic_best.pt"}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("RG_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "FLUX2/Resolution Guard"

    def load_model(self, checkpoint_path: str, device: str):
        model = load_model_checkpoint(checkpoint_path, device=device)
        return (RGModelHandle(model=model, device=device, checkpoint_path=checkpoint_path),)


class Flux2RGApplyCorrection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("RG_MODEL",),
                "image": ("IMAGE",),
                "mp_ratio": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 8.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "anchor_image": ("IMAGE",),
                "mask": ("MASK",),
                "original_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "FLUX2/Resolution Guard"

    def apply(
        self,
        model: RGModelHandle,
        image: torch.Tensor,
        mp_ratio: float,
        strength: float,
        anchor_image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        original_image: Optional[torch.Tensor] = None,
    ):
        net = model.model
        device = next(net.parameters()).device
        image_b = _comfy_image_to_bchw(image).to(device)
        anchor_b = _comfy_image_to_bchw(anchor_image).to(device) if anchor_image is not None else image_b
        original_b = _comfy_image_to_bchw(original_image).to(device) if original_image is not None else image_b
        mask_b = _normalize_mask(mask, image_b.shape[-2], image_b.shape[-1], device, image_b.dtype)

        with torch.no_grad():
            outputs = net(
                edited=image_b,
                original=original_b,
                anchor=anchor_b,
                mask=mask_b,
                mp_ratio=mp_ratio,
                strength=strength,
            )
        return (_bchw_to_comfy_image(outputs["corrected"].cpu()),)


class Flux2RGAnalyticCompand:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mp_ratio": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 8.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 3.0, "step": 0.01}),
                "blur_sigma": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 32.0, "step": 0.1}),
            },
            "optional": {
                "anchor_image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "FLUX2/Resolution Guard"

    def apply(
        self,
        image: torch.Tensor,
        mp_ratio: float,
        strength: float,
        blur_sigma: float,
        anchor_image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        image_b = _comfy_image_to_bchw(image)
        anchor_b = _comfy_image_to_bchw(anchor_image) if anchor_image is not None else None
        mask_b = _normalize_mask(mask, image_b.shape[-2], image_b.shape[-1], image_b.device, image_b.dtype) if mask is not None else None
        with torch.no_grad():
            corrected = analytic_compand_correction_tensor(
                image=image_b,
                anchor_image=anchor_b,
                mask=mask_b,
                mp_ratio=mp_ratio,
                strength=strength,
                blur_sigma=blur_sigma,
            )
        return (_bchw_to_comfy_image(corrected.cpu()),)


NODE_CLASS_MAPPINGS = {
    "Flux2RGLoadModel": Flux2RGLoadModel,
    "Flux2RGApplyCorrection": Flux2RGApplyCorrection,
    "Flux2RGAnalyticCompand": Flux2RGAnalyticCompand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2RGLoadModel": "FLUX2 RG Load Model",
    "Flux2RGApplyCorrection": "FLUX2 RG Apply Correction",
    "Flux2RGAnalyticCompand": "FLUX2 RG Analytic Compand",
}

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from ..utils.image_ops import pil_to_tensor, mask_pil_to_tensor, split_frequencies
from ..utils.warp import radial_inward_flow, warp_image


def _image_files(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


class SyntheticFlux2DriftDataset(Dataset):
    """
    Creates synthetic FLUX.2-like drift examples.

    This is for pretraining the correction model before real FLUX.2 triplet finetuning.
    It supports both whole-image and masked perturbations.
    """

    def __init__(
        self,
        image_dir: str | Path,
        image_size: int = 256,
        whole_image_probability: float = 0.5,
        min_mp_ratio: float = 1.05,
        max_mp_ratio: float = 2.25,
        seed: int = 0,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.whole_image_probability = float(whole_image_probability)
        self.min_mp_ratio = float(min_mp_ratio)
        self.max_mp_ratio = float(max_mp_ratio)
        self.rng = random.Random(seed)
        self.files = _image_files(self.image_dir)
        if not self.files:
            raise FileNotFoundError(f"No images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _random_mask(self, image_size: int) -> Image.Image:
        if self.rng.random() < self.whole_image_probability:
            return Image.new("L", (image_size, image_size), 255)

        mask = Image.new("L", (image_size, image_size), 0)
        draw = ImageDraw.Draw(mask)

        cx = self.rng.randint(image_size // 4, image_size * 3 // 4)
        cy = self.rng.randint(image_size // 4, image_size * 3 // 4)
        rx = self.rng.randint(image_size // 8, image_size // 3)
        ry = self.rng.randint(image_size // 8, image_size // 3)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)

        if self.rng.random() < 0.4:
            pad = self.rng.randint(8, image_size // 10)
            draw.rounded_rectangle((pad, pad, image_size - pad, image_size - pad), radius=pad, outline=255, width=pad // 2 or 1)
        return mask

    def _make_bad(self, original: torch.Tensor, mask: torch.Tensor, mp_ratio: float) -> torch.Tensor:
        original_b = original.unsqueeze(0)
        mask_b = mask.unsqueeze(0)

        # Simulate outward drift by inverting an inward companding field.
        flow = -radial_inward_flow(
            mask_b,
            strength=self.rng.uniform(0.5, 1.0),
            mp_ratio=mp_ratio,
            anisotropy_x=self.rng.uniform(0.95, 1.05),
            anisotropy_y=self.rng.uniform(0.95, 1.05),
        )
        warped = warp_image(original_b, flow)

        # Low-frequency washout biased by MP ratio.
        split = split_frequencies(warped, sigma=4.0)
        base = split.base
        detail = split.detail

        factor = min(max((mp_ratio - 1.0) / 1.25, 0.0), 1.0)
        washed = base * (1.0 - 0.08 * factor) + 0.04 * factor
        oklab_like = washed  # stay simple here; real color correction loss happens during training
        bad = (oklab_like + detail * (1.0 - 0.08 * factor)).clamp(0.0, 1.0)

        # Keep outside-mask region untouched to simulate crop/detail workflows.
        bad = original_b * (1.0 - mask_b) + bad * mask_b
        return bad[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | float]:
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        original = pil_to_tensor(image)
        mask_img = self._random_mask(self.image_size)
        mask = mask_pil_to_tensor(mask_img)
        mp_ratio = self.rng.uniform(self.min_mp_ratio, self.max_mp_ratio)
        edited = self._make_bad(original, mask, mp_ratio=mp_ratio)

        return {
            "original": original,
            "anchor": original.clone(),
            "edited": edited,
            "target": original,
            "mask": mask,
            "mp_ratio": torch.tensor(mp_ratio, dtype=torch.float32),
        }

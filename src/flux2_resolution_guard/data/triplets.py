from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset

from ..utils.image_ops import mask_pil_to_tensor, pil_to_tensor, split_frequencies


class Flux2TripletDataset(Dataset):
    """
    Dataset for FLUX.2 self-distillation triplets.

    Each sample uses:
    - original image
    - anchor image (stable lower-resolution FLUX.2 result)
    - highres image (high-resolution FLUX.2 result with drift)
    - mask (optional)
    - mp_ratio

    The default target is:
      target = lowfreq(anchor) + highfreq(highres)
    """

    def __init__(self, manifest_path: str | Path, image_size: int = 256) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = int(image_size)
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.records: list[dict[str, Any]] = json.load(f)
        if not self.records:
            raise ValueError("Manifest is empty")

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, path: str | Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return pil_to_tensor(image)

    def _load_mask(self, path: str | Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        return mask_pil_to_tensor(image)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        original = self._load_rgb(record["original"])
        anchor = self._load_rgb(record["anchor"])
        highres = self._load_rgb(record["highres"])
        mask = self._load_mask(record["mask"]) if record.get("mask") else torch.ones(1, self.image_size, self.image_size)

        anchor_split = split_frequencies(anchor, sigma=4.0)
        highres_split = split_frequencies(highres, sigma=4.0)
        target = (anchor_split.base + highres_split.detail).clamp(0.0, 1.0)[0]

        return {
            "original": original,
            "anchor": anchor,
            "edited": highres,
            "target": target,
            "mask": mask,
            "mp_ratio": torch.tensor(float(record.get("mp_ratio", 1.0)), dtype=torch.float32),
        }

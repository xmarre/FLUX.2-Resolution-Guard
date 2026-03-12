from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import torch

from flux2_resolution_guard.data import Flux2TripletDataset, SyntheticFlux2DriftDataset


def _write_rgb(path: Path, size=(64, 64), color=(128, 100, 90)):
    Image.new("RGB", size, color).save(path)


def _write_mask(path: Path, size=(64, 64), value=255):
    Image.new("L", size, value).save(path)


def test_synthetic_dataset(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _write_rgb(img_dir / "a.png")
    ds = SyntheticFlux2DriftDataset(img_dir, image_size=64, seed=123)
    sample = ds[0]
    assert sample["edited"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)


def test_triplet_dataset(tmp_path: Path):
    base = tmp_path / "data"
    base.mkdir()
    _write_rgb(base / "orig.png", color=(120, 90, 80))
    _write_rgb(base / "anchor.png", color=(118, 88, 78))
    _write_rgb(base / "highres.png", color=(130, 95, 85))
    _write_mask(base / "mask.png")

    manifest = [
        {
            "original": str(base / "orig.png"),
            "anchor": str(base / "anchor.png"),
            "highres": str(base / "highres.png"),
            "mask": str(base / "mask.png"),
            "mp_ratio": 1.75,
        }
    ]
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ds = Flux2TripletDataset(manifest_path, image_size=64)
    sample = ds[0]
    assert sample["target"].shape == (3, 64, 64)
    assert torch.isfinite(sample["target"]).all()

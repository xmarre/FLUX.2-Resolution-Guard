from __future__ import annotations

import torch

from flux2_resolution_guard.inference.image import analytic_compand_correction_tensor
from flux2_resolution_guard.models import SMICCorrectionModel


def test_analytic_compand_shape():
    image = torch.rand(1, 3, 64, 64)
    anchor = torch.rand(1, 3, 64, 64)
    mask = torch.ones(1, 1, 64, 64)
    out = analytic_compand_correction_tensor(image=image, anchor_image=anchor, mask=mask, mp_ratio=1.6, strength=0.8)
    assert out.shape == image.shape
    assert torch.isfinite(out).all()


def test_checkpointless_model_correction():
    model = SMICCorrectionModel()
    image = torch.rand(1, 3, 64, 64)
    out = model(edited=image, original=image, anchor=image, mask=torch.ones(1, 1, 64, 64), mp_ratio=1.5)
    assert out["corrected"].shape == image.shape

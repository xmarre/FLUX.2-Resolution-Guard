from __future__ import annotations

import torch

from flux2_resolution_guard.models import SMICCorrectionModel, SMICConfig


def test_model_forward_shapes():
    model = SMICCorrectionModel(SMICConfig())
    edited = torch.rand(2, 3, 64, 64)
    original = torch.rand(2, 3, 64, 64)
    anchor = torch.rand(2, 3, 64, 64)
    mask = torch.rand(2, 1, 64, 64)

    out = model(edited=edited, original=original, anchor=anchor, mask=mask, mp_ratio=torch.tensor([1.2, 1.8]))
    assert out["corrected"].shape == (2, 3, 64, 64)
    assert out["flow"].shape == (2, 2, 64, 64)
    assert out["gate"].shape == (2, 1, 64, 64)

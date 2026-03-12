from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainConfig:
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    num_workers: int = 4

    w_pix: float = 1.0
    w_lf: float = 0.75
    w_seam: float = 0.75
    w_oklab: float = 0.5
    w_identity: float = 0.2
    w_warp: float = 0.1

    warp_grid_size: int = 16

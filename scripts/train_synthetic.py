#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader, random_split

from flux2_resolution_guard.config import TrainConfig
from flux2_resolution_guard.data import SyntheticFlux2DriftDataset
from flux2_resolution_guard.models import SMICConfig, SMICCorrectionModel
from flux2_resolution_guard.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = SyntheticFlux2DriftDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
    )
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = max(1, len(dataset) - n_val)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SMICCorrectionModel(SMICConfig())
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        config=TrainConfig(
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    main()

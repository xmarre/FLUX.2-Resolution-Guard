from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..models.smic import SMICCorrectionModel, SMICConfig
from .losses import LossWeights, total_loss


class Trainer:
    def __init__(
        self,
        model: SMICCorrectionModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str | Path = "runs/default",
        config: Optional[TrainConfig] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or TrainConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() or self.config.device == "cpu" else "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.weights = LossWeights(
            pix=self.config.w_pix,
            lowfreq=self.config.w_lf,
            seam=self.config.w_seam,
            oklab=self.config.w_oklab,
            identity=self.config.w_identity,
            warp=self.config.w_warp,
        )
        self.best_val = float("inf")

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _step(self, batch: dict[str, torch.Tensor], train: bool = True) -> dict[str, float]:
        batch = self._move_batch(batch)
        outputs = self.model(
            edited=batch["edited"],
            original=batch["original"],
            anchor=batch["anchor"],
            mask=batch["mask"],
            mp_ratio=batch["mp_ratio"],
        )
        loss, metrics = total_loss(
            corrected=outputs["corrected"],
            target=batch["target"],
            edited=batch["edited"],
            flow=outputs["flow"],
            mask=outputs["mask"],
            mp_ratio=batch["mp_ratio"],
            weights=self.weights,
        )
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        return metrics

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        all_metrics: list[dict[str, float]] = []
        torch.set_grad_enabled(train)
        for batch in loader:
            metrics = self._step(batch, train=train)
            all_metrics.append(metrics)
        torch.set_grad_enabled(True)
        if not all_metrics:
            return {"loss_total": 0.0}
        keys = all_metrics[0].keys()
        return {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        path = self.output_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "model_config": self.model.config.__dict__,
                "metrics": metrics,
            },
            path,
        )

    def fit(self) -> None:
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

            if self.val_loader is not None:
                val_metrics = self._run_epoch(self.val_loader, train=False)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})
                if val_metrics["loss_total"] < self.best_val:
                    self.best_val = val_metrics["loss_total"]
                    self._save_checkpoint("smic_best.pt", epoch, val_metrics)
            else:
                self._save_checkpoint("smic_last.pt", epoch, train_metrics)

            history.append(record)
            with (self.output_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

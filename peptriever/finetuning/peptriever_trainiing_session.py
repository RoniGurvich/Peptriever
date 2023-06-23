import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from peptriever.acceleration import get_device
from peptriever.finetuning.session_metrics import FinetuningSessionMetrics
from peptriever.tensorboard_writer import Tensorboard


@dataclass
class SessionState:
    epoch: int = 0
    batch_i: int = 0
    min_val_loss: float = 1e8


@dataclass
class SessionParams:
    grad_accumulation: int = 1


class PeptrieverTrainingSession:
    def __init__(
        self,
        model,
        model_name: str,
        loss_func,
        optimizer: torch.optim.Optimizer,
        dist_metric: str,
        output_path: Path,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        session_params: SessionParams = SessionParams(),
    ):
        self.device = get_device()
        self.model = model.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_name = model_name
        self.tensorboard = Tensorboard(
            output_dir=f"tensorboard/{model_name}", prefix="peptriever"
        )
        self.state = SessionState()
        self.train_metrics = FinetuningSessionMetrics(dist_metric=dist_metric)
        self.val_metrics = FinetuningSessionMetrics(dist_metric=dist_metric)
        self.session_params = session_params
        self.output_path = output_path

    def train(self, train_data: DataLoader, val_data: DataLoader, epochs: int):
        for epoch in range(epochs):
            self._reset_metrics_and_optimizer()
            self.state.epoch = epoch
            self.train_epoch(train_data)
            self.val_epoch(val_data)
            self._maybe_lr_scheduler_step()
            self.save_last_and_best()

    def _maybe_lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _reset_metrics_and_optimizer(self):
        for metrics in (self.train_metrics, self.val_metrics):
            metrics.reset()
        self.optimizer.zero_grad(set_to_none=True)

    def train_epoch(self, data):
        self.model.train()

        w_bar = tqdm(data)
        for batch_i, batch in enumerate(w_bar):
            self.state.batch_i = batch_i
            loss, outputs, labels = self._batch_forward(batch)
            self._batch_backward(loss, batch_i)
            self._update_metrics(loss, outputs, labels, train=True)
            self._update_status_msg(w_bar, train=True)

        self._log_metrics(train=True)

    def val_epoch(self, data):
        self.model.eval()

        with torch.no_grad():
            w_bar = tqdm(data)
            for batch_i, batch in enumerate(w_bar):
                self.state.batch_i = batch_i
                loss, outputs, labels = self._batch_forward(batch)
                self._update_metrics(loss, outputs, labels, train=False)
                self._update_status_msg(w_bar, train=False)
            self._log_metrics(train=False)

    def _update_status_msg(self, w_bar, train: bool):
        if train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics
        msg = f"Epoch: {self.state.epoch} loss: {metrics['loss']:.4f}"
        w_bar.set_description(msg)

    def _update_metrics(self, loss, outputs, labels, train: bool):
        if train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        metrics.update("loss", value=loss.detach().cpu())
        metrics.calc(outputs, labels)

    def _batch_forward(self, batch):
        outputs = self.model(batch["x1"].to(self.device), batch["x2"].to(self.device))
        labels = (batch["labels1"].to(self.device), batch["labels2"].to(self.device))
        loss = self.loss_func(outputs, labels)
        return loss, outputs, labels

    def _batch_backward(self, loss, batch_i):
        adjusted_loss = loss / self.session_params.grad_accumulation
        adjusted_loss.backward()
        if batch_i > 0 and batch_i % self.session_params.grad_accumulation == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def _log_metrics(self, train: bool):
        if train:
            metrics = self.train_metrics
            prefix = ""
        else:
            metrics = self.val_metrics
            prefix = "val_"
        scalars = {}
        for key, value in metrics.items():
            scalars[prefix + key] = value

        if train and self.lr_scheduler is not None:
            lr = self.lr_scheduler.get_last_lr()[0]
            scalars["lr"] = lr
        self.tensorboard.write_scalars(scalars, increment_step=not train)

    @property
    def output_models_path(self):
        return self.output_path / self.model_name

    def save(self, output_path: Path):
        os.makedirs(output_path, exist_ok=True)
        self.model.save_pretrained(output_path)

    def save_last_and_best(self):
        self.save(self.output_models_path / "last")
        val_loss = self.val_metrics["loss"]
        if val_loss < self.state.min_val_loss:
            self.state.min_val_loss = val_loss
            best_path = self.output_models_path / "best"
            self.save(best_path)
            print(f"saving best model {best_path}")

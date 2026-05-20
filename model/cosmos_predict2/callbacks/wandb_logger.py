import torch
import wandb

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed


class WandbLogger(EveryN):
    """Callback that logs training metrics to Weights & Biases.

    Logs every ``every_n`` optimizer steps (same cadence as IterSpeed).
    Metrics logged:
      - All scalar values from ``output_batch`` (loss, Var[x_0], …)
      - Learning rate
      - Gradient / weight norms (when GradClip has ``log_wandb=True``)

    Respects ``WANDB_MODE=disabled`` — no-ops silently in that case.

    Args:
        every_n: Log every this many optimizer steps.
        project: W&B project name. Falls back to ``config.job.project`` at runtime.
        name: W&B run name. Falls back to ``config.job.name`` at runtime.
        tags: Optional list of tags to attach to the run.
    """

    def __init__(
        self,
        every_n: int = 100,
        project: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        super().__init__(every_n=every_n, barrier_after_run=False)
        self._project = project
        self._name = name
        self._tags = tags
        self._last_lr: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not distributed.is_rank0():
            return
        if wandb.run is not None:
            return  # already initialised (e.g. resumed run)

        project = self._project or getattr(getattr(self, "config", None), "job", None) and self.config.job.project
        name = self._name or getattr(getattr(self, "config", None), "job", None) and self.config.job.name

        wandb.init(
            entity="julie-trrsr",
            project=project or "cosmos-predict2",
            name=name or None,
            tags=self._tags,
            resume="allow",
        )

    def on_before_optimizer_step(
        self,
        model_ddp,
        optimizer: torch.optim.Optimizer,
        scheduler,
        grad_scaler,
        iteration: int = 0,
    ) -> None:
        if optimizer.param_groups:
            self._last_lr = optimizer.param_groups[0]["lr"]

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if distributed.is_rank0() and wandb.run is not None:
            wandb.finish()

    # ------------------------------------------------------------------
    # Per-N logging
    # ------------------------------------------------------------------

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        if not distributed.is_rank0():
            return
        if wandb.run is None:
            return

        metrics: dict[str, float] = {}
        for key, value in output_batch.items():
            if isinstance(value, (int, float)):
                metrics[f"train/{key}"] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[f"train/{key}"] = value.item()

        if self._last_lr is not None:
            metrics["train/lr"] = self._last_lr

        if metrics:
            wandb.log(metrics, step=iteration)
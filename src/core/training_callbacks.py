"""
Training Callbacks System

Provides hooks for epoch-level events during training.
Allows strategies to report progress and trainers to respond (e.g., save checkpoints).
"""

from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
import logging

class TrainingCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_epoch_end(
        self,
        epoch: int,
        task_idx: int,
        task_name: str,
        metrics: Dict[str, float],
        is_final: bool = False
    ):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            task_idx: Current task index
            task_name: Current task name
            metrics: Dictionary of metrics (e.g., {'train_loss': 0.5, 'val_accuracy': 0.85})
            is_final: Whether this is the final epoch of the task
        """
        pass

    def on_task_start(self, task_idx: int, task_name: str):
        """Called when starting a new task."""
        pass

    def on_task_end(self, task_idx: int, task_name: str):
        """Called when finishing a task."""
        pass

class CallbackList:
    """Manages a list of callbacks."""

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: TrainingCallback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_epoch_end(
        self,
        epoch: int,
        task_idx: int,
        task_name: str,
        metrics: Dict[str, float],
        is_final: bool = False
    ):
        """Trigger all callbacks for epoch end."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, task_idx, task_name, metrics, is_final)

    def on_task_start(self, task_idx: int, task_name: str):
        """Trigger all callbacks for task start."""
        for callback in self.callbacks:
            callback.on_task_start(task_idx, task_name)

    def on_task_end(self, task_idx: int, task_name: str):
        """Trigger all callbacks for task end."""
        for callback in self.callbacks:
            callback.on_task_end(task_idx, task_name)

class CheckpointCallback(TrainingCallback):
    """
    Callback that saves checkpoints using CheckpointManager.

    This is the bridge between the training loop and checkpoint management.
    """

    def __init__(
        self,
        checkpoint_manager,
        model_state_fn: Callable,
        strategy_state_fn: Callable,
        metric_key: str = 'val_accuracy',
        additional_state_fn: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize CheckpointCallback.

        Args:
            checkpoint_manager: CheckpointManager instance
            model_state_fn: Function that returns model state dict
            strategy_state_fn: Function that returns strategy state dict
            metric_key: Key in metrics dict to use for checkpoint comparison
            additional_state_fn: Optional function that returns additional state to save
            logger: Logger instance
        """
        self.checkpoint_manager = checkpoint_manager
        self.model_state_fn = model_state_fn
        self.strategy_state_fn = strategy_state_fn
        self.metric_key = metric_key
        self.additional_state_fn = additional_state_fn
        self.logger = logger or logging.getLogger(__name__)

    def on_epoch_end(
        self,
        epoch: int,
        task_idx: int,
        task_name: str,
        metrics: Dict[str, float],
        is_final: bool = False
    ):
        """Save checkpoint at the end of each epoch."""
        # Get the metric value for this epoch
        if self.metric_key not in metrics:
            self.logger.warning(
                f"Metric '{self.metric_key}' not found in metrics. "
                f"Available: {list(metrics.keys())}. Skipping checkpoint save."
            )
            return

        metric_value = metrics[self.metric_key]

        # Get states
        model_state = self.model_state_fn()
        strategy_state = self.strategy_state_fn()

        # Get additional state if provided
        additional_state = None
        if self.additional_state_fn:
            additional_state = self.additional_state_fn()

        # Save checkpoint
        saved_paths = self.checkpoint_manager.save_checkpoint(
            task_idx=task_idx,
            task_name=task_name,
            epoch=epoch,
            model_state=model_state,
            strategy_state=strategy_state,
            metric_value=metric_value,
            additional_state=additional_state,
            is_final=is_final
        )

        # Log saved checkpoints
        checkpoint_types = list(saved_paths.keys())
        self.logger.debug(f"Saved checkpoints: {', '.join(checkpoint_types)}")

class MetricsLoggingCallback(TrainingCallback):
    """Callback that logs metrics during training."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def on_epoch_end(
        self,
        epoch: int,
        task_idx: int,
        task_name: str,
        metrics: Dict[str, float],
        is_final: bool = False
    ):
        """Log metrics at the end of each epoch."""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        epoch_type = "final" if is_final else ""
        self.logger.info(
            f"Task {task_idx} ({task_name}) - Epoch {epoch + 1} {epoch_type}: {metrics_str}"
        )

    def on_task_start(self, task_idx: int, task_name: str):
        """Log task start."""
        self.logger.info(f"=" * 60)
        self.logger.info(f"Starting Task {task_idx}: {task_name}")
        self.logger.info(f"=" * 60)

    def on_task_end(self, task_idx: int, task_name: str):
        """Log task end."""
        self.logger.info(f"Completed Task {task_idx}: {task_name}")
        self.logger.info(f"=" * 60)

class EarlyStoppingCallback(TrainingCallback):
    """
    Callback that implements early stopping based on validation metrics.

    Note: This doesn't stop training directly but can signal to stop.
    """

    def __init__(
        self,
        metric_key: str = 'val_accuracy',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'max',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize EarlyStoppingCallback.

        Args:
            metric_key: Metric to monitor
            patience: Number of epochs with no improvement before signaling stop
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics where higher is better, 'min' for lower is better
            logger: Logger instance
        """
        self.metric_key = metric_key
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)

        # State
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.wait = 0
        self.should_stop = False

    def on_task_start(self, task_idx: int, task_name: str):
        """Reset state for new task."""
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')
        self.wait = 0
        self.should_stop = False

    def on_epoch_end(
        self,
        epoch: int,
        task_idx: int,
        task_name: str,
        metrics: Dict[str, float],
        is_final: bool = False
    ):
        """Check if training should stop early."""
        if self.metric_key not in metrics:
            return

        current_metric = metrics[self.metric_key]

        # Check if there's improvement
        if self.mode == 'max':
            improved = current_metric > self.best_metric + self.min_delta
        else:
            improved = current_metric < self.best_metric - self.min_delta

        if improved:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience: {self.patience})"
                )

    def reset(self):
        """Reset early stopping state."""
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')
        self.wait = 0
        self.should_stop = False

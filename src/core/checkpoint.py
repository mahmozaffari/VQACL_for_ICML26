"""
Checkpoint Handling Module

This module provides a robust checkpoint manager that:
- Saves checkpoints after every epoch
- Tracks and saves the best checkpoint based on validation metrics
- Supports multiple checkpoint retention strategies
- Handles cleanup of old checkpoints
"""

import os
import json
import torch
import shutil
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from enum import Enum

class CheckpointMetric(Enum):
    """Metrics for determining best checkpoint."""
    ACCURACY = 'accuracy'
    LOSS = 'loss'

    def is_better(self, new_value: float, old_value: float) -> bool:
        """Determine if new value is better than old value."""
        if self == CheckpointMetric.ACCURACY:
            return new_value > old_value
        elif self == CheckpointMetric.LOSS:
            return new_value < old_value
        else:
            raise ValueError(f"Unknown metric: {self}")


@dataclass
class CheckpointMetaData:
    """Metadata for a checkpoint."""
    task_idx: int
    task_name: str
    epoch: int
    metric_name: str
    metric_value: float
    timestamp: str
    checkpoint_type: str #'best', 'latest', 'epoch'
    global_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and cleanup.

    Features:
    - saves checkpoints after every epoch
    - tracks and saves best checkpoint based on validation metrics
    - keeps last N checkpoints
    - automatic cleanup of old checkpoints
    """

    def __init__(self,
                 checkpoint_dir: str,
                 metric: CheckpointMetric = CheckpointMetric.ACCURACY,
                 keep_last_n_checkpoints: int = 3,
                 save_every_n_epochs: int = 1,
                 logger: Optional[logging.Logger] = None):

        """Initialize CheckpointManager.
        Args:
            checkpoint_dir: Directory to save checkpoints
            metric: Metric to determine best checkpoint
            keep_last_n_checkpoints: Number of last checkpoints to keep
            save_every_n_epochs: Frequency of saving checkpoints
            logger: Optional logger for logging messages
        """

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric = metric
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.save_every_n_epochs = save_every_n_epochs
        self.logger = logger or logging.getLogger(__name__)

        # Track best metrics per task
        self.best_metrics: Dict[int, float] = {}

        # Track all checkpoints for cleanup
        self.checkpoint_history: Dict[int, List[Path]] = {}

        # Initialize best metric values
        self._init_best_metrics()

    def get_best_checkpoint_path(self, task_name: str) -> Optional[Path]:
        """Get the path to the best checkpoint for a given task."""
        checkpoint_name = f"{task_name}_best.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path
        return None

    def _init_best_metrics(self):
        """Initialize best metric values based on metric type."""
        if self.metric == CheckpointMetric.ACCURACY:
            self.best_metric_default = float('-inf')
        elif self.metric == CheckpointMetric.LOSS:
            self.best_metric_default = float('inf')
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def save_checkpoint(self,
                        task_idx: int,
                        task_name: str,
                        epoch: int,
                        model_state: Dict[str, Any],
                        strategy_state: Dict[str, Any],
                        metric_value: float,
                        additional_state: Optional[Dict[str, Any]] = None,
                        is_final: bool = False) -> Dict[str, Any]:
        """Save checkpoint after an epoch.

        Args:
        - task_idx: Index of the current task
        - task_name: Name of the current task
        - epoch: Current epoch number
        - model_state: State dict of the model
        - strategy_state: State dict of the training strategy
        - metric_value: Validation metric value for this epoch
        - additional_state: Any additional state to save
        - is_final: If True, indicates this is the final checkpoint for the task

        Returns:
            Dictionary mapping checkpint types to their paths.
        """

        saved_paths = {}

        # Always save latest checkpoint
        if epoch % self.save_every_n_epochs == 0 or is_final:
            latest_path = self._save_checkpoint_file(
                task_idx, task_name, epoch, model_state,
                strategy_state, metric_value, additional_state, checkpoint_type='latest')
            saved_paths['latest'] = latest_path

            # Epoch-specific checkpoints are disabled.
            if False:
                epoch_path = self._save_checkpoint_file(
                    task_idx, task_name, epoch, model_state,
                    strategy_state, metric_value, additional_state, checkpoint_type=f'epoch_{epoch}')

                saved_paths[f'epoch_{epoch}'] = epoch_path

                # Track for cleanup
                if task_idx not in self.checkpoint_history:
                    self.checkpoint_history[task_idx] = []
                self.checkpoint_history[task_idx].append(epoch_path)

        # Check if this is the best checkpoint
        if self._is_best_checkpoint(task_idx, metric_value):
            self.best_metrics[task_idx] = metric_value
            best_path = self._save_checkpoint_file(
                task_idx, task_name, epoch, model_state, strategy_state,
                metric_value, additional_state, checkpoint_type='best'
            )
            saved_paths['best'] = best_path
            self.logger.info(
                f"New best checkpoint for task {task_name} "
                f"(epoch {epoch}): {self.metric.value}={metric_value:.4f}"
            )

        # Save final checkpoint if this is the last epoch
        #         task_idx, task_name, epoch, model_state, strategy_state,

        # Cleanup of old epoch-specific checkpoints is disabled.
        if False:
            self._cleanup_old_checkpoints(task_idx)

        return saved_paths

    def _save_checkpoint_file(
        self,
        task_idx: int,
        task_name: str,
        epoch: int,
        model_state: Dict[str, Any],
        strategy_state: Dict[str, Any],
        metric_value: float,
        additional_state: Optional[Dict[str, Any]],
        checkpoint_type: str
    ) -> Path:

        """Save a single checkpoint file."""
        checkpoint_name = f"{task_name}_{checkpoint_type}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Create metadata
        metadata = CheckpointMetaData(
            task_idx=task_idx,
            task_name=task_name,
            epoch=epoch,
            metric_name=self.metric.value,
            metric_value=metric_value,
            timestamp=datetime.now().isoformat(),
            checkpoint_type=checkpoint_type
        )

        # Prepare checkpoint
        checkpoint = {
            'task_idx': task_idx,
            'task_name': task_name,
            'epoch': epoch,
            'model_state': model_state,
            'strategy_state': strategy_state,
            'metric_name': self.metric.value,
            'metric_value': metric_value,
            'metadata': metadata.to_dict(),
            'checkpoint_type': checkpoint_type,
        }

        # Add additional state if provided
        if additional_state:
            checkpoint.update(additional_state)

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Save metadata as JSON for easy inspection
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        self.logger.info(
            f"Saved {checkpoint_type} checkpoint: {checkpoint_path.name} "
            f"(epoch {epoch}, {self.metric.value}={metric_value:.4f})"
        )

        return checkpoint_path

    def _is_best_checkpoint(self, task_idx: int, metric_value: float) -> bool:
        """Check if current metric is the best so far."""
        if metric_value is None:
            return False

        if task_idx not in self.best_metrics:
            return True

        current_best = self.best_metrics[task_idx]
        return self.metric.is_better(metric_value, current_best)

    def _cleanup_old_checkpoints(self, task_idx: int):
        """Remove old epoch checkpoints, keeping only the last N."""
        if self.keep_last_n_checkpoints <= 0:
            return  # keep all

        if task_idx not in self.checkpoint_history:
            return

        checkpoints = self.checkpoint_history[task_idx]

        if len(checkpoints) > self.keep_last_n_checkpoints:
            # Sort by modification time
            checkpoints.sort(key=lambda p: p.stat().st_mtime)

            # Remove oldest checkpoints
            to_remove = checkpoints[:-self.keep_last_n_checkpoints]
            for checkpoint_path in to_remove:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    # Remove metadata file
                    metadata_path = checkpoint_path.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    self.logger.info(f"Remove old checkpoint: {checkpoint_path.name}")

            # Update history
            self.checkpoint_history[task_idx] = checkpoints[-self.keep_last_n_checkpoints:]

    def load_checkpoint(
        self,
        task_idx: int,
        task_name: str,
        checkpoint_type: str = 'best',
        checkpoint_dir: Optional[str] = None,   # path to checkpoint directory: default to self.checkpoint_dir (in args.output_dir/checkpoints)
        checkpoint_path: Optional[Path] = None  # path to specific checkpoint file
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        Args:
            task_idx: Index of the task
            task_name: Name of the task
            checkpoint_type: Type of checkpoint to load ('best', 'latest', 'final', or 'epoch_X')
        Returns:
            Loaded checkpoint dictionary.
        """
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            self.logger.info(f"Loading checkpoint from specified path: {checkpoint_path.name}")

        else:
            # Look for a checkpoint inside checkpoint directory

            if checkpoint_dir is None:
                checkpoint_dir = self.checkpoint_dir

            checkpoint_name = f"{task_name}_{checkpoint_type}.pth"
            checkpoint_path = checkpoint_dir / checkpoint_name

            if not checkpoint_path.exists():
                # Try alternative checkpoint types
                alternatives = ['best', 'latest', 'final']
                for alt_type in alternatives:
                    if alt_type != checkpoint_type:
                        alt_path = checkpoint_dir / f"{task_name}_{alt_type}.pth"

                        if alt_path.exists():
                            self.logger.warning(
                                f"Requested checkpoint '{checkpoint_type}' not found. "
                                f"Loading alternative '{alt_type}' instead."
                            )
                            checkpoint_path = alt_path
                            break

                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        return checkpoint

    def get_best_metric(self, task_idx: int) -> Optional[float]:
        """Get the best metric value for a task"""
        return self.best_metrics.get(task_idx, None)

    def get_checkpoint_info(self, task_name: str, checkpoint_type: str = 'best') -> Dict[str, Any]:
        """Get metadata info for a specific checkpoint without loading the full checkpoint."""
        checkpoint_name = f"{task_name}_{checkpoint_type}.pth"
        metadata_path = (self.checkpoint_dir / checkpoint_name).with_suffix('.json')

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return {}

    def list_checkpoints(self, task_name: Optional[str] = None) -> List[Path]:
        """List all checkpoints in the checkpoint directory."""
        pattern = f"{task_name}_*.pth" if task_name else "*.pth"
        return sorted(self.checkpoint_dir.glob(pattern))

    def __repr__(self) -> str:
        return (
            f"CheckpointManager(dir={self.checkpoint_dir}, "
            f"metric={self.metric}, "
            f"keep_last_n_checkpoints={self.keep_last_n_checkpoints}, "
        )

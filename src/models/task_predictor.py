"""
Task predictors for MoE-based continual learning.

The oracle predictor routes each sample to the expert of its ground-truth task.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

@dataclass
class TaskIDClassifierConfig:
    """Configuration for task-ID classifier."""
    num_tasks: int = 1
    use_distillation: bool = False
    temperature: float = 2.0
    distill_weight: float = 0.5

class TaskPredictorBase(nn.Module):
    """Base class for task predictors."""

    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.is_oracle = False  # Flag to indicate if this is an oracle predictor

    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for task prediction.

        Args:
            batch: Input batch (may contain 'features', 'questions', etc.)
            ground_truth_task_ids: Ground truth task IDs (optional, for oracle predictors)

        Returns:
            Dictionary with predictions, confidences, and probabilities
        """
        raise NotImplementedError

    def predict(self, batch: Dict[str, Any], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict task IDs and return predictions, confidences, and probabilities.

        Args:
            batch: Input batch
            ground_truth_task_ids: Ground truth task IDs (optional)

        Returns:
            Tuple of (predictions, confidences, probabilities)
        """
        outputs = self.forward(batch, **kwargs)
        return outputs['predictions'], outputs['confidences'], outputs['probabilities']

    def _get_device_from_batch(self, batch: Dict[str, Any]) -> Optional[torch.device]:
        """
        Extract device from any tensor in the batch.

        Args:
            batch: Input batch dictionary

        Returns:
            Device if found, None otherwise
        """
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.device
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                if isinstance(value[0], torch.Tensor):
                    return value[0].device
        return None

class PerfectTaskPredictor(TaskPredictorBase):
    """
    Oracle task predictor that uses ground truth task IDs.

    Used for oracle routing, which sends each sample to the expert of its own task.
    """

    def __init__(self, num_tasks: int):
        super().__init__(num_tasks)
        self.logger = logging.getLogger('CL.strategy.oracle_task_predictor')
        self.confidence_score = 1.0  # Perfect confidence (hardcoded)
        self.is_oracle = True  # Mark as oracle predictor

    def forward(self, batch: torch.Tensor, ground_truth_task_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict task IDs (using ground truth for perfect prediction).

        Args:
            features: Input features [batch_size, feature_dim]
            ground_truth_task_ids: Ground truth task IDs [batch_size] (for perfect prediction)

        Returns:
            Dictionary with predictions, confidences, and probabilities
        """
        # Simply use the device and batch size from ground_truth_task_ids
        device = ground_truth_task_ids.device
        batch_size = ground_truth_task_ids.size(0)

        if ground_truth_task_ids is not None:
            # Use ground truth for perfect prediction
            predicted_task_ids = ground_truth_task_ids
        else:
            # Default to task 0 if no ground truth (fallback)
            raise ValueError("Ground truth task IDs must be provided for PerfectTaskPredictor")

        # Create perfect confidence scores
        confidences = torch.full((batch_size,), self.confidence_score, device=device)

        # Create one-hot probabilities (perfect certainty)
        probabilities = torch.zeros(batch_size, self.num_tasks, device=device)
        probabilities.scatter_(1, predicted_task_ids.unsqueeze(1), 1.0)

        return {
            'predictions': predicted_task_ids,
            'confidences': confidences,
            'probabilities': probabilities,
            'logits': torch.log(probabilities + 1e-8)  # Convert to logits
        }

    def predict(self, batch: Dict[str, Any], ground_truth_task_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict task IDs and return predictions, confidences, and probabilities.

        Returns:
            Tuple of (predictions, confidences, probabilities)
        """
        outputs = self.forward(batch, ground_truth_task_ids)
        return outputs['predictions'], outputs['confidences'], outputs['probabilities']

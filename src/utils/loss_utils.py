"""
Centralized loss computation for VQA continual learning.
Handles different target formats (soft labels, hard labels) and loss functions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import torch.nn.functional as F
import logging

class LossManager:
    """
    Centralized loss computation manager.
    Handles BCE loss for soft targets and CrossEntropy for hard targets.
    """

    def __init__(self, config, device: Optional[torch.device] = None):
        """
        Initialize the loss manager.

        Args:
            device: Device to move tensors to (if needed)
        """
        self.config = config
        self.device = device
        self.vqa_loss_weight = config.vqa_loss_weight

        # DRO options (all optional)
        self.use_dro_loss = getattr(config, "use_dro_loss", False)
        self.dro_lambda = float(getattr(config, "dro_lambda", 1.0))

        self._eps = 1e-12

    def compute_ce_vqa_loss(self, outputs, batch, mask=None):
        """Cross-Entropy VQA loss for hard labels"""
        logits = outputs.logits  # (batch_size, num_classes)
        targets = batch['targets']  # (batch_size,)
        assert targets.dim() == 2, "Targets should be 1D (batch_size,)"
        if mask is not None:
            logits = logits[:, mask]
            targets = targets[:, mask]

        return F.cross_entropy(logits, targets)

    def compute_dro_vqa_loss(self, outputs, batch, mask=None, scale=False, task_id: Optional[int] = None):
        """
        DRO loss:
          elem = BCEWithLogits(reduction='none')   # (B, C)
          per_sample = elem.sum(dim=1)             # (B,)
          weights = softmax((per_sample - max)/lambda)
          return sum(weights * per_sample)

        Scaling:
          - If scale=False: divide per-sample sum by num_classes to match your current BCE(mean) scale
          - If scale=True : keep sum over classes (matches your scale_vqa_loss behavior)
        """
        logits = outputs.logits
        targets = batch["targets"]

        if mask is not None:
            logits = logits[:, mask]
            targets = targets[:, mask]

        elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B,C)
        per_sample = elem.sum(dim=1)  # (B,)

        if not scale:
            C = elem.size(1)
            per_sample = per_sample / max(C, 1)

        lam = self.dro_lambda
        if lam <= 0:
            raise ValueError(f"dro_lambda must be > 0, got {lam}")

        mx = torch.max(per_sample)
        weights = torch.softmax((per_sample - mx) / (lam + self._eps), dim=0)  # (B,)

        return torch.sum(weights * per_sample)

    def compute_vqa_loss(self, outputs, batch, mask=None, scale=False):

        if self.use_dro_loss:
            return self.compute_dro_vqa_loss(outputs, batch, mask, scale)
        """Standard VQA loss"""
        logits = outputs.logits
        targets = batch['targets']
        if mask is not None:
            logits = logits[:, mask]
            targets = targets[:, mask]
            if targets.sum() == 0:
                print("All target values are zero after applying mask.")

        loss = F.binary_cross_entropy_with_logits(logits, targets) # * targets.shape[1]
        if scale:
            loss = loss * targets.shape[1]
        return loss #F.binary_cross_entropy_with_logits(logits, targets) # * targets.shape[1]

    def compute_taskid_loss(self, routing_outputs, batch):
        """Task-ID classification loss"""
        if 'task_ids' not in batch:
            return torch.tensor(0.0, device=batch['input_ids'].device)

        pred_logits = routing_outputs['task_predictions']['logits']
        true_task_ids = batch['task_ids']
        return F.cross_entropy(pred_logits, true_task_ids)

    def compute_routing_entropy_loss(self, routing_outputs):
        """Encourage confident routing (optional)"""
        routing_weights = routing_outputs['routing_weights']
        entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-8), dim=-1)
        return entropy.mean()

    def compute_total_loss(self, outputs, routing_outputs, batch, mode='staged', mask=None, scale=False):
        """
        Compute total loss based on training mode

        Args:
            mode: 'staged' (only VQA) or 'end-to-end' (VQA + task prediction)
        """
        losses = {}

        # VQA loss (always computed)
        losses['vqa_loss'] = self.compute_vqa_loss(outputs, batch, mask, scale)

        total_loss = losses['vqa_loss']

        losses['total_loss'] = total_loss
        return losses['total_loss']

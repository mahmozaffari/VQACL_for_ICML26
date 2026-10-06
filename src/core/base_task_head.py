"""
Abstract base class for task heads in continual learning.

This module provides a unified interface for different task head management strategies:
- Separate task heads (MoE)
- Expanding task head
- Single joint task head
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TaskHeadOutput:
    """Output from a task head."""
    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    routing_weights: Optional[torch.Tensor] = None
    expert_usage: Optional[Dict[str, Any]] = None

class TaskHeadBase(ABC, nn.Module):
    """Abstract base class for task head management."""

    def __init__(self, input_size: int, hidden_size: int = None, logger: Optional[Any] = None):
        """
        Initialize task head base.

        Args:
            input_size: Hidden size of input features
        """
        super().__init__()
        self.input_size = input_size
        if hidden_size is None:
            self.hidden_size = 2 * input_size
        else:
            self.hidden_size = hidden_size
        self.task_info: Dict[int, Dict[str, Any]] = {}
        self.logger = logger

    @abstractmethod
    def add_task(self, task_id: int, task_name: str, num_classes: int,
                 label2ans: Optional[List[str]] = None, device = None) -> None:
        """
        Add a new task to the head manager.

        Args:
            task_id: Task identifier
            task_name: Task name
            num_classes: Number of classes for this task
            label2ans: Mapping from label index to answer string
        """
        pass

    @abstractmethod
    def forward(self, features: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass through task-specific head.

        Args:
            features: Input features [batch_size, input_size]
            task_id: Task identifier

        Returns:
            Logits [batch_size, num_classes_for_task]
        """
        pass

    @abstractmethod
    def get_answer_from_logits(self, logits: torch.Tensor, task_id: int,
                               return_indices: bool = False) -> Any:
        """
        Convert logits to actual answers (strings or indices).

        Args:
            logits: Model logits [batch_size, num_classes]
            task_id: Task identifier
            return_indices: If True, return prediction indices; if False, return answer strings

        Returns:
            Either prediction indices or answer strings based on return_indices flag
        """
        pass

    @abstractmethod
    def get_num_classes(self, task_id: int) -> int:
        """
        Get number of classes for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            Number of classes
        """
        pass

    def freeze_task(self, task_id: int) -> None:
        """
        Freeze parameters for a specific task.

        Args:
            task_id: Task to freeze
        """
        pass

    def unfreeze_task(self, task_id: int) -> None:
        """
        Unfreeze parameters for a specific task.

        Args:
            task_id: Task to unfreeze
        """
        pass

    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """
        Get trainable parameters for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            List of parameters for the task
        """
        return []

    def get_head_parameters(self) -> List[nn.Parameter]:
        """
        Get all head parameters.
        """
        pass

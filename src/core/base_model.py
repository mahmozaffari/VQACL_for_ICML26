"""
Abstract base class for model wrappers in continual learning.

This class provides a unified interface for different models to work with CL strategies.
It handles task-specific components, parameter management, and forward pass logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from dataclasses import dataclass
from core.base_task_head import TaskHeadBase

@dataclass
class ModelOutput:
    """Standardized output from model forward pass."""
    logits: torch.Tensor  # Raw logits for current prediction
    probabilities: torch.Tensor  # Softmax probabilities
    features: Optional[torch.Tensor] = None  # Intermediate features
    attention_weights: Optional[torch.Tensor] = None  # Attention weights
    task_specific_outputs: Optional[Dict[str, torch.Tensor]] = None  # Task-specific outputs

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = torch.softmax(self.logits, dim=-1)

class BaseModelWrapper(ABC, nn.Module):
    """Abstract base class for model wrappers in continual learning."""

    def __init__(self, base_model: nn.Module, args, task_head_type: str = 'separate', **kwargs):
        """
        Initialize the model wrapper.

        Args:
            base_model: The underlying model (e.g., ViLT, BERT, etc.)
            args: Configuration arguments
            task_head_type: Type of task head to use ('separate', 'expanding', 'joint')
            **kwargs: Model-specific arguments
        """
        super().__init__()

        self.base_model = base_model
        self.args = args
        self.task_head_type = task_head_type
        self.classifier_hidden_size = getattr(args, 'classifier_hidden_size', None)

        # Model configuration
        self.hidden_size = self._get_hidden_size()
        self.device = next(base_model.parameters()).device

        # Initialize wrapper-specific components
        self._initialize_wrapper_components()

        # Initialize task head manager (will be set in subclass)
        self.task_head_manager: Optional[TaskHeadBase] = None

        # Task management
        self.current_task_id: Optional[int] = None
        self.num_tasks: int = 0

    @abstractmethod
    def _get_hidden_size(self) -> int:
        """Get the hidden size of the base model."""
        pass

    @abstractmethod
    def _initialize_wrapper_components(self):
        """Initialize wrapper-specific components."""
        pass

    @abstractmethod
    def _extract_features(self, batch: Dict[str, Any], task_id: int) -> torch.Tensor:
        """
        Extract features from input using the base model.

        Args:
            batch: Input batch
            task_id: Current task identifier

        Returns:
            Extracted features tensor
        """
        pass

    def _initialize_task_head_manager(self, task_head_type: str = 'separate') -> TaskHeadBase:
        """
        Initialize the appropriate task head manager.

        Args:
            task_head_type: Type of task head ('separate', 'expanding', 'joint')

        Returns:
            Initialized task head manager
        """
        if task_head_type == 'separate':
            from models.task_heads import SeparateTaskHeads
            return SeparateTaskHeads(self.hidden_size, self.classifier_hidden_size, self.logger)
        elif task_head_type == 'expanding':
            from models.task_heads import ExpandingTaskHead
            return ExpandingTaskHead(self.hidden_size, self.classifier_hidden_size, self.logger)
        elif task_head_type == 'joint':
            from models.task_heads import JointTaskHead
            return JointTaskHead(self.hidden_size, self.classifier_hidden_size, self.logger)
        else:
            raise ValueError(f"Unknown task head type: {task_head_type}")

    def add_task(self,
                 task_id: int,
                 task_name: str,
                 num_classes: int,
                 label2ans: Optional[Dict[int, str]] = None) -> None:
        """
        Add a new task to the model.

        Args:
            task_id: Unique task identifier
            task_name: Human-readable task name
            num_classes: Number of classes for this task
            label2ans: Mapping from label indices to answer strings
        """

        if self.task_head_manager is None:
            raise ValueError("Task head manager not initialized")

        if task_id in self.task_head_manager.task_info:
            raise ValueError(f"Task {task_id} already exists")

        # Convert label2ans to list format if needed
        #     # Dict format: {0: "yes", 1: "no", ...}
        #     # Already list format

        # Add task through the head manager
        self.task_head_manager.add_task(task_id, task_name, num_classes, label2ans, device=self.device)
        self.num_tasks += 1

        # Move to correct device
        self.to(self.device)

    def _create_task_adapter(self, task_id: int) -> Optional[nn.Module]:
        """
        Create a task-specific adapter (optional).

        Args:
            task_id: Task identifier

        Returns:
            Task-specific adapter or None
        """
        # Base implementation returns None
        # Subclasses can override to add adapters
        return None

    def set_current_task(self, task_id: int) -> None:
        """
        Set the current active task.

        Args:
            task_id: ID of the task to set as current
        """
        if self.task_head_manager is None:
            raise ValueError("Task head manager not initialized")

        if task_id not in self.task_head_manager.task_info:
            raise ValueError(f"Task {task_id} not found")

        self.current_task_id = task_id

    def forward(self,
                batch: Dict[str, Any],
                task_id: Optional[int] = None,
                return_features: bool = False, **kwargs) -> ModelOutput:
        """
        Forward pass through the model.

        Args:
            batch: Input batch
            task_id: Specific task ID to use (if None, uses current_task_id)
            return_features: Whether to return intermediate features

        Returns:
            ModelOutput containing predictions and optional features
        """

        # Determine which task to use

        # Extract features using base model
        features = self._extract_features(batch, task_id=task_id)

        # Get task-specific logits through head manager
        logits = self.task_head_manager.forward(features, task_id)

        # Create output
        output = ModelOutput(
            logits=logits,
            probabilities=torch.softmax(logits, dim=-1),
            features=features if return_features else None
        )

        return output

    def get_predictions(self,
                       batch: Dict[str, Any],
                       task_id: Optional[int] = None,
                       return_indices: bool = False) -> Any:
        """
        Get predictions as answer strings or indices.

        Args:
            batch: Input batch
            task_id: Task ID to use for prediction
            return_indices: If True, return prediction indices; if False, return answer strings

        Returns:
            Either prediction indices or answer strings based on return_indices flag
        """
        target_task_id = task_id if task_id is not None else self.current_task_id

        if target_task_id is None:
            raise ValueError("No task ID specified and no current task set")

        # Get model output
        outputs = self.forward(batch, task_id=target_task_id)

        # Convert logits to answers through head manager
        return self.task_head_manager.get_answer_from_logits(
            outputs.logits, target_task_id, return_indices=return_indices
        )
    #
    def get_trainable_parameters(self, task_id: Optional[int] = None) -> List[nn.Parameter]:
        """
        Get trainable parameters for a specific task or all parameters.

        Args:
            task_id: Specific task ID (if None, returns all trainable parameters)

        Returns:
            List of trainable parameters
        """
        if task_id is None:
            return [p for p in self.parameters() if p.requires_grad]

        params = []

        # Base model parameters (if trainable)
        params.extend([p for p in self.base_model.parameters() if p.requires_grad])

        # Task-specific head parameters
        if self.task_head_manager is not None:
            params.extend(self.task_head_manager.get_task_parameters(task_id))

        return params

    def freeze_base_model(self) -> None:
        """Freeze the base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self) -> None:
        """Unfreeze the base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def freeze_task(self, task_id: int) -> None:
        """
        Freeze parameters for a specific task.

        Args:
            task_id: Task to freeze
        """
        if self.task_head_manager is not None:
            self.task_head_manager.freeze_task(task_id)

    def unfreeze_task(self, task_id: int) -> None:
        """
        Unfreeze parameters for a specific task.

        Args:
            task_id: Task to unfreeze
        """
        if self.task_head_manager is not None:
            self.task_head_manager.unfreeze_task(task_id)

    def get_task_info(self, task_id: int) -> Dict[str, Any]:
        """Get information about a specific task."""
        if self.task_head_manager is None:
            raise ValueError("Task head manager not initialized")

        if task_id not in self.task_head_manager.task_info:
            raise ValueError(f"Task {task_id} not found")

        return self.task_head_manager.task_info[task_id]

    #     Args:
    #         task_id: Task identifier

    #     Args:
    #         batch: Input batch
    #         task_ids: Specific task IDs to use (if None, uses all learned tasks)

    #     # LoRA case: Extract features separately for each task with correct routing
    #                 # Set task routing for LoRA adapters

    def get_model_size_info(self) -> Dict[str, int]:
        """
        Get information about model size and parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        base_model_params = sum(p.numel() for p in self.base_model.parameters())
        task_head_params = sum(p.numel() for p in self.task_head_manager.get_head_parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'base_model_parameters': base_model_params,
            'task_head_parameters': task_head_params,
            'num_tasks': self.num_tasks
        }

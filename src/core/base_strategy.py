"""
Abstract base class for continual learning strategies.

This class defines the interface that all continual learning approaches must implement.
It encapsulates the core CL algorithm logic (how to learn new tasks while retaining old ones).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TaskInfo:
    """Information about a specific task."""
    task_id: int
    task_name: str
    num_classes: int
    label2ans: Dict[int, str]
    dataset_size: int

@dataclass
class TrainingConfig:
    """Configuration for training a specific task."""
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer_type: str
    scheduler_type: str
    warmup_steps: int
    gradient_clip_norm: float

class BaseStrategy(ABC):
    """Abstract base class for continual learning strategies."""

    def __init__(self, model_wrapper, args, **kwargs):
        """
        Initialize the continual learning strategy.

        Args:
            model_wrapper: Wrapped model that provides CL-specific functionality
            args: Configuration arguments
            **kwargs: Strategy-specific arguments
        """
        self.model_wrapper = model_wrapper
        self.args = args

        # Flag to indicate if strategy requires unified answer space
        self.requires_unified_answer_space = False

        # Allow override from args if needed
        if hasattr(args, 'force_unified_answer_space'):
            self.requires_unified_answer_space = args.force_unified_answer_space

        # Unified answer space (will be set externally if needed)
        self.unified_ans2label = None
        self.unified_label2ans = None

        # Track learned tasks
        self.learned_tasks: List[TaskInfo] = []
        self.current_task_info: Optional[TaskInfo] = None

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.training_config: Optional[TrainingConfig] = None

        # Strategy-specific state (to be used by subclasses)
        self.strategy_state = {}

        # Initialize strategy-specific components
        self._initialize_strategy_components()

        if hasattr(model_wrapper, 'device'):
            self.device = model_wrapper.device
        elif torch.cuda.is_available():
            if hasattr(args, 'gpu'):
                self.device = torch.device(f'cuda:{args.gpu}')
                self.logger.info(f"Strategy using GPU {args.gpu}")
            else:
                self.device = torch.device('cuda:0')
                self.logger.info("Strategy using default GPU 0")
        else:
            self.device = torch.device('cpu')
            self.logger.warning("Strategy running on CPU.")

        # Setup AMP and other performance optimizations
        self._setup_amp_if_needed(args)

        # Move model to CUDA and enable cudnn optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Early stopping configuration
        self.early_stopping_enabled = getattr(args, 'early_stopping', False)
        self.early_stopping_patience = getattr(args, 'early_stopping_patience', 5)
        self.early_stopping_delta = getattr(args, 'early_stopping_delta', 0.001)
        self.restore_best_weights = getattr(args, 'restore_best_weights', True)

    def _setup_scaler(self, args):
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            init_gscale = getattr(args, 'init_grad_scaler', None)
            if init_gscale is not None:
                self.scaler = GradScaler(init_scale=init_gscale)
            else:
                self.scaler = GradScaler()
            self.logger.info("AMP enabled.")
        else:
            self.scaler = None
            self.logger.warning("AMP disabled. Add --use_amp to enable mixed precision.")

    def _setup_amp_if_needed(self, args):
        """Setup automatic mixed precision (AMP) if enabled."""
        # Performance optimizations
        self.use_amp = getattr(args, 'use_amp', False)  # Mixed precision - default to False for safety
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.compile_model = getattr(args, 'compile_model', False)  # Default to False

        # Initialize AMP scaler
        self._setup_scaler(args)

        self.logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps} steps")

        # Compile model for faster execution (PyTorch 2.0+)
        if self.compile_model and hasattr(torch, 'compile'):
            try:
                self.model_wrapper = torch.compile(self.model_wrapper, mode='reduce-overhead')
            except:
                self.logger.warning("Warning: torch.compile not available or failed, continuing without compilation")

    @abstractmethod
    def _initialize_strategy_components(self):
        """Initialize strategy-specific components (e.g., memory buffer, regularizers)."""
        pass

    @abstractmethod
    def prepare_for_task(self, task_info: TaskInfo) -> None:
        """
        Prepare the strategy for learning a new task.

        This method is called before training starts on a new task.
        Strategies should use this to:
        - Initialize task-specific components
        - Prepare the model architecture
        - Set up any task-specific training procedures

        Args:
            task_info: Information about the task to be learned
        """
        pass

    @abstractmethod
    def train_task(self, task_info: TaskInfo, train_loader, val_loader=None, callbacks=None) -> Dict[str, Any]:
        """
        Train the model on a specific task.

        This is the core method where the CL strategy implements its learning algorithm.

        Args:
            task_info: Information about the current task
            train_loader: Training data loader
            val_loader: Validation data loader (optional)

        Returns:
            Dictionary containing training results and metrics
        """
        pass

    @abstractmethod
    def predict(self, batch: Dict[str, Any], task_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Make predictions using the continual learning strategy.

        Args:
            batch: Input batch
            task_id: Known task ID (for oracle evaluation), None for realistic setting

        Returns:
            Dictionary containing predictions and confidence scores
        """
        pass

    @abstractmethod
    def consolidate_knowledge(self, task_info: TaskInfo) -> None:
        """
        Consolidate knowledge after learning a task.

        This method is called after training on a task is complete.
        Strategies can use this to:
        - Update regularization terms
        - Consolidate memory buffers
        - Freeze certain model parameters
        - Update importance weights

        Args:
            task_info: Information about the just-completed task
        """
        pass

    def before_training_step(self, batch: Dict[str, Any], step: int, epoch: int) -> Dict[str, Any]:
        """
        Hook called before each training step.

        Strategies can override this for custom pre-processing.

        Args:
            batch: Training batch
            step: Current step number
            epoch: Current epoch number

        Returns:
            Modified batch or additional information
        """
        return batch

    def after_training_step(self,
                          batch: Dict[str, Any],
                          outputs: Dict[str, Any],
                          loss: torch.Tensor,
                          step: int,
                          epoch: int) -> Dict[str, Any]:
        """
        Hook called after each training step.

        Strategies can override this for custom post-processing.

        Args:
            batch: Training batch
            outputs: Model outputs
            loss: Computed loss
            step: Current step number
            epoch: Current epoch number

        Returns:
            Additional metrics or information
        """
        return {}

    def compute_loss(self, batch: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the loss for VQA soft targets.

        Args:
            batch: Training batch
            outputs: Model outputs

        Returns:
            Computed loss tensor
        """
        # Handle both ModelOutput objects and dictionaries
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            raise ValueError("No logits found in outputs")

        # Get targets with priority order for VQA
        targets = None
        target_keys = ['targets', 'scores', 'labels']

        for key in target_keys:
            if key in batch and batch[key] is not None:
                targets = batch[key]
                break

        if targets is None:
            raise ValueError(f"No targets found in batch. Available keys: {list(batch.keys())}")

        # Convert targets to tensor if needed
        if isinstance(targets, list):
            targets = torch.tensor(targets, dtype=torch.float)

        # Ensure targets are on the same device

        # Choose loss function based on target format
        if len(targets.shape) == 2:
            # Soft targets [batch_size, num_classes] - VQA format
            if targets.shape[1] == logits.shape[1]:
                # Same number of classes - use BCE loss
                loss_fn = nn.functional.binary_cross_entropy_with_logits
                return loss_fn(logits, targets) * targets.shape[1]
            else:
                raise ValueError(f"Logits shape {logits.shape} doesn't match targets shape {targets.shape}")

        elif len(targets.shape) == 1:
            # Hard targets [batch_size] - classification format
            if targets.dtype in [torch.long, torch.int]:
                loss_fn = nn.CrossEntropyLoss()
                return loss_fn(logits, targets)
            else:
                # Convert to long if needed
                targets = targets.long()
                loss_fn = nn.CrossEntropyLoss()
                return loss_fn(logits, targets)

        else:
            raise ValueError(f"Unexpected target shape: {targets.shape}. Expected [batch_size] or [batch_size, num_classes]")

    def setup_optimizer(self, training_config: TrainingConfig) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Setup optimizer and learning rate scheduler.

        Args:
            training_config: Training configuration

        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Get trainable parameters
        params = [p for p in self.model_wrapper.parameters() if p.requires_grad]

        # Create optimizer
        if training_config.optimizer_type.lower() == 'adamw':
            # eps and weight decay match the defaults of the former transformers AdamW
            optimizer = torch.optim.AdamW(params, lr=training_config.learning_rate, eps=1e-6, weight_decay=0.0)
        elif training_config.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=training_config.learning_rate)
        elif training_config.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=training_config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {training_config.optimizer_type}")

        # Create scheduler
        scheduler = None
        if training_config.scheduler_type.lower() == 'linear':
            from transformers.optimization import get_linear_schedule_with_warmup
            total_steps = training_config.epochs * 1000  # Approximate
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=training_config.warmup_steps,
                num_training_steps=total_steps
            )
        elif training_config.scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=training_config.epochs
            )

        return optimizer, scheduler

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get strategy state for checkpointing.

        Returns:
            Dictionary containing strategy state
        """
        state = {
            'learned_tasks': self.learned_tasks,
            'current_task_info': self.current_task_info,
            'strategy_state': self.strategy_state,
        }

        # Add optimizer and scheduler state if they exist
        if self.optimizer is not None:
            state['optimizer_state'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load strategy state from checkpoint.

        Args:
            state_dict: Dictionary containing strategy state
        """
        self.learned_tasks = state_dict.get('learned_tasks', [])
        self.current_task_info = state_dict.get('current_task_info', None)
        self.strategy_state = state_dict.get('strategy_state', {})

        # Load optimizer state if available
        if 'optimizer_state' in state_dict and hasattr(self, 'optimizer') and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer_state'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")

        # Load scheduler state if available
        if 'scheduler_state' in state_dict and hasattr(self, 'scheduler') and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler_state'])
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics for this strategy.

        Returns:
            Dictionary with memory usage information
        """
        total_params = sum(p.numel() for p in self.model_wrapper.parameters())
        trainable_params = sum(p.numel() for p in self.model_wrapper.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        }

    def get_task_prediction_info(self) -> Dict[str, Any]:
        """
        Get information about how this strategy handles task prediction.

        Returns:
            Dictionary with task prediction capabilities
        """
        return {
            'requires_task_id': True,  # Most strategies need task ID
            'can_predict_task_id': False,  # Most strategies cannot predict task ID
            'task_prediction_method': None
        }

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Efficiently move batch to GPU with non-blocking transfers."""
        if not torch.cuda.is_available():
            return batch

        device = self.device

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Non-blocking transfer for better GPU utilization
                if value.device != device:
                    batch[key] = value.to(device, non_blocking=True)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch[key] = [v.to(device, non_blocking=True) if v.device != device else v for v in value]

        return batch

    def _extract_targets(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract targets from batch."""
        target_keys = ['targets', 'scores', 'labels', 'answers', 'answer_labels']

        for key in target_keys:
            if key in batch:
                targets = batch[key]

                if isinstance(targets, torch.Tensor):
                    return targets
                elif isinstance(targets, list):
                    if len(targets) == 0:
                        continue
                    if isinstance(targets[0], str):
                        continue  # Skip string targets
                    else:
                        try:
                            targets = torch.tensor(targets, dtype=torch.float)
                            return targets
                        except Exception:
                            continue

        return None

    def _should_stop_early(self, callbacks) -> bool:
        """Check if any early stopping callback signals to stop."""
        if callbacks is None:
            return False
        from core.training_callbacks import EarlyStoppingCallback
        if hasattr(callbacks, 'callbacks'):
            for callback in callbacks.callbacks:
                if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                    return True
        return False

    @property
    def num_learned_tasks(self) -> int:
        """Get the number of tasks learned so far."""
        return len(self.learned_tasks)

    @property
    def task_list(self) -> List[str]:
        """Get list of learned task names."""
        return [task.task_name for task in self.learned_tasks]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(learned_tasks={self.num_learned_tasks})"

    def supports_meaningful_oracle_evaluation(self) -> bool:
        """Indicate support for meaningful oracle evaluation."""
        # By default, assume strategies do not support meaningful oracle evaluation
        return False

    def supports_bayesian_evaluation(self) -> bool:
        """Indicate support for Bayesian evaluation."""
        # By default, assume strategies do not support Bayesian evaluation
        return False

    def _set_early_stopping_metric(self, val_loader):
        if val_loader is not None and not getattr(self.args, 'skip_validation', False):
            best_metric = float('-inf')
        else:
            best_metric = float('inf')

        return best_metric

    def _setup_early_stopping_state(self, val_loader=None):
        """Setup early stopping state variables."""
        self.best_metric = self._set_early_stopping_metric(val_loader)
        self.patience_counter = 0
        self.best_epoch = 0
        self.num_improvements = 0

    def _is_better_metric(self, current_metric: float) -> bool:
        """Determine if the current metric is better than the best metric."""
        improved = current_metric > self.best_metric     # higher is better (e.g., accuracy or negative loss)
        significant_improvement = abs(current_metric - self.best_metric) >= self.early_stopping_delta
        return improved, significant_improvement

    def _should_early_stop(self, checkpoint_metric, epoch):
        """Check if early stopping criteria are met."""
        if self.early_stopping_enabled is False:
            return False

        # Check if current metric is better than best
        improved, significant_improvement = self._is_better_metric(checkpoint_metric)

        if improved and significant_improvement:
            self.num_improvements += 1
            self.best_metric = checkpoint_metric
            self.best_epoch = epoch
            self.patience_counter = 0
            self.logger.info(f"Improvement detected. Best metric updated to {self.best_metric:.4f} at epoch {epoch}.")
        else:
            self.patience_counter += 1
            self.logger.info(f"No improvement. Patience counter: {self.patience_counter}/{self.early_stopping_patience}")

        should_stop = (self.patience_counter >= self.early_stopping_patience) and (self.num_improvements > 5)
        return should_stop

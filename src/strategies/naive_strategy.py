"""
Naive (Sequential) strategy for continual learning.

This strategy trains tasks sequentially without any mechanisms to prevent
catastrophic forgetting. Uses a single head for the entire combined answer space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import numpy as np
import logging
from models.task_heads import JointTaskHead, SeparateTaskHeads
from torch.cuda.amp import autocast, GradScaler
from core.base_strategy import BaseStrategy, TaskInfo, TrainingConfig

import os
on_slurm = os.getenv('SLURM_JOB_ID') is not None
# Configure tqdm based on environment
tqdm_config = {
    'disable': False,
    'mininterval': 60.0 if on_slurm else 0.1,
    'dynamic_ncols': not on_slurm,
    'ncols': 80 if on_slurm else None,  # Fixed width on SLURM
    'ascii': on_slurm,  # ASCII characters instead of Unicode
}

class NaiveStrategy(BaseStrategy):
    """
    Naive continual learning strategy that trains tasks sequentially
    without any forgetting prevention mechanisms. Uses a single head
    for the entire combined answer space.
    """

    def __init__(self, model_wrapper, args, **kwargs):
        """
        Initialize naive strategy.

        Args:
            model_wrapper: Model wrapper instance
            args: Configuration arguments
            **kwargs: Additional arguments
        """

        # Add logger
        self.logger = logging.getLogger('CL.strategy.naive')

        super().__init__(model_wrapper, args, **kwargs)

        self.unified_label2ans = kwargs.get('unified_label2ans', None)

        if not self.unified_label2ans:
            raise ValueError("NaiveStrategy requires a predefined unified_label2ans mapping")

        # Set flags for unified answer space
        self.requires_unified_answer_space = True       # Force unified answer space
        self.overridable_unified_answer_space = False   # Do not allow override

        self._initialize_unified_task_head()

        # Training configuration
        self.allow_base_model_updates = True  # Always train base model in naive

        self._setup_optimizer()

        self.logger.info("Initialized Naive Strategy (Sequential Training with Single Unified Head)")
        self.logger.info(f"   Unified vocabulary size: {len(self.unified_label2ans)}")

    def _setup_optimizer(self):
        # Setup training configuration
        self.training_config = TrainingConfig(
            epochs=getattr(self.args, 'epochs', 10),
            batch_size=getattr(self.args, 'batch_size', 32),
            learning_rate=getattr(self.args, 'lr', 1e-4),
            optimizer_type=getattr(self.args, 'optimizer', 'adamw'),
            scheduler_type=getattr(self.args, 'scheduler', 'linear'),
            warmup_steps=getattr(self.args, 'warmup_steps', 100),
            gradient_clip_norm=getattr(self.args, 'clip_grad_norm', 1.0)
        )

        # Setup optimizer for all parameters
        self.optimizer, self.scheduler = self.setup_optimizer(self.training_config)

    def _initialize_unified_task_head(self):
        """Initialize the single unified task head."""
        # Create JointTaskHead with predefined unified vocabulary
        self.model_wrapper.task_head_manager = JointTaskHead(self.model_wrapper.hidden_size, self.model_wrapper.classifier_hidden_size, self.logger, unified_vocab=self.unified_label2ans)
        # Move to device
        self.model_wrapper = self.model_wrapper.to(self.model_wrapper.device)
        self.model_wrapper.task_head_manager = self.model_wrapper.task_head_manager.to(self.model_wrapper.device)

        self.logger.info(f"Initialized JointTaskHead with {len(self.unified_label2ans)} unified classes")

    def _initialize_strategy_components(self):
        """Initialize strategy-specific components."""
        pass

    def prepare_for_task(self, task_info: TaskInfo) -> None:
        """
        Prepare for learning a new task.

        Args:
            task_info: Information about the task to be learned
        """
        self.logger.info(f"Preparing naive strategy for task {task_info.task_id}: {task_info.task_name}")

        self.model_wrapper.add_task(
            task_id=task_info.task_id,
            task_name=task_info.task_name,
            num_classes=task_info.num_classes,
            label2ans=task_info.label2ans,
        )

        # Set current task
        self.model_wrapper.current_task_id = task_info.task_id
        self.current_task_info = task_info

        # Enable training for all parameters (no freezing in naive)
        for param in self.model_wrapper.parameters():
            param.requires_grad = True

        self.logger.info(f"Prepared for task {task_info.task_name}")
        self.logger.info(f"Unified vocabulary size: {len(self.unified_label2ans)}")

    def train_task(self, task_info: TaskInfo, train_loader, val_loader=None, callbacks=None) -> Dict[str, Any]:
        """
        Train on a task sequentially (naive approach).

        Args:
            task_info: Information about the current task
            train_loader: Training data loader
            val_loader: Validation data loader (optional)

        Returns:
            Training results and metrics
        """
        self.logger.info(f"Naive training on task {task_info.task_name}")

        # Training loop
        self.model_wrapper.train()

        self._setup_optimizer()
        self._setup_scaler(self.args)

        training_losses = []
        validation_accuracies = []
        best_val_acc = 0.0

        for epoch in range(self.training_config.epochs):
            # Train epoch
            epoch_loss = self._train_epoch(epoch, train_loader, self.training_config, task_info.task_id)
            training_losses.append(epoch_loss)

            checkpoint_metric = None

            # Validation
            if val_loader is not None and not getattr(self.args, 'skip_validation', False):
                val_acc = self._validate_epoch(epoch, val_loader, task_info.task_id)
                validation_accuracies.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.logger.info(f"  -> New best validation accuracy: {val_acc:.4f}")
                checkpoint_metric = val_acc
            else:
                checkpoint_metric = -epoch_loss  # Use negative loss if no val

            # Trigger callbacks
            if callbacks is not None:
                metrics = {
                    'train_loss': epoch_loss,
                    'val_accuracy': checkpoint_metric,
                }
                is_final = (epoch == self.training_config.epochs - 1)

                callbacks.on_epoch_end(
                    epoch=epoch,
                    task_idx=task_info.task_id,
                    task_name=task_info.task_name,
                    metrics=metrics,
                    is_final=is_final
                )
                if self._should_stop_early(callbacks):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}/{self.training_config.epochs}")
                    break

            self.logger.info(f"Epoch {epoch+1}/{self.training_config.epochs}: Loss={epoch_loss:.6f}")

        return {
            'task_id': task_info.task_id,
            'task_name': task_info.task_name,
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'best_val_accuracy': best_val_acc,
            'training_losses': training_losses,
            'validation_accuracies': validation_accuracies,
            'total_epochs': self.training_config.epochs,
            'strategy': 'naive'
        }

    def _train_epoch(self, epoch: int, train_loader, training_config: TrainingConfig, task_id: int) -> float:
        """Train for one epoch."""
        from torch.cuda.amp import autocast

        self.model_wrapper.train()
        self.model_wrapper.base_model.train()

        if self.model_wrapper.task_head_manager is not None:
            self.model_wrapper.task_head_manager.head.train()

        assert self.model_wrapper.base_model.training, "Base model must be in eval mode!"

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, **tqdm_config)

        accumulation_counter = 0

        for batch_idx, batch in enumerate(pbar):
            actual_accumulated_steps = 1  # Default for non-accumulation steps

            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Add ground truth task IDs
            batch_size = len(batch.get('question_ids', batch.get('questions', [''])))

            if self.use_amp:
                with autocast():
                    outputs = self.model_wrapper(batch, task_id=task_id, return_features=True)
                    loss = self.compute_loss(batch, outputs)
                    loss = loss / self.gradient_accumulation_steps

                # Backward with scaled gradients
                self.scaler.scale(loss).backward()
            else:
                # Regular forward/backward
                outputs = self.model_wrapper(batch, task_id=task_id, return_features=True)
                loss = self.compute_loss(batch, outputs)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            accumulation_counter += 1

            is_last_batch = (batch_idx + 1) == len(train_loader)
            should_update = (accumulation_counter % self.gradient_accumulation_steps == 0) or is_last_batch

            # Track if we've logged this epoch
            if not hasattr(self, '_debug_logged_epochs'):
                self._debug_logged_epochs = set()

            # Update weights after accumulation
            if should_update:

                if is_last_batch and accumulation_counter % self.gradient_accumulation_steps != 0:
                    # Last batch with incomplete accumulation
                    actual_accumulated_steps = accumulation_counter % self.gradient_accumulation_steps
                else:
                    # Normal full accumulation
                    actual_accumulated_steps = self.gradient_accumulation_steps

                if self.use_amp:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)

                    # Run diagnostics once per epoch.
                    epoch_task_key = f"{epoch}_{task_id}"
                    if epoch_task_key not in self._debug_logged_epochs:
                        self.debug_gradient_flow(task_id, loss)
                        self._debug_logged_epochs.add(epoch_task_key)

                    # Optional check for inf/nan gradients.

                    if training_config.gradient_clip_norm > 0:
                        model_params = self._get_parameters(task_id)
                        torch.nn.utils.clip_grad_norm_(model_params, training_config.gradient_clip_norm)

                    # Step optimizer
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular gradient clipping
                    if training_config.gradient_clip_norm > 0:
                        model_params = self._get_parameters(task_id)
                        torch.nn.utils.clip_grad_norm_(model_params, training_config.gradient_clip_norm)

                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.info(f"  Epoch {epoch+1} - Loss: {epoch_loss:.4f}.")

        return total_loss / max(num_batches, 1)

    def _validate_epoch(self, epoch: int, val_loader, task_id: int) -> float:
        """Validate for one epoch."""

        original_benchmark = torch.backends.cudnn.benchmark
        original_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.model_wrapper.eval()
        self.model_wrapper.set_current_task(task_id)

        total_predictions = 0
        total_accuracy = 0.0

        with torch.no_grad():
            with autocast(False):
                for batch in tqdm(val_loader, desc=f"Validation", leave=False, **tqdm_config):
                    batch = self._move_batch_to_device(batch)

                    batch_size = len(batch.get('question_id', batch.get('questions', [''])))

                    # Forward pass
                    outputs = self.model_wrapper(batch, task_id='unified', return_features=True)

                    # Calculate accuracy
                    if hasattr(outputs, 'logits'):
                        predictions = torch.argmax(outputs.logits, dim=-1)
                        targets = self._extract_targets(batch)

                        if targets is not None:
                            if len(targets.shape) == 2:
                                try:
                                    individual_accuracies = targets.gather(1, predictions.unsqueeze(1)).squeeze(1)
                                    batch_accuracy = individual_accuracies.mean().item()
                                except Exception as e:
                                    self.logger.error(f"Error computing accuracy: {e}")
                                    batch_accuracy = 0.0

                            else:
                                raise NotImplementedError("Target tensors with more than 2 dimensions are not supported.")

                            total_accuracy += batch_accuracy * targets.size(0)
                            total_predictions += targets.size(0)

        accuracy = total_accuracy / total_predictions if total_predictions > 0 else 0.0

        # Restore original settings
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic

        return accuracy

    def predict(self, batch: Dict[str, Any], task_id: Optional[int] = None, return_answer_strings: bool = True) -> Dict[str, Any]:
        """
        Make predictions using the unified head.

        Args:
            batch: Input batch
            task_id: Task ID (used for mapping back to task-specific labels if needed)

        Returns:
            Dictionary containing predictions
        """
        from torch.cuda.amp import autocast

        original_benchmark = torch.backends.cudnn.benchmark
        original_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.model_wrapper.eval()

        assert not self.model_wrapper.training, "Model must be in eval mode for prediction!"

        # Also check dropout specifically
        for module in self.model_wrapper.modules():
            if isinstance(module, nn.Dropout):
                assert module.training == False, "Dropout should be disabled!"

        with torch.no_grad():
            with autocast(enabled=False):  # Disable autocast for stability during inference for consistency
                batch = self._move_batch_to_device(batch)
                outputs = self.model_wrapper(batch, task_id=task_id, return_features=True)

                prediction_indices = torch.argmax(outputs.logits, dim=-1)
                confidences = torch.max(outputs.probabilities, dim=-1)[0]
                predictions = self.model_wrapper.task_head_manager.get_answer_from_logits(outputs.logits, 0, return_indices=False)    # Convert (no forward!)

        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic

        return {
            'predictions': predictions,
            'confidences': confidences,
            'prediction_indices': prediction_indices,
            'strategy': 'naive'
        }

    def consolidate_knowledge(self, task_info: TaskInfo) -> None:
        """
        Consolidate knowledge after learning a task (no special action for naive).

        Args:
            task_info: Information about the just-completed task
        """
        self.learned_tasks.append(task_info)
        self.logger.info(f"Completed naive training for task {task_info.task_name}")
        self.logger.info(f"Total tasks learned: {len(self.learned_tasks)}")

    def _get_parameters(self, task_id: int) -> List:
        """Get parameters for specific task."""
        task_params = [p for p in self.model_wrapper.base_model.parameters() if p.requires_grad]

        # Get task head parameters
        if self.model_wrapper.task_head_manager is not None:
            task_params.extend(self.model_wrapper.task_head_manager.get_head_parameters())

        return task_params

    def get_task_prediction_info(self) -> Dict[str, Any]:
        """Get task prediction info for naive strategy."""
        return {
            'requires_task_id': False,  # Naive doesn't need task ID for prediction
            'can_predict_task_id': False,
            'uses_unified_head': True,
            'unified_vocab_size': self.unified_answer_space.unified_vocab_size,
            'strategy': 'naive'
        }

    def __repr__(self) -> str:
        return f"NaiveStrategy(learned_tasks={len(self.learned_tasks)}, unified_vocab={len(self.unified_label2ans)})"

    def debug_gradient_flow(self, task_id: int, loss: torch.Tensor):
        """Debug gradient flow after backward pass."""
        print(f"\n{'='*60}")
        print(f"Gradient flow for task {task_id}")
        print(f"{'='*60}")

        has_gradients = {}
        no_gradients = {}

        for name, param in self.model_wrapper.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()

                # Categorize by component
                if 'task_heads' in name:
                    task_key = name.split('.')[1]
                    component = f"task_head_{task_key}"
                else:
                    component = "base_model"

                if component not in has_gradients:
                    has_gradients[component] = []
                has_gradients[component].append((name, grad_norm))

            elif param.requires_grad and param.grad is None:
                if 'task_heads' in name:
                    task_key = name.split('.')[1]
                    component = f"task_head_{task_key}"
                else:
                    component = "base_model"

                if component not in no_gradients:
                    no_gradients[component] = []
                no_gradients[component].append(name)

        # Print components with gradients
        print("\nCOMPONENTS WITH GRADIENTS:")
        for component, params in has_gradients.items():
            print(f"\n  {component}:")
            avg_grad = sum(p[1] for p in params) / len(params)
            print(f"    Params with gradients: {len(params)}")
            print(f"    Avg gradient norm: {avg_grad:.6f}")
            # Show top 3 gradients
            top_grads = sorted(params, key=lambda x: x[1], reverse=True)[:3]
            for name, grad_norm in top_grads:
                print(f"      {name}: {grad_norm:.6f}")

        # Print components without gradients (but trainable)
        if no_gradients:
            print("\nTrainable parameters without gradients:")
            for component, params in no_gradients.items():
                print(f"\n  {component}: {len(params)} params")
                for name in params[:3]:  # Show first 3
                    print(f"    - {name}")

        print(f"\n{'='*60}\n")

    def supports_meaningful_oracle_evaluation(self) -> bool:
        """Indicate support for meaningful oracle evaluation."""
        return False

"""
Mixture-of-experts strategy: one LoRA expert and answer head per task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
import logging
from utils.loss_utils import LossManager
from core.base_strategy import BaseStrategy, TaskInfo, TrainingConfig
from models.task_predictor import PerfectTaskPredictor

from core.checkpoint import CheckpointManager, CheckpointMetric
from pathlib import Path
import time
import torch.profiler

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

class MoEStrategy(BaseStrategy):
    """Mixture-of-experts strategy with one LoRA expert per task."""

    def __init__(self, model_wrapper, args, task_list: List[str], **kwargs):

        self._task_list = task_list
        self.num_experts = len(task_list)
        self.num_trained_tasks = 0

        self.task_predictor_type = getattr(args, 'task_predictor_type', 'perfect')  # Choices: 'perfect', 'learned'

        # Add logger instance
        self.logger = logging.getLogger('CL.strategy.moe')

        self.scale_loss = getattr(args, 'scale_vqa_loss', False)

        super().__init__(model_wrapper, args, **kwargs)

        self.task_configs = kwargs.get('task_configs', None) if getattr(args, "init_unified_answer_space", False) else None

        self.logger.info(f"Initialized MoE strategy with {self.num_experts} experts")

        #  Multi-component checkpoint tracking
        self.vqa_loaded_from_checkpoint = set()

    def _initialize_strategy_components(self):
        """Initialize MoE-specific components with unified answer space."""
        # Initialize perfect task predictor
        if self.task_predictor_type == 'perfect':
            self.task_predictor = PerfectTaskPredictor(self.num_experts)
        elif self.task_predictor_type == 'learned':
            raise NotImplementedError("Learned task predictor not implemented yet.")

        # Move to device if available
        if hasattr(self.model_wrapper, 'device'):
            self.device = self.model_wrapper.device
            self.task_predictor = self.task_predictor.to(self.model_wrapper.device)
        else:
            self.device = torch.device("cpu")
            self.logger.warning("Warning: model_wrapper has no device attribute, skipping device transfer for components")

        self.loss_manager = LossManager(self.args, self.device)
        # Expert training history
        self.expert_training_history: Dict[int, Dict[str, Any]] = {}

        self._setup_checkpoint_managers()

    def get_checkpoint_paths_for_task(self, task_idx: int, task_name: str, base_dir: str, checkpoint_type: str = None) -> Dict[str, Optional[str]]:
        """
        Get paths to all component checkpoints for a task.
        Called by base trainer during checkpoint extraction.
        """
        from pathlib import Path

        base_path = Path(base_dir)

        # VQA expert checkpoint (main component)
        vqa_dir = base_path / 'checkpoints' / 'vqa'
        print(f"Looking for VQA checkpoints in: {vqa_dir}")
        vqa_path = self._find_component_checkpoint_helper(vqa_dir, task_name, checkpoint_type = checkpoint_type)

        paths = {
            'vqa': str(vqa_path) if vqa_path else None,
            'ae': None,   # Not used
            'mlp': None,  # Not used
            'main': str(vqa_path) if vqa_path else None
        }
        return paths

    def _find_component_checkpoint_helper(self, checkpoint_dir, task_name, checkpoint_type: str = None) -> Optional[Path]:
        """Find best available checkpoint for a component."""
        from pathlib import Path

        if not checkpoint_dir.exists():
            return None

        if checkpoint_type is None:
            for suffix in ['_best.pth', '_final.pth', '_latest.pth']:
                path = checkpoint_dir / f"{task_name}{suffix}"
                if path.exists():
                    return path
        else:
            assert checkpoint_type in ['best', 'final', 'latest'], "Invalid checkpoint type"
            path = checkpoint_dir / f"{task_name}_{checkpoint_type}.pth"
            if path.exists():
                return path

        return None

    def _setup_checkpoint_managers(self):
        checkpoint_dir = Path(self.args.output) / 'checkpoints'
        # VQA expert checkpoint manager
        self.vqa_checkpoint_dir = Path(checkpoint_dir) / 'vqa'
        self.vqa_checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.vqa_checkpoint_dir),
            metric=CheckpointMetric.ACCURACY,
            keep_last_n_checkpoints=getattr(self.args, 'keep_last_n_checkpoints', 1),
            save_every_n_epochs=getattr(self.args, 'save_every_n_epochs', 1),
            logger=self.logger
        )
        self.logger.info(f"VQA checkpoints: {self.vqa_checkpoint_dir}")

    def _set_lora_train_mode(self, task_id: int):
        """Set LoRA adapters for current task to train mode."""
        def set_train_recursive(module):
            for child in module.modules():
                # Check if this is a LoRA wrapper
                if hasattr(child, 'task_lora_adapters'):
                    # Set the LoRA wrapper itself to train
                    child.train()
                    # Set the specific task adapter to train
                    task_key = str(task_id)
                    if task_key in child.task_lora_adapters:
                        child.task_lora_adapters[task_key].train()

        set_train_recursive(self.model_wrapper.base_model)

    def _train_expert_epoch(self, epoch: int, train_loader, training_config: TrainingConfig, task_id: int) -> float:
        """Train for one epoch with mixed precision and gradient accumulation."""

        self.model_wrapper.train()

        if hasattr(self, 'task_vocab_masks'):
            assert task_id in self.task_vocab_masks, f"No vocab mask found for task {task_id}"
            mask = self.task_vocab_masks[task_id]
        else:
            mask = None

        # Set task head to train

        # Keep base model in eval mode to prevent BN/LN stats updates

        # Set LoRA adapters to train mode

        # Keep base model in eval mode to prevent BN/LN stats updates

        # Keep base model backbone in eval (but LoRA adapters are now in train)
        # We need to selectively set non-LoRA parts to eval
        for module in self.model_wrapper.base_model.modules():
            if not hasattr(module, 'task_lora_adapters'):  # If not a LoRA module
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.Dropout)):
                    module.eval()  # Keep normalization/dropout in eval

        self.model_wrapper.set_current_task(task_id)

        total_loss = 0.0
        num_batches = 0

        # Use optimized tqdm settings
        pbar = tqdm(train_loader, desc=f"Expert {task_id} Epoch {epoch+1}",
                   leave=False, **tqdm_config)

        # Gradient accumulation counter
        accumulation_counter = 0

        task_params = self._get_task_parameters(task_id)

        for batch_idx, batch in enumerate(pbar):
            actual_accumulated_steps = 1  # Default for non-accumulation steps

            # Move batch to GPU efficiently
            batch = self._move_batch_to_device(batch)

            # Add ground truth task IDs
            batch_size = len(batch.get('question_ids', batch.get('questions', [''])))
            batch['ground_truth_task_ids'] = torch.full((batch_size,), task_id,
                                                    dtype=torch.long, device=self.device, pin_memory=False)

            if self.use_amp:
                with autocast():
                    outputs = self.model_wrapper(batch, task_id=task_id) #, return_features=True)
                    loss = self.compute_loss(batch, outputs, mask, self.scale_loss)
                    loss = loss / self.gradient_accumulation_steps

                # Backward with scaled gradients
                self.scaler.scale(loss).backward()
            else:
                # Regular forward/backward
                outputs = self.model_wrapper(batch, task_id=task_id) #, return_features=True)
                loss = self.compute_loss(batch, outputs, mask, self.scale_loss)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            accumulation_counter += 1

            is_last_batch = (batch_idx + 1) == len(train_loader)
            should_update = (accumulation_counter % self.gradient_accumulation_steps == 0) or is_last_batch

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

                    if training_config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(task_params, training_config.gradient_clip_norm)

                    # Step optimizer
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular gradient clipping
                    if training_config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(task_params, training_config.gradient_clip_norm)

                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item() * actual_accumulated_steps #self.gradient_accumulation_steps
            num_batches += 1

        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0

        self.logger.info(f"  Epoch {epoch} - Loss: {epoch_loss:.4f}.")

        return epoch_loss

    def _validate_expert_epoch(self, epoch: int, val_loader, task_id: int) -> float:
        """Validate with mixed precision."""

        # Ensure deterministic evaluation
        original_benchmark = torch.backends.cudnn.benchmark
        original_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.model_wrapper.eval()
        self.model_wrapper.set_current_task(task_id)

        total_predictions = 0
        total_accuracy = 0.0

        with torch.no_grad():
            with autocast(False): # Disable AMP for validation for safety
                for batch in tqdm(val_loader, desc=f"Validation", leave=False, **tqdm_config):
                    batch = self._move_batch_to_device(batch)

                    # Add ground truth task IDs
                    batch_size = len(batch.get('question_id', batch.get('questions', [''])))
                    batch['ground_truth_task_ids'] = torch.full((batch_size,), task_id,
                                                               dtype=torch.long, device=self.device, pin_memory=False)

                    # Forward pass
                    outputs = self.model_wrapper(batch, task_id=task_id) #, return_features=True)

                    # Calculate accuracy
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
                            raise NotImplementedError("Only 2D target tensors are supported for accuracy calculation.")

                        total_accuracy += batch_accuracy * targets.size(0)
                        total_predictions += targets.size(0)

        accuracy = total_accuracy / total_predictions if total_predictions > 0 else 0.0

        # Restore original settings
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic

        return accuracy

    def _get_task_parameters(self, task_id: int) -> List:
        return self.model_wrapper.get_trainable_parameters(task_id)

    def setup_optimizer_for_task(self, training_config: TrainingConfig, task_id: int, train_loader=None):
        """Set up the optimizer."""
        # Get task parameters
        task_params = self._get_task_parameters(task_id)

        # Count parameters for logging
        total_params = sum(p.numel() for p in task_params)
        self.logger.info(f"Expert {task_id} - Total trainable parameters: {total_params:,}")

        # Create optimizer with optimized settings
        if training_config.optimizer_type.lower() == 'adamw':
            from torch.optim import AdamW
            optimizer = AdamW(
                task_params,
                lr=training_config.learning_rate,
                betas=(0.9, 0.999),  # Standard AdamW betas
                eps=1e-6,  # Better numerical stability
                weight_decay=0.01,  # Standard weight decay
                fused=True  # fused AdamW kernel
            )
        elif training_config.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                task_params,
                lr=training_config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-6
            )
        else:
            optimizer = torch.optim.SGD(
                task_params,
                lr=training_config.learning_rate,
                momentum=0.9,
                nesterov=True  # Use Nesterov momentum
            )

        # Create scheduler with better warmup
        scheduler = None
        if training_config.scheduler_type.lower() == 'linear':
            from transformers.optimization import get_linear_schedule_with_warmup

            if train_loader is not None and hasattr(train_loader, '__len__'):
                steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
                total_steps = training_config.epochs * steps_per_epoch
            else:
                estimated_batches = max(200 // training_config.batch_size, 10)
                steps_per_epoch = estimated_batches // self.gradient_accumulation_steps
                total_steps = training_config.epochs * steps_per_epoch

            # Adjust warmup steps for gradient accumulation
            warmup_steps = min(training_config.warmup_steps, total_steps // 10)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif training_config.scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config.epochs,
                eta_min=training_config.learning_rate * 0.01  # Don't go to 0
            )

        return optimizer, scheduler

    def _create_task_vocab_mask(self, task_info: TaskInfo, task_config: Dict[str, Any]) -> None:
        """
        Create a mask indicating which classes in the shared space are valid for this task.
        """
        task_id = task_info.task_id
        shared_vocab_size = len(task_info.label2ans)  # Your unified answer space
        shared_ans2idx = {ans: idx for idx, ans in enumerate(task_info.label2ans)}
        self.logger.debug(f"Creating vocab mask for task {task_id} with {len(task_config['label2ans'])} valid classes out of {shared_vocab_size} total.")

        # Create mask (True = invalid class, False = valid class)
        temp_mask = torch.ones(shared_vocab_size, dtype=torch.bool, device=self.device)

        # Mark valid classes as False
        for _, ans in enumerate(task_config['label2ans']):
            temp_mask[shared_ans2idx[ans]] = False

        # Store mask
        if not hasattr(self, 'task_vocab_masks'):
            self.task_vocab_masks = {}
        self.task_vocab_masks[task_id] = ~temp_mask

        valid_classes = (~temp_mask).sum().item()
        self.logger.info(f"  Task {task_id}: {valid_classes}/{shared_vocab_size} classes are valid")

    def prepare_for_task(self, task_info: TaskInfo) -> None:
        """Prepare for learning a new task - adds task vocabulary to unified space."""

        self.logger.info(f"Preparing MoE for task {task_info.task_id}: {task_info.task_name}")

        self.model_wrapper.zero_grad(set_to_none=True)

        if getattr(self.args, "init_unified_answer_space", False):
            assert self.task_configs is not None, "Task configs must be provided for unified answer space initialization."
            task_config = self.task_configs.get(task_info.task_id, None)
            if task_config is None:
                raise ValueError(f"No task config found for task ID {task_info.task_id}")
            assert task_config['task_name'] == task_info.task_name, "Task name mismatch in config."

            self._create_task_vocab_mask(task_info, task_config)

        # Add task to model wrapper (creates expert)
        self.model_wrapper.add_task(
            task_id=task_info.task_id,
            task_name=task_info.task_name,
            num_classes=task_info.num_classes,
            label2ans=task_info.label2ans
        )

        # Freeze previous tasks if not the first task
        if task_info.task_id > 0:
            self.logger.info(f"Freezing parameters for tasks < {task_info.task_id}")

            # Freeze all previous task heads
            for prev_task_id in range(task_info.task_id):
                self.model_wrapper.freeze_task(prev_task_id)

            # Freeze all previous LoRA adapters
            if hasattr(self.model_wrapper, 'freeze_other_task_lora'):
                self.model_wrapper.freeze_other_task_lora(task_info.task_id)

        # Set current task for routing
        self.model_wrapper.set_current_task(task_info.task_id)

        # Store task info
        self.current_task_info = task_info

        # Initialize expert training history
        self.expert_training_history[task_info.task_id] = {
            'task_name': task_info.task_name,
            'training_losses': [],
            'validation_accuracies': [],
            'best_accuracy': 0.0
        }

        self.logger.info(f"Prepared for task: {task_info.task_id}.")

    def consolidate_knowledge(self, task_info: TaskInfo) -> None:
        """Consolidate after task completion."""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()  # Free up memory after task

    def train_task(self, task_info: TaskInfo, train_loader: DataLoader, val_loader: Optional[DataLoader]=None, callbacks=None) -> Dict[str, Any]:
        import gc
        """
        Train on a single task using MoE.

        Args:
            task_info: Information about the current task
            train_loader: Training data loader
            val_loader: Validation data loader (optional)

        Returns:
            Dictionary containing training results and metrics
        """
        self.logger.info(f"\nTraining MoE expert {task_info.task_id} on task: {task_info.task_name}")

        # Log scaler state at VQA start
        if self.use_amp:
            current_scale = self.scaler.get_scale()
            growth_tracker = self.scaler._growth_tracker
            self.logger.debug(f"GradScaler at VQA start:")
            self.logger.debug(f"   scale={current_scale}")
            self.logger.debug(f"   _growth_tracker={growth_tracker}")

        task_id = task_info.task_id

        # Setup training configuration
        training_config = TrainingConfig(
            epochs=getattr(self.args, 'epochs', 10),
            batch_size=getattr(self.args, 'batch_size', 32),
            learning_rate=getattr(self.args, 'lr', 1e-4),
            optimizer_type=getattr(self.args, 'optimizer', 'adamw'),
            scheduler_type=getattr(self.args, 'scheduler', 'linear'),
            warmup_steps=getattr(self.args, 'warmup_steps', 100),
            gradient_clip_norm=getattr(self.args, 'clip_grad_norm', 1.0)
        )

        # Setup optimizer
        self.optimizer, self.scheduler = self.setup_optimizer_for_task(
            training_config, task_info.task_id, train_loader
        )

        self.model_wrapper.train()

        # Training loop
        training_losses = []
        validation_accuracies = []
        best_val_acc = 0.0

        # early stopping state
        self._setup_early_stopping_state(val_loader)    # resets best_metric, patience_counter best_epoch
        should_stop = False

        for epoch in range(training_config.epochs):
            epoch_loss = self._train_expert_epoch(epoch, train_loader, training_config, task_info.task_id)
            training_losses.append(epoch_loss)

            checkpoint_metric = None

            if val_loader is not None and not getattr(self.args, 'skip_validation', False):
                val_acc = self._validate_expert_epoch(epoch, val_loader, task_info.task_id)
                validation_accuracies.append(val_acc)
                self.logger.info(f"  Epoch {epoch} - Validation Accuracy: {val_acc:.4f}.")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.logger.info(f"  ↑ New best validation accuracy: {val_acc:.4f} (epoch {epoch})")

                checkpoint_metric = val_acc
            else:
                checkpoint_metric = -epoch_loss
                self.logger.debug("  Validation skipped, using negative loss for checkpointing.")

            is_final = (epoch == training_config.epochs - 1)
            self._save_vqa_checkpoint(
                task_info=task_info,
                epoch=epoch,
                metric_value=checkpoint_metric,
                is_final=is_final
            )
            # Early stopping check
            should_stop = self._should_early_stop(checkpoint_metric, epoch)
            if should_stop:
                self.logger.info(f"  Early stopping triggered at epoch {epoch}. Best metric: {self.best_metric:.4f} at epoch {self.best_epoch}.")
                break

        self._restore_best_vqa_checkpoint(task_info)

        # Store training history
        self.expert_training_history[task_info.task_id].update({
            'training_losses': training_losses,
            'validation_accuracies': validation_accuracies,
            'best_accuracy': best_val_acc,
            'final_loss': training_losses[-1] if training_losses else 0.0
        })
        if self.early_stopping_enabled:
            self.expert_training_history[task_info.task_id]['best_metric'] = self.best_metric
            self.expert_training_history[task_info.task_id]['best_epoch'] = self.best_epoch

        # Clear cache after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug(f"Cleared CUDA cache after VQA expert training")

        return {
            'task_id': task_info.task_id,
            'task_name': task_info.task_name,
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'best_val_accuracy': best_val_acc,
            'training_losses': training_losses,
            'validation_accuracies': validation_accuracies,
            'stopped_early': should_stop,
            'total_epochs': epoch + 1
        }

    def _restore_best_vqa_checkpoint(self, task_info: TaskInfo):
        """Restore the best VQA checkpoint for the given task."""
        checkpoint_path = self.vqa_checkpoint_manager.get_best_checkpoint_path(f"{task_info.task_name}")
        if checkpoint_path is not None and checkpoint_path.exists():
            self.logger.info(f"Restoring best VQA checkpoint from: {checkpoint_path}")
            self._load_vqa_checkpoint_from_path(task_info.task_id, task_info.task_name, checkpoint_path)
            self.logger.info(f"Restored best VQA checkpoint for task {task_info.task_id}")
        else:
            self.logger.warning(f"No best VQA checkpoint found for task {task_info.task_id} at expected path: {checkpoint_path}")

    def predict(self, batch: Dict[str, Any], task_id: Optional[int] = None, return_answer_strings: bool = True) -> Dict[str, Any]:
        """
        Make predictions using MoE routing.

        Args:
            batch: Input batch
            task_id: Task ID for oracle routing (if None, uses task predictor)
            return_answer_strings: If True, return actual answer strings; if False, return indices

        Returns:
            Dictionary containing:
                - predictions: Answer strings (if return_answer_strings=True) or indices
                - confidences: Prediction confidences
                - selected_expert: Which expert was used
                - prediction_indices: Raw prediction indices (always included for internal use)
        """
        self.model_wrapper.eval()

        assert not self.model_wrapper.training, "Model must be in eval mode for prediction!"

        # Also check dropout specifically
        for module in self.model_wrapper.modules():
            if isinstance(module, nn.Dropout):
                assert module.training == False, "Dropout should be disabled!"

        # Store original settings for deterministic evaluation
        original_benchmark = torch.backends.cudnn.benchmark
        original_deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        with torch.no_grad():
            with autocast(enabled=False):  # Disable autocast for stability during inference for consistency
                batch = self._move_batch_to_device(batch)
                batch_size = len(batch.get('questions', []))

                # Route through experts
                if task_id is not None:
                    # Oracle routing
                    selected_expert = task_id
                else:
                    # Oracle routing: use the ground-truth task IDs.
                    # Get features
                    mock_features = torch.zeros((batch['input_ids'].size(0), 768), device=self.device)  # Dummy features
                    # Use task predictor
                    ground_truth_ids = batch.get('task_ids', None)
                    if ground_truth_ids is not None and ground_truth_ids.dim() > 1:
                        ground_truth_ids = ground_truth_ids.squeeze(-1)
                    pred_outputs = self.task_predictor(mock_features, ground_truth_ids)
                    selected_expert = pred_outputs['predictions'][0].item()

                # Forward through selected expert
                self.model_wrapper.set_current_task(selected_expert)
                outputs = self.model_wrapper(batch, task_id=selected_expert) #, return_features=True)
                prediction_indices = torch.argmax(outputs.logits, dim=-1)                 # Extract
                predictions = self.model_wrapper.task_head_manager.get_answer_from_logits(outputs.logits, selected_expert, return_indices=False)    # Convert (no forward!)
                confidences = torch.max(outputs.probabilities, dim=-1)[0]

        # Restore original settings
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic

        return {
            'predictions': predictions,
            'confidences': confidences,
            'task_id_predictions': [selected_expert]*len(predictions),
            'task_id_confidences': [1.0]*len(predictions),
            'prediction_indices': prediction_indices,  # Always include for compatibility
        }

    def compute_loss(self, batch: Dict[str, Any], outputs: Dict[str, Any], mask = None, scale = False) -> torch.Tensor:
        """
        Compute the loss for training - FIXED for VQA soft targets.

        Args:
            batch: Training batch
            outputs: Model outputs
            mask: Optional mask for valid classes

        Returns:
            Computed loss tensor
        """

        return self.loss_manager.compute_total_loss(
            outputs,
            None,
            batch,
            mode='staged',
            mask=mask,
            scale=scale
        )

    def debug_parameter_states(self, task_id: int, stage: str = ""):
        """Debug function to show which parameters are trainable vs frozen."""
        print(f"\n{'='*80}")
        print(f"Parameter states for task {task_id} - {stage}")
        print(f"{'='*80}")

        # Categories for tracking
        categories = {
            'base_model': {'trainable': [], 'frozen': []},
            'lora_adapters': {},  # Will have task-specific subcategories
            'task_heads': {},  # Will have task-specific subcategories
            'other': {'trainable': [], 'frozen': []}
        }

        # Analyze model parameters
        for name, param in self.model_wrapper.named_parameters():
            param_info = {
                'name': name,
                'shape': list(param.shape),
                'numel': param.numel(),
                'requires_grad': param.requires_grad
            }

            # Categorize parameters
            if 'task_head_manager.heads' in name:
                # Extract task ID from name (e.g., "task_head_manager.heads.0.linear.weight")
                parts = name.split('.')
                if len(parts) > 1:
                    task_key = parts[1]  # Get the task ID
                    if task_key not in categories['task_heads']:
                        categories['task_heads'][task_key] = {'trainable': [], 'frozen': []}

                    if param.requires_grad:
                        categories['task_heads'][task_key]['trainable'].append(param_info)
                    else:
                        categories['task_heads'][task_key]['frozen'].append(param_info)

            elif 'lora_A' in name or 'lora_B' in name:
                # Extract task ID from LoRA adapter name
                # Format: "base_model.encoder.layer.0.attention.attention.query.task_lora_adapters.0.lora_A.weight"
                if 'task_lora_adapters' in name:
                    parts = name.split('task_lora_adapters.')
                    if len(parts) > 1:
                        task_key = parts[1].split('.')[0]  # Get the task ID
                        if task_key not in categories['lora_adapters']:
                            categories['lora_adapters'][task_key] = {'trainable': [], 'frozen': []}

                        if param.requires_grad:
                            categories['lora_adapters'][task_key]['trainable'].append(param_info)
                        else:
                            categories['lora_adapters'][task_key]['frozen'].append(param_info)

            elif 'base_model' in name or 'vilt' in name:
                if param.requires_grad:
                    categories['base_model']['trainable'].append(param_info)
                else:
                    categories['base_model']['frozen'].append(param_info)

            else:
                if param.requires_grad:
                    categories['other']['trainable'].append(param_info)
                else:
                    categories['other']['frozen'].append(param_info)

        # Print summary
        print("\nPARAMETER SUMMARY:")
        print("-" * 60)

        # Base model
        base_trainable = sum(p['numel'] for p in categories['base_model']['trainable'])
        base_frozen = sum(p['numel'] for p in categories['base_model']['frozen'])
        print(f"\nBASE MODEL:")
        print(f"  Trainable: {base_trainable:,} params ({len(categories['base_model']['trainable'])} tensors)")
        print(f"  Frozen: {base_frozen:,} params ({len(categories['base_model']['frozen'])} tensors)")

        # LoRA adapters by task
        print(f"\nLoRA ADAPTERS:")
        for task_key in sorted(categories['lora_adapters'].keys()):
            lora_trainable = sum(p['numel'] for p in categories['lora_adapters'][task_key]['trainable'])
            lora_frozen = sum(p['numel'] for p in categories['lora_adapters'][task_key]['frozen'])

            status = "ACTIVE" if task_key == str(task_id) else "FROZEN"
            print(f"  Task {task_key} {status}:")
            print(f"    Trainable: {lora_trainable:,} params")
            print(f"    Frozen: {lora_frozen:,} params")

        # Task heads by task
        print(f"\nTASK HEADS:")
        for task_key in sorted(categories['task_heads'].keys()):
            head_trainable = sum(p['numel'] for p in categories['task_heads'][task_key]['trainable'])
            head_frozen = sum(p['numel'] for p in categories['task_heads'][task_key]['frozen'])

            status = "ACTIVE" if task_key == str(task_id) else "FROZEN"
            print(f"  Task {task_key} {status}:")
            print(f"    Trainable: {head_trainable:,} params")
            print(f"    Frozen: {head_frozen:,} params")

        # Total summary
        total_trainable = base_trainable
        total_frozen = base_frozen

        for task_key in categories['lora_adapters']:
            total_trainable += sum(p['numel'] for p in categories['lora_adapters'][task_key]['trainable'])
            total_frozen += sum(p['numel'] for p in categories['lora_adapters'][task_key]['frozen'])

        for task_key in categories['task_heads']:
            total_trainable += sum(p['numel'] for p in categories['task_heads'][task_key]['trainable'])
            total_frozen += sum(p['numel'] for p in categories['task_heads'][task_key]['frozen'])

        print(f"\nTOTAL:")
        print(f"  Total Trainable: {total_trainable:,} params")
        print(f"  Total Frozen: {total_frozen:,} params")
        print(f"  Total Model: {total_trainable + total_frozen:,} params")

        # Check optimizer parameters
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            print(f"\nOPTIMIZER:")
            opt_params = []
            for param_group in self.optimizer.param_groups:
                opt_params.extend(param_group['params'])

            opt_param_count = sum(p.numel() for p in opt_params)
            print(f"  Parameters in optimizer: {len(opt_params)} tensors")
            print(f"  Total params in optimizer: {opt_param_count:,}")

            # Verify optimizer params match trainable params
            if opt_param_count != total_trainable:
                print(f"  WARNING: Optimizer has {opt_param_count:,} params but model has {total_trainable:,} trainable params!")
            else:
                print(f"  Optimizer params match trainable params")

        # Sample some parameter names
        if categories['base_model']['trainable']:
            print(f"\nSAMPLE BASE MODEL TRAINABLE PARAMS:")
            for p in categories['base_model']['trainable'][:3]:
                print(f"  - {p['name']}: {p['shape']}")

        print(f"\n{'='*80}\n")

    def get_state_dict(self) -> Dict[str, Any]:
        """Override to include training history."""
        # Get base state from parent class
        state = super().get_state_dict()

        # Add expert training history
        state['expert_training_history'] = self.expert_training_history
        # Save training configuration
        state['training_config'] = {
            'use_amp': self.use_amp,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Override to load training history."""
        # Load base state
        super().load_state_dict(state_dict)

        # Load expert training history if available
        self.expert_training_history = state_dict.get('expert_training_history', {})
        # Load training configuration
        if 'training_config' in state_dict:
            config = state_dict['training_config']
            self.use_amp = config.get('use_amp', self.use_amp)

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
                elif 'lora_A' in name or 'lora_B' in name:
                    if 'task_lora_adapters' in name:
                        task_key = name.split('task_lora_adapters.')[1].split('.')[0]
                        component = f"lora_task_{task_key}"
                    else:
                        component = "lora_unknown"
                else:
                    component = "base_model"

                if component not in has_gradients:
                    has_gradients[component] = []
                has_gradients[component].append((name, grad_norm))

            elif param.requires_grad and param.grad is None:
                if 'task_heads' in name:
                    task_key = name.split('.')[1]
                    component = f"task_head_{task_key}"
                elif 'lora' in name.lower():
                    component = "lora"
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
            print("\nTRAINABLE PARAMS WITHOUT GRADIENTS:")
            for component, params in no_gradients.items():
                print(f"\n  {component}: {len(params)} params")
                for name in params[:3]:  # Show first 3
                    print(f"    - {name}")

        print(f"\n{'='*60}\n")

    def _save_vqa_checkpoint(
        self,
        task_info: TaskInfo,
        epoch: int,
        metric_value: float,
        is_final: bool = False
    ):
        """Save VQA expert checkpoint."""
        task_id = task_info.task_id
        task_name = task_info.task_name

        # Collect task head state
        task_head_key = str(task_id)
        task_head_state = None
        if task_head_key in self.model_wrapper.task_head_manager.heads:
            task_head_state = self.model_wrapper.task_head_manager.heads[task_head_key].state_dict()

        # Collect LoRA adapter states
        lora_state = self._collect_lora_adapters(task_id)

        # Prepare model state
        model_state = {
            'task_head_state_dict': task_head_state,
            'lora_adapters_state_dict': lora_state
        }

        # Prepare strategy state
        strategy_state = {
            'expert_training_history': self.expert_training_history[task_id],
            'current_task_info': {
                'task_id': task_info.task_id,
                'task_name': task_info.task_name,
                'num_classes': task_info.num_classes,
                'label2ans': task_info.label2ans,
                'dataset_size': task_info.dataset_size,
            },
            'num_tasks': task_id + 1,
            'vqa_epochs': getattr(self.args, 'epochs', 10),
            'vqa_lr': getattr(self.args, 'lr', 1e-4),
            'training_stage': 'vqa_expert',
            'current_epoch': epoch,
            'is_complete': is_final,
        }

        # Save checkpoint
        saved_paths = self.vqa_checkpoint_manager.save_checkpoint(
            task_idx=task_id,
            task_name=task_name,
            epoch=epoch,
            model_state=model_state,
            strategy_state=strategy_state,
            metric_value=metric_value,
            is_final=is_final
        )

        self.logger.info(f"Saved VQA checkpoint for task {task_id}: {saved_paths}")

    def load_checkpoints_for_task(self, task_idx: int, task_config: Dict[str, Any]):
        """
        Load all component checkpoints for a task.
        Called during test mode or resume.
        """
        task_name = task_config['task_info']['task_name']
        checkpoint_paths = task_config.get('checkpoint_paths', {})

        self.logger.info(f"Loading checkpoints for task {task_idx}: {task_name}")

        # Load vqa component if available
        if checkpoint_paths.get('vqa'):
            success = self._load_vqa_checkpoint_from_path(
                task_idx, task_name, checkpoint_paths['vqa']
            )
            if success:
                self.vqa_loaded_from_checkpoint.add(task_idx)
                self.logger.info(f"  Loaded VQA component")

    def _load_vqa_checkpoint_from_path(self, task_idx, task_name, checkpoint_path):
        """Load VQA expert from specific path."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_state = checkpoint.get('model_state', checkpoint)

            # Load task head
            if 'task_head_state_dict' in model_state:
                task_head_key = str(task_idx)
                if task_head_key in self.model_wrapper.task_head_manager.heads:
                    self.model_wrapper.task_head_manager.heads[task_head_key].load_state_dict(
                        model_state['task_head_state_dict']
                    )

            # Load LoRA adapters
            if 'lora_adapters_state_dict' in model_state:
                self._load_lora_adapters(task_idx, model_state['lora_adapters_state_dict'])

            # Freeze
            self.model_wrapper.freeze_task(task_idx)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load VQA Expert: {e}")
            return False

    def _collect_lora_adapters(self, task_id: int) -> Dict[str, torch.Tensor]:
        """
        Collect LoRA adapter parameters for a specific task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary containing LoRA adapter state dicts
        """
        task_key = str(task_id)
        lora_state = {}

        # Iterate through the base model to find LoRA wrappers
        for name, module in self.model_wrapper.base_model.named_modules():
            if hasattr(module, 'task_lora_adapters'):
                if task_key in module.task_lora_adapters:
                    # Get adapter state dict
                    adapter_state = module.task_lora_adapters[task_key].state_dict()
                    # Add with module name prefix
                    for param_name, param_value in adapter_state.items():
                        full_name = f"{name}.{param_name}"
                        lora_state[full_name] = param_value

        return lora_state

    def _load_lora_adapters(self, task_id: int, lora_state_dict: Dict[str, torch.Tensor]):
        """
        Load LoRA adapters for a specific task.

        Args:
            task_id: Task ID
            lora_state_dict: State dictionary containing LoRA adapter parameters
        """
        task_key = str(task_id)

        # Iterate through the base model to find LoRA wrappers
        for name, module in self.model_wrapper.base_model.named_modules():
            if hasattr(module, 'task_lora_adapters'):
                if task_key in module.task_lora_adapters:
                    # Find matching keys in the state dict
                    module_prefix = f"{name}."
                    matching_keys = {
                        k: v for k, v in lora_state_dict.items()
                        if k.startswith(module_prefix)
                    }

                    if matching_keys:
                        # Remove prefix and load
                        adapter_state = {
                            k.replace(module_prefix, ''): v
                            for k, v in matching_keys.items()
                        }
                        module.task_lora_adapters[task_key].load_state_dict(
                            adapter_state, strict=False
                        )

    def _try_load_vqa_checkpoint(self, task_info: TaskInfo) -> bool:
        """
        Try to load VQA expert checkpoint for a task.

        Args:
            task_info: Information about the current task

        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        try:
            task_id = task_info.task_id
            task_name = task_info.task_name

            # Try to load from provided checkpoint path
            if isinstance(self.vqa_checkpoint_path, str):
                checkpoint_dir = Path(self.vqa_checkpoint_path)
            else:
                checkpoint_dir = self.vqa_checkpoint_dir

            # Try different checkpoint types in order of preference
            checkpoint_types = ['best', 'final', 'latest']
            checkpoint_path = None

            for ckpt_type in checkpoint_types:
                potential_path = checkpoint_dir / f"{task_name}_{ckpt_type}.pth"
                if potential_path.exists():
                    checkpoint_path = potential_path
                    self.logger.info(f"Found VQA checkpoint: {checkpoint_path}")
                    break

            if checkpoint_path is None:
                self.logger.warning(f"No VQA checkpoint found for task {task_name}")
                return False

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Extract model state
            if 'model_state' in checkpoint:
                model_state = checkpoint['model_state']
            else:
                model_state = checkpoint

            # Load task head
            if 'task_head_state_dict' in model_state:
                task_head_key = str(task_id)
                if task_head_key in self.model_wrapper.task_head_manager.heads:
                    self.model_wrapper.task_head_manager.heads[task_head_key].load_state_dict(
                        model_state['task_head_state_dict']
                    )
                    self.logger.info(f"  Loaded task head for task {task_id}")

            # Load LoRA adapters
            if 'lora_adapters_state_dict' in model_state:
                # Load LoRA adapters into the base model
                self._load_lora_adapters(task_id, model_state['lora_adapters_state_dict'])
                self.logger.info(f"  Loaded LoRA adapters for task {task_id}")

            # Freeze the loaded components
            self.model_wrapper.freeze_task(task_id)
            if hasattr(self.model_wrapper, 'freeze_other_task_lora'):
                self.model_wrapper.freeze_other_task_lora(task_id)

            self.logger.info(f"Successfully loaded VQA checkpoint for task {task_id}: {task_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load VQA checkpoint for task {task_info.task_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def supports_meaningful_oracle_evaluation(self) -> bool:
        """Indicate support for meaningful oracle evaluation."""
        return True

    def _profile(self, epoch, train_loader, task_id):
        from torch.profiler import profile, ProfilerActivity, schedule

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=lambda p: print(p.key_averages().table(
                sort_by="cuda_time_total", row_limit=20
            )),
            record_shapes=True,
            with_stack=False
        ) as prof:

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 10:  # Profile first 5 batches
                    break

                batch = self._move_batch_to_device(batch)

                batch_size = len(batch.get('question_ids', batch.get('questions', [''])))
                batch['ground_truth_task_ids'] = torch.full((batch_size,), task_id,
                                                        dtype=torch.long, device=self.device, pin_memory=False)
                with autocast():
                    outputs = self.model_wrapper(batch, task_id=task_id) #, return_features=True)
                    loss = self.compute_loss(batch, outputs, None, self.scale_loss)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                prof.step()  # Tell profiler this iteration is done
        self.logger.debug("Profiling finished!")

    def _profile2(self, epoch, train_loader, task_id):
        self.logger.info("Starting transfer tracking...")
        from utils.transfer_tracker import TransferTracker
        tracker = TransferTracker()
        tracker.start()

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 20:  # Profile first 5 batches
                break

            batch = self._move_batch_to_device(batch)

            batch_size = len(batch.get('question_ids', batch.get('questions', [''])))
            batch['ground_truth_task_ids'] = torch.full((batch_size,), task_id,
                                                    dtype=torch.long, device=self.device, pin_memory=False)
            with autocast():
                outputs = self.model_wrapper(batch, task_id=task_id) #, return_features=True)
                loss = self.compute_loss(batch, outputs, None, self.scale_loss)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

        tracker.stop()
        tracker.report()
        self.logger.debug("Transfer tracking finished!")

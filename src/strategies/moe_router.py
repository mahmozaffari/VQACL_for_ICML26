"""
MoE router strategy with task prediction
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
from models.taskid_classifier import TaskIDClassifier, create_model_config_from_args
from models.text_embedder import BERTEmbedder
from strategies.modules.memory_buffer import MLPMemoryManager
from utils.data_utils import cycle
from utils.data_utils import FiniteMemoryIterator
from core.checkpoint import CheckpointManager, CheckpointMetric
from pathlib import Path

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

class MoERouterStrategy(BaseStrategy):
    """MoE router strategy with task prediction."""

    def __init__(self, model_wrapper, args, task_list: List[str], **kwargs):

        self._task_list = task_list
        self.num_experts = len(task_list)
        self.num_trained_tasks = 0

        # Add logger instance
        self.logger = logging.getLogger('CL.strategy.moe_router')

        self.scale_loss = getattr(args, 'scale_vqa_loss', False)

        super().__init__(model_wrapper, args, **kwargs)

        # Checkpoint configuration
        self.vqa_checkpoint_path = getattr(args, 'vqa_checkpoint_path', None)  # Path to pre-trained VQA checkpoints
        self.mlp_checkpoint_path = getattr(args, 'mlp_checkpoint_path', None)  # Path to pre-trained MLP checkpoints

        #  Multi-component checkpoint tracking
        self.vqa_loaded_from_checkpoint = set()
        self.mlp_loaded_from_checkpoint = set()

        # Skip flags
        self.skip_mlp_training_if_loaded = getattr(args, 'skip_mlp_training_if_loaded', True)
        self.skip_vqa_training_if_loaded = getattr(args, 'skip_vqa_training_if_loaded', True)

        self._verify_checkpoint_configs()

        self.logger.info(f"Initialized MoE strategy with {self.num_experts} experts")

    def get_checkpoint_paths_for_task(
        self, task_idx: int, task_name: str, base_dir: str, checkpoint_type: str = None
    ) -> Dict[str, Optional[str]]:
        """
        Get paths to all component checkpoints for a task.
        Called by base trainer during checkpoint extraction.
        """
        from pathlib import Path

        paths = {}
        base_path = Path(base_dir)

        # VQA expert checkpoint (main component)
        vqa_dir = base_path / 'checkpoints' / 'vqa'
        vqa_path = self._find_component_checkpoint_helper(vqa_dir, task_name)
        paths['vqa'] = str(vqa_path) if vqa_path else None

        # Router/MLP checkpoint
        mlp_dir = base_path / 'checkpoints' / 'router'
        mlp_path = self._find_component_checkpoint_helper(mlp_dir, task_name, checkpoint_type)
        paths['mlp'] = str(mlp_path) if mlp_path else None

        # Set main component
        paths['main'] = paths['vqa']

        return paths

    def _find_component_checkpoint_helper(self, checkpoint_dir, task_name, checkpoint_type=None) -> Optional[Path]  :
        """Find best available checkpoint for a component."""
        from pathlib import Path

        if not checkpoint_dir.exists():
            return None

        if checkpoint_type is not None:
            suffix = f"_{checkpoint_type}.pth"
            path = checkpoint_dir / f"{task_name}{suffix}"
            if path.exists():
                return path
            self.logger.warning(f"No {checkpoint_type} checkpoint found for task {task_name} in {checkpoint_dir}")

        for suffix in ['_best.pth', '_final.pth', '_latest.pth']:
            path = checkpoint_dir / f"{task_name}{suffix}"
            if path.exists():
                return path
        return None

    def _verify_checkpoint_configs(self):
        """Verify checkpoint configurations."""

        if self.mlp_checkpoint_path is not None:
            self.logger.info(f"MLP checkpoints will be loaded from: {self.mlp_checkpoint_path}")
        else:
            self.logger.info("No MLP checkpoint path provided; training MLP from scratch.")
        if self.vqa_checkpoint_path is not None:
            self.logger.info(f"VQA checkpoints will be loaded from: {self.vqa_checkpoint_path}")
        else:
            self.logger.info("No VQA checkpoint path provided; training VQA experts from scratch.")

    def _initialize_strategy_components(self):
        """Initialize MoE-specific components with unified answer space."""
        # Initialize perfect task predictor

        self.training_mode = getattr(self.args, 'training_mode', 'full')  # 'full', 'mlp_only', 'vqa_only'

        # Move to device if available
        if hasattr(self.model_wrapper, 'device'):
            self.device = self.model_wrapper.device
        else:
            self.device = torch.device("cpu")
            self.logger.warning("Warning: model_wrapper has no device attribute, skipping device transfer for components")

        self.loss_manager = LossManager(self.args, self.device)
        # Expert training history
        self.expert_training_history: Dict[int, Dict[str, Any]] = {}
        self.mlp_training_history: Dict[int, Dict[str, Any]] = {}

        self.text_embedder = BERTEmbedder(getattr(self.args, 'bert_model', 'bert-base-uncased'))
        self.text_embedder = self.text_embedder.to(self.device)

        self.memory_manager = MLPMemoryManager(total_buffer_size=getattr(self.args, 'memory_buffer_size', 5000))

        self._mlp_configure()

        self.task_id_classifier = TaskIDClassifier(self.mlp_config).to(self.device)

        self._setup_checkpoint_managers()

        self.logger.info(f"MLP initialized with config: {self.mlp_config}")
        self.logger.debug(self.task_id_classifier)
        self.logger.info(" Initialized text embedder")

    def _setup_checkpoint_managers(self):
        checkpoint_dir = Path(self.args.output) / 'checkpoints'

        # MLP checkpoint manager
        self.mlp_checkpoint_dir = Path(checkpoint_dir) / 'router'
        self.mlp_checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.mlp_checkpoint_dir),
            metric=CheckpointMetric.ACCURACY,
            keep_last_n_checkpoints=getattr(self.args, 'keep_last_n_checkpoints', 1),
            save_every_n_epochs=getattr(self.args, 'save_every_n_epochs', 1),
            logger=self.logger
        )

        # VQA expert checkpoint manager
        self.vqa_checkpoint_dir = Path(checkpoint_dir) / 'vqa'
        self.vqa_checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.vqa_checkpoint_dir),
            metric=CheckpointMetric.ACCURACY,
            keep_last_n_checkpoints=getattr(self.args, 'keep_last_n_checkpoints', 2),
            save_every_n_epochs=getattr(self.args, 'save_every_n_epochs', 1),
            logger=self.logger
        )

        self.logger.info(f"MLP checkpoints: {self.mlp_checkpoint_dir}")
        self.logger.info(f"VQA checkpoints: {self.vqa_checkpoint_dir}")

    def _mlp_configure(self):
        self._router_input_dim = self.text_embedder.get_embedding_dim()
        self.mlp_config = create_model_config_from_args(
            self.args,
            input_dim=self._router_input_dim,
            expand_input=False
        )
        self.mlp_epochs = getattr(self.args, 'mlp_epochs', 10)
        self.mlp_lr = getattr(self.args, 'mlp_lr', 1e-4)

    def _get_router_inputs(self, questions: List[str]) -> torch.Tensor:
        """Get router inputs directly from question embeddings."""
        assert questions is not None, "_get_router_inputs called with None questions"
        requires_grad = any(p.requires_grad for p in self.text_embedder.parameters())
        if requires_grad and self.text_embedder.training:
            return self.text_embedder.embed_batch(questions)
        with torch.no_grad():
            return self.text_embedder.embed_batch(questions)

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

        for batch_idx, batch in enumerate(pbar):
            actual_accumulated_steps = 1  # Default for non-accumulation steps

            # Move batch to GPU efficiently
            batch = self._move_batch_to_device(batch)

            # Add ground truth task IDs
            batch_size = len(batch.get('question_ids', batch.get('questions', [''])))
            batch['ground_truth_task_ids'] = torch.full((batch_size,), task_id,
                                                       dtype=torch.long, device=self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model_wrapper(batch, task_id=task_id, return_features=True)
                    loss = self.compute_loss(batch, outputs, scale=self.scale_loss)
                    loss = loss / self.gradient_accumulation_steps

                # Backward with scaled gradients
                self.scaler.scale(loss).backward()
            else:
                # Regular forward/backward
                outputs = self.model_wrapper(batch, task_id=task_id, return_features=True)
                loss = self.compute_loss(batch, outputs, scale=self.scale_loss)
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

                    if training_config.gradient_clip_norm > 0:
                        task_params = self._get_task_parameters(task_id)
                        torch.nn.utils.clip_grad_norm_(task_params, training_config.gradient_clip_norm)

                    # Step optimizer
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular gradient clipping
                    if training_config.gradient_clip_norm > 0:
                        task_params = self._get_task_parameters(task_id)
                        torch.nn.utils.clip_grad_norm_(task_params, training_config.gradient_clip_norm)

                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                if batch_idx == 0:  # Check once per epoch
                    # Verify frozen parameters haven't changed
                    for prev_task_id in range(task_id):
                        head_key = str(prev_task_id)
                        if head_key in self.model_wrapper.task_head_manager.heads:
                            for name, param in self.model_wrapper.task_head_manager.heads[head_key].named_parameters():
                                assert not param.requires_grad, f"Task head {prev_task_id} param {name} is not frozen!"

                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item() * actual_accumulated_steps #self.gradient_accumulation_steps
            num_batches += 1

        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0

        self.logger.info(f"  Epoch {epoch+1} - Loss: {epoch_loss:.4f}.")

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
        self.task_id_classifier.eval()
        self.text_embedder.eval()

        total_predictions = 0
        total_accuracy = 0.0

        with torch.no_grad():
            with autocast(False): # Disable AMP for validation for safety
                for batch in tqdm(val_loader, desc=f"Validation", leave=False, **tqdm_config):
                    batch = self._move_batch_to_device(batch)

                    # Add ground truth task IDs
                    batch_size = len(batch.get('question_id', batch.get('questions', [''])))
                    batch['ground_truth_task_ids'] = torch.full((batch_size,), task_id,
                                                               dtype=torch.long, device=self.device)

                    # Forward pass
                    outputs = self.model_wrapper(batch, task_id=task_id, return_features=True)

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
                weight_decay=0.01  # Standard weight decay
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

    def prepare_for_task(self, task_info: TaskInfo) -> None:
        """Prepare for learning a new task - adds task vocabulary to unified space."""

        self.logger.info(f"Preparing MoE for task {task_info.task_id}: {task_info.task_name}")

        self.model_wrapper.zero_grad(set_to_none=True)

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

        self.mlp_training_history[task_info.task_id] = {
            'task_name': task_info.task_name,
            'training_losses': [],
            'validation_losses': [],
            'validation_accuracies': [],
            'best_val_acc': 0.0
        }

        task_id = task_info.task_id

        # Try to load AE_MLP checkpoint if path is provided
        if self.mlp_checkpoint_path is not None and task_id not in self.mlp_loaded_from_checkpoint:
            checkpoint_dir = Path(self.mlp_checkpoint_path)
            mlp_path = self._find_component_checkpoint_helper(checkpoint_dir, task_info.task_name)
            if mlp_path and self._load_mlp_checkpoint_from_path(task_id, task_info.task_name, str(mlp_path)):
                self.mlp_loaded_from_checkpoint.add(task_id)
                self.logger.info(f"Loaded MLP checkpoint for task {task_info.task_id}")

        # Try to load VQA checkpoint if path is provided
        if self.vqa_checkpoint_path is not None and task_id not in self.vqa_loaded_from_checkpoint:
            checkpoint_dir = Path(self.vqa_checkpoint_path)
            vqa_path = self._find_component_checkpoint_helper(checkpoint_dir, task_info.task_name)
            if vqa_path and self._load_vqa_checkpoint_from_path(task_id, task_info.task_name, str(vqa_path)):
                self.vqa_loaded_from_checkpoint.add(task_id)
                self.logger.info(f"Loaded VQA checkpoint for task {task_info.task_id}")

        self.logger.info(f"Prepared for task: {task_info.task_id}.")

    def consolidate_knowledge(self, task_info: TaskInfo) -> None:
        """Consolidate after task completion."""
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()  # Free up memory after task

    def _train_task_mlp(self, task_info: TaskInfo, train_loader: DataLoader, val_loader: Optional[DataLoader]=None) -> Dict[str, Any]:
        """
        Train MLP for a single task.
        Args:
            task_info: Information about the current task
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        Returns:
            Dictionary containing training results and metrics
        """

        task_id = task_info.task_id

        #  Skip if checkpoint was loaded and flag is set
        if task_id in self.mlp_loaded_from_checkpoint and self.skip_mlp_training_if_loaded:
            self.logger.info(f" Skipping MLP training for task {task_id} - loaded from checkpoint")
            return {
                'task_id': task_id,
                'task_name': task_info.task_name,
                'skipped': True,
                'reason': 'loaded_from_checkpoint'
            }

        # instantiate memory loader
        if self.memory_manager.get_num_tasks()>0:
            memory_loader = self.memory_manager.create_dataloader_from_samples(batch_size=train_loader.batch_size)
        else:
            memory_loader = None

        self.logger.debug('Memory loader: {}'.format(memory_loader))

        # setup optimizer
        optimizer = torch.optim.Adam(self.task_id_classifier.parameters(), lr=self.mlp_lr, weight_decay=0)
        # Setup scheduler
        total_steps = len(train_loader) * self.mlp_epochs

        from transformers.optimization import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(100, total_steps // 10),
            num_training_steps=total_steps
        )

        best_val_acc = 0.0
        training_losses = []
        validation_losses = []
        validation_accuracies = []

        self.task_id_classifier.train()  # set router to train mode

        for epoch in range(self.mlp_epochs):
            epoch_loss = self._train_mlp_epoch(epoch, task_id, self.task_id_classifier, train_loader, memory_loader, optimizer, scheduler)

            training_losses.append(epoch_loss)

            checkpoint_metric = None

            if val_loader is not None and not getattr(self.args, 'skip_validation', False):
                val_loss, val_acc = self._validate_mlp_epoch(epoch, task_id, self.task_id_classifier, val_loader)
                validation_losses.append(val_loss)
                validation_accuracies.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.logger.info(f"  ↑ New best validation loss: {val_loss:.4f}")
                checkpoint_metric = val_acc
            else:
                checkpoint_metric = -epoch_loss
                self.logger.debug("  Validation skipped, using train loss for checkpointing.")

            is_final = (epoch == self.mlp_epochs - 1)
            self._save_mlp_checkpoint(
                task_info=task_info,
                epoch=epoch,
                metric_value=checkpoint_metric,
                is_final=is_final
            )

        self.mlp_training_history[task_id].update({
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'validation_accuracies': validation_accuracies,
            'best_val_acc': best_val_acc,
            'final_loss': training_losses[-1] if training_losses else 0.0
        })

        if memory_loader is not None:
            del memory_loader
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.debug(f"Cleaned up memory_loader for Task {task_id}")
            self.logger.debug(f"Cleaned up memory_loader for Task {task_id}")

        self.memory_manager.add_new_task_buffer(task_id, train_loader)

        del optimizer
        del scheduler
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.debug("MLP optimizer/scheduler deleted, memory freed")

        return {
            'task_id': task_id,
            'task_name': task_info.task_name,
            'final_loss': training_losses[-1] if training_losses else 0.0,
            'best_val_acc': best_val_acc,
            'training_losses': training_losses,
            'validation_losses': validation_losses,
        }

    def _validate_mlp_epoch(self, epoch:int, task_id:int, mlp_model: nn.Module, val_loader: DataLoader) -> float:

        mlp_model.eval()
        self.text_embedder.eval()
        self.model_wrapper.eval()

        total_loss = 0.0
        num_batches = 0
        total_predictions = 0
        total_accuracy = 0.0

        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                pbar = tqdm(val_loader, desc=f"MLP Task {task_id} Val Epoch {epoch+1}", leave=False, **tqdm_config)
                for batch in pbar:
                    batch = self._move_batch_to_device(batch)
                    questions = batch.get('questions', [])
                    targets = batch.get('task_ids').squeeze()
                    router_inputs = self._get_router_inputs(questions)
                    logits = mlp_model(router_inputs)
                    predictions = torch.argmax(logits, dim=-1)

                    individual_accuracies = (predictions == targets).float()
                    batch_accuracy = individual_accuracies.mean().item()
                    total_accuracy += batch_accuracy * targets.size(0)
                    total_predictions += targets.size(0)

                    batch_loss = F.cross_entropy(logits, targets, reduction='none')

                    loss = batch_loss.mean()
                    total_loss += loss.item()
                    num_batches += 1

                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Accuracy': f'{(total_accuracy / total_predictions):.4f}'
                    })

        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_accuracy = total_accuracy / total_predictions if total_predictions > 0 else 0.0

        self.logger.info(f"  MLP Val Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        return epoch_loss, epoch_accuracy

    def _train_mlp_epoch(self, epoch:int, task_id:int, mlp_model:nn.Module, train_loader: DataLoader, memory_loader: Optional[DataLoader], optimizer: torch.optim.Optimizer, scheduler: Any ) -> float:

        mlp_model.train()
        self.text_embedder.eval()
        self.model_wrapper.eval()

        total_loss = 0.0
        num_batches = 0
        total_task_accuracy = 0.0
        total_mem_accuracy = 0.0
        task_predictions = 0
        mem_predictions = 0

        # Use FiniteMemoryIterator instead of cycle() to prevent memory leaks
        if memory_loader is not None and len(memory_loader.dataset)>0:
            memory_iterator = FiniteMemoryIterator(memory_loader)
            use_memory = True
        else:
            memory_iterator = None
            use_memory = False

        pbar = tqdm(train_loader, desc=f"MLP Task {task_id} Epoch {epoch+1}", leave=False, **tqdm_config)

        for batch in pbar:

            batch = self._move_batch_to_device(batch)

            if use_memory:
                mem_batch = next(memory_iterator)
                mem_batch = self._move_batch_to_device(mem_batch)
            else:
                mem_batch = None

            if self.use_amp:
                with autocast():
                    questions, task_ids = batch.get('questions', []), batch.get('task_ids').squeeze()
                    router_inputs = self._get_router_inputs(questions)
                    logits = mlp_model(router_inputs)
                    batch_loss = F.cross_entropy(logits, task_ids, reduction='mean')
                    batch_accuracy = (torch.argmax(logits, dim=-1) == task_ids).float().mean().item()
                    total_task_accuracy += batch_accuracy * len(task_ids)
                    task_predictions += len(task_ids)

                    if mem_batch is not None:
                        mem_questions = mem_batch.get('questions', [])
                        mem_task_ids = mem_batch.get('task_ids').squeeze()
                        mem_router_inputs = self._get_router_inputs(mem_questions)
                        mem_logits = mlp_model(mem_router_inputs)
                        mem_loss = F.cross_entropy(mem_logits, mem_task_ids, reduction='mean')
                        mem_accuracy = (torch.argmax(mem_logits, dim=-1) == mem_task_ids).float().mean().item()
                        total_mem_accuracy += mem_accuracy * len(mem_task_ids)
                        mem_predictions += len(mem_task_ids)

                        loss = (batch_loss + mem_loss) / 2
                    else:
                        loss = batch_loss

                # Backward with scaled gradients
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

            else:
                questions, task_ids = batch.get('questions', []), batch.get('task_ids').squeeze()
                router_inputs = self._get_router_inputs(questions)
                logits = mlp_model(router_inputs)
                batch_loss = F.cross_entropy(logits, task_ids, reduction='mean')
                batch_accuracy = (torch.argmax(logits, dim=-1) == task_ids).float().mean().item()
                total_task_accuracy += batch_accuracy * len(task_ids)
                task_predictions += len(task_ids)

                if mem_batch is not None:
                    mem_questions = mem_batch.get('questions', [])
                    mem_task_ids = mem_batch.get('task_ids').squeeze()
                    mem_router_inputs = self._get_router_inputs(mem_questions)
                    mem_logits = mlp_model(mem_router_inputs)
                    mem_loss = F.cross_entropy(mem_logits, mem_task_ids, reduction='mean')
                    loss = (batch_loss + mem_loss) / 2
                    mem_accuracy = (torch.argmax(mem_logits, dim=-1) == mem_task_ids).float().mean().item()
                    total_mem_accuracy += mem_accuracy * len(mem_task_ids)
                    mem_predictions += len(mem_task_ids)
                else:
                    loss = batch_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Explicitly delete batch references to release memory
            del batch
            if mem_batch is not None:
                del mem_batch

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'Task Acc': f'{(total_task_accuracy / task_predictions):.4f}' if task_predictions > 0 else 'N/A',
                'Mem Acc': f'{(total_mem_accuracy / mem_predictions):.4f}' if mem_predictions > 0 else 'N/A'
            })

        # Clean up memory iterator
        if memory_iterator is not None:
            del memory_iterator

        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.info(f"  MLP Epoch {epoch+1} - Loss: {epoch_loss:.4f}")
        return epoch_loss

    def train_task(self, task_info: TaskInfo, train_loader: DataLoader, val_loader: Optional[DataLoader]=None, callbacks=None) -> Dict[str, Any]:
        import gc
        """
        Train on a single task
        Args:
            task_info: Information about the current task
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        Returns:
            Dictionary containing training results and metrics
        """

        results = {}

        if self.training_mode in ['full', 'mlp_only']:
            mlp_results = self._train_task_mlp(task_info, train_loader, val_loader)
            results['mlp'] = mlp_results

            gc.collect()
            gc.collect()  # Call twice for cyclic references
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.debug("Memory cleaned up after MLP training")

            if self.training_mode == 'mlp_only':
                self.logger.info(f"MLP-only mode: Skipping VQA training")
                return results

        if self.training_mode in ['full', 'vqa_only']:
            vqa_results = self._train_vqa_expert(task_info, train_loader, val_loader)
            results['vqa'] = vqa_results

            # Clean up after vqa phase
            if hasattr(self, 'optimizer'):
                del self.optimizer
            if hasattr(self, 'scheduler'):
                del self.scheduler
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.debug(f"Memory cleaned up after VQA training")

        self.num_trained_tasks = task_info.task_id + 1

        return results

    def _train_vqa_expert(self, task_info: TaskInfo, train_loader: DataLoader, val_loader: Optional[DataLoader]=None) -> Dict[str, Any]:
        """
        Train VQA expert for a single task

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

        # CHECK: Skip if checkpoint was loaded and flag is set
        if task_id in self.vqa_loaded_from_checkpoint and self.skip_vqa_training_if_loaded:
            self.logger.info(f" Skipping VQA training for task {task_id} - loaded from checkpoint")
            return {
                'task_id': task_id,
                'task_name': task_info.task_name,
                'skipped': True,
                'reason': 'loaded_from_checkpoint'
            }

        self.debug_parameter_states(task_info.task_id, "BEFORE OPTIMIZER SETUP")

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

        # Check parameter states after optimizer setup
        self.debug_parameter_states(task_info.task_id, "AFTER OPTIMIZER SETUP")

        self.model_wrapper.train()

        # Training loop
        training_losses = []
        validation_accuracies = []
        best_val_acc = 0.0

        str_ = f"\nBEFORE EPOCH LOOP - Task {task_info.task_id}\n"

        for name, param in self.model_wrapper.named_parameters():
            if 'lora' in name and param.requires_grad:
                str_ += f"  {name}: requires_grad={param.requires_grad}, is_leaf={param.is_leaf}, grad_fn={param.grad_fn}\n"
                break  # Just show one example
        self.logger.debug(str_)

        for epoch in range(training_config.epochs):
            epoch_loss = self._train_expert_epoch(epoch, train_loader, training_config, task_info.task_id)
            training_losses.append(epoch_loss)

            checkpoint_metric = None

            if val_loader is not None  and not getattr(self.args, 'skip_validation', False):
                val_acc = self._validate_expert_epoch(epoch, val_loader, task_info.task_id)
                validation_accuracies.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.logger.info(f"  ↑ New best validation accuracy: {val_acc:.4f}")
                checkpoint_metric = val_acc
            else:
                checkpoint_metric = -epoch_loss
                self.logger.debug("  Validation skipped, using negative train loss for checkpointing.")

            is_final = (epoch == training_config.epochs - 1)
            self._save_vqa_checkpoint(
                task_info=task_info,
                epoch=epoch,
                metric_value=checkpoint_metric,
                is_final=is_final
            )

        # Store training history
        self.expert_training_history[task_info.task_id].update({
            'training_losses': training_losses,
            'validation_accuracies': validation_accuracies,
            'best_accuracy': best_val_acc,
            'final_loss': training_losses[-1] if training_losses else 0.0
        })

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
        }

    def predict_ensemble(self, batch: Dict[str, Any], return_answer_strings: bool = True, k:int = 3, temperature: float = 1.0) -> Dict[str, Any]:
        """
        Make predsictions using ensemble of top-k experts.
        Args:
            batch: Input batch
            return_answer_strings: If True, return actual answer strings; if False, return indices
            k: Number of top experts to ensemble
            temperature: Softmax temperature for weighting expert contributions
        Returns:
            Dictionary containing:
                - predictions: Answer strings (if return_answer_strings=True) or indices
                - confidences: Prediction confidences
                - selected_experts: Which experts were used
                - prediction_indices: Raw prediction indices (always included for internal use)
        """
        self.model_wrapper.eval()
        self.text_embedder.eval()
        self.task_id_classifier.eval()

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
            with autocast(enabled=self.use_amp):
                batch = self._move_batch_to_device(batch)
                batch_size = len(batch.get('questions', []))

                # predict task-ids
                questions = batch.get('questions', [])
                router_inputs = self._get_router_inputs(questions)
                gate_logits = self.task_id_classifier(router_inputs)
                num_trained_tasks = len(self.model_wrapper.task_head_manager.heads)
                gate_logits = gate_logits[:, :num_trained_tasks]  # Slice to trained tasks!
                expert_weights = F.softmax(gate_logits / temperature, dim=-1)

                k = min(k, num_trained_tasks)

                all_expert_unified_probs, all_expert_local_outputs = self._get_all_experts_unified(batch, num_trained_tasks)

                # Get top-k experts
                topk_weights, topk_indices = torch.topk(expert_weights, k=k, dim=-1)
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

                # Combine top-k experts at answer level
                combined_probs = self._combine_topk_unified(all_expert_unified_probs,
                                                                                    topk_indices,
                                                                                    topk_weights,
                                                                                    batch_size,
                                                                                    k)

                # Step 5: Get predictions
                pred_indices = torch.argmax(combined_probs, dim=-1)  # [B]
                confidences = combined_probs.max(dim=-1).values      # [B]

                unified_vocab = self.model_wrapper.task_head_manager.unified_vocab
                predictions = [unified_vocab[idx.item()] for idx in pred_indices]

                per_expert_predictions = {
                    expert_id: outputs['pred_answers'] for expert_id, outputs in all_expert_local_outputs.items()
                }

                per_expert_confidences = {
                    expert_id: outputs['confidences'].cpu().tolist() for expert_id, outputs in all_expert_local_outputs.items()
                }

                per_expert_pred_indices = {
                    expert_id: outputs['pred_indices'].cpu().tolist() for expert_id, outputs in all_expert_local_outputs.items()
                }

                per_sample_experts = []
                for sample_idx in range(batch_size):
                    sample_data = {
                        'selected_expert': topk_indices[sample_idx, 0].item(),
                        'experts': []
                    }
                    for expert_id in range(num_trained_tasks):
                        router_entropy = -(expert_weights[sample_idx] * torch.log(expert_weights[sample_idx] + 1e-8)).sum()
                        sample_data['router_entropy'] = router_entropy.item()
                        sample_data['N_effective'] = torch.exp(router_entropy).item()
                        expert_info = {
                            'expert_id': expert_id,
                            'weight': expert_weights[sample_idx, expert_id].item(),
                            'prediction': all_expert_local_outputs[expert_id]['pred_answers'][sample_idx],
                            'confidence': all_expert_local_outputs[expert_id]['confidences'][sample_idx].item()
                        }
                        sample_data['experts'].append(expert_info)
                    per_sample_experts.append(sample_data)

        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic

        return {
            'predictions': predictions,
            'confidences': confidences.cpu().tolist(),
            'task_id_predictions': topk_indices[:, 0].cpu().tolist(),
            'task_id_confidences': topk_weights[:, 0].cpu().tolist(),
            'expert_weights': topk_weights.cpu().tolist(),
            'expert_indices': topk_indices.cpu().tolist(),
            'prediction_indices': [-1]*batch_size,  # Not applicable in ensemble
            'all_expert_weights': expert_weights.cpu().tolist(),
            'combined_probs': combined_probs.cpu().tolist(),
            'per_sample_experts': per_sample_experts
        }

    def _get_all_experts_unified(self, batch: Dict[str, Any], num_experts: int) -> Dict[int, Dict[str, Any]]:
        """
        Run batch through all experts and collect outputs.

        Returns:
            Dictionary mapping expert_id to their outputs:
                - probs: [B, num_classes] probability distribution
                - pred_indidces: [B] predicted indices
                - pred_answers: List of predicted answer strings
                - confidences: [B] prediction confidences
        """

        questions = batch.get('questions')
        batch_size = len(questions)
        device = self.device
        unified_vocab_size = self.model_wrapper.task_head_manager._get_unified_vocab_size()

        all_unified_probs = torch.zeros(num_experts, batch_size, unified_vocab_size, device=device)
        local_outputs = {}

        for expert_id in range(num_experts):
            self.model_wrapper.set_current_task(expert_id)
            outputs = self.model_wrapper(batch, task_id=expert_id, return_features=True)

            # Get local logits
            local_logits = outputs.logits
            local_probs = outputs.probabilities

            # Transform to unified space
            unified_probs = self.model_wrapper.task_head_manager.local_to_unified_probs(local_probs, expert_id, fill_value=0.0)
            all_unified_probs[expert_id] = unified_probs

            # Store local outputs for diagnostics
            pred_indices = torch.argmax(local_logits, dim=-1)
            confidences = local_probs.max(dim=-1).values
            pred_answers = self.model_wrapper.task_head_manager.get_answer_from_logits(local_logits, expert_id, return_indices=False)
            local_outputs[expert_id] = {
                'local_logits': local_logits,
                'local_probs': local_probs,
                'unified_probs': unified_probs,
                'pred_indices': pred_indices,
                'pred_answers': pred_answers,
                'confidences': confidences
            }
        return all_unified_probs, local_outputs

    def _combine_topk_unified(self, all_expert_unified_probs: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor, batch_size: int, k: int) -> torch.Tensor:
        """
        Weighted combination of top-k experts in unified space.

        Returns:
            combined_probs: [B, unified_vocab_size]
        """
        unified_vocab_size = all_expert_unified_probs.size(-1)
        device = all_expert_unified_probs.device

        combined_probs = torch.zeros(batch_size, unified_vocab_size, device=device)

        for j in range(k):
        # Get expert indices for this rank: [B]
            expert_ids = topk_indices[:, j]
            # Get weights for this rank: [B]
            weights = topk_weights[:, j]

            # Gather expert probs for each sample
            # all_expert_unified_probs[expert_ids[b], b, :] for each b
            for b in range(batch_size):
                eid = expert_ids[b].item()
                combined_probs[b] += weights[b] * all_expert_unified_probs[eid, b]

        return combined_probs
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
        self.text_embedder.eval()
        self.task_id_classifier.eval()

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
                    selected_expert, task_confidences = task_id, 1.0
                    predictions, prediction_indices, confidences = self._predict_by_expert(batch, selected_expert, return_answer_strings)
                else:
                    # predict task-ids
                    questions = batch.get('questions', [])
                    router_inputs = self._get_router_inputs(questions)
                    logits = self.task_id_classifier(router_inputs)
                    num_trained_tasks = len(self.model_wrapper.task_head_manager.heads)
                    logits = logits[:, :num_trained_tasks]  # Slice to trained tasks!
                    selected_expert = torch.argmax(logits, dim=1)
                    task_probabilities = F.softmax(logits, dim=1)
                    task_confidences = torch.max(task_probabilities, dim=1)[0]

                    # Initialize outputs with correct size
                    all_predictions = [None] * batch_size
                    all_prediction_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                    all_confidences = torch.zeros(batch_size, device=self.device)

                    for expert_id in torch.unique(selected_expert):
                        mask = (selected_expert == expert_id)
                        if mask.sum() == 0:
                            continue

                        # Get indices for reordering
                        indices = torch.where(mask)[0]

                        # Create sub-batch
                        sub_batch = {k: (v[mask] if isinstance(v, torch.Tensor)
                                    else [v[i] for i in range(len(v)) if mask[i]])
                                    for k, v in batch.items()}

                        preds, pred_indices, confs = self._predict_by_expert(sub_batch, expert_id.item(), return_answer_strings)

                        # Store predictions in ORIGINAL positions
                        for i, idx in enumerate(indices):
                            all_predictions[idx.item()] = preds[i]
                            all_prediction_indices[idx] = pred_indices[i]
                            all_confidences[idx] = confs[i]

                    # Verify all predictions are filled
                    assert all(p is not None for p in all_predictions), "Some predictions were not filled!"
                    assert len(all_predictions) == batch_size, f"Mismatch in predictions: {len(all_predictions)} vs {batch_size}"

                    predictions = all_predictions
                    prediction_indices = all_prediction_indices
                    confidences = all_confidences

        # Restore original settings
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_deterministic

        return {
            'predictions': predictions,
            'confidences': confidences,
            'task_id_predictions': selected_expert if task_id is None else [selected_expert]*len(predictions),
            'task_id_confidences': task_confidences if task_id is None else [1.0]*len(predictions),
            'prediction_indices': prediction_indices,  # Always include for compatibility,

        }

    def _predict_by_expert(self, batch: Dict[str, Any], selected_expert: int, return_answer_strings: bool = True) -> Dict[str, Any]:

        self.model_wrapper.set_current_task(selected_expert)
        outputs = self.model_wrapper(batch, task_id=selected_expert, return_features=True)
        prediction_indices = torch.argmax(outputs.logits, dim=-1)                 # Extract
        predictions = self.model_wrapper.task_head_manager.get_answer_from_logits(outputs.logits, selected_expert, return_indices=False)    # Convert (no forward!)
        confidences = torch.max(outputs.probabilities, dim=-1)[0]

        return predictions, prediction_indices, confidences

    def compute_loss(self, batch: Dict[str, Any], outputs: Dict[str, Any], scale: bool = False) -> torch.Tensor:
        """
        Compute the loss for training - FIXED for VQA soft targets.

        Args:
            batch: Training batch
            outputs: Model outputs

        Returns:
            Computed loss tensor
        """

        return self.loss_manager.compute_total_loss(
            outputs,
            None,
            batch,
            mode='staged',
            scale=scale
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """Override to include training history."""
        # Get base state from parent class
        state = super().get_state_dict()

        state['expert_training_history'] = self.expert_training_history
        state['mlp_training_history'] = self.mlp_training_history

        # Save training configuration
        state['training_config'] = {
            'training_mode': self.training_mode,
            'mlp_lr': self.mlp_lr,
            'mlp_epochs': self.mlp_epochs,
            'router_input_dim': self._router_input_dim,
            'use_amp': self.use_amp,
        }

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Override to load training history."""
        # Load base state
        self.logger.info('Loading state dictionary ...')
        super().load_state_dict(state_dict)

        self.expert_training_history = state_dict.get('expert_training_history', {})
        self.mlp_training_history = state_dict.get('mlp_training_history', {})

        training_config = state_dict.get('training_config', {})

        if 'mlp_lr' in training_config:
            self.mlp_lr = training_config['mlp_lr']
        if 'mlp_epochs' in training_config:
            self.mlp_epochs = training_config['mlp_epochs']
        if 'router_input_dim' in training_config:
            self._router_input_dim = training_config['router_input_dim']
        if 'use_amp' in training_config:
            self.use_amp = training_config['use_amp']

        routing_state = state_dict.get('routing', {})

        # Load router MLP
        if 'router' in routing_state and hasattr(self, 'task_id_classifier'):
            self.task_id_classifier.load_state_dict(routing_state['router'])

        # Load memory buffer
        if 'memory' in routing_state and hasattr(self, 'memory_manager'):
            self.memory_manager.load_state(routing_state['memory'])

    # Save MLP checkpoint
    def _save_mlp_checkpoint(
        self,
        task_info: TaskInfo,
        epoch: int,
        metric_value: float,
        is_final: bool = False
    ):
        """Save MLP checkpoint including memory buffer."""
        task_id = task_info.task_id
        task_name = task_info.task_name

        # Prepare model state
        model_state = {
            'mlp_classifier_state_dict': self.task_id_classifier.state_dict(),
            'memory_buffer_state': self.memory_manager.get_state(),
            'mlp_config': self.mlp_config
        }

        # Prepare strategy state
        strategy_state = {
            'mlp_training_history': self.mlp_training_history[task_id],
            'num_tasks': task_id + 1,
            'mlp_epochs': self.mlp_epochs,
            'mlp_lr': self.mlp_lr,
            'training_stage': 'mlp',
            'current_epoch': epoch,
            'is_complete': is_final,
        }

        # Save checkpoint
        saved_paths = self.mlp_checkpoint_manager.save_checkpoint(
            task_idx=task_id,
            task_name=task_name,
            epoch=epoch,
            model_state=model_state,
            strategy_state=strategy_state,
            metric_value=metric_value,
            is_final=is_final
        )

        self.logger.info(f"Saved MLP checkpoint for task {task_id}: {saved_paths}")

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
            'num_classes': task_info.num_classes,
            'label2ans': task_info.label2ans,
            'num_tasks': task_id + 1,
            'vqa_epochs': getattr(self.args, 'epochs', 10),
            'vqa_lr': getattr(self.args, 'lr', 1e-4),
            'training_stage': 'vqa',
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

    def supports_meaningful_oracle_evaluation(self) -> bool:
        """Indicate support for meaningful oracle evaluation."""
        return True

    def supports_bayesian_evaluation(self) -> bool:
        """Indicate support for Bayesian evaluation."""
        # By default, assume strategies do not support Bayesian evaluation
        return True
    def load_checkpoints_for_task(self, task_idx: int, task_config: Dict[str, Any]):
        """
        Load all component checkpoints for a task.
        Called during test mode or resume.
        """
        task_name = task_config['task_info']['task_name']
        checkpoint_paths = task_config.get('checkpoint_paths', {})

        self.logger.info(f"Loading checkpoints for task {task_idx}: {task_name}")

        if checkpoint_paths.get('mlp'):
            success = self._load_mlp_checkpoint_from_path(
                task_idx, task_name, checkpoint_paths['mlp']
            )
            if success:
                self.mlp_loaded_from_checkpoint.add(task_idx)
                self.logger.info(f"  Loaded Router component")

        if checkpoint_paths.get('vqa'):
            success = self._load_vqa_checkpoint_from_path(
                task_idx, task_name, checkpoint_paths['vqa']
            )
            if success:
                self.vqa_loaded_from_checkpoint.add(task_idx)
                self.logger.info(f"  Loaded VQA component")

    def _verify_routing_loaded(self):
        """Verify routing components loaded (for logging)."""
        status = []
        if hasattr(self, 'task_id_classifier'):
            status.append("router MLP")
        if hasattr(self, 'memory_manager'):
            n_samples = len(self.memory_manager.get_all_samples())
            n_tasks = len(self.memory_manager.task_buffers)
            status.append(f"memory ({n_samples} samples, {n_tasks} tasks)")

        self.logger.info(f"  Routing: {', '.join(status)}")

    def _load_mlp_checkpoint_from_path(self, task_idx, task_name, checkpoint_path):
        """Load MLP from specific path."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model_state = checkpoint.get('model_state', checkpoint)

            if 'mlp_classifier_state_dict' in model_state:
                self.task_id_classifier.load_state_dict(
                    model_state['mlp_classifier_state_dict']
                )

            if 'memory_buffer_state' in model_state:
                self.memory_manager.load_state(model_state['memory_buffer_state'])

            return True
        except Exception as e:
            self.logger.error(f"Failed to load Router: {e}")
            return False

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

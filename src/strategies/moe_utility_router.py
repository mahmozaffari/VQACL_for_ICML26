"""
MoE utility router strategy
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
from models.taskid_classifier import TaskIDClassifier, create_model_config_from_args
from models.text_embedder import BERTEmbedder
from strategies.modules.memory_buffer import MLPMemoryManager
from utils.data_utils import cycle
from utils.data_utils import FiniteMemoryIterator
from pathlib import Path
from strategies.moe_router import MoERouterStrategy

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

VALID_LOSS_TYPES = ['ACC', 'KL']

class MoEUtilityRouterStrategy(MoERouterStrategy):
    """MoE utility router strategy."""

    def __init__(self, model_wrapper, args, task_list: List[str], **kwargs):

        super().__init__(model_wrapper, args, task_list, **kwargs)

        # Add logger instance
        self.logger = logging.getLogger('CL.strategy.moe_utility_router')

        self.router_alpha = getattr(self.args, 'router_alpha', 0.0)  # Task-ID regularization weight
        self.router_beta = getattr(self.args, 'router_beta', 0.0)   # Entropy regularization weight
        self.router_gamma = getattr(self.args, 'router_gamma', 0.0) # Load balancing weight

        # Router loss type
        self.router_loss_type = getattr(self.args, 'router_loss_type', 'ACC')
        if self.router_loss_type not in VALID_LOSS_TYPES:
            raise ValueError(f"Invalid router_loss_type '{self.router_loss_type}'. Must be one of {VALID_LOSS_TYPES}.")

        # Oracle bootstrapping is not supported
        self.use_oracle_bootstrap = getattr(self.args, 'use_oracle_bootstrap', False)
        if self.use_oracle_bootstrap:
            self.logger.warning("Oracle bootstrapping requested but not yet implemented!")

        self._unified_label2ans = kwargs.get('unified_label2ans', None)
        if self._unified_label2ans is None:
            raise NotImplementedError("Unified label2ans handling not yet implemented in MoEUtilityRouterStrategy.")
        self._unified_ans2idx = {ans: idx for idx, ans in enumerate(self._unified_label2ans)}

        # DRO hyperparameters
        self.use_dro = getattr(self.args, 'use_dro_loss', False)  # Default to True for this class
        self.dro_lambda = getattr(self.args, 'dro_lambda', 0.1)

        # Will be built incrementally as tasks are added
        self.expert_to_unified = {}  # expert_id -> {expert_ans_idx: unified_ans_idx}
        self.unified_to_expert = {}  # expert_id -> {unified_ans_idx: expert_ans_idx}

        self.logger.info("MoE Utility Router Strategy Initialized")
        self.logger.info(f"   Router loss type: {self.router_loss_type}")
        self.logger.info(f"   Router hyperparameters: α={self.router_alpha}, β={self.router_beta}, γ={self.router_gamma}")
        if self.use_dro:
            self.logger.info(f"   DRO enabled with λ={self.dro_lambda}")

    def _mlp_configure(self):
        self._router_input_dim = self.text_embedder.get_embedding_dim()
        self.mlp_config = create_model_config_from_args(
            self.args,
            input_dim=self._router_input_dim,
            expand_input=False,
        )
        self.mlp_epochs = getattr(self.args, 'mlp_epochs', 10)
        self.mlp_lr = getattr(self.args, 'mlp_lr', 1e-4)

    def prepare_for_task(self, task_info: TaskInfo) -> None:
        """Prepare for learning a new task - adds task vocabulary to unified space."""

        super().prepare_for_task(task_info)
        self._build_expert_unified_mappings(task_info)

    def _build_expert_unified_mappings(self, task_info: TaskInfo):
        """
        Build mappings between expert's vocabulary and unified vocabulary.

        This allows us to convert targets from unified space to any expert's space.
        """
        expert_id = task_info.task_id
        expert_label2ans = task_info.label2ans  # Expert's vocabulary

        # unified_label2ans comes from __init__ (passed from trainer)
        if not hasattr(self, '_unified_ans2idx'):
            raise ValueError("Unified label2ans mapping not found!")

        unified_vocab_size = len(self._unified_label2ans)
        expert_vocab_size = len(expert_label2ans)

        # Build expert_to_unified: expert_ans_idx -> unified_ans_idx
        expert_to_unified = {}
        for idx, answer in enumerate(expert_label2ans):
            if answer in self._unified_ans2idx:
                expert_to_unified[idx] = self._unified_ans2idx[answer]
            else:
                self.logger.warning(f"Answer '{answer}' from expert {expert_id} not in unified space!")

        # Build unified_to_expert: unified_ans_idx -> expert_ans_idx
        unified_to_expert = {v: k for k, v in expert_to_unified.items()}

        self.expert_to_unified[expert_id] = expert_to_unified
        self.unified_to_expert[expert_id] = unified_to_expert

        # Optional: Cache conversion matrices for faster batched operations
        # expert → unified matrix
        expert_to_unified_matrix = torch.zeros(expert_vocab_size, unified_vocab_size)
        for expert_idx, unified_idx in expert_to_unified.items():
            expert_to_unified_matrix[expert_idx, unified_idx] = 1.0

        # unified → expert matrix
        unified_to_expert_matrix = torch.zeros(unified_vocab_size, expert_vocab_size)
        for unified_idx, expert_idx in unified_to_expert.items():
            unified_to_expert_matrix[unified_idx, expert_idx] = 1.0

        if not hasattr(self, 'expert_to_unified_matrices'):
            self.expert_to_unified_matrices = {}
            self.unified_to_expert_matrices = {}

        self.expert_to_unified_matrices[expert_id] = expert_to_unified_matrix.to(self.device)
        self.unified_to_expert_matrices[expert_id] = unified_to_expert_matrix.to(self.device)

        self.logger.info(
            f"Built mappings for expert {expert_id}: "
            f"{len(expert_to_unified)}/{len(expert_label2ans)} answers mapped"
        )

    def _get_all_expert_accuracies(
        self,
        batch: Dict[str, Any],
        unified_targets: torch.Tensor,
        num_experts: int
    ) -> torch.Tensor:
        """
        Run all experts (frozen previous + trainable current) and stack their predictions.

        Args:
            batch: Input batch
            current_task_id: ID of the task currently being trained
            num_experts: Total number of experts available

        Returns:
            expert_logits_stacked: [batch_size, num_experts, vocab_size]
        """
        expert_accuracy_list = []

        for expert_id in range(num_experts):
            # Set the current expert
            self.model_wrapper.set_current_task(expert_id)

            with torch.no_grad():
                outputs = self.model_wrapper(batch, task_id=expert_id, return_features=True)
                expert_logits = outputs.logits  # [batch_size, task_vocab_size]

                # FAST target conversion using matrix multiplication
                expert_targets = self._get_expert_targets_fast_internal(
                    unified_targets, expert_id
                )

                pred_indices = torch.argmax(expert_logits, dim=-1)
                pred_scores = expert_targets.gather(1, pred_indices.unsqueeze(1)).squeeze(1)
                accuracy = torch.clamp(pred_scores, 0.0, 1.0)

                expert_accuracy_list.append(accuracy)

                del outputs, expert_logits, expert_targets, pred_indices, pred_scores

        # Stack all expert predictions: [batch_size, num_experts, vocab_size]
        all_expert_accuracies = torch.stack(expert_accuracy_list, dim=1)

        return all_expert_accuracies

    def _convert_task_targets_to_unified(
        self,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        """
        Fast version using matrix operations where possible.
        """
        unified_vocab_size = len(self._unified_label2ans)

        unified_targets = torch.matmul(targets, self.expert_to_unified_matrices[task_id])
        assert unified_targets.size(1) == unified_vocab_size

        return unified_targets

    def _convert_targets_to_unified(
        self,
        targets: torch.Tensor,          # [batch_size, task_vocab_size] or varying sizes
        task_ids: torch.Tensor          # [batch_size] - which task each sample belongs to
    ) -> torch.Tensor:
        """
        Convert targets from task-specific spaces to unified space.

        Each sample in the batch may come from a different task, so we process
        them individually based on their task_id.

        Args:
            targets: Task-specific targets (can have varying vocab sizes across batch)
            task_ids: Task ID for each sample

        Returns:
            unified_targets: [batch_size, unified_vocab_size]
        """
        batch_size = task_ids.size(0)
        unified_vocab_size = len(self._unified_label2ans)

        # Initialize unified targets
        unified_targets = torch.zeros(
            batch_size, unified_vocab_size,
            device=targets.device,
            dtype=targets.dtype
        )

        # Group samples by task_id to process efficiently
        unique_task_ids = torch.unique(task_ids)

        for task_id in unique_task_ids:
            task_id_int = task_id.item()

            # Find samples belonging to this task
            mask = (task_ids == task_id_int)
            sample_indices = torch.where(mask)[0]

            # Get mapping for this task
            if task_id_int not in self.expert_to_unified:
                raise ValueError(f"Mappings for task_id {task_id_int} not found!")

            expert_to_unified = self.expert_to_unified[task_id_int]

            # Convert each sample's targets
            for sample_idx in sample_indices:
                sample_targets = targets[sample_idx]  # [task_vocab_size]

                # Map each expert answer index to unified index
                for expert_ans_idx, unified_ans_idx in expert_to_unified.items():
                    unified_targets[sample_idx, unified_ans_idx] = sample_targets[expert_ans_idx]

        return unified_targets

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

        best_val_vqa_acc = 0.0
        best_val_task_acc = 0.0
        training_losses = []
        validation_losses = []
        validation_vqa_accs = []
        validation_task_accs = []

        self.task_id_classifier.train()  # set router to train mode

        # Generate oracle accuracies once per task
        self.logger.info("Generating oracle expert accuracies (this will take time, but only once)...")
        oracle_accuracies, qid_to_idx = self._generate_oracle_accuracies_v2(train_loader, memory_loader, task_id)
        self.logger.info(f"Oracle accuracies generated for {oracle_accuracies.shape[0]} batches")

        for epoch in range(self.mlp_epochs):
            epoch_loss = self._train_mlp_epoch_with_oracle_v2(epoch, task_id, self.task_id_classifier, train_loader, memory_loader, oracle_accuracies, qid_to_idx, optimizer, scheduler)

            training_losses.append(epoch_loss)
            self.logger.info(f"  Router Train Epoch {epoch+1} - Loss: {epoch_loss:.4f}")

            checkpoint_metric = None

            if val_loader is not None and not getattr(self.args, 'skip_validation', False):
                val_loss, val_vqa_acc, val_task_acc = self._validate_mlp_epoch(epoch, task_id, self.task_id_classifier, val_loader)
                validation_losses.append(val_loss)
                validation_task_accs.append(val_task_acc)
                validation_vqa_accs.append(val_vqa_acc)

                if val_vqa_acc > best_val_vqa_acc:
                    best_val_vqa_acc = val_vqa_acc
                    self.logger.info(f"  ↑ New best validation loss: {val_loss:.4f}, VQA Acc: {val_vqa_acc:.4f}, Task Acc: {val_task_acc:.4f}, Best VQA Acc: {best_val_vqa_acc:.4f}")
                checkpoint_metric = val_vqa_acc
            else:
                checkpoint_metric = -epoch_loss  # Use negative loss if no val
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
            'validation_vqa_accs': validation_vqa_accs,
            'validation_task_accs': validation_task_accs,
            'best_val_vqa_acc': best_val_vqa_acc,
            'final_loss': training_losses[-1] if training_losses else 0.0
        })

        # Cleanup oracle accuracies to free memory
        del oracle_accuracies

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
            'best_val_vqa_acc': best_val_vqa_acc,
            'training_losses': training_losses,
            'validation_losses': validation_losses,
            'validation_vqa_accs': validation_vqa_accs,
            'validation_task_accs': validation_task_accs
        }

    def _validate_mlp_epoch(self, epoch:int, task_id:int, mlp_model: nn.Module, val_loader: DataLoader) -> float:

        mlp_model.eval()
        self.text_embedder.eval()
        self.model_wrapper.eval()

        num_batches = 0

        # Loss component tracking
        total_loss = 0.0
        total_vqa_loss = 0.0
        total_task_loss = 0.0

        total_predictions = 0
        total_task_accuracy = 0.0
        total_vqa_accuracy = 0.0
        total_upperbound_accuracy = 0.0
        total_oracle_accuracy = 0.0

        num_experts = task_id + 1

        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                pbar = tqdm(val_loader, desc=f"MLP Task {task_id} Val Epoch {epoch+1}", leave=False, **tqdm_config)
                for batch in pbar:
                    batch = self._move_batch_to_device(batch)

                    task_ids_gt = batch.get('task_ids').squeeze()
                    local_targets = self._extract_targets(batch)   # task local targets
                    unified_targets = self._convert_task_targets_to_unified(local_targets, task_id)

                    # Get router predictions
                    questions = batch.get('questions', [])
                    router_inputs = self._get_router_inputs(questions)
                    gate_logits = mlp_model(router_inputs)[:, :num_experts]
                    gate_weights = torch.argmax(gate_logits, dim=-1)

                    # Get expert predictions
                    expert_accuracies_stacked = self._get_all_expert_accuracies(
                        batch, unified_targets, num_experts
                    )

                    # Compute loss
                    loss, loss_components = self._compute_router_loss(
                        gate_logits=gate_logits,
                        expert_vqa_accuracies_stacked=expert_accuracies_stacked,
                        answer_targets=unified_targets,
                        task_ids=task_ids_gt,
                        alpha=self.router_alpha,
                        beta=self.router_beta,
                        gamma=self.router_gamma
                    )

                    # Accumulate metrics from loss_components
                    batch_size = len(task_ids_gt)
                    total_loss += loss.item()
                    total_task_accuracy += loss_components['task_acc'] * batch_size
                    total_vqa_accuracy += loss_components['vqa_acc'] * batch_size
                    total_upperbound_accuracy += loss_components['upperbound_accuracy'] * batch_size
                    total_oracle_accuracy += loss_components['oracle_accuracy'] * batch_size
                    total_predictions += batch_size
                    num_batches += 1

                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Primary Loss': f'{loss_components["vqa_loss"]:.4f}' if 'vqa_loss' in loss_components else f'{loss_components["kl_divergence"]:.4f}',
                        'VQA_Acc': f'{(total_vqa_accuracy / total_predictions if total_predictions > 0 else 0):.4f}',
                        'Task_Acc': f'{(total_task_accuracy / total_predictions if total_predictions > 0 else 0):.4f}',
                        'Upperbound_Acc': f'{(total_upperbound_accuracy / total_predictions if total_predictions > 0 else 0):.4f}',
                        'Oracle_Acc': f'{(total_oracle_accuracy / total_predictions if total_predictions > 0 else 0):.4f}',
                        'Entropy': f'{loss_components["entropy"]:.4f}',

                    })

        # Compute epoch averages
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_vqa_acc = total_vqa_accuracy / total_predictions if total_predictions > 0 else 0.0
        epoch_task_acc = total_task_accuracy / total_predictions if total_predictions > 0 else 0.0
        epoch_upperbound_acc = total_upperbound_accuracy / total_predictions if total_predictions > 0 else 0.0

        self.logger.info(
            f"  Router Val Epoch {epoch+1} - "
            f"Loss: {epoch_loss:.4f}, "
            f"VQA_Acc: {epoch_vqa_acc:.4f} (PRIMARY), "
            f"Upperbound: {epoch_upperbound_acc:.4f}, "
            f"Task_Acc: {epoch_task_acc:.4f}",
        )
        return epoch_loss, epoch_vqa_acc, epoch_task_acc

    def _train_mlp_epoch_with_oracle_v2(
        self,
        epoch: int,
        task_id: int,
        mlp_model: nn.Module,
        train_loader: DataLoader,
        memory_loader: Optional[DataLoader],
        # oracle_accuracies: Dict[str, torch.Tensor],  # Pre-computed!
        oracle_tensor: torch.Tensor, qid_to_idx: Dict[str, int],
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ) -> float:

        mlp_model.train()
        self.text_embedder.eval()
        self.model_wrapper.eval()

        total_loss = 0.0
        num_batches = 0
        # Loss component accumulators
        total_vqa_loss = 0.0
        total_task_loss = 0.0
        total_entropy = 0.0
        total_load_balance = 0.0
        # Accuracy tracking
        total_vqa_accuracy = 0.0
        total_task_accuracy = 0.0
        total_upperbound_accuracy = 0.0
        total_oracle_accuracy = 0.0
        task_predictions = 0

        num_experts = task_id + 1

        # Setup memory iterator
        if memory_loader is not None and len(memory_loader.dataset) > 0:
            memory_iterator = FiniteMemoryIterator(memory_loader)
            use_memory = True
        else:
            memory_iterator = None
            use_memory = False

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"MLP Task {task_id} Epoch {epoch+1}",
            leave=False, **tqdm_config
        )

        mlp_params = mlp_model.parameters()

        for batch_idx, batch in pbar:

            batch = self._move_batch_to_device(batch)

            # GET QUESTION IDs (same as oracle generation)
            question_ids = batch.get('question_id')
            questions = batch.get('questions', [])
            batch_size = len(question_ids)

            batch_indices = torch.tensor([qid_to_idx[qid] for qid in question_ids], device=self.device)

            if use_memory:
                mem_batch = next(memory_iterator)
                mem_batch = self._move_batch_to_device(mem_batch)
                mem_question_ids = mem_batch.get('question_id')
                mem_batch_indices = torch.tensor([qid_to_idx[qid] for qid in mem_question_ids], device=self.device)
                # Extract task IDs and questions for router
                mem_task_ids = mem_batch.get('task_ids').view(-1)
                all_batch_indices = torch.cat([batch_indices, mem_batch_indices], dim=0)
                all_questions = questions + mem_batch.get('questions')
                all_task_ids = torch.cat([batch.get('task_ids').view(-1), mem_task_ids], dim=0)
            else:
                all_batch_indices = batch_indices
                all_questions = questions
                all_task_ids = batch.get('task_ids').view(-1)

        # LOAD ORACLE ACCURACIES BY QUESTION ID
            total_size = all_batch_indices.size(0)
            expert_accuracies_stacked = oracle_tensor[all_batch_indices]

            # Split back to current batch and memory batch

            if self.use_amp:
                with autocast():
                    router_inputs = self._get_router_inputs(all_questions)
                    gate_logits = mlp_model(router_inputs)[:, :num_experts]
                    gate_weights = F.softmax(gate_logits, dim=-1)

                    # Compute loss using cached accuracies
                    loss, loss_components = self._compute_router_loss(
                        gate_logits=gate_logits,
                        expert_vqa_accuracies_stacked=expert_accuracies_stacked,
                        answer_targets=None,
                        task_ids=all_task_ids,
                        alpha=self.router_alpha,
                        beta=self.router_beta,
                        gamma=self.router_gamma
                    )

                # Backward with scaled gradients
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mlp_params, max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

            else:
                router_inputs = self._get_router_inputs(all_questions)
                gate_logits = mlp_model(router_inputs)[:, :num_experts]
                gate_weights = F.softmax(gate_logits, dim=-1)

                # Compute loss using cached accuracies
                loss, loss_components = self._compute_router_loss(
                    gate_logits=gate_logits,
                    expert_vqa_accuracies_stacked=expert_accuracies_stacked,
                    answer_targets=None,
                    task_ids=all_task_ids,
                    alpha=self.router_alpha,
                    beta=self.router_beta,
                    gamma=self.router_gamma
                )

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mlp_params, max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler:
                scheduler.step()

            # Compute metrics (without expert forwards!)
            with torch.no_grad():
                predicted_tasks = torch.argmax(gate_weights, dim=-1)
                task_acc = (predicted_tasks == all_task_ids).float().mean().item()
                entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean().item()
                vqa_acc = expert_accuracies_stacked.gather(1, predicted_tasks.unsqueeze(1)).squeeze(1).mean().item()
                best_vqa_acc = expert_accuracies_stacked.max(dim=-1)[0].mean().item()
                oracle_vqa_acc = expert_accuracies_stacked.gather(1, all_task_ids.unsqueeze(1)).squeeze(1).mean().item()

            # Accumulate metrics
            batch_size = len(all_task_ids)
            total_loss += loss.item()
            total_vqa_loss += -loss.item()  # Store as positive for logging
            total_task_loss += 0.0  # Not used in this formulation
            total_entropy += entropy
            total_load_balance += 0.0  # Not used

            total_task_accuracy += task_acc * batch_size
            total_vqa_accuracy += vqa_acc * batch_size
            total_upperbound_accuracy += best_vqa_acc * batch_size
            total_oracle_accuracy += oracle_vqa_acc * batch_size
            task_predictions += batch_size
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'VQA_Acc': f'{(total_vqa_accuracy / task_predictions if task_predictions > 0 else 0):.4f}',
                'Task_Acc': f'{(total_task_accuracy / task_predictions if task_predictions > 0 else 0):.4f}',
                'Upperbound': f'{(total_upperbound_accuracy / task_predictions if task_predictions > 0 else 0):.4f}',
                'Oracle_Acc': f'{(total_oracle_accuracy / task_predictions if task_predictions > 0 else 0):.4f}',
                'Entropy': f'{(total_entropy / num_batches if num_batches > 0 else 0):.4f}',
                'l2_reg_loss': f'{loss_components["l2_regularization"]:.4f}' if 'l2_regularization' in loss_components else 'N/A',
                'entropy_reg_loss': f'{loss_components["entropy_loss"]:.4f}' if 'entropy_loss' in loss_components else 'N/A',
            })

            # Clean up
            del batch, expert_accuracies_stacked

        # Compute epoch averages
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_vqa_acc = total_vqa_accuracy / task_predictions if task_predictions > 0 else 0.0
        epoch_task_acc = total_task_accuracy / task_predictions if task_predictions > 0 else 0.0
        epoch_entropy = total_entropy / num_batches if num_batches > 0 else 0.0

        self.logger.info(
            f"  Router Epoch {epoch+1} (Oracle) - "
            f"Loss: {epoch_loss:.4f}, "
            f"VQA_Acc: {epoch_vqa_acc:.4f}, "
            f"Task_Acc: {epoch_task_acc:.4f}, "
            f"Entropy: {epoch_entropy:.4f}, "
        )

        return epoch_loss

    def _compute_router_loss(
        self,
        gate_logits: torch.Tensor,      # [batch_size, num_experts]
        expert_vqa_accuracies_stacked: torch.Tensor,  # [batch_size, num_experts]
        answer_targets: torch.Tensor,    # [batch_size, vocab_size] - soft targets
        task_ids: torch.Tensor,          # [batch_size] - ground truth task IDs
        alpha: float,                    # Task-ID regularization weight
        beta: float,                     # Entropy weight
        gamma: float                     # Load balancing weight
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-component loss for task-agnostic router training.

        Loss = VQA_loss + α * TaskID_loss + β * Entropy + γ * LoadBalance

        Args:
            gate_weights: Softmax weights from router [B, num_experts]
            expert_logits_stacked: Predictions from all experts [B, num_experts, vocab_size]
            answer_targets: Ground truth answer scores [B, vocab_size]
            task_ids: Ground truth task IDs [B]
            alpha, beta, gamma: Loss component weights

        Returns:
            total_loss: Combined loss
            loss_components: Dictionary with individual loss values for logging
        """
        batch_size, num_experts = expert_vqa_accuracies_stacked.shape

        # DEFENSIVE: Ensure task_ids is 1D and Long
        task_ids = task_ids.view(-1).long()

        # 1. Main VQA Loss: Soft mixture of expert predictions
        gate_weights = F.softmax(gate_logits, dim=-1)
        expected_accuracy = (gate_weights * expert_vqa_accuracies_stacked).sum(dim=-1)  # [B]

        if self.router_loss_type == 'ACC':
            loss, loss_details = self._compute_expected_accuracy_loss(
                gate_weights, expert_vqa_accuracies_stacked, beta
            )
        elif self.router_loss_type == 'KL':
            loss, loss_details = self._compute_kl_divergence_loss(
                gate_logits, gate_weights, expert_vqa_accuracies_stacked, beta
            )
        else:
            raise ValueError(f"Unknown router_loss_type: {self.router_loss_type}")

        with torch.no_grad():
            predicted_tasks = torch.argmax(gate_weights, dim=-1)
            task_acc = (predicted_tasks == task_ids).float().mean().item()
            vqa_acc = expert_vqa_accuracies_stacked.gather(1, predicted_tasks.unsqueeze(1)).squeeze(1).mean().item()
            best_vqa_acc = expert_vqa_accuracies_stacked.max(dim=-1)[0].mean().item()
            oracle_vqa_acc = expert_vqa_accuracies_stacked.gather(1, task_ids.unsqueeze(1)).squeeze(1).mean().item()

            expected_accuracy = (gate_weights * expert_vqa_accuracies_stacked).sum(dim=-1).mean().item()

            # Entropy
            entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean().item()

        # Prepare logging dictionary
        loss_components = {
            'loss_type': self.router_loss_type,
            'total_loss': loss.item(),
            'vqa_acc': vqa_acc,
            'task_acc': task_acc,
            'expected_accuracy': expected_accuracy,
            'upperbound_accuracy': best_vqa_acc,
            'oracle_accuracy': oracle_vqa_acc,
            'entropy': entropy,
            **loss_details
        }

        return loss, loss_components

    def _compute_dro_router_loss(
        self,
        per_sample_losses: torch.Tensor,  # [batch_size] - negative expected accuracy
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply DRO reweighting to per-sample losses.

        DRO formula:
          weights = softmax((per_sample_loss - max) / lambda)
          dro_loss = sum(weights * per_sample_loss)

        Args:
            per_sample_losses: [batch_size] - individual sample losses
            scale: If True, don't divide by batch size (not typically used for router)

        Returns:
            dro_loss: Scalar DRO loss
            weights: [batch_size] - DRO weights for logging
        """
        # Numerical stability: subtract max before exp
        mx = torch.max(per_sample_losses)

        L = (per_sample_losses - mx) / self.dro_lambda  # [B]
        weights = torch.exp(L) / torch.sum(torch.exp(L))  # [B]

        # Weighted loss
        dro_loss = torch.sum(weights * per_sample_losses)

        return dro_loss, weights

    def _compute_expected_accuracy_loss(self, gate_weights: torch.Tensor, expert_accuracies: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute expected accuracy loss for router training: Maximize expected vqa accuracy

        loss = -E[accuracy] - beta * H(gate_weights)
        Args:
            gate_weights: Softmax weights from router [B, num_experts]
            expert_accuracies: Accuracies from all experts [B, num_experts]
            beta: Entropy regularization weight
        Returns:
            loss: Negative expected accuracy
            loss_computents: Dictionary with individual loss values for logging
        """
        expected_accuracy = (gate_weights * expert_accuracies).sum(dim=-1)  # [B]
        if beta > 0:
            entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1)  # [B]
            entropy_loss = -entropy.mean()
            per_sample_loss = -expected_accuracy - beta * entropy  # [B]

        else:
            entropy_loss = torch.tensor(0.0, device=gate_weights.device)
            per_sample_loss = -expected_accuracy  # [B]

        if self.use_dro:
            total_loss, weights = self._compute_dro_router_loss(per_sample_loss)
        else:
            total_loss = per_sample_loss.mean()
            weights = torch.ones_like(expected_accuracy) / expected_accuracy.size(0)  # uniform weights

        loss_components = {
            'vqa_loss': - torch.sum(weights*expected_accuracy).item() if self.use_dro else -expected_accuracy.mean().item(),
            'task_loss': 0.0,  # No explicit task loss in this formulation
            'load_balance': 0.0,  # Not computed here
            'expected_accuracy': expected_accuracy.mean().item(),
            'entropy_loss': entropy_loss if isinstance(entropy_loss, float) else entropy_loss.item(),
            'beta_weight': beta,
        }

        return total_loss, loss_components

    def _compute_kl_divergence_loss(self, gate_logits: torch.Tensor, gate_weights: torch.Tensor, expert_accuracies: torch.Tensor, beta: float) -> torch.Tensor:

        batch_size, num_experts = expert_accuracies.shape

        # Create target distribution by normalizing oracle accuracies
        oracle_sum = expert_accuracies.sum(dim=-1, keepdim=True)

        uniform_dist = torch.ones_like(expert_accuracies) / num_experts

        p = torch.where(oracle_sum > 1e-6, expert_accuracies / (oracle_sum + 1e-8), uniform_dist)

        # predicted distribution
        q = gate_weights

        # Compute KL divergence: KL(p || q) = sum(p * log(p/q))
        # = sum(p * (log(p) - log(q)))
        log_q = F.log_softmax(gate_logits, dim=-1)  # # log_softmax for numerical stability (more stable than log(softmax))
        log_p = torch.log(p + 1e-8)  # Add epsilon to avoid log(0)

        # KL divergence per sample
        kl_div = (p * (log_p - log_q)).sum(dim=-1)  # [B]

        # Mean over batch

        # L2 regularization on gate logits to prevent overconfidence
        if beta > 0:
            l2_reg = (gate_logits **2).sum(dim=-1)
            l2_value = l2_reg.mean().item()
            per_sample_loss = kl_div + beta * l2_reg  # [B]
        else:
            per_sample_loss = kl_div  # [B]
            l2_value = 0.0

        if self.use_dro:
            total_loss, weights = self._compute_dro_router_loss(per_sample_loss)
        else:
            total_loss = per_sample_loss.mean()
            weights = torch.ones_like(kl_div) / kl_div.size(0)  # uniform weights

        with torch.no_grad():
            uniform_fallback_rate = (oracle_sum == 0).float().mean().item()

            # average KL divergenece
            avg_kl = kl_div.mean().item()

            p_entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean().item()
            q_entropy = -(q * torch.log(q + 1e-8)).sum(dim=-1).mean().item()

        loss_components = {
            'vqa_loss': 0.0,
            'task_loss': 0.0,
            'load_balance': 0.0,
            'kl_divergence': avg_kl,
            'l2_regularization': l2_value,
            'beta_weight': beta,
            'uniform_fallback_rate': uniform_fallback_rate,
            'target_entropy': p_entropy,
        }

        return total_loss, loss_components

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
            with autocast(enabled=self.use_amp):  # Disable autocast for stability during inference for consistency
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
                    gate_logits = self.task_id_classifier(router_inputs)
                    num_trained_tasks = len(self.model_wrapper.task_head_manager.heads)
                    gate_logits = gate_logits[:, :num_trained_tasks]  # Slice to trained tasks!
                    selected_expert = torch.argmax(gate_logits, dim=1)
                    task_probabilities = F.softmax(gate_logits, dim=1)
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
            'prediction_indices': prediction_indices,  # Always include for compatibility,
            'task_id_confidences': task_confidences if task_id is None else [1.0]*len(predictions),

        }

    def _predict_by_expert(self, batch: Dict[str, Any], selected_expert: int, return_answer_strings: bool = True) -> Dict[str, Any]:

        self.model_wrapper.set_current_task(selected_expert)
        outputs = self.model_wrapper(batch, task_id=selected_expert, return_features=True)
        outputs.probabilities = F.softmax(outputs.logits, dim=-1)
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

    def _get_expert_targets_fast_internal(
        self,
        unified_targets: torch.Tensor,  # Already extracted
        expert_id: int
    ) -> torch.Tensor:
        """Internal fast version that takes pre-extracted unified targets."""
        conversion_matrix = self.unified_to_expert_matrices[expert_id]
        return torch.matmul(unified_targets, conversion_matrix)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Override to load training history."""
        # Load base state
        self.logger.info('Loading state dictionary ...')
        super().load_state_dict(state_dict)

        self.expert_training_history = state_dict.get('expert_training_history', {})
        self.mlp_training_history = state_dict.get('mlp_training_history', {})

        training_config = state_dict.get('training_config', {})
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
    def _generate_oracle_accuracies_v2(
        self,
        train_loader: DataLoader,
        memory_loader: Optional[DataLoader],
        task_id: int
    ) -> Dict[str, torch.Tensor]:  # Change: str keys (question IDs)
        """
        Generate oracle expert accuracies indexed by question ID.

        Returns:
            oracle_accuracies: Dict mapping question_id -> expert accuracies [num_experts]
        """

        # Collect all question IDs first
        all_qids = []

        self.logger.info(f"Collecting question IDs from training and memory data...")

        def extract_qids_from_dataset(dataset):
            qids = []
            if hasattr(dataset, 'data'):
                for item in dataset.data:
                    if isinstance(item, dict) and 'question_id' in item:
                        qids.append(item['question_id'])
                    else:
                        self.logger.warning(f"Item in dataset.data is not a dict or lacks 'question_id': {item}")
                        break
                if len(qids) == len(dataset.data):
                    return qids
            self.logger.info("Using fallback method for QID extraction")
            for idx in range(len(dataset)):
                item = dataset[idx]
                if isinstance(item, dict) and 'question_id' in item:
                    qids.append(item['question_id'])
                else:
                    raise ValueError(f"Dataset item at index {idx} is not a dict or lacks 'question_id': {item}")
            return qids

        try:
            all_qids = extract_qids_from_dataset(train_loader.dataset)
        except Exception as e:
            self.logger.warning(f"Failed to extract QIDs directly from dataset: ({e}), falling back to DataLoader iteration")
            for batch in tqdm(train_loader, desc="Collecting QIDs from Train", leave=False, **tqdm_config):
                all_qids.extend(batch.get('question_id', []))

        if memory_loader is not None and len(memory_loader.dataset) > 0:
            for batch in tqdm(memory_loader, desc="Collecting QIDs from Memory", leave=False, **tqdm_config):
                all_qids.extend(batch.get('question_id', []))
        self.logger.info(f"Collected total of {len(all_qids)} question IDs")

        qids_to_idx = {qid:idx for idx, qid in enumerate(all_qids)}
        num_experts = task_id + 1
        num_samples = len(all_qids)

        self.logger.info(f"Generating oracle for {num_samples} samples, {num_experts} experts")

        oracle_tensor = torch.zeros(num_samples, num_experts, device=self.device, dtype=torch.float32)

        sample_idx = 0

        # Setup memory iterator
        use_memory = memory_loader is not None and len(memory_loader.dataset) > 0

        self.model_wrapper.eval()
        self.text_embedder.eval()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Generating oracle accuracies for Task {task_id}",
            leave=False, **tqdm_config
        )

        with torch.no_grad():
            self.logger.info(f"Processing Training data for oracle accuracies...")
            for batch_idx, batch in pbar:
                batch = self._move_batch_to_device(batch)

                local_targets = self._extract_targets(batch)
                unified_targets = self._convert_task_targets_to_unified(local_targets, task_id)

                # GET QUESTION IDs (must be in your batch!)
                question_ids = batch.get('question_id')
                assert question_ids is not None, "Batch must contain 'question_id' for oracle generation"

                # Generate accuracies
                expert_accuracies = self._get_all_expert_accuracies(
                    batch, unified_targets, num_experts
                )  # [B, num_experts]

                # Store per sample, not per batch!
                batch_size = len(question_ids)
                oracle_tensor[torch.tensor([qids_to_idx[qid] for qid in question_ids], device=self.device)] = expert_accuracies

                pbar.set_postfix({
                    'Samples': sample_idx,
                })

                del batch, expert_accuracies

            if use_memory:
                self.logger.info(f"Processing Memory data for oracle accuracies...")
                pbar = tqdm(
                    enumerate(memory_loader),
                    total=len(memory_loader),
                    desc=f"Generating oracle accuracies for Memory data",
                    leave=False,
                    **tqdm_config
                )
                for batch_idx, mem_batch in pbar:
                    mem_batch = self._move_batch_to_device(mem_batch)

                    unified_targets = self._convert_mem_targets_to_unified(mem_batch)

                    question_ids = mem_batch.get('question_id')
                    assert question_ids is not None, "Memory batch must contain 'question_id' for oracle generation"

                    expert_accuracies = self._get_all_expert_accuracies(
                        mem_batch, unified_targets, num_experts
                    )  # [B, num_experts]

                    batch_size = len(question_ids)
                    oracle_tensor[torch.tensor([qids_to_idx[qid] for qid in question_ids], device=self.device)] = expert_accuracies

                    pbar.set_postfix({
                        'Samples': sample_idx,
                        'Avg_Best_Acc': f'{expert_accuracies.max(dim=-1)[0].mean().item():.3f}'
                    })

                    del mem_batch, expert_accuracies

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(f"Generated oracle for {sample_idx} samples")

        return oracle_tensor, qids_to_idx

    def _convert_mem_targets_to_unified(self, mem_batch: Dict[str, Any]) -> torch.Tensor:
        """
        Convert memory batch targets to unified vocabulary space.

        Args:
            mem_batch: Memory batch containing 'task_ids' and 'targets'

        Returns:
            unified_targets: [B, unified_vocab_size]
        """
        mem_task_ids = mem_batch.get('task_ids')  # [B]
        mem_local_targets = mem_batch.get('targets')

        # Handle task_ids shape
        if mem_task_ids.dim() > 1:
            mem_task_ids = mem_task_ids.squeeze()

        batch_size = mem_task_ids.size(0)
        unified_vocab_size = len(self._unified_label2ans)

        if isinstance(mem_local_targets, list):
            dtype = mem_local_targets[0].dtype if isinstance(mem_local_targets[0], torch.Tensor) else torch.float32
        elif isinstance(mem_local_targets, torch.Tensor):
            dtype = mem_local_targets.dtype
        else:
            dtype = torch.float32

        unified_targets = torch.zeros(batch_size, unified_vocab_size, device=self.device, dtype=dtype)

        for i in range(batch_size):
            task_id = mem_task_ids[i].item()
            local_target = mem_local_targets[i]  # [local_vocab_size]

            # Get mapping for this task
            expert_to_unified = self.expert_to_unified.get(task_id, {})

            for local_idx in range(local_target.size(0)):
                if local_target[local_idx] > 0:
                    unified_idx = expert_to_unified.get(local_idx, None)
                    if unified_idx is not None:
                        unified_targets[i, unified_idx] = local_target[local_idx]

        assert unified_targets.shape[1] == len(self._unified_label2ans), "Unified targets shape mismatch"

        return unified_targets

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

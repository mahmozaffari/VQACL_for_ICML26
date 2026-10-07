"""
A unified trainer for VQA continual learning.
"""

import os
import json
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

from core.base_trainer import BaseTrainer
from utils.data_utils import get_image_directory, initialize_dataset
from evaluation.vqa_evaluator import VQAEvaluator
from models.vilt_wrapper import ViLTWrapper
from core.training_callbacks import (
    CallbackList,
    CheckpointCallback,
)
from core.base_strategy import TaskInfo

class UnifiedTrainer(BaseTrainer):
    """
    Fixed unified trainer for VQA continual learning that supports multiple strategies.
    """

    def __init__(self, args, task_list: List[str], train: bool = True):
        """
        Initialize the unified trainer for VQA continual learning.

        Args:
            args: Configuration arguments
            task_list: List of task names/identifiers
            train: Whether this is for training or testing only
        """
        # Initialize base trainer
        super().__init__(args, task_list, train)

        # VQA-specific setup
        self.image_dir = get_image_directory(args.dataset)
        self.processor = None  # Will be initialized with model

        # Data loaders (organized by task)
        self.train_loader = None
        self.val_loader = None
        self.test_loaders: Dict[str, Any] = {}

        # Initialize datasets
        self.train_dset, self.val_dset, self.test_dset = initialize_dataset(args)

        # Cache dataset label mappings for consistency checks.
        self.dataset_a2l = {}
        self.dataset_l2a = {}
        # Logger setup

        # GPU and distributed training setup
        self._setup_gpu()

        self.logger.info(f'Unified Trainer initialized for {len(task_list)} tasks')
        self.logger.info(f'Strategy: {args.strategy}')
        self.logger.info(f'Model: {args.model_name}')

    def _setup_callbacks(self):
        """Set up training callbacks."""
        self.callbacks = CallbackList()

        # Add checkpoint callback
        checkpoint_callback = CheckpointCallback(
            checkpoint_manager=self.checkpoint_manager,
            model_state_fn=self._get_model_state,
            strategy_state_fn=self._get_strategy_state,
            metric_key=getattr(self.args, 'checkpoint_metric', 'val_accuracy'),
            additional_state_fn=self._get_additional_state,
            logger=self.logger
        )
        self.callbacks.add_callback(checkpoint_callback)

    def _setup_gpu(self):
        """Setup GPU and distributed training configuration."""
        if hasattr(self.args, 'gpu'):
            self.device = torch.device(f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.logger.info(f'Using device: {self.device}')

        # Setup distributed training if specified
        if getattr(self.args, 'distributed', False):
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.logger.info('Distributed training initialized')

    def _initialize_strategy(self):
        """Initialize the continual learning strategy based on configuration."""
        strategy_name = getattr(self.args, 'strategy', 'moe')
        if strategy_name == 'moe':
            if getattr(self.args, 'init_unified_answer_space', False):
                all_task_configs = self._get_all_task_configs()
                self._unified_ans2label, self._unified_label2ans = self._build_unified_answer_space(all_task_configs)
            else:
                all_task_configs = None

            from strategies.moe import MoEStrategy
            self.strategy = MoEStrategy(
                model_wrapper=self.model_wrapper,
                args=self.args,
                task_list=self.task_list,
                unified_label2ans=getattr(self, '_unified_label2ans', None),
                task_configs=all_task_configs
            )
        elif strategy_name == 'moe_utility_router':
            all_task_configs = self._get_all_task_configs()
            _unified_ans2label, _unified_label2ans = self._build_unified_answer_space(all_task_configs)
            from strategies.moe_utility_router import MoEUtilityRouterStrategy
            self.strategy = MoEUtilityRouterStrategy(
                model_wrapper=self.model_wrapper,
                args=self.args,
                task_list=self.task_list,
                unified_label2ans=_unified_label2ans,
            )
        elif strategy_name == 'moe_router':
            all_task_configs = self._get_all_task_configs()
            _unified_ans2label, _unified_label2ans = self._build_unified_answer_space(all_task_configs)
            from strategies.moe_router import MoERouterStrategy
            self.strategy = MoERouterStrategy(
                model_wrapper=self.model_wrapper,
                args=self.args,
                task_list=self.task_list,
                unified_label2ans=_unified_label2ans,
            )
        elif strategy_name == 'naive':
            from strategies.naive_strategy import NaiveStrategy
            all_task_configs = self._get_all_task_configs()
            self._unified_ans2label, self._unified_label2ans = self._build_unified_answer_space(all_task_configs)

            self.strategy = NaiveStrategy(
                model_wrapper=self.model_wrapper,
                args=self.args,
                unified_label2ans=getattr(self, '_unified_label2ans', None)
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        self.logger.info(f'Initialized strategy: {self.strategy}')

    def _initialize_model(self):
        """Initialize the model wrapper based on configuration."""
        model_name = getattr(self.args, 'model_name', 'vilt')

        if model_name == 'vilt':
            # Initialize ViLT model and processor
            from transformers import ViltProcessor

            # Load processor
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", size=384, do_resize=True)

            from transformers import ViltModel
            self.logger.info("Using ViltModel for continual learning with task-specific heads")
            base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

            self.model_wrapper = ViLTWrapper(
                base_model=base_model,
                processor=self.processor,
                args=self.args
            )

        elif model_name == 'flava':

            from transformers import FlavaModel, FlavaProcessor
            self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
            base_model = FlavaModel.from_pretrained("facebook/flava-full")

            from models.flava_wrapper import FlavaWrapper
            self.model_wrapper = FlavaWrapper(
                base_model=base_model,
                processor=self.processor,
                args=self.args
            )

        # Move to device
        self.model_wrapper = self.model_wrapper.to(self.device)

        # Setup distributed training
        if getattr(self.args, 'distributed', False):
            self.model_wrapper = DDP(
                self.model_wrapper,
                device_ids=[self.args.gpu] if hasattr(self.args, 'gpu') else None,
                find_unused_parameters=True
            )

        self.logger.info(f'Initialized model: {self.model_wrapper}')
        model_info = self.model_wrapper.get_model_size_info()
        self.logger.info(f'Model parameters: {model_info}')

    def _initialize_evaluator(self, task_configs: Optional[List[Dict[str, Any]]] = None):
        """Initialize the evaluator based on configuration."""
        # Build test loaders for all tasks

        self._task_configs = task_configs

        # Create evaluator
        self.evaluator = VQAEvaluator(
            args=self.args,
            test_loaders=self.test_loaders,
            task_list=self.task_list
        )

        self.logger.info(f'Initialized evaluator (test loaders will be built on-demand)')

    def _ensure_test_loaders_available(self):
        """
        Ensure test loaders are available for evaluation.
        Builds them if they don't exist or were cleared.
        """
        if self.test_loaders:
            # Already have test loaders
            self.logger.debug(f"Test loaders already available ({len(self.test_loaders)} tasks)")
            return

        self.logger.info("Building test loaders for evaluation...")

        # Build test loaders
        if hasattr(self, '_task_configs') and self._task_configs is not None:
            self._build_test_loaders_with_configs(self._task_configs)
        else:
            self._build_test_loaders()

        self.logger.info(f"Test loaders built: {len(self.test_loaders)} tasks")

    def _build_test_loaders(self):
        """Build test data loaders for all tasks."""
        self.logger.info('Using original dataset for test loaders.')
        from vqa_dataset import get_loader_qlevel

        self.logger.info('Building test loaders for all tasks...')

        ans2label = self._unified_ans2label if hasattr(self, '_unified_ans2label') else None
        if ans2label:
            self.logger.info(f'Using unified answer space with {len(ans2label)} answers.')

        for task_idx, task_name in enumerate(self.task_list):
            test_loader, _, _, _, _ = get_loader_qlevel(
                processor=self.processor,
                image_dir=self.image_dir,
                args=self.args,
                coco_Ours=self.task_list,
                Examplar_set = [],
                _dset=self.test_dset,
                split=getattr(self.args, 'test_split', 'test'),
                mode='val',  # Use val mode for testing
                batch_size=getattr(self.args, 'test_batch_size', self.args.batch_size),
                distributed=getattr(self.args, 'distributed', False),
                gpu=getattr(self.args, 'gpu', 0),
                workers=min(getattr(self.args, 'num_workers', 4), 2),  # test loaders for every task stay alive; keep them light
                topk=getattr(self.args, 'test_topk', -1),
                ans2label=ans2label,
                task=task_name,
                task_id=task_idx,
                partition_name=getattr(self.args, 'partition_name', None),
                persistent_workers=False,
                prefetch_factor=2, #getattr(self.args, 'prefetch_factor', 2),
                pin_memory=getattr(self.args, 'pin_memory', True)
            )

            self.test_loaders[task_name] = test_loader
            self.logger.info(f'Built test loader for task {task_name}: {len(test_loader)} batches')

    def _build_test_loaders_with_configs(self, task_configs: List[Dict[str, Any]]):
        """Build test data loaders for all tasks."""
        self.logger.info('Using original dataset for test loaders.')
        from vqa_dataset import get_loader_qlevel

        self.logger.info('Building test loaders for all tasks...')

        for task_idx, task_config in task_configs.items():
            task_info = task_config.get('task_info', {})
            assert task_idx == task_info['task_id'], "Task index mismatch in task configs"
            task_idx = task_info['task_id']
            task_name = task_info['task_name']
            ans2label = task_info.get('ans2label', None)

            if ans2label is None and 'label2ans' in task_info:
                label2ans = task_info['label2ans']
                self.logger.debug(f'label2ans for task {task_name}: {label2ans[:5]}... (total {len(label2ans)})')
                assert isinstance(label2ans, list), "label2ans should be a list"
                ans2label = {ans: idx for idx, ans in enumerate(label2ans)}
                self.logger.info(f'Constructed ans2label mapping for task {task_name} from label2ans')
            elif ans2label is None:
                raise ValueError(f"ans2label mapping not found in task config for task {task_name}")

            test_loader, _, _, _, _ = get_loader_qlevel(
                processor=self.processor,
                image_dir=self.image_dir,
                args=self.args,
                coco_Ours=self.task_list,
                Examplar_set = [],
                _dset=self.test_dset,
                split=getattr(self.args, 'test_split', 'test'),
                mode='val',  # Use val mode for testing
                batch_size=getattr(self.args, 'test_batch_size', self.args.batch_size),
                distributed=getattr(self.args, 'distributed', False),
                gpu=getattr(self.args, 'gpu', 0),
                workers=min(getattr(self.args, 'num_workers', 4), 2),  # test loaders for every task stay alive; keep them light
                topk=getattr(self.args, 'test_topk', -1),
                task=task_name,
                task_id=task_idx,
                ans2label=ans2label,
                partition_name=getattr(self.args, 'partition_name', None),
                persistent_workers=False,
                prefetch_factor=getattr(self.args, 'prefetch_factor', 2),
                pin_memory=getattr(self.args, 'pin_memory', True)
            )

            self.test_loaders[task_name] = test_loader
            self.logger.info(f'Built test loader for task {task_name}: {len(test_loader)} batches')

    def _train_single_task(self, task_idx: int, task_name: str) -> Dict[str, Any]:
        """
        Train on a single task using the strategy.

        Args:
            task_idx: Current task index
            task_name: Current task name

        Returns:
            Task training results
        """
        self.logger.info(f"Training on task {task_idx}: {task_name}")

        self.callbacks.on_task_start(task_idx, task_name)

        # Build data loaders for this task
        train_loader, val_loader, task_info = self._build_task_loaders(task_idx, task_name)

        if hasattr(self, '_unified_ans2label') and self._unified_ans2label is not None:
            self.dataset_l2a[task_idx] = train_loader.dataset.label2ans
            self.dataset_a2l[task_idx] = train_loader.dataset.ans2label

            if task_idx > 0:
                # Check answer-label consistency with previous tasks.
                for prev_task_idx in range(task_idx):
                    prev_l2a = self.dataset_l2a[prev_task_idx]
                    prev_a2l = self.dataset_a2l[prev_task_idx]
                    for ans, label in prev_a2l.items():
                        assert ans in train_loader.dataset.ans2label, f"Answer '{ans}' from task {prev_task_idx} not found in task {task_idx}"
                        curr_label = train_loader.dataset.ans2label[ans]
                        assert curr_label == label, f"Answer '{ans}' has inconsistent labels between tasks {prev_task_idx} and {task_idx}: {curr_label} vs {label}"

                    for ans1, ans2 in zip(prev_l2a, train_loader.dataset.label2ans):
                        assert ans1 == ans2, f"Label2Ans mismatch between tasks {prev_task_idx} and {task_idx}: '{ans1}' vs '{ans2}'"
        # Store loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Prepare strategy for this task
        self.strategy.prepare_for_task(task_info)

        # Train using the strategy
        training_results = self.strategy.train_task(
            task_info=task_info,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=self.callbacks
        )

        # Save training results for this task
        results_file = os.path.join(self.args.output, f'{task_name}_training_results.json')
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=4)

        # Consolidate knowledge after training
        self.strategy.consolidate_knowledge(task_info)

        # Load best checkpoint to ensure consistency between training and evaluation
        if hasattr(self.strategy, 'get_checkpoint_paths_for_task'):
            self.logger.info(f"Multi-component strategy - best weights already restored during training for task {task_name}")
        else:
            # Single-component strategy - load best checkpoint from disk
            try:
                self.logger.info(f"Loading best checkpoint for task {task_name} to ensure consistency...")
                self.load_checkpoint(task_idx, task_name, checkpoint_type='best')
            except Exception as e:
                self.logger.warning(f"Best checkpoint not found for task {task_name}, using current model state")

        # Execute on_task_end callbacks
        self.callbacks.on_task_end(task_idx, task_name)

        self.logger.info(f"Completed training on task {task_name}")
        self.logger.info(f"Training results: {training_results}")

        self._release_loaders(train_loader, val_loader)

        self.train_loader = None
        self.val_loader = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure cleanup completes

        self.logger.debug(f"Cleared loader references for task {task_name}")

        return training_results

    def _release_loaders(self, *loaders) -> None:
        """
        Shut down DataLoader worker processes and drop strong references so that
        Python's GC can reclaim the Dataset (and its open HDF5 files).

        Call this as soon as you no longer need the loader.
        """
        import gc, torch
        from contextlib import suppress

        for loader in loaders:
            if loader is None:
                continue

            # 1. ask the DataLoader to shut down its workers (PyTorch >= 1.8)
            with suppress(Exception):
                loader._shutdown_workers()

            # 2. explicitly close any open HDF5 files that the Dataset kept
            ds = getattr(loader, 'dataset', None)
            if ds is not None and hasattr(ds, 'source_to_h5'):
                for f in ds.source_to_h5.values():
                    with suppress(Exception):
                        if f and hasattr(f, 'close'):
                            f.close()

            # 3. drop references
            del loader

        # 4. run garbage-collection and clear CUDA cache (get_loader_qlevel froze the heap
        #    before forking workers; unfreeze so the released datasets can be reclaimed)
        gc.unfreeze()
        gc.collect()
        torch.cuda.empty_cache()

    def _build_task_loaders(self, task_idx: int, task_name: str) -> Tuple[Any, Any, TaskInfo]:
        """
        Build train and validation loaders for a specific task.

        Args:
            task_idx: Task index
            task_name: Task name

        Returns:
            Tuple of (train_loader, val_loader, task_info)
        """
        self.logger.info('Using original dataset for test loaders.')
        from vqa_dataset import get_loader_qlevel

        ans2label = self._unified_ans2label if hasattr(self, '_unified_ans2label') else None
        if ans2label:
            self.logger.info(f'Using unified answer space with {len(ans2label)} answers for task {task_name}')

        # Build training loader
        train_loader, total_num_Q, task_num_answers, task_label2ans, _ = get_loader_qlevel(
            processor=self.processor,
            image_dir=self.image_dir,
            args=self.args,
            coco_Ours=self.task_list,
            Examplar_set = [],
            _dset=self.train_dset,
            split=getattr(self.args, 'train_split', 'train'),
            mode='train',
            batch_size=self.args.batch_size,
            distributed=getattr(self.args, 'distributed', False),
            gpu=getattr(self.args, 'gpu', 0),
            workers=getattr(self.args, 'num_workers', 8),
            topk=getattr(self.args, 'train_topk', -1),
            ans2label=ans2label,
            task=task_name,
            task_id=task_idx,
            partition_name=getattr(self.args, 'partition_name', None),
            persistent_workers=getattr(self.args, 'persistent_workers', False),
            prefetch_factor=getattr(self.args, 'prefetch_factor', 2),
            pin_memory=getattr(self.args, 'pin_memory', True)
        )

        # Build validation loader
        val_loader, _, _, val_task_lbl_2_ans, _ = get_loader_qlevel(
            processor=self.processor,
            image_dir=self.image_dir,
            args=self.args,
            coco_Ours=self.task_list,
            Examplar_set = [],
            _dset=self.val_dset,
            split=getattr(self.args, 'val_split', 'val'),
            mode='val',
            batch_size=getattr(self.args, 'val_batch_size', self.args.batch_size),
            distributed=getattr(self.args, 'distributed', False),
            gpu=getattr(self.args, 'gpu', 0),
            workers=min(getattr(self.args, 'num_workers', 4), 2),  # fewer, non-persistent workers for validation
            topk=getattr(self.args, 'val_topk', -1),
            ans2label=ans2label,
            task=task_name,
            task_id=task_idx,
            partition_name=getattr(self.args, 'partition_name', None),
            persistent_workers=False,
            prefetch_factor=getattr(self.args, 'prefetch_factor', 2),
            pin_memory=getattr(self.args, 'pin_memory', True)
        )

        for a,b in zip(task_label2ans, val_task_lbl_2_ans):
            assert a==b, "Label2Ans mismatch between train and val loaders"

        # Create task info
        task_info = TaskInfo(
            task_id=task_idx,
            task_name=task_name,
            num_classes=task_num_answers,
            label2ans=task_label2ans,
            dataset_size=total_num_Q
        )

        return train_loader, val_loader, task_info

    def get_current_model(self):
        """Get the current model for external access."""
        if isinstance(self.model_wrapper, DDP):
            return self.model_wrapper.module
        return self.model_wrapper

    def get_current_strategy(self):
        """Get the current strategy for external access."""
        return self.strategy

    def save_checkpoint(self, task_idx: int, checkpoint_type: str = 'latest'):
        """
        Save model checkpoint after completing a task.

        Args:
            task_idx: Index of completed task
            checkpoint_type: Type of checkpoint ('latest', 'best', etc.)
        """
        task_name = self.task_list[task_idx]
        checkpoint_name = f"{task_name}_{checkpoint_type}.pth"
        checkpoint_path = os.path.join(self.args.output, checkpoint_name)

        # Get model state
        if isinstance(self.model_wrapper, torch.nn.parallel.DistributedDataParallel):
            model_state = self.model_wrapper.module.state_dict()
        else:
            model_state = self.model_wrapper.state_dict()

        checkpoint = {
            'task_idx': task_idx,
            'task_name': task_name,
            'model_state': model_state,
            'strategy_state': self.strategy.get_state_dict(),
            'result_matrices': self.result_matrices,
            'args': vars(self.args),
            'task_list': self.task_list,
            'checkpoint_type': checkpoint_type,
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Saved checkpoint: {checkpoint_path}')

        # Verify checkpoint was saved correctly
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f"Failed to save checkpoint to {checkpoint_path}")

    def load_checkpoint(self, task_idx: int, task_name: str, checkpoint_type: str = 'best', load_checkpoint_dir: Optional[str] = None, checkpoint_path: Optional[str] = None):
        """
        Load model checkpoint for a specific task.

        Args:
            task_idx: Task index to load
            task_name: task name to load
            checkpoint_type: Type of checkpoint to load
        """
        if hasattr(self.strategy, 'get_checkpoint_paths_for_task'):
            return

        checkpoint = self.checkpoint_manager.load_checkpoint(
            task_idx=task_idx,
            task_name=task_name,
            checkpoint_type=checkpoint_type,
            checkpoint_dir = load_checkpoint_dir,
            checkpoint_path=checkpoint_path
        )

        # Load model state
        if isinstance(self.model_wrapper, torch.nn.parallel.DistributedDataParallel):
            self.model_wrapper.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model_wrapper.load_state_dict(checkpoint['model_state'])

        self.strategy.load_state_dict(checkpoint['strategy_state'])

        if 'optimizer_state' in checkpoint and hasattr(self.strategy, 'optimizer') and self.strategy.optimizer is not None:
            self.strategy.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'scheduler_state' in checkpoint and hasattr(self.strategy, 'scheduler') and self.strategy.scheduler is not None:
            self.strategy.scheduler.load_state_dict(checkpoint['scheduler_state'])

        # Load result matrices if available
        if 'result_matrices' in checkpoint:
            self.result_matrices = checkpoint['result_matrices']

        self.logger.info(f"Loaded {checkpoint_type} checkpoint for task {task_name}")

        # Log checkpoint info
        metadata = checkpoint.get('metadata', {})
        if metadata:
            epoch = metadata.get('epoch', 'unknown')
            metric_value = metadata.get('metric_value', 'unknown')
            self.logger.info(
                f"  Checkpoint from epoch {epoch}, "
                f"metric={metric_value}"
            )

    def _update_results(self, task_idx: int, eval_results: Dict[str, Any]):
        """
        Update result matrices with new evaluation results.

        Args:
            task_idx: Index of completed task
            eval_results: Evaluation results from evaluator
        """
        current_task = self.task_list[task_idx]
        self.logger.info(f"="*60)
        self.logger.info(f"Evaluation Results after Task {task_idx}: {current_task}")
        self.logger.info(f"-"*60)

        # Log accuracy on each learned task
        if 'standard' in eval_results:
            for test_task, result in eval_results['standard'].items():
                if 'additional_metrics' in result.__dict__ and 'task_id_accuracy' in result.additional_metrics and result.additional_metrics['task_id_accuracy'] is not None:
                    task_id_acc = result.additional_metrics['task_id_accuracy']
                    self.logger.info(f"  {test_task}: Acc={result.accuracy:.4f}, ECE={result.ece:.4f}, TaskID_Acc={task_id_acc:.4f}")
                else:
                    self.logger.info(f"  {test_task}: Acc={result.accuracy:.4f}, ECE={result.ece:.4f}")

        if 'oracle' in eval_results:
            for test_task, result in eval_results['oracle'].items():
                self.logger.info(f"  [Oracle] {test_task}: Acc={result.accuracy:.4f}, ECE={result.ece:.4f}")

        if 'bayesian' in eval_results:
            self.logger.info(f"  [Bayes]: {eval_results['bayesian'].get('bayesian_accuracy', 0):.4f}")

        # Log average performance
        if 'joint' in eval_results:
            self.logger.info(f"  Joint Accuracy: {eval_results['joint'].get('joint_accuracy', 0):.4f}")

        # Log forgetting if not first task
        if task_idx > 0:
            # Calculate and log forgetting
            self.logger.info(f"  Forgetting: [calculate here]")

        # Update accuracy matrices for different scenarios
        for scenario, scenario_results in eval_results.items():
            if scenario == 'expert_matrix':
                continue
            if scenario == 'joint':
                # Handle joint evaluation results
                if f'{scenario}_accuracy' not in self.result_matrices:
                    self.result_matrices[f'{scenario}_accuracy'] = {}
                self.result_matrices[f'{scenario}_accuracy'][current_task] = scenario_results.get('joint_accuracy', 0.0)

                if f'{scenario}_ece' not in self.result_matrices:
                    self.result_matrices[f'{scenario}_ece'] = {}
                self.result_matrices[f'{scenario}_ece'][current_task] = scenario_results.get('joint_ece', 1.0)

            elif isinstance(scenario_results, dict):
                # Handle per-task evaluation results
                for test_task, eval_result in scenario_results.items():
                    # Update accuracy matrix
                    if f'{scenario}_accuracy' not in self.result_matrices:
                        self.result_matrices[f'{scenario}_accuracy'] = {task: {} for task in self.task_list}
                    self.result_matrices[f'{scenario}_accuracy'][current_task][test_task] = eval_result.accuracy

                    # Update ECE matrix
                    if f'{scenario}_ece' not in self.result_matrices:
                        self.result_matrices[f'{scenario}_ece'] = {task: {} for task in self.task_list}
                    self.result_matrices[f'{scenario}_ece'][current_task][test_task] = eval_result.ece

    def _save_checkpoint(self, task_idx: int):

        pass

    def get_checkpoint_info(self, task_name: str) -> Dict[str, Any]:
        """Get information about checkpoints for a specific task."""
        return {
            'best': self.checkpoint_manager.get_checkpoint_info(task_name, 'best'),
            'latest': self.checkpoint_manager.get_checkpoint_info(task_name, 'latest'),
            'checkpoints': [
                str(p.name)
                for p in self.checkpoint_manager.list_checkpoints(task_name)
            ]
        }

    def print_results_summary(self):
        """Print a summary of all results."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING COMPLETE - RESULTS SUMMARY")
        self.logger.info("=" * 80)

        # Print final metrics if available
        try:
            final_metrics = self._compute_final_metrics()
            for metric_name, value in final_metrics.items():
                self.logger.info(f"{metric_name}: {value:.4f}")
        except Exception as e:
            self.logger.warning(f"Could not compute final metrics: {e}")

        # Print accuracy matrix
        if 'standard_accuracy' in self.result_matrices:
            self.logger.info("\nAccuracy Matrix:")
            acc_matrix = self.result_matrices['standard_accuracy']

            # Print header
            header = "Train\\Test".ljust(12)
            for test_task in self.task_list:
                header += f"{test_task[:8]:>8}"
            self.logger.info(header)

            # Print rows
            for train_task in self.task_list:
                if train_task in acc_matrix:
                    row = train_task[:10].ljust(12)
                    for test_task in self.task_list:
                        if test_task in acc_matrix[train_task]:
                            acc = acc_matrix[train_task][test_task]
                            row += f"{acc:8.3f}"
                        else:
                            row += f"{'--':>8}"
                    self.logger.info(row)

        self.logger.info("=" * 80)

    def __repr__(self) -> str:
        strategy_name = getattr(self.args, 'strategy', 'unknown')
        return f"UnifiedTrainer(strategy={strategy_name}, tasks={len(self.task_list)})"

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state dict for checkpointing."""
        if isinstance(self.model_wrapper, torch.nn.parallel.DistributedDataParallel):
            return self.model_wrapper.module.state_dict()
        return self.model_wrapper.state_dict()

    def _get_strategy_state(self) -> Dict[str, Any]:
        """Get strategy state dict for checkpointing."""
        return self.strategy.get_state_dict()

    def _get_additional_state(self) -> Dict[str, Any]:
        """Get additional state for checkpointing."""
        additional_state = {
            "result_matrices": self.result_matrices,
            "args": vars(self.args),
            "task_list": self.task_list
        }

        # Add optimizer and scheduler states if they exist
        if hasattr(self.strategy, 'optimizer') and self.strategy.optimizer:
            additional_state['optimizer_state'] = self.strategy.optimizer.state_dict()

        if hasattr(self.strategy, 'scheduler') and self.strategy.scheduler:
            additional_state['scheduler_state'] = self.strategy.scheduler.state_dict()

        return additional_state

    def _clear_test_loaders(self):
        """Clear all test loaders to free memory."""
        if not self.test_loaders:
            return

        self.logger.info(f"Clearing {len(self.test_loaders)} test loaders...")

        for task_name, loader in self.test_loaders.items():
            self._release_loaders(loader)

        self.test_loaders.clear()

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Test loaders cleared")

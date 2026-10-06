"""
Abstract base class for continual learning trainers.

This class defines the main interface that all continual learning trainers must implement.
It coordinates the overall training/testing process and delegates CL-specific logic to strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import os
import json
import logging

from core.checkpoint import CheckpointManager, CheckpointMetric
from core.training_callbacks import (
    CallbackList,
    CheckpointCallback,
    MetricsLoggingCallback,
    EarlyStoppingCallback
)

import psutil
import os

def log_memory(phase, logger=None):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024**3)
    if logger:
        logger.info(f"MEMORY LOG: {phase}: {mem_gb:.1f} GB")
    else:
        print(f"MEMORY LOG: {phase}: {mem_gb:.1f} GB")

class BaseTrainer(ABC):
    """Abstract base trainer for continual learning approaches."""

    def __init__(self, args, task_list: List[str], train: bool = True):
        """
        Initialize the base trainer.

        Args:
            args: Configuration arguments
            task_list: List of task names/identifiers
            train: Whether this is for training or testing only
        """
        self.args = args
        self.task_list = task_list
        self.is_training = train
        self.current_task_idx = 0
        self.current_task = None

        # Initialize result tracking
        self.result_matrices = self._initialize_result_matrices()

        # Setup logging and output directory
        self._setup_logging()
        self._setup_output_dir()

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir= args.output,
            metric=CheckpointMetric.ACCURACY,
            keep_last_n_checkpoints=getattr(args, 'keep_last_n_checkpoints', 3),
            save_every_n_epochs=getattr(args, 'checkpoint_interval', 1),
            logger=self.logger
        )

        # Initialize callbacks
        self._setup_callbacks()

        # Initialize strategy, model, and evaluator (to be implemented by subclasses)
        self.strategy = None
        self.model_wrapper = None
        self.evaluator = None

    @abstractmethod
    def _setup_callbacks(self):
        """Setup callbacks"""
        pass

    @abstractmethod
    def _initialize_strategy(self):
        """Initialize the continual learning strategy."""
        pass

    @abstractmethod
    def _initialize_model(self):
        """Initialize the model wrapper."""
        pass

    @abstractmethod
    def _initialize_evaluator(self, task_configs: Optional[List[Dict[str, Any]]] = None):
        """Initialize the evaluator."""
        pass

    def _initialize_result_matrices(self) -> Dict[str, Dict]:
        """Initialize result tracking matrices."""
        matrices = {}

        # Different evaluation scenarios
        scenarios = [
            'standard',      # Standard CL evaluation
            'oracle',        # With ground-truth task IDs
            'ood',          # Out-of-distribution task IDs
            'joint',        # Joint evaluation across all tasks
            'bayesian'      # Bayesian aggregation evaluation
        ]

        # Different metrics
        metrics = ['accuracy', 'ece', 'forget', 'transfer']

        for scenario in scenarios:
            for metric in metrics:
                key = f'{scenario}_{metric}'
                matrices[key] = {task: {} for task in self.task_list}

                # Joint metrics don't need per-task breakdown
                if scenario == 'joint':
                    matrices[key] = {task: 0.0 for task in self.task_list}

        return matrices

    def _setup_logging(self):
        """Setup logging configuration."""
        # This will be implemented based on the existing logging setup
        logger_name = f'CL.trainer.{self.__class__.__name__.lower()}'
        self.logger = logging.getLogger(logger_name)

        # Check for distributed mode and rank
        if hasattr(self.args, 'distributed') and self.args.distributed:
            if hasattr(self.args, 'gpu') and self.args.gpu != 0:
                # Non-primary ranks only log warnings and above
                self.logger.setLevel(logging.WARNING)
            else:
                # Primary rank gets normal logging
                self.logger.setLevel(logging.INFO)
        else:
            # Single GPU/CPU mode
            self.logger.setLevel(logging.INFO)

        # Log initialization
        self.logger.debug(f"{self.__class__.__name__} logger initialized")
        pass

    def _setup_output_dir(self):
        """Setup output directory for saving results."""
        os.makedirs(self.args.output, exist_ok=True)

        # Save configuration
        config_path = os.path.join(self.args.output, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=4)

    def _get_all_task_configs(self) -> List[Dict[str, Any]]:
        """Get configurations for all tasks."""
        all_task_configs = {}
        from vqa_dataset import get_loader_qlevel
        for task_idx, task_name in enumerate(self.task_list):
            # Quick load to get label2ans for each task
            _, _, task_num_answers, task_label2ans, _ = get_loader_qlevel(
                # ... params for a minimal load just to get label2ans
                processor=self.processor,
                image_dir=self.image_dir,
                args=self.args,
                coco_Ours=self.task_list,
                Examplar_set = [],
                _dset=self.train_dset,
                split=getattr(self.args, 'train_split', 'train'),
                mode='train',
                batch_size=1,  # Minimal batch size for config collection
                distributed=False,
                gpu=getattr(self.args, 'gpu', 0),
                workers=0,  # No workers needed for config
                topk=1,  # Minimal data loading
                task=task_name,
                task_id=task_idx,
                ans2label=None,
                partition_name=getattr(self.args, 'partition_name', None)
            )

            all_task_configs[task_idx] = {
                'label2ans': task_label2ans,
                'num_answers': task_num_answers,
                'task_name': task_name
            }
        return all_task_configs

    def _build_unified_answer_space(self, all_task_configs):
        """Build unified answer space from all task configurations."""
        all_answers = set()

        # Collect all unique answers from all tasks
        for task_config in all_task_configs.values():
            task_label2ans = task_config.get('label2ans', [])
            if task_label2ans:
                all_answers.update(task_label2ans)

        # Create unified mappings
        unified_label2ans = sorted(list(all_answers))
        unified_ans2label = {ans: idx for idx, ans in enumerate(unified_label2ans)}

        for idx, ans in enumerate(unified_ans2label):
            assert unified_ans2label[ans] == idx, "Mismatch in unified answer space"

        self.logger.info(f"Created unified answer space with {len(unified_label2ans)} unique answers")
        return unified_ans2label, unified_label2ans

    def train(self) -> Dict[str, Any]:
        """
        Main training loop for continual learning.

        Args:
            resume_from_task: Task name to resume training from

        Returns:
            Training results and metrics
        """

        # Handle resuming from checkpoint
        start_idx = self._handle_resume()
        print("start_idx:", start_idx)

        # Initialize components
        self._initialize_model()
        self._initialize_strategy()
        self._initialize_evaluator()

        # If resuming, prepare previous tasks (load checkpoints, skip training)
        if start_idx > 0:
            self._prepare_previous_tasks(start_idx)
            self.logger.info(f"Model state prepared with {start_idx} previous task(s)")
            self.logger.info(f"Resuming training from task {start_idx}: {self.task_list[start_idx]}")

        # Sequential task training
        for task_idx in range(start_idx, len(self.task_list)):
            self.current_task_idx = task_idx
            self.current_task = self.task_list[task_idx]

            # Train on current task
            log_memory(f"Before training on task {task_idx}", self.logger)
            task_results = self._train_single_task(task_idx, self.current_task)
            log_memory(f"After training on task {task_idx}", self.logger)

            # Evaluate after task completion
            if not self.args.skip_progressive_eval or task_idx == len(self.task_list)-1:
                # Ensure test loaders before evaluation
                log_memory("Before eval", self.logger)
                self._ensure_test_loaders_available()
                log_memory("Test loaders built", self.logger)

                eval_results = self._evaluate_after_task(task_idx)

                # Update result matrices
                self.logger.info(f"Updating results for task: {self.current_task}")
                self._update_results(task_idx, eval_results)

                # Clear test loaders after evaluation
                self._clear_test_loaders()
                log_memory("After clearing", self.logger)

        # Final evaluation and cleanup
        final_results = self._finalize_results()

        return final_results

    def test(self, checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        ## Test method needs to be updated to load checkpoints properly! outdated!
        """
        Test-only mode for evaluating trained models.

        Args:
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Test results and metrics
        """
        from core.base_strategy import TaskInfo

        if checkpoint_dir is None:
            checkpoint_dir = self.args.output

        # Initialize components
        self._initialize_model()
        self._initialize_strategy()

        task_configs = self._extract_task_configs_from_checkpoints(checkpoint_dir)

        self._initialize_evaluator(task_configs)

        # Load checkpoints and test each task
        results = {}

        for task_idx, task_name in enumerate(self.task_list):
            self.current_task_idx = task_idx
            self.current_task = task_name

            self.logger.info(f"Testing on task {task_idx}: {task_name}")

            task_config = task_configs[task_idx]

            self.logger.info(f'Preparing for task: {task_name}')
            self.strategy.prepare_for_task(TaskInfo(**task_config['task_info']))
            # Load checkpoint for this task
            self._load_checkpoint_for_task(task_idx, task_config)

            # Ensure test loaders before testing
            self._ensure_test_loaders_available()

            # Run all evaluation scenarios
            task_results = self._evaluate_after_task(task_idx)
            results[task_name] = task_results

            # Update result matrices
            self._update_results(task_idx, task_results)

        # Compute final metrics
        final_results = self._finalize_results()

        return final_results

    def _train_single_task(self, task_idx: int, task_name: str) -> Dict[str, Any]:
        """
        Train on a single task using the strategy.

        Args:
            task_idx: Current task index
            task_name: Current task name

        Returns:
            Task training results
        """
        # Delegate to strategy
        return self.strategy.train_task(task_idx, task_name)

    def _evaluate_after_task(self, task_idx: int) -> Dict[str, Any]:
        """
        Evaluate model after completing a task.

        Args:
            task_idx: Index of just completed training task
            task_name: Task name

        Returns:
            Evaluation results
        """
        return self.evaluator.evaluate_after_task(
            task_idx,
            self.task_list[:task_idx+1],
            model_wrapper=self.model_wrapper,
            strategy=self.strategy,
            total_tasks=len(self.task_list)
        )

    def _evaluate_task(self, task_idx: int, task_name: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a specific task.

        Args:
            task_idx: Task index
            task_name: Task name

        Returns:
            Task evaluation results
        """
        return self.evaluator.evaluate_task(
            task_idx,
            task_name,
            self.task_list,
            model_wrapper=self.model_wrapper,
            strategy=self.strategy
        )

    def _handle_resume(self) -> int:
        """
        Handle resuming training from a specific task.

        Priority:
        1. Use --resume_task_idx if provided
        2. Fall back to --resume_checkpoint (old method)
        3. Default to 0 (start from beginning)
        """

        # Priority to resume_task_idx argument
        if hasattr(self.args, 'resume_task_idx') and self.args.resume_task_idx is not None:
            resume_idx = self.args.resume_task_idx

            if resume_idx < 0 or resume_idx >= len(self.task_list):
                raise ValueError(
                    f"Invalid resume_task_idx: {resume_idx}. "
                    f"Must be between 0 and {len(self.task_list)-1}"
                )

            self.logger.info(f"Resuming from task index {resume_idx}: {self.task_list[resume_idx]}")
            self.logger.info(f"   Tasks 0-{resume_idx-1} will be prepared (not trained)")
            return resume_idx

        return 0

    def _prepare_previous_tasks(self, resume_idx: int):
        """
        Prepare model state for all tasks before resume_idx.

        This method:
        1. Loads task configurations from checkpoints
        2. Calls prepare_for_task() to add model components (experts, heads, etc.)
        3. Loads checkpoint weights
        4. SKIPS actual training

        Args:
            resume_idx: Index of task to resume from (0-indexed)
        """
        from core.base_strategy import TaskInfo

        if resume_idx == 0:
            self.logger.info("No previous tasks to prepare (starting from task 0)")
            return

        # Determine checkpoint directory
        checkpoint_dir = getattr(self.args, 'resume_from', None) or self.args.output

        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_dir}\n"
                f"Use --resume_from to specify checkpoint location"
            )

        self.logger.info(f"Preparing model for tasks 0 to {resume_idx-1}")
        self.logger.info(f"Loading checkpoints from: {checkpoint_dir}")

        # Extract task configs from checkpoints
        try:
            task_configs = self._extract_task_configs_from_checkpoints(
                checkpoint_dir,
                num_tasks=resume_idx,
                allow_missing=False)
            self.logger.debug(f'TASK CONFIGS RETURNED FOR PREPARATION: {len(task_configs)}')

        except Exception as e:
            self.logger.error(f"Failed to extract task configs: {e}")
            raise

        # Verify we got all the configs we need
        if len(task_configs) < resume_idx:
            missing_tasks = [
                self.task_list[i] for i in range(resume_idx)
                if i not in task_configs
            ]
            raise FileNotFoundError(
                f"Missing checkpoints for tasks: {missing_tasks}. "
                f"Cannot resume from task {resume_idx}"
            )

        # Prepare each previous task
        for task_idx in range(resume_idx):
            task_name = self.task_list[task_idx]
            self.logger.info(f"   [{task_idx+1}/{resume_idx}] Preparing task: {task_name}")

            train_loader, val_loader, task_info = self._build_task_loaders(task_idx, task_name)

            if hasattr(self, '_unified_ans2label') and self._unified_ans2label is not None:
                self.dataset_l2a[task_idx] = train_loader.dataset.label2ans
                self.dataset_a2l[task_idx] = train_loader.dataset.ans2label

            try:
                task_config = task_configs[task_idx]
                # 1. Prepare strategy (adds components, heads, freezes params, etc.)
                self.strategy.prepare_for_task(TaskInfo(**task_config['task_info']))

                # 2. Load checkpoint weights
                self._load_checkpoint_for_task(task_idx, task_config)

                self.logger.info(f"      Task {task_idx} prepared successfully")

                self._release_loaders(train_loader, val_loader)

            except Exception as e:
                self.logger.error(f"Failed to prepare task {task_idx} ({task_name}): {e}")
                raise

        self.logger.info(f"Successfully prepared {resume_idx} previous task(s)")

    def _load_checkpoint_for_task(self, task_idx: int, task_config: Dict[str, Any]):
        """Load model checkpoint for a specific task using task configuration."""

        # Check if strategy has custom multi-checkpoint loading
        if task_config.get('is_multi_component', False):
            if hasattr(self.strategy, 'load_checkpoints_for_task'):
                # Multi-component strategies handle their own loading
                self.strategy.load_checkpoints_for_task(task_idx, task_config)
            else:
                raise NotImplementedError(
                    f"Strategy claims multi-component support but doesn't implement "
                    f"load_checkpoints_for_task method"
                )
        else:
            # Single-component strategies use standard loading
            checkpoint_path = task_config.get('checkpoint_path')
            self.logger.debug(f"Loading checkpoint for task {task_idx} from {checkpoint_path}")
            task_name = task_config['task_info']['task_name']
            self.load_checkpoint(task_idx, task_name, checkpoint_type='best', checkpoint_path=checkpoint_path)

            self.logger.info(f"Loaded checkpoint for task {task_config['task_info']['task_name']}")

    def _update_results(self, task_idx: int, eval_results: Dict[str, Any]):
        """Update result matrices with new evaluation results."""
        # This will be implemented based on the specific result structure
        pass

    def _compute_final_metrics(self) -> Dict[str, float]:
        """Compute final continual learning metrics."""
        metrics = {}

        # Average accuracy across all tasks
        if 'standard_accuracy' in self.result_matrices:
            acc_matrix = self.result_matrices['standard_accuracy']

            # Compute average accuracy for each training stage
            for train_task in self.task_list:
                if train_task in acc_matrix:
                    accuracies = list(acc_matrix[train_task].values())
                    if accuracies:
                        metrics[f'avg_acc_after_{train_task}'] = sum(accuracies) / len(accuracies)

            # Final average accuracy
            final_task = self.task_list[-1]
            if final_task in acc_matrix:
                final_accuracies = list(acc_matrix[final_task].values())
                if final_accuracies:
                    metrics['final_avg_accuracy'] = sum(final_accuracies) / len(final_accuracies)

        # Compute forgetting
        if 'standard_forget' in self.result_matrices:
            forget_matrix = self.result_matrices['standard_forget']
            final_task = self.task_list[-1]
            if final_task in forget_matrix:
                forget_values = [v for v in forget_matrix[final_task].values() if v > 0]
                if forget_values:
                    metrics['avg_forgetting'] = sum(forget_values) / len(forget_values)

        return metrics

    def _finalize_results(self) -> Dict[str, Any]:
        """Finalize results with comprehensive CL metrics."""
        # Compute CL metrics using the evaluator
        try:
            cl_metrics = self.evaluator.compute_cl_metrics()
            cl_metrics_dict = cl_metrics.to_dict()
        except Exception as e:
            self.logger.warning(f"Could not compute CL metrics: {e}")
            cl_metrics_dict = None

        # Build final results
        final_results = {
            'task_list': self.task_list,
            'result_matrices': self.result_matrices,
            'continual_learning_metrics': cl_metrics_dict,
            'args': vars(self.args)
        }

        # Save results
        self._save_results(final_results)

        # Print summary
        if cl_metrics_dict:
            self.evaluator.print_cl_metrics_summary()

        return final_results

    def _save_results(self, results: Dict[str, Any]):
        """Save results to files."""
        # Save as JSON
        results_path = os.path.join(self.args.output, 'results.json')

        # Convert any non-serializable objects
        serializable_results = self._make_serializable(results)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)

        # Save individual result matrices
        for matrix_name, matrix_data in self.result_matrices.items():
            matrix_path = os.path.join(self.args.output, f'{matrix_name}.json')
            with open(matrix_path, 'w') as f:
                json.dump(matrix_data, f, indent=4)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert complex objects to string representation
        else:
            return obj

    def _extract_task_configs_from_checkpoints(
            self, checkpoint_dir: str,
            num_tasks: Optional[int] = None,
            allow_missing: bool = False
        ) -> List[Dict[str, Any]]:
        """
        Extract task configurations from saved checkpoints.
        Returns:
            List of task configurations
        """
        self.logger.info(f"Extracting task configs from checkpoints in {checkpoint_dir}")

        if num_tasks is None:
            num_tasks = len(self.task_list)
        else:
            num_tasks = min(num_tasks, len(self.task_list))

        task_configs = {}

        for task_idx in range(num_tasks):
            task_name = self.task_list[task_idx]

            # Check if strategy has multi-component checkpoint support
            if hasattr(self.strategy, 'get_checkpoint_paths_for_task'):
                # Multi-component strategy
                self.logger.debug(f"Extracting multi-component checkpoints for task {task_name}")
                checkpoint_paths = self.strategy.get_checkpoint_paths_for_task(
                    task_idx, task_name, checkpoint_dir, checkpoint_type=getattr(self.args, 'checkpoint_type', None)
                )

                if not checkpoint_paths or checkpoint_paths.get('main') is None:
                    if allow_missing:
                        self.logger.warning(f"No checkpoints found for task {task_name}; skipping")
                        continue
                    else:
                        raise FileNotFoundError(
                            f"No checkpoints found for task {task_name} in {checkpoint_dir}"
                        )

                # Load metadata from main checkpoint
                main_checkpoint_path = checkpoint_paths['main']
                task_config = self._extract_config_from_checkpoint(
                    task_idx, task_name, main_checkpoint_path
                )
                task_config['checkpoint_paths'] = checkpoint_paths
                task_config['is_multi_component'] = True
            else:
                # Single-component strategy
                checkpoint_path = self._find_checkpoint_path(checkpoint_dir, task_name)

                if checkpoint_path is None:
                    if allow_missing:
                        self.logger.warning(
                            f"No checkpoint found for task {task_name}, skipping"
                        )
                        continue
                    else:
                        raise FileNotFoundError(
                            f"No checkpoint found for task {task_name} in {checkpoint_dir}"
                        )
                task_config = self._extract_config_from_checkpoint(
                    task_idx, task_name, checkpoint_path
                )
                task_config['checkpoint_path'] = checkpoint_path
                task_config['is_multi_component'] = False

            task_configs[task_idx] = task_config
            self.logger.info(
                f"Extracted config for task {task_name} "
                f"(multi-component: {task_config['is_multi_component']})"
            )

        task_configs[task_idx] = task_config
        self.logger.info(f"Extracted config for task {task_name}: (multi-component: {task_config['is_multi_component']}) \n label2ans: {task_config['task_info']['label2ans'][:10]}\n num_classes: {task_config['task_info'].get('num_classes')}"
        )
        return task_configs

    def _extract_config_from_checkpoint(
        self, task_idx: int, task_name: str, checkpoint_path: str
    ) -> Dict[str, Any]:
        """Extract task configuration from a single checkpoint file."""
        self.logger.debug(f"  Loading checkpoint: {os.path.basename(checkpoint_path)}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        task_config = {
            'task_info': {'task_id': task_idx, 'task_name': task_name},
        }

        # Extract task info from strategy state
        if 'strategy_state' in checkpoint:
            if 'current_task_info' in checkpoint['strategy_state']:
                ct = checkpoint['strategy_state']['current_task_info']

                if hasattr(ct, '__dict__'):
                    ct = ct.__dict__
                if ct.get('task_name') == task_name or ct.get('task_idx') == task_idx:
                    task_config['task_info'].update({
                        'num_classes': ct.get('num_classes', None),
                        'label2ans': ct.get('label2ans', {}),
                        'dataset_size': ct.get('dataset_size', 0),
                    })
            elif 'num_classes' in checkpoint['strategy_state']:
                task_config['task_info']['num_classes'] = checkpoint['strategy_state']['num_classes']
                task_config['task_info']['label2ans'] = checkpoint['strategy_state'].get('label2ans', {})
                task_config['task_info']['dataset_size'] = checkpoint['strategy_state'].get('dataset_size', 0)

        # Fallback: try to get from top-level checkpoint keys
        if 'num_classes' not in task_config['task_info'] and 'num_answers' in checkpoint:
            task_config['task_info']['num_classes'] = checkpoint['num_answers']
        if 'label2ans' not in task_config['task_info'] and 'label2ans' in checkpoint:
            task_config['task_info']['label2ans'] = checkpoint['label2ans']

        # Validate required fields
        if 'num_classes' not in task_config['task_info'] and 'num_answers' not in task_config['task_info']:
            raise ValueError(
                f"Neither num_classes nor num_answers found in checkpoint for task {task_name}"
            )

        return task_config

    def _find_checkpoint_path(self, checkpoint_dir: str, task_name: str) -> Optional[str]:
        candidates = [
            f"{task_name}_best.pth",
            f"{task_name}_latest.pth",
            f"{task_name}_final.pth",
            f"{task_name}.pth",
        ]

        for candidate in candidates:
            path = os.path.join(checkpoint_dir, candidate)
            if os.path.exists(path):
                return path

        return None

    @abstractmethod
    def get_current_model(self):
        """Get the current model for external access."""
        pass

    @abstractmethod
    def get_current_strategy(self):
        """Get the current strategy for external access."""
        pass

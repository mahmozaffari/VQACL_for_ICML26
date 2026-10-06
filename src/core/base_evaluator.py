
"""
Abstract base class for evaluators in continual learning.

This class provides a unified interface for evaluating different CL strategies
across various scenarios and computing standard CL metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
import os
import time
import traceback
from tqdm import tqdm

class EvaluationScenario(Enum):
    """Different evaluation scenarios for continual learning."""
    STANDARD = "standard"  # Use strategy's task prediction
    ORACLE = "oracle"      # Use ground-truth task IDs
    BAYESIAN = "bayesian"  # Use ensemble of all task heads

@dataclass
class EvaluationResult:
    """Result from a single evaluation run."""
    scenario: EvaluationScenario
    task_name: str
    accuracy: float
    ece: float  # Expected Calibration Error
    predictions: Dict[str, Any]  # Question ID -> (answer, confidence, metadata)
    per_class_accuracy: Optional[Dict[str, float]] = None
    per_question_type_accuracy: Optional[Dict[str, float]] = None
    per_answer_type_accuracy: Optional[Dict[str, float]] = None
    additional_metrics: Optional[Dict[str, float]] = None
    logits: Optional[Dict[str, Any]] = None  # Question ID -> logits

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'scenario': self.scenario.value,
            'task_name': self.task_name,
            'accuracy': self.accuracy,
            'ece': self.ece,
            'per_class_accuracy': self.per_class_accuracy,
            'per_question_type_accuracy': self.per_question_type_accuracy,
            'per_answer_type_accuracy': self.per_answer_type_accuracy,
            'additional_metrics': self.additional_metrics or {}
        }

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a summary dictionary with key metrics."""
        return {
            'scenario': self.scenario.value,
            'task_name': self.task_name,
            'accuracy': self.accuracy,
            'ece': self.ece,
            'num_predictions': len(self.predictions),
            'per_class_accuracy': self.per_class_accuracy,
            'per_question_type_accuracy': self.per_question_type_accuracy,
            'per_answer_type_accuracy': self.per_answer_type_accuracy,
            'additional_metrics': self.additional_metrics or {}
        }

    def get_predictions_list(self) -> List[Dict[str, Any]]:
        """Get predictions as a list of dictionaries."""
        predictions_list = []
        for qid, pred in self.predictions.items():
            pred_entry = {
                'question_id': qid,
                'answer': pred['answer'],
                'confidence': pred.get('confidence', None),
                'answer_id': pred.get('answer_id', None)
            }
            for key, value in pred.items():
                if key not in pred_entry:
                    pred_entry[key] = value
            predictions_list.append(pred_entry)
        return predictions_list

    def save_predictions(self, output_dir: str, training_task: str, test_task: str, scenario: str, expert_id: Optional[int] = None):
        """
        Save predictions to a JSON file with proper naming and structure.

        Args:
            output_dir: Base output directory
            training_task: Name of the training task (training stage)
            test_task: Name of the test task
            scenario: Evaluation scenario name
        """
        if isinstance(scenario, EvaluationScenario):
            scenario = scenario.value
        scenario = str(scenario).lower()

        # Create predictions directory structure
        predictions_dir = os.path.join(output_dir, 'predictions', scenario)
        os.makedirs(predictions_dir, exist_ok=True)

        # Filename indicates training stage and test task (optionally forced expert)
        if expert_id is None:
            filename = f"after_{training_task}_on_{test_task}.json"
        else:
            filename = f"after_{training_task}_by_expert_{expert_id}_on_{test_task}.json"
        filepath = os.path.join(predictions_dir, filename)

        if expert_id is None:
            logits_filename = f"after_{training_task}_on_{test_task}_logits.pth"
        else:
            logits_filename = f"after_{training_task}_by_expert_{expert_id}_on_{test_task}_logits.pth"
        logits_filepath = os.path.join(predictions_dir, logits_filename)

        if self.logits:
            torch.save(self.logits, logits_filepath)

        # Format predictions for saving
        predictions_list = []
        for qid, pred in self.predictions.items():
            pred_entry = {
                'question_id': qid,
                'answer': pred['answer'],
                'confidence': float(pred.get('confidence', 1.0)),
                'answer_id': int(pred.get('answer_id', -1)),
            }

            # Include task_id prediction if available (for methods with task-id prediction)
            if 'pred_task_id' in pred and pred['pred_task_id'] is not None:
                pred_entry['pred_task_id'] = int(pred['pred_task_id'])

            # Include any additional prediction-related fields
            for key, value in pred.items():
                if key not in pred_entry and key not in ['answer', 'confidence', 'answer_id', 'pred_task_id']:
                    pred_entry[key] = value

            predictions_list.append(pred_entry)

        # Create metadata
        save_data = {
            'training_task': training_task,
            'test_task': test_task,
            'scenario': scenario,
            'num_predictions': len(predictions_list),
            'accuracy': float(self.accuracy),
            'ece': float(self.ece),
            'additional_metrics': self.additional_metrics or {},
            'predictions': predictions_list
        }
        if expert_id is not None:
            save_data['expert_id'] = int(expert_id)

        def _json_fallback(obj: Any):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            return str(obj)

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=_json_fallback)

        return filepath

@dataclass
class CLMetrics:
    """Continual learning specific metrics."""
    average_accuracy: float
    forgetting: float
    backward_transfer: float
    forward_transfer: float
    average_incremental_accuracy: List[float]
    final_performance: Dict[str, float]
    per_task_forgetting: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'average_accuracy': float(self.average_accuracy),
            'forgetting': float(self.forgetting),
            'backward_transfer': float(self.backward_transfer),
            'forward_transfer': float(self.forward_transfer),
            'average_incremental_accuracy': [float(x) for x in self.average_incremental_accuracy],
            'final_performance': {k: float(v) for k, v in self.final_performance.items()},
            'per_task_forgetting': {k: float(v) for k, v in self.per_task_forgetting.items()}
        }

class BaseEvaluator(ABC):
    """Abstract base class for continual learning evaluators."""

    def __init__(self, args, test_loaders: Dict[str, Any], **kwargs):
        """
        Initialize the evaluator.

        Args:
            args: Configuration arguments
            test_loaders: Dictionary mapping task names to test data loaders
            **kwargs: Evaluator-specific arguments
        """
        self.args = args
        self.test_loaders = test_loaders

        # Evaluation results storage
        self.evaluation_history: List[EvaluationResult] = []
        self.accuracy_matrix: Dict[str, Dict[str, float]] = {}  # train_task -> {test_task: acc}
        self.ece_matrix: Dict[str, Dict[str, float]] = {}
        self.bayesian_accuracy_matrix: Dict[str, Dict[str, float]] = {}  # ADD
        self.bayesian_ece_matrix: Dict[str, Dict[str, float]] = {}

        # Cache oracle results since they don't change after a task is trained
        self.oracle_cache: Dict[str, EvaluationResult] = {}

        # Metrics computation
        self.metric_functions = self._initialize_metric_functions()

        # Create predictions output directory
        self.predictions_dir = os.path.join(args.output, 'predictions')
        os.makedirs(self.predictions_dir, exist_ok=True)

    @abstractmethod
    def _initialize_metric_functions(self) -> Dict[str, Callable]:
        """
        Initialize metric computation functions.

        Returns:
            Dictionary mapping metric names to computation functions
        """
        pass

    @abstractmethod
    def _compute_accuracy(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Compute accuracy given predictions and ground truth.

        Args:
            predictions: Dictionary with question IDs as keys
            ground_truth: Dictionary with question IDs as keys

        Returns:
            Accuracy score
        """
        pass

    @abstractmethod
    def _compute_ece(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Compute Expected Calibration Error.

        Args:
            predictions: Dictionary with question IDs as keys, values contain confidences
            ground_truth: Dictionary with question IDs as keys

        Returns:
            ECE score
        """
        pass

    def evaluate_after_task(self,
                          current_task_idx: int,
                          learned_tasks: List[str],
                          model_wrapper,
                          strategy,
                          total_tasks: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the model after completing training on a task.

        Args:
            current_task_idx: Index of the just-completed training task
            learned_tasks: List of task names learned so far
            model_wrapper: Model wrapper instance
            strategy: CL strategy instance

        Returns:
            Dictionary containing evaluation results
        """
        current_task = learned_tasks[current_task_idx]
        results = {}

        # Check if oracle evaluation is meaningful for this strategy
        should_run_oracle = strategy.supports_meaningful_oracle_evaluation()

        # Evaluate on all learned tasks using different scenarios
        for scenario in EvaluationScenario:
            if scenario == EvaluationScenario.ORACLE and not should_run_oracle:
                print(f"Skipping oracle evaluation for {strategy.__class__.__name__} "
                            f"(not meaningful for this strategy)")
                continue

            if scenario == EvaluationScenario.BAYESIAN and not strategy.supports_bayesian_evaluation():
                print(f"Skipping bayesian evaluation for {strategy.__class__.__name__} "
                            f"(not supported by this strategy)")
                continue

            scenario_results = {}

            for test_task_idx, test_task in enumerate(learned_tasks):

                if self.check_evaluation_scenario_skip(scenario):
                    continue

                if scenario == EvaluationScenario.ORACLE and test_task in self.oracle_cache: # skip ORACLE evaluation, if we already have oracle result for this test task
                    # Reuse cached oracle result
                    eval_result = self.oracle_cache[test_task]
                    print(f"[CACHED] Oracle evaluation for {test_task} "
                            f"(trained at stage {test_task_idx}, reusing cached result)")
                    scenario_results[test_task] = eval_result
                    continue

                print(f"{scenario.value} evaluation after training stage {current_task}.")
                eval_result = self._evaluate_single_task(
                    test_task_idx, test_task, scenario, model_wrapper, strategy
                )

                # Save predictions to file
                try:
                    saved_path = eval_result.save_predictions(
                        output_dir=self.args.output,
                        training_task=current_task,
                        test_task=test_task,
                        scenario=scenario.value
                    )
                    print(f"  Saved predictions to: {saved_path}")
                except Exception as e:
                    print(f"  Warning: Failed to save predictions: {e}")

                if scenario == EvaluationScenario.ORACLE:  # Cache this oracle result for future reuse
                    self.oracle_cache[test_task] = eval_result
                    print(f"  [CACHED] Oracle result for {test_task} saved to cache")

                scenario_results[test_task] = eval_result
                self.evaluation_history.append(eval_result)

            results[scenario.value] = scenario_results

        # Optionally evaluate expert-by-task matrix at final stage
        total_tasks = total_tasks or len(learned_tasks)
        if (getattr(self.args, 'evaluate_expert_matrix', False)
                and current_task_idx == total_tasks - 1):
            expert_matrix = self._evaluate_expert_matrix(
                learned_tasks=learned_tasks,
                model_wrapper=model_wrapper,
                strategy=strategy,
                training_task=current_task
            )
            results['expert_matrix'] = expert_matrix

        # Update accuracy matrices
        self._update_accuracy_matrices(current_task, results)

        results['joint'] = {}

        return results

    def check_evaluation_scenario_skip(self, scenario):
        if scenario == EvaluationScenario.STANDARD and getattr(self.args, 'skip_standard_evaluation', False):
            return True
        if scenario == EvaluationScenario.ORACLE and getattr(self.args, 'skip_oracle_evaluation', False):
            return True
        if scenario == EvaluationScenario.BAYESIAN and getattr(self.args, 'skip_bayesian_evaluation', False):
            return True
        return False

    def evaluate_task(self,
                     task_idx: int,
                     task_name: str,
                     all_tasks: List[str],
                     model_wrapper,
                     strategy) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a specific task (for test-only mode).

        Args:
            task_idx: Task index
            task_name: Task name
            all_tasks: List of all task names
            model_wrapper: Model wrapper instance
            strategy: CL strategy instance

        Returns:
            Dictionary containing evaluation results
        """
        results = {}

        # Test this task against all other tasks using different scenarios
        for scenario in EvaluationScenario:
            scenario_results = {}

            for test_task_idx, test_task in enumerate(all_tasks):
                eval_result = self._evaluate_single_task(
                    test_task_idx, test_task, scenario, model_wrapper, strategy,
                    training_task_idx=task_idx
                )
                scenario_results[test_task] = eval_result

            results[scenario.value] = scenario_results

        return results

    def _evaluate_single_task(self,
                             test_task_idx: int,
                             test_task: str,
                             scenario: EvaluationScenario,
                             model_wrapper,
                             strategy,
                             training_task_idx: Optional[int] = None,
                             forced_expert_id: Optional[int] = None) -> EvaluationResult:
        """
        Evaluate on a single task using a specific scenario.

        Args:
            test_task: Name of task to test on
            scenario: Evaluation scenario to use
            model_wrapper: Model wrapper instance
            strategy: CL strategy instance
            training_task_idx: Index of training task (for OOD evaluation)

        Returns:
            Evaluation result
        """
        if test_task not in self.test_loaders:
            raise ValueError(f"Test loader not found for task: {test_task}")

        test_loader = self.test_loaders[test_task]
        predictions = {}
        logits = {}

        # Set model to evaluation mode
        model_wrapper.eval()

        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"{scenario.value} eval on {test_task}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                batch_predictions, batch_logits = self._get_predictions_for_scenario(
                    batch, scenario, model_wrapper, strategy, training_task_idx,
                    forced_expert_id=forced_expert_id
                )

                predictions.update(batch_predictions)
                if getattr(self.args, 'save_logits', False) and batch_logits is not None:
                    logits.update(batch_logits)

        # ---------- task-ID accuracy ----------
        task_id_total   = 0
        task_id_correct = 0
        for p in predictions.values():
            tid = p.get("task_id_pred")          # may be None
            if tid is not None:
                task_id_total += 1
                if tid == test_task_idx:
                    task_id_correct += 1
        task_id_accuracy = (task_id_correct / task_id_total) if task_id_total else None
        # ---------- VQA accuracy ----------
        evaluator = test_loader.evaluator
        acc_dict = evaluator.evaluate_raw_ece(predictions)
        accuracy = acc_dict['overall']
        ece = acc_dict['ece']

        for qid in predictions.keys():
            if qid in evaluator.evalQA:
                predictions[qid]['vqa_accuracy'] = evaluator.evalQA[qid] / 100.0

        additional_metrics = {'task_id_accuracy': task_id_accuracy}
        if forced_expert_id is not None:
            additional_metrics['forced_expert_id'] = int(forced_expert_id)

        return EvaluationResult(
            scenario=scenario,
            task_name=test_task,
            accuracy=accuracy,
            ece=ece,
            predictions=predictions,
            additional_metrics=additional_metrics,
            logits=logits if getattr(self.args, 'save_logits', False) and logits else None
        )

    def _get_predictions_for_scenario(self,
                                    batch: Dict[str, Any],
                                    scenario: EvaluationScenario,
                                    model_wrapper,
                                    strategy,
                                    training_task_idx: Optional[int] = None,
                                    forced_expert_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get predictions for a specific evaluation scenario.

        Args:
            batch: Input batch
            scenario: Evaluation scenario
            model_wrapper: Model wrapper instance
            strategy: CL strategy instance
            training_task_idx: Training task index (for OOD)

        Returns:
            Dictionary with predictions
        """
        predictions = {}
        logits_dict = {}

        if forced_expert_id is not None:
            # Force routing to a specific expert, regardless of true task.
            results = strategy.predict(batch, task_id=forced_expert_id, return_answer_strings=True)
        elif scenario == EvaluationScenario.STANDARD:
            # Use strategy's normal prediction method
            results = strategy.predict(batch, return_answer_strings=True)

        elif scenario == EvaluationScenario.ORACLE:
            # Use ground-truth task IDs
            if 'task_ids' not in batch:
                self.logger.warning("Batch does not contain 'task_ids'; skipping Oracle evaluation.")
            task_id = batch.get('task_ids')[0].item()  # Assume batch contains true task ID
            results = strategy.predict(batch, task_id=task_id, return_answer_strings=True)
        elif scenario == EvaluationScenario.BAYESIAN:
            # Use ensemble of all task heads
            results = strategy.predict_ensemble(batch, return_answer_strings=True, temperature=self.args.router_temperature_scale)
        elif scenario == EvaluationScenario.CONF:
            # Use confidence-based task selection
            results = strategy.predict_confidence_based(batch, return_answer_strings=True)
        else:
            raise ValueError(f"Unsupported scenario: {scenario}")

        # Extract prediction information
        questions = batch['questions']
        question_ids = batch['question_id']
        pred_answers = results['predictions']  # These are actual answer strings
        confidences = results.get('confidences', [1.0] * len(pred_answers))
        prediction_indices = results.get('prediction_indices', [0] * len(pred_answers))
        task_id_preds = results.get('task_id_predictions', [None] * len(pred_answers)) # For strategies that provide this
        task_id_confs = results.get('task_id_confidences', [None] * len(pred_answers))
        meta_data = results.get('meta_data', {})
        per_sample_experts = results.get('per_sample_experts', None)
        logits = results.get('logits', None)

        for key, value in meta_data.items():
            if isinstance(value, torch.Tensor):
                meta_data[key] = value.cpu().tolist()

        def get_meta_data_for_index(index: int) -> Dict[str, Any]:
            data_entry = {}
            for key, value in meta_data.items():
                if isinstance(value, list):
                    data_entry[key] = value[index]
            return data_entry
        def get_logits_for_index(index: int) -> Any:
            if logits is not None:
                if isinstance(logits, torch.Tensor):
                    return logits[index].cpu().numpy()
                elif isinstance(logits, np.ndarray):
                    return logits[index]
            return None
        def get_expert_data_for_index(index: int) -> Dict[str, Any]:
                sample = per_sample_experts[index]
                return {
                    'selected_expert': sample['selected_expert'],
                    'expert_predictions': {exp['expert_id']: {'answer': exp['prediction'], 'confidence': exp['confidence'], 'weight': exp['weight'], 'accuracy': exp['accuracy'] if 'accuracy' in exp else None} for exp in sample['experts']},
                }

        # Handle case where predictions is a single string (batch size = 1)
        if isinstance(pred_answers, str):
            pred_answers = [pred_answers]

        # Handle case where confidences/indices are tensors
        if isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().tolist()
        if isinstance(prediction_indices, torch.Tensor):
            prediction_indices = prediction_indices.cpu().tolist()

        if isinstance(task_id_preds, torch.Tensor):
            task_id_preds = task_id_preds.cpu().tolist()
        if isinstance(task_id_confs, torch.Tensor):
            task_id_confs = task_id_confs.cpu().tolist()

        # Format predictions
        for idx, (qid, q, answer, conf, ans_id, tid, tid_conf) in enumerate(zip(question_ids, questions, pred_answers,
                                             confidences, prediction_indices, task_id_preds, task_id_confs)):
            # Extract scalar values if needed
            if isinstance(conf, torch.Tensor):
                conf = conf.item()
            if isinstance(ans_id, torch.Tensor):
                ans_id = ans_id.item()
            if isinstance(tid, torch.Tensor):
                tid = tid.item()

            predictions[qid] = {
                'question': q,
                'answer': answer,  # Actual string answer
                'confidence': conf,
                'answer_id': ans_id,  # Kept for backward compatibility
                'task_id_pred': tid,  # Which task was predicted (if available)
                'task_id_conf': tid_conf  # Confidence of task ID prediction
            }
            if per_sample_experts is not None:
                predictions[qid].update(get_expert_data_for_index(idx))
            logits_dict[qid] = get_logits_for_index(idx)

            meta_data_entry = get_meta_data_for_index(idx)
            predictions[qid].update(meta_data_entry)

        return predictions, logits_dict

    def _evaluate_expert_matrix(self,
                                learned_tasks: List[str],
                                model_wrapper,
                                strategy,
                                training_task: str) -> Dict[str, Any]:
        """
        Evaluate every expert on every task (final stage only).

        Returns:
            Dictionary with accuracy/ece matrices keyed by expert id.
        """
        num_experts = len(learned_tasks)
        expert_accuracy_matrix: Dict[str, Dict[str, float]] = {}
        expert_ece_matrix: Dict[str, Dict[str, float]] = {}

        for expert_id in range(num_experts):
            expert_key = str(expert_id)
            expert_accuracy_matrix[expert_key] = {}
            expert_ece_matrix[expert_key] = {}

            for test_task_idx, test_task in enumerate(learned_tasks):
                eval_result = self._evaluate_single_task(
                    test_task_idx=test_task_idx,
                    test_task=test_task,
                    scenario=EvaluationScenario.ORACLE,
                    model_wrapper=model_wrapper,
                    strategy=strategy,
                    forced_expert_id=expert_id
                )

                try:
                    saved_path = eval_result.save_predictions(
                        output_dir=self.args.output,
                        training_task=training_task,
                        test_task=test_task,
                        scenario="OOD",
                        expert_id=expert_id
                    )
                    print(f"  Saved expert-matrix predictions to: {saved_path}")
                except Exception as e:
                    print(f"  Warning: Failed to save expert-matrix predictions: {type(e).__name__}: {e}")
                    print(traceback.format_exc())

                expert_accuracy_matrix[expert_key][test_task] = eval_result.accuracy
                expert_ece_matrix[expert_key][test_task] = eval_result.ece

        summary = {
            'training_task': training_task,
            'num_experts': num_experts,
            'accuracy_matrix': expert_accuracy_matrix,
            'ece_matrix': expert_ece_matrix
        }
        summary_path = os.path.join(self.args.output, 'expert_matrix_summary.json')
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            print(f"Expert-matrix summary saved to {summary_path}")
        except Exception as e:
            print(f"Warning: Failed to save expert-matrix summary: {e}")

        return summary

    def _update_accuracy_matrices(self, current_task: str, results: Dict[str, Any]) -> None:
        """
        Update accuracy and ECE matrices with new results.

        Args:
            current_task: Name of the current training task
            results: Evaluation results
        """
        if current_task not in self.accuracy_matrix:
            self.accuracy_matrix[current_task] = {}
            self.ece_matrix[current_task] = {}
            self.bayesian_accuracy_matrix[current_task] = {}
            self.bayesian_ece_matrix[current_task] = {}

        # Update with standard scenario results (primary metric)
        if 'standard' in results:
            for test_task, eval_result in results['standard'].items():
                self.accuracy_matrix[current_task][test_task] = eval_result.accuracy
                self.ece_matrix[current_task][test_task] = eval_result.ece

        # Bayesian scenario results
        if 'bayesian' in results:
            for test_task, eval_result in results['bayesian'].items():
                self.bayesian_accuracy_matrix[current_task][test_task] = eval_result.accuracy
                self.bayesian_ece_matrix[current_task][test_task] = eval_result.ece

    def compute_cl_metrics(self) -> CLMetrics:
        """
        Compute continual learning specific metrics.

        Returns:
            CLMetrics object with computed metrics
        """
        if not self.accuracy_matrix:
            raise ValueError("No evaluation results available")

        task_names = list(self.accuracy_matrix.keys())

        # Get final accuracies
        final_task = task_names[-1]
        final_accuracies = self.accuracy_matrix[final_task]

        # Average accuracy
        avg_accuracy = np.mean(list(final_accuracies.values()))

        # Compute forgetting
        forgetting, per_task_forgetting = self._compute_forgetting_comprehensive()

        # Compute transfer metrics
        backward_transfer = self._compute_backward_transfer()
        forward_transfer = self._compute_forward_transfer()

        # Incremental accuracy (average accuracy after each task)
        incremental_accuracies = self._compute_incremental_accuracies()

        return CLMetrics(
            average_accuracy=avg_accuracy,
            forgetting=forgetting,
            backward_transfer=backward_transfer,
            forward_transfer=forward_transfer,
            average_incremental_accuracy=incremental_accuracies,
            final_performance=final_accuracies,
            per_task_forgetting=per_task_forgetting
        )

    def _compute_forgetting_comprehensive(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute forgetting metric comprehensively.

        Forgetting measures how much performance drops on old tasks after learning new ones.
        For each task i, forgetting is: max_j(acc[j,i]) - acc[T,i]
        where j ranges from i to T-1, and T is the final task.

        Returns:
            Tuple of (average_forgetting, per_task_forgetting_dict)
        """
        if len(self.accuracy_matrix) < 2:
            return 0.0, {}

        task_names = list(self.accuracy_matrix.keys())
        forgetting_scores = []
        per_task_forgetting = {}

        # For each task except the last (last task has no future tasks to forget it)
        for i, task_name in enumerate(task_names[:-1]):
            # Find best accuracy achieved on this task during training
            best_acc = 0.0
            best_at_stage = None

            # Check accuracy at each training stage from when task was learned onwards
            for j in range(i, len(task_names)):
                train_stage = task_names[j]
                if task_name in self.accuracy_matrix[train_stage]:
                    acc = self.accuracy_matrix[train_stage][task_name]
                    if acc > best_acc:
                        best_acc = acc
                        best_at_stage = train_stage

            # Final accuracy on this task
            final_train_stage = task_names[-1]
            final_acc = self.accuracy_matrix[final_train_stage].get(task_name, 0.0)

            # Forgetting is the drop from best to final
            task_forgetting = max(0, best_acc - final_acc)
            forgetting_scores.append(task_forgetting)
            per_task_forgetting[task_name] = task_forgetting

            # Log non-trivial forgetting values.
            if task_forgetting > 0.01:
                print(f"  Task {task_name}: best={best_acc:.4f} (at {best_at_stage}), "
                      f"final={final_acc:.4f}, forgetting={task_forgetting:.4f}")

        avg_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0.0

        return avg_forgetting, per_task_forgetting

    def _compute_backward_transfer(self) -> float:
        """
        Compute backward transfer metric.

        Backward transfer measures how learning new tasks affects performance on old tasks.
        BWT = (1/(T-1)) * sum_{i=1}^{T-1} (acc[T,i] - acc[i,i])

        Returns:
            Backward transfer score
        """
        if len(self.accuracy_matrix) < 2:
            return 0.0

        task_names = list(self.accuracy_matrix.keys())
        transfer_scores = []

        # For each task except the last
        for i, task_name in enumerate(task_names[:-1]):
            # Accuracy right after learning the task
            acc_after_learning = self.accuracy_matrix[task_name].get(task_name, 0.0)

            # Accuracy after learning all tasks
            final_task = task_names[-1]
            acc_final = self.accuracy_matrix[final_task].get(task_name, 0.0)

            # Transfer is the difference
            transfer = acc_final - acc_after_learning
            transfer_scores.append(transfer)

        bwt = np.mean(transfer_scores) if transfer_scores else 0.0

        return bwt

    def _compute_forward_transfer(self) -> float:
        """
        Compute forward transfer metric.

        Forward transfer measures how learning previous tasks helps with new tasks.
        FWT = (1/(T-1)) * sum_{i=2}^{T} (acc[i-1,i] - baseline[i])

        Returns:
            Forward transfer score
        """
        # Forward-transfer baselines are not computed by this evaluator.
        return 0.0

    def _compute_incremental_accuracies(self) -> List[float]:
        """
        Compute average accuracy after each task.

        Incremental accuracy shows how average performance evolves as more tasks are learned.

        Returns:
            List of average accuracies after each task
        """
        task_names = list(self.accuracy_matrix.keys())
        incremental_accuracies = []

        for i, train_task in enumerate(task_names):
            # Get all test tasks up to and including current training task
            test_tasks = task_names[:i+1]

            # Compute average accuracy on these tasks
            accuracies = []
            for test_task in test_tasks:
                if test_task in self.accuracy_matrix[train_task]:
                    accuracies.append(self.accuracy_matrix[train_task][test_task])

            if accuracies:
                avg_acc = np.mean(accuracies)
                incremental_accuracies.append(avg_acc)

        return incremental_accuracies

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evaluations performed.

        Returns:
            Dictionary with evaluation summary
        """
        try:
            cl_metrics = self.compute_cl_metrics()
        except ValueError:
            cl_metrics = None

        summary = {
            'total_evaluations': len(self.evaluation_history),
            'accuracy_matrix': self.accuracy_matrix,
            'ece_matrix': self.ece_matrix,
            'continual_learning_metrics': cl_metrics.to_dict() if cl_metrics else None,
        }

        return summary

    def save_all_results(self, output_dir: Optional[str] = None):
        """
        Save all evaluation results including predictions.
        """
        if output_dir is None:
            output_dir = self.args.output

        os.makedirs(output_dir, exist_ok=True)

        # Save evaluation summary
        summary = self.get_evaluation_summary()

        if summary.get('continual_learning_metrics'):
            cl_metrics = summary['continual_learning_metrics']
            if hasattr(cl_metrics, 'to_dict'):
                summary['continual_learning_metrics'] = cl_metrics.to_dict()

        with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        # Save matrices
        for matrix_name in ['accuracy_matrix', 'ece_matrix']:
            if matrix_name in summary:
                with open(os.path.join(output_dir, f'{matrix_name}.json'), 'w') as f:
                    json.dump(summary[matrix_name], f, indent=4)

        print(f"All results saved to {output_dir}")
        print(f"Predictions saved in {os.path.join(output_dir, 'predictions')}")

    def save_results(self, output_dir: str) -> None:
        """
        Save evaluation results to files.

        Args:
            output_dir: Directory to save results
        """
        import json
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Save evaluation summary
        summary = self.get_evaluation_summary()

        # Make summary JSON-serializable
        if summary['continual_learning_metrics']:
            # Convert CLMetrics to dict
            cl_metrics = summary['continual_learning_metrics']
            if hasattr(cl_metrics, 'to_dict'):
                summary['continual_learning_metrics'] = cl_metrics.to_dict()

        with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        # Save individual result matrices
        for matrix_name in ['accuracy_matrix', 'ece_matrix']:
            if matrix_name in summary:
                with open(os.path.join(output_dir, f'{matrix_name}.json'), 'w') as f:
                    json.dump(summary[matrix_name], f, indent=4)

    def print_cl_metrics_summary(self):
        """Print a formatted summary of CL metrics."""
        try:
            metrics = self.compute_cl_metrics()

            print("\n" + "="*70)
            print("CONTINUAL LEARNING METRICS SUMMARY")
            print("="*70)

            print(f"\nAverage Final Accuracy: {metrics.average_accuracy:.4f}")
            print(f"Average Forgetting:     {metrics.forgetting:.4f}")
            print(f"Backward Transfer:      {metrics.backward_transfer:.4f}")
            print(f"Forward Transfer:       {metrics.forward_transfer:.4f}")

            print("\nPer-Task Forgetting:")
            for task_name, forget_score in metrics.per_task_forgetting.items():
                print(f"  {task_name:20s}: {forget_score:.4f}")

            print("\nIncremental Accuracies:")
            task_names = list(self.accuracy_matrix.keys())
            for i, (task_name, inc_acc) in enumerate(zip(task_names, metrics.average_incremental_accuracy)):
                print(f"  After {task_name:15s}: {inc_acc:.4f}")

            print("="*70 + "\n")

        except Exception as e:
            print(f"Could not print CL metrics: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_evaluations={len(self.evaluation_history)})"

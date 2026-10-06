
"""
VQA-specific evaluator for continual learning.

This module provides evaluation functionality specifically designed for
Visual Question Answering tasks, including VQA-specific metrics and
evaluation scenarios.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict
import torch

from core.base_evaluator import BaseEvaluator, EvaluationResult, EvaluationScenario, CLMetrics

class VQAEvaluator(BaseEvaluator):
    """VQA-specific evaluator for continual learning."""

    def __init__(self, args, test_loaders: Dict[str, Any], task_list: List[str], **kwargs):
        """
        Initialize VQA evaluator.

        Args:
            args: Configuration arguments
            test_loaders: Dictionary mapping task names to test data loaders
            task_list: List of task names
            **kwargs: Additional evaluator arguments
        """
        super().__init__(args, test_loaders, **kwargs)
        self.task_list = task_list

        # VQA-specific configuration
        self.use_vqa_accuracy = getattr(args, 'use_vqa_accuracy', True)
        self.min_answer_confidence = getattr(args, 'min_answer_confidence', 0.0)

        # Question type analysis (if available)
        self.analyze_question_types = getattr(args, 'analyze_question_types', True)
        self.question_type_mapping = self._load_question_type_mapping()

        # Answer type analysis
        self.analyze_answer_types = getattr(args, 'analyze_answer_types', True)

    def _initialize_metric_functions(self) -> Dict[str, Callable]:
        """Initialize VQA-specific metric computation functions."""
        return {
            'vqa_accuracy': self._compute_vqa_accuracy,
            'exact_match_accuracy': self._compute_exact_match_accuracy,
            'per_question_type_accuracy': self._compute_per_question_type_accuracy,
            'per_answer_type_accuracy': self._compute_per_answer_type_accuracy,
            'confidence_statistics': self._compute_confidence_statistics,
            'top_k_accuracy': lambda pred, gt: self._compute_top_k_accuracy(pred, gt, k=3)
        }

    def _load_question_type_mapping(self) -> Dict[str, str]:
        """
        Load question type mapping for analysis.

        Returns:
            Dictionary mapping question IDs or patterns to question types
        """
        # Simple keyword-based mapping from question words to types
        return {
            'what': 'object',
            'where': 'location',
            'when': 'time',
            'who': 'person',
            'why': 'reason',
            'how many': 'counting',
            'how': 'method',
            'is': 'yes/no',
            'does': 'yes/no',
            'can': 'yes/no',
            'are': 'yes/no',
        }

    def _compute_accuracy(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Compute VQA accuracy (default metric).

        Args:
            predictions: Dictionary with question IDs as keys
            ground_truth: Dictionary with question IDs as keys

        Returns:
            VQA accuracy score
        """
        if self.use_vqa_accuracy:
            return self._compute_vqa_accuracy(predictions, ground_truth)
        else:
            return self._compute_exact_match_accuracy(predictions, ground_truth)

    def _compute_vqa_accuracy(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Compute VQA accuracy (min(1, #correct_answers/3)).

        This follows the standard VQA evaluation protocol where an answer is
        considered correct if at least 3 out of 10 human annotators agree.

        Args:
            predictions: Dictionary with question IDs as keys
            ground_truth: Dictionary with question IDs as keys

        Returns:
            VQA accuracy score
        """
        if not predictions:
            return 0.0

        total_score = 0.0
        num_questions = 0

        for qid in predictions:
            if qid not in ground_truth:
                continue

            pred_answer = self._normalize_answer(predictions[qid]['answer'])

            # Handle multiple ground truth answers (if available)
            gt_data = ground_truth[qid]
            if isinstance(gt_data, dict) and 'answers' in gt_data:
                gt_answers = [self._normalize_answer(ans) for ans in gt_data['answers']]
            elif isinstance(gt_data, dict) and 'answer' in gt_data:
                gt_answers = [self._normalize_answer(gt_data['answer'])]
            else:
                gt_answers = [self._normalize_answer(str(gt_data))]

            # Count how many times the prediction appears in ground truth
            answer_count = gt_answers.count(pred_answer)

            # VQA accuracy: min(answer_count/3, 1)
            vqa_score = min(answer_count / 3.0, 1.0)
            total_score += vqa_score
            num_questions += 1

        return total_score / num_questions if num_questions > 0 else 0.0

    def _compute_exact_match_accuracy(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Compute exact match accuracy.

        Args:
            predictions: Dictionary with question IDs as keys
            ground_truth: Dictionary with question IDs as keys

        Returns:
            Exact match accuracy score
        """
        if not predictions:
            return 0.0

        correct = 0
        total = 0

        for qid in predictions:
            if qid not in ground_truth:
                continue

            pred_answer = self._normalize_answer(predictions[qid]['answer'])

            # Get ground truth answer
            gt_data = ground_truth[qid]
            if isinstance(gt_data, dict) and 'answer' in gt_data:
                gt_answer = self._normalize_answer(gt_data['answer'])
            else:
                gt_answer = self._normalize_answer(str(gt_data))

            if pred_answer == gt_answer:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def _compute_ece(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """
        Compute Expected Calibration Error for VQA predictions.

        Args:
            predictions: Dictionary with question IDs and confidence scores
            ground_truth: Dictionary with question IDs as keys

        Returns:
            ECE score
        """
        if not predictions:
            return 1.0

        # Collect confidence scores and correctness
        confidences = []
        correct_flags = []

        for qid in predictions:
            if qid not in ground_truth:
                continue

            confidence = predictions[qid].get('confidence', 1.0)
            confidences.append(confidence)

            # Check if prediction is correct
            pred_answer = self._normalize_answer(predictions[qid]['answer'])
            gt_data = ground_truth[qid]

            if isinstance(gt_data, dict) and 'answer' in gt_data:
                gt_answer = self._normalize_answer(gt_data['answer'])
            else:
                gt_answer = self._normalize_answer(str(gt_data))

            correct_flags.append(pred_answer == gt_answer)

        if not confidences:
            return 1.0

        # Compute ECE using binning
        return self._compute_ece_from_confidences(confidences, correct_flags)

    def _compute_ece_from_confidences(self, confidences: List[float], correct_flags: List[bool], num_bins: int = 10) -> float:
        """
        Compute ECE from confidence scores and correctness flags.

        Args:
            confidences: List of confidence scores
            correct_flags: List of boolean correctness flags
            num_bins: Number of bins for ECE computation

        Returns:
            ECE score
        """
        if len(confidences) != len(correct_flags) or len(confidences) == 0:
            return 1.0

        # Convert to numpy arrays
        confidences = np.array(confidences)
        correct_flags = np.array(correct_flags, dtype=float)

        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = correct_flags[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _compute_per_question_type_accuracy(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute accuracy per question type.

        Args:
            predictions: Dictionary with question IDs as keys
            ground_truth: Dictionary with question IDs as keys

        Returns:
            Dictionary mapping question types to accuracy scores
        """
        if not self.analyze_question_types:
            return {}

        type_correct = defaultdict(int)
        type_total = defaultdict(int)

        for qid in predictions:
            if qid not in ground_truth:
                continue

            # Determine question type
            question_text = predictions[qid].get('question', '')
            question_type = self._get_question_type(question_text)

            # Check correctness
            pred_answer = self._normalize_answer(predictions[qid]['answer'])
            gt_data = ground_truth[qid]

            if isinstance(gt_data, dict) and 'answer' in gt_data:
                gt_answer = self._normalize_answer(gt_data['answer'])
            else:
                gt_answer = self._normalize_answer(str(gt_data))

            if pred_answer == gt_answer:
                type_correct[question_type] += 1
            type_total[question_type] += 1

        # Compute accuracy per type
        type_accuracy = {}
        for qtype in type_total:
            if type_total[qtype] > 0:
                type_accuracy[qtype] = type_correct[qtype] / type_total[qtype]

        return type_accuracy

    def _compute_per_answer_type_accuracy(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute accuracy per answer type (yes/no, number, other).

        Args:
            predictions: Dictionary with question IDs as keys
            ground_truth: Dictionary with question IDs as keys

        Returns:
            Dictionary mapping answer types to accuracy scores
        """
        if not self.analyze_answer_types:
            return {}

        type_correct = defaultdict(int)
        type_total = defaultdict(int)

        for qid in predictions:
            if qid not in ground_truth:
                continue

            # Get ground truth answer and determine type
            gt_data = ground_truth[qid]
            if isinstance(gt_data, dict) and 'answer' in gt_data:
                gt_answer = str(gt_data['answer'])
            else:
                gt_answer = str(gt_data)

            answer_type = self._get_answer_type(gt_answer)

            # Check correctness
            pred_answer = self._normalize_answer(predictions[qid]['answer'])
            gt_answer_norm = self._normalize_answer(gt_answer)

            if pred_answer == gt_answer_norm:
                type_correct[answer_type] += 1
            type_total[answer_type] += 1

        # Compute accuracy per type
        type_accuracy = {}
        for atype in type_total:
            if type_total[atype] > 0:
                type_accuracy[atype] = type_correct[atype] / type_total[atype]

        return type_accuracy

    def _compute_confidence_statistics(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute statistics about prediction confidence scores.

        Args:
            predictions: Dictionary with question IDs as keys
            ground_truth: Dictionary with question IDs as keys

        Returns:
            Dictionary with confidence statistics
        """
        confidences = []
        correct_confidences = []
        incorrect_confidences = []

        for qid in predictions:
            if qid not in ground_truth:
                continue

            confidence = predictions[qid].get('confidence', 1.0)
            confidences.append(confidence)

            # Check correctness
            pred_answer = self._normalize_answer(predictions[qid]['answer'])
            gt_data = ground_truth[qid]

            if isinstance(gt_data, dict) and 'answer' in gt_data:
                gt_answer = self._normalize_answer(gt_data['answer'])
            else:
                gt_answer = self._normalize_answer(str(gt_data))

            if pred_answer == gt_answer:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)

        if not confidences:
            return {}

        stats = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }

        if correct_confidences:
            stats.update({
                'mean_correct_confidence': np.mean(correct_confidences),
                'std_correct_confidence': np.std(correct_confidences)
            })

        if incorrect_confidences:
            stats.update({
                'mean_incorrect_confidence': np.mean(incorrect_confidences),
                'std_incorrect_confidence': np.std(incorrect_confidences)
            })

        return stats

    def _compute_top_k_accuracy(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any], k: int = 3) -> float:
        """
        Compute top-k accuracy (if multiple predictions are available).

        Args:
            predictions: Dictionary with question IDs as keys
            ground_truth: Dictionary with question IDs as keys
            k: Number of top predictions to consider

        Returns:
            Top-k accuracy score
        """
        # Top-k prediction tensors are not available in this evaluator path.
        return self._compute_exact_match_accuracy(predictions, ground_truth)

    def _get_question_type(self, question: str) -> str:
        """
        Determine the type of a question.

        Args:
            question: Question text

        Returns:
            Question type string
        """
        question_lower = question.lower().strip()

        # Check for question patterns
        for pattern, qtype in self.question_type_mapping.items():
            if question_lower.startswith(pattern):
                return qtype

        # Default type
        return 'other'

    def _get_answer_type(self, answer: str) -> str:
        """
        Determine the type of an answer.

        Args:
            answer: Answer text

        Returns:
            Answer type string
        """
        answer = answer.lower().strip()

        # Yes/No answers
        if answer in ['yes', 'no']:
            return 'yes/no'

        # Number answers
        try:
            float(answer)
            return 'number'
        except ValueError:
            pass

        # Check for common number words
        number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        if answer in number_words:
            return 'number'

        return 'other'

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer text for comparison.

        Args:
            answer: Raw answer text

        Returns:
            Normalized answer text
        """
        if not isinstance(answer, str):
            answer = str(answer)

        # Convert to lowercase and strip whitespace
        answer = answer.lower().strip()

        # Remove articles
        articles = ['a', 'an', 'the']
        words = answer.split()
        words = [word for word in words if word not in articles]

        # Remove punctuation
        import string
        answer = ''.join(words)
        answer = answer.translate(str.maketrans('', '', string.punctuation))

        return answer

    def _extract_ground_truth(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract ground truth from VQA batch.

        Args:
            batch: Input batch

        Returns:
            Dictionary with ground truth answers
        """
        ground_truth = {}
        question_ids = batch.get('question_ids', [])

        # Try different possible answer keys
        answers = None
        for key in ['label', 'targets']:
            if key in batch:
                answers = batch[key]
                break

        if answers is None:
            return ground_truth

        # Handle different answer formats
        for i, qid in enumerate(question_ids):
            if i < len(answers):
                if isinstance(answers[i], (list, tuple)):
                    # Multiple answers
                    ground_truth[qid] = {'answers': answers[i]}
                else:
                    # Single answer
                    ground_truth[qid] = {'answer': answers[i]}

        return ground_truth

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive evaluation summary with VQA-specific information."""
        summary = super().get_evaluation_summary()

        # Add VQA-specific summary information
        vqa_summary = {
            'evaluation_protocol': 'VQA' if self.use_vqa_accuracy else 'Exact Match',
            'question_type_analysis': self.analyze_question_types,
            'answer_type_analysis': self.analyze_answer_types,
            'min_answer_confidence': self.min_answer_confidence,
            'total_tasks': len(self.task_list),
            'task_list': self.task_list
        }

        summary['vqa_config'] = vqa_summary
        return summary

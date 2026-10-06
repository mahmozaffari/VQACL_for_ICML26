"""
Data utilities for VQA continual learning.

This module provides helper functions for dataset initialization,
image directory setup, and data loading coordination.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, List
from torch.utils.data import DataLoader, ConcatDataset

def get_image_directory(dataset: str) -> str:
    """
    Get the image directory path for a specific dataset.

    Args:
        dataset: Name of the dataset (e.g., 'coco', 'vqav2', 'tdiuc')

    Returns:
        Path to the image directory
    """
    # Default image directories for different datasets
    dataset_image_dirs = {
        'vqav2': 'datasets/coco/images',
        'tdiuc': 'datasets/tdiuc/Images',
    }

    # Try to get from environment variables first
    env_var = f"{dataset.upper()}_IMAGE_DIR"
    if env_var in os.environ:
        return os.environ[env_var]

    # Use default mapping
    if dataset.lower() in dataset_image_dirs:
        return dataset_image_dirs[dataset.lower()]

    # Fallback to current directory
    return str(Path.cwd() / 'images' / dataset)

def initialize_dataset(args) -> Tuple[Any, Any, Any]:
    """
    Initialize train, validation, and test datasets.

    Args:
        args: Configuration arguments containing dataset information

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    dataset_name = getattr(args, 'dataset', 'vqav2')

    if dataset_name.lower() in ['vqav2', 'coco']:
        # Use VQA dataset
        from vqa_dataset import VQADataset

        # Initialize datasets
        train_dset = VQADataset(
            splits=getattr(args, 'train_split', 'karpathy_train'),
            vqa_dir=getattr(args, 'vqa_dir', None),
        )

        val_dset = VQADataset(
            splits=getattr(args, 'val_split', 'karpathy_val'),
            vqa_dir=getattr(args, 'vqa_dir', None),
        )

        test_dset = VQADataset(
            splits=getattr(args, 'test_split', 'karpathy_test'),
            vqa_dir=getattr(args, 'vqa_dir', None),
        )

    elif dataset_name.lower() == 'tdiuc':
        # Use TDIUC dataset
        from vqa_dataset import TDIUCDataset

        train_dset = TDIUCDataset(
            splits=getattr(args, 'train_split', 'train'),
            vqa_dir=getattr(args, 'vqa_dir', None),
        )

        val_dset = TDIUCDataset(
            splits=getattr(args, 'val_split', 'val'),
            vqa_dir=getattr(args, 'vqa_dir', None),
        )

        test_dset = TDIUCDataset(
            splits=getattr(args, 'test_split', 'test'),
            vqa_dir=getattr(args, 'vqa_dir', None),
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_dset, val_dset, test_dset

def get_task_data_path(dataset: str, task: str, split: str = 'train') -> str:
    """
    Get the data path for a specific task and split.

    Args:
        dataset: Dataset name
        task: Task name
        split: Data split (train/val/test)

    Returns:
        Path to the task data file
    """
    # This function can be customized based on your data organization
    base_data_dir = os.environ.get('VQA_DATA_DIR', '/path/to/vqa/data')
    return os.path.join(base_data_dir, dataset, task, f'{split}.json')

def create_data_split(data, split_ratio: float = 0.8, seed: int = 42) -> Tuple[Any, Any]:
    """
    Create train/validation split from data.

    Args:
        data: Input data to split
        split_ratio: Ratio for train split (remaining goes to validation)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data)
    """
    import random

    random.seed(seed)

    if isinstance(data, list):
        random.shuffle(data)
        split_idx = int(len(data) * split_ratio)
        return data[:split_idx], data[split_idx:]
    else:
        raise NotImplementedError("Split creation only implemented for lists")

def load_task_questions(task_file: str) -> list:
    """
    Load questions for a specific task from file.

    Args:
        task_file: Path to task question file

    Returns:
        List of questions for the task
    """
    import json

    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with open(task_file, 'r') as f:
        task_data = json.load(f)

    # Handle different task file formats
    if isinstance(task_data, list):
        return task_data
    elif isinstance(task_data, dict) and 'questions' in task_data:
        return task_data['questions']
    else:
        raise ValueError(f"Unknown task file format: {task_file}")

def filter_questions_by_task(questions: list, task_name: str) -> list:
    """
    Filter questions to only include those belonging to a specific task.

    Args:
        questions: List of all questions
        task_name: Name of the task to filter for

    Returns:
        List of questions belonging to the specified task
    """
    # This function assumes questions have a 'task' field
    filtered = []

    for q in questions:
        if isinstance(q, dict):
            if q.get('task') == task_name or q.get('task_type') == task_name:
                filtered.append(q)
        else:
            # Handle other question formats as needed
            filtered.append(q)

    return filtered

def get_dataset_stats(dataset) -> dict:
    """
    Get statistics about a dataset.

    Args:
        dataset: Dataset object

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_samples': len(dataset) if hasattr(dataset, '__len__') else 0,
        'dataset_type': type(dataset).__name__
    }

    # Try to get additional stats if available
    if hasattr(dataset, 'get_stats'):
        stats.update(dataset.get_stats())

    return stats

def setup_data_directories(args) -> None:
    """
    Setup necessary data directories based on configuration.

    Args:
        args: Configuration arguments
    """
    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create subdirectories for different types of outputs
    subdirs = ['checkpoints', 'logs', 'results', 'plots']
    for subdir in subdirs:
        os.makedirs(os.path.join(args.output, subdir), exist_ok=True)

    # Setup cache directory if specified
    if hasattr(args, 'cache_dir'):
        os.makedirs(args.cache_dir, exist_ok=True)

def validate_data_paths(args) -> bool:
    """
    Validate that all required data paths exist.

    Args:
        args: Configuration arguments

    Returns:
        True if all paths are valid, False otherwise
    """
    required_paths = []

    # Add dataset-specific paths
    dataset = getattr(args, 'dataset', '')
    if dataset:
        image_dir = get_image_directory(dataset)
        required_paths.append(image_dir)

    # Check if paths exist
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Warning: Required path does not exist: {path}")
            return False

    return True

def get_collate_fn(task_type: str = 'vqa'):
    """
    Get appropriate collate function for batching data.

    Args:
        task_type: Type of task (vqa, classification, etc.)

    Returns:
        Collate function for DataLoader
    """
    if task_type == 'vqa':
        def vqa_collate_fn(batch):
            """Collate function for VQA batches."""
            # Extract different components
            images = [item['image'] for item in batch if 'image' in item]
            questions = [item['question'] for item in batch if 'question' in item]
            answers = [item['answer'] for item in batch if 'answer' in item]
            question_ids = [item['question_id'] for item in batch if 'question_id' in item]

            # Create batch dictionary
            batch_dict = {}

            if images:
                # Stack images if they're tensors
                import torch
                if torch.is_tensor(images[0]):
                    batch_dict['images'] = torch.stack(images)
                else:
                    batch_dict['images'] = images

            if questions:
                batch_dict['questions'] = questions

            if answers:
                batch_dict['answers'] = answers

            if question_ids:
                batch_dict['question_ids'] = question_ids

            return batch_dict

        return vqa_collate_fn

    else:
        # Return default collate function
        from torch.utils.data.dataloader import default_collate
        return default_collate

def balance_dataset(dataset, max_samples_per_class: Optional[int] = None):
    """
    Balance dataset by limiting samples per class.

    Args:
        dataset: Input dataset
        max_samples_per_class: Maximum samples to keep per class

    Returns:
        Balanced dataset
    """
    # No class-balancing policy is configured; keep the dataset unchanged.
    return dataset

class FiniteMemoryIterator:
    """
    Replacement for cycle() that properly releases references.
    Creates a fresh iterator each epoch instead of keeping infinite reference.

    This fixes memory leaks by:
    1. Not holding references to all batches forever
    2. Allowing garbage collection of old batches
    3. Resetting iterator after each epoch
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.dataloader)

        try:
            return next(self.iterator)
        except StopIteration:
            # Reset iterator for next epoch - this releases references!
            del self.iterator
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

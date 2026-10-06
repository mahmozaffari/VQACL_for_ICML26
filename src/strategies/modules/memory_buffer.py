"""
Memory Buffer for MLP Task-ID Classifier Training
Maintains separate buffers per task with automatic resampling when new tasks are added.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Sampler
import gc
import logging

def release_gpu_memory():
    # put the freed blocks back into CUDA caching allocator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class TaskMemoryBuffer:
    """
    Buffer for storing examples from a single task.
    Stores raw samples including question text, answers, and any other fields.
    """

    def __init__(self, task_id: int, max_size: int, logger: Optional[logging.Logger] = None):
        """
        Initialize task-specific buffer.

        Args:
            task_id: Identifier for the task
            max_size: Maximum number of samples to store
        """
        self.task_id = task_id
        self.max_size = max_size
        self.samples = []
        self.logger = logger or logging.getLogger(f'CL.strategy.moe_ae.buffer.task{task_id}')

    def populate(self, dataloader):
        self.samples = []
        seen = 0

        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Populating Task {self.task_id} Buffer", dynamic_ncols=True, leave=False)
        for batch in pbar:
            batch_size = len(batch['question_id'])

            for i in range(batch_size):
                sample = self._extract_sample(batch, i)
                seen += 1

                if len(self.samples) < self.max_size:
                    self.samples.append(sample)  # Fill reservoir
                else:
                    j = np.random.randint(0, seen)
                    if j < self.max_size:
                        self.samples[j] = sample  # Random replacement

                # Update progress bar
                pbar.set_postfix({
                    'Buffer': f'{len(self.samples)}/{self.max_size}',
                    'Sampled': seen
                })

    def _extract_sample(self, batch: Dict, index: int) -> Dict:
        """Extract a single sample from a batched dictionary."""
        sample = {}

        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                # Extract single tensor and move to CPU
                sample[key] = val[index].detach().cpu().clone()

            elif isinstance(val, (list, tuple)):
                # Extract single element from list/tuple
                sample[key] = val[index]

            elif isinstance(val, np.ndarray):
                # Extract from numpy array
                sample[key] = val #np.copy(val[index])

            else:
                # For scalars or other types, check if it's batch-independent
                # (e.g., dataset name, task_id, etc.)
                # We replicate it for each sample
                sample[key] = val

        return sample

    def resample(self, new_size: int):
        """
        Resample buffer to new size using random sampling.

        Args:
            new_size: New buffer size (should be <= current size)
        """
        if new_size >= len(self.samples):
            self.max_size = new_size
            return  # No need to resample if new size is larger

        # Random sampling without replacement
        indices = np.random.choice(len(self.samples), new_size, replace=False)

        new_samples = [self.samples[i] for i in indices]
        # drop old tensors ASAP
        self.samples = new_samples
        self.max_size = new_size
        gc.collect()
        gc.collect()

        self.logger.info(f"Task {self.task_id} buffer resampled to {len(self.samples)} samples")

    def get_samples(self) -> List[Dict[str, Any]]:
        """Return all samples in buffer."""
        return self.samples

    def get_sample_count(self) -> int:
        """Return number of samples in buffer."""
        return len(self.samples)

    def __len__(self):
        return len(self.samples)

    def clear(self):
        """Clear all samples from buffer."""
        self.samples.clear()
        gc.collect()

class MLPMemoryManager:
    """
    Manager for task-specific memory buffers used in MLP task-id classifier training.
    Maintains a fixed total buffer size across all tasks.
    """

    def __init__(self, total_buffer_size: int = 5000):
        """
        Initialize memory manager.

        Args:
            total_buffer_size: Total memory budget across all tasks
        """
        self.total_buffer_size = total_buffer_size
        self.task_buffers: Dict[int, TaskMemoryBuffer] = {}
        self.task_order = []  # Track order of task addition
        self.logger = logging.getLogger(f'CL.strategy.moe_ae')

    def add_new_task_buffer(self, task_id: int, dataloader):
        """
        Add buffer for new task and rebalance existing buffers.

        Args:
            task_id: Identifier for the task
            dataloader: DataLoader containing samples for this task
        """
        # Calculate new buffer size per task (balanced allocation)
        num_tasks = len(self.task_buffers) + 1
        buffer_size_per_task = self.total_buffer_size // num_tasks

        self.logger.debug(f"\n{'='*60}")
        self.logger.debug(f"Adding Task {task_id} Buffer")
        self.logger.debug(f"Total tasks: {num_tasks}")
        self.logger.debug(f"Buffer size per task: {buffer_size_per_task}")
        self.logger.debug(f"{'='*60}")

        # Resample existing buffers to maintain total budget
        for buf in self.task_buffers.values():
            buf.resample(buffer_size_per_task)

        # Create and populate new task buffer
        new_buffer = TaskMemoryBuffer(task_id, buffer_size_per_task, logger=self.logger)
        new_buffer.populate(dataloader)

        self.task_buffers[task_id] = new_buffer
        self.task_order.append(task_id)

        # Resample existing buffers to maintain total budget

        # Force garbage collection after buffer operations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Print buffer statistics
        self._print_buffer_stats()

    def get_task_samples(self, task_id: int) -> List[Dict[str, Any]]:
        """
        Get samples for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            List of samples for the specified task
        """
        if task_id not in self.task_buffers:
            return []
        return self.task_buffers[task_id].get_samples()

    def get_all_samples(self, include_task_id: bool = True) -> List[Dict[str, Any]]:
        """
        Get all samples from all task buffers.

        Args:
            include_task_id: Whether to add task_id field to each sample

        Returns:
            List of all samples across all tasks
        """
        all_samples = []
        for task_id, buffer in self.task_buffers.items():
            samples = buffer.get_samples()
            if include_task_id:
                for sample in samples:
                    sample['task_id'] = task_id
            all_samples.extend(samples)
        return all_samples

    def get_buffer_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics about buffer contents.

        Returns:
            Dictionary mapping task_id to buffer statistics
        """
        stats = {}
        for task_id, buffer in self.task_buffers.items():
            stats[task_id] = {
                'num_samples': len(buffer),
                'max_size': buffer.max_size,
            }
        return stats

    def _print_buffer_stats(self):
        """Print current buffer statistics."""
        self.logger.debug(f"\nCurrent Buffer Statistics:")
        self.logger.debug(f"{'Task ID':<10} {'Samples':<10} {'Max Size':<10}")
        self.logger.debug(f"{'-'*30}")

        total = 0
        for task_id in self.task_order:
            buffer = self.task_buffers[task_id]
            self.logger.debug(f"{task_id:<10} {len(buffer):<10} {buffer.max_size:<10}")
            total += len(buffer)
        self.logger.debug(f"{'-'*30}")
        self.logger.debug(f"{'Total':<10} {total:<10} {self.total_buffer_size:<10}")
        self.logger.debug("")

    def save_buffers(self, filepath: str):
        """
        Save all buffers to file.

        Args:
            filepath: Path to save buffer checkpoint
        """
        buffer_data = {
            'total_buffer_size': self.total_buffer_size,
            'task_order': self.task_order,
            'task_buffers': {},
        }

        for task_id, buffer in self.task_buffers.items():
            buffer_data['task_buffers'][task_id] = {
                'task_id': buffer.task_id,
                'max_size': buffer.max_size,
                'question_ids': [sample['question_id'] for sample in buffer.samples],
            }

        torch.save(buffer_data, filepath)
        self.logger.info(f"Buffers saved to {filepath}")
        self.logger.info(f"Saved {len(self.task_buffers)} task buffers with total {sum(len(b) for b in self.task_buffers.values())} samples")

    def load_buffers(self, filepath: str):
        """
        Load buffers from file.

        Args:
            filepath: Path to buffer checkpoint
        """
        self.logger.info(f"Loading buffers from {filepath}...")
        buffer_data = torch.load(filepath, map_location='cpu')

        self.total_buffer_size = buffer_data['total_buffer_size']
        self.task_order = buffer_data['task_order']
        self.task_buffers = {}

        for task_id, data in buffer_data['task_buffers'].items():
            buffer = TaskMemoryBuffer(data['task_id'], data['max_size'])

            if 'question_ids' in data:
                buffer.question_ids = data['question_ids']
                buffer.samples = []  # Samples need to be repopulated
                self.logger.info(f"Task {task_id} buffer: loaded {len(buffer.question_ids)} with question IDs only, samples need to be repopulated.")
            elif 'samples' in data:
                buffer.samples = data['samples']
                self.logger.warning(f"Task {task_id}: Loaded {len(buffer.samples)} full samples (old format)")
            else:
                buffer.samples = []

            self.task_buffers[task_id] = buffer

        self.logger.info(f"Buffers loaded successfully!")
        self.logger.info(f"Tasks: {self.task_order}")

        needs_repopulation = any(hasattr(buf, 'question_ids') and buf.question_ids
                                for buf in self.task_buffers.values())
        if needs_repopulation:
            self.logger.info(f"Note: Call repopulate_from_dataloaders() to restore full samples")

        self._print_buffer_stats()

    def get_state(self) -> Dict[str, Any]:
        """Get memory buffer state for checkpointing."""
        return {
            'task_buffers': {
                task_id: {
                    'question_ids': [sample['question_id'] for sample in buffer.samples],
                    'max_size': buffer.max_size,
                } for task_id, buffer in self.task_buffers.items()
            },
            'total_buffer_size': self.total_buffer_size,
            'num_tasks': len(self.task_buffers)
        }

    def load_state(self, state: Dict[str, Any]):
        """Load memory buffer state from checkpoint."""
        from .memory_buffer import TaskMemoryBuffer  # Import if needed

        self.task_buffers = {}
        for task_id, buffer_state in state['task_buffers'].items():
            # Recreate TaskMemoryBuffer objects
            task_buffer = TaskMemoryBuffer(int(task_id), buffer_state['max_size'])

            # Check if this is the new format (question_ids) or old format (full samples)
            if 'question_ids' in buffer_state:
                # New format: only question_ids
                task_buffer.question_ids = buffer_state['question_ids']
                task_buffer.samples = []  # Empty until repopulated
                self.logger.info(f"Task {task_id}: Loaded {len(task_buffer.question_ids)} question_ids (new format)")
            elif 'samples' in buffer_state:
                # Old format: full samples (backward compatibility)
                task_buffer.samples = buffer_state['samples']
                self.logger.warning(f"Task {task_id}: Loaded {len(task_buffer.samples)} full samples (old format)")
                self.logger.warning(f"Consider re-saving checkpoints to use the new space-efficient format")
            else:
                task_buffer.samples = []
                self.logger.warning(f"Task {task_id}: No samples or question_ids found")

            task_buffer.max_size = buffer_state['max_size']
            self.task_buffers[int(task_id)] = task_buffer

        self.total_buffer_size = state['total_buffer_size']
        self.logger.info(f"Loaded memory buffer metadata for {len(self.task_buffers)} tasks")

        # Check if any buffers need repopulation
        needs_repopulation = any(hasattr(buf, 'question_ids') and buf.question_ids
                                for buf in self.task_buffers.values())
        if needs_repopulation:
            self.logger.info(f"Note: Some buffers need to be repopulated from dataloaders using repopulate_from_dataloaders()")

    def repopulate_from_task_dataloader(self, task_id, dataloader: Dict[int, DataLoader]):
        """
        Repopulate buffers from dataloaders using saved question_ids.

        Args:
            task_dataloaders: Dictionary mapping task_id to DataLoader for that task
        """
        if task_id not in self.task_buffers:
            self.logger.error(f"No buffer found for task {task_id}, cannot repopulate!")
            return
        buffer = self.task_buffers[task_id]

        if not hasattr(buffer, 'question_ids') or not buffer.question_ids:
            self.logger.warning(f"Task {task_id} has no question_ids to repopulate, skipping")
            return

        self.logger.info(f"Repopulating buffer for task {task_id} with {len(buffer.question_ids)} samples...")

        # Build question_id to sample mapping
        question_id_set = set(buffer.question_ids)
        buffer.samples = []

        found_count = 0

        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Repopulating Task {task_id} Buffer", dynamic_ncols=True, leave=False)

        for batch in pbar:
            batch_size = len(batch['question_id'])

            for i in range(batch_size):
                q_id = batch['question_id'][i]

                # Check if this question_id is in our saved list
                if q_id in question_id_set:
                    sample = buffer._extract_sample(batch, i)
                    buffer.samples.append(sample)
                    found_count += 1
                    question_id_set.remove(q_id)  # Remove to avoid duplicates

                # Update progress
                pbar.set_postfix({
                    'Found': f'{found_count}/{len(buffer.question_ids)}'
                })

                # Early exit if we found all samples
                if found_count >= len(buffer.question_ids):
                    break

            if found_count >= len(buffer.question_ids):
                break

        if found_count < len(buffer.question_ids):
            self.logger.warning(f"Only found {found_count}/{len(buffer.question_ids)} samples for task {task_id}")
        else:
            self.logger.info(f"Successfully repopulated {found_count} samples for task {task_id}")

        # Remove the restored question ID cache after repopulation.
        delattr(buffer, 'question_ids')

        # Garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_num_tasks(self) -> int:
        """Return number of tasks with buffers."""
        return len(self.task_buffers)

    def has_task(self, task_id: int) -> bool:
        """Check if buffer exists for given task."""
        return task_id in self.task_buffers

    def create_dataloader_from_samples(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """
        Create a DataLoader from given samples.

        Args:
            samples: List of sample dictionaries
            batch_size: Batch size for DataLoader
        Returns:
            DataLoader yielding batches of samples

        """
        samples = self.get_all_samples(include_task_id=True)
        dataset = MemoryBufferDataset(samples)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=dataset.collate_fn,
            pin_memory=False,
            drop_last=False
        )

class MemoryBufferDataset(Dataset):
    # Ignore targets when using this dataset; samples already carry labels.
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        if len(batch) == 0:
            return {}

        batch_dict = {}

        keys = batch[0].keys()

        for key in keys:

            values = [sample[key] for sample in batch]

            if key == 'task_id':
                batch_dict['task_ids'] = torch.tensor(values, dtype=torch.long).unsqueeze(1)

            # Try to stack as tensors if all are tensors
            elif all(isinstance(v, torch.Tensor) for v in values):
                try:
                    batch_dict[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    # If shapes don't match, keep as list
                    batch_dict[key] = values

            # Everything else stays as list
            else:
                batch_dict[key] = values

        return batch_dict

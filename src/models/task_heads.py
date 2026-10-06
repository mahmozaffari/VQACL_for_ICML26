from typing import Dict, List, Optional, Any, Tuple, Set
import torch
import torch.nn as nn
from core.base_task_head import TaskHeadBase


class SeparateTaskHeads(TaskHeadBase):
    """Separate task-specific heads (used in MoE strategy)."""

    def __init__(self, input_size: int, hidden_size: int=None, logger: Optional[Any] = None):
        """
        Initialize separate task heads manager.

        Args:
            input_size: Hidden size of input features
        """
        super().__init__(input_size, hidden_size, logger)
        self.heads = nn.ModuleDict()

        # Unified vocabulary tracking
        self.total_classes = 0

        self.unified_vocab: List[str] = []  # Global answer vocabulary
        self.ans2idx: Dict[str, int] = {}   # answer -> unified index

        self.task_local_to_unified: Dict[int, List[int]] = {}  # task_id -> [unified indices]
        self.task_active_indices: Dict[int, Set[int]] = {}  # task_id -> set of active unified indices

    def add_task(self, task_id: int, task_name: str, num_classes: int,
                 label2ans: Optional[List[str]] = None, device=None) -> None:
        """Add a new task with its own classification head."""
        if str(task_id) in self.heads:
            raise ValueError(f"Task {task_id} already exists")

        # Create task-specific head
        head = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, num_classes)
        )

        # Better initialization
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Scale down final layer
        with torch.no_grad():
            head[3].weight.mul_(0.1)

        if device is not None:
            head = head.to(device, non_blocking=True)

        self.heads[str(task_id)] = head

        # Store task info
        self.task_info[task_id] = {
            'task_name': task_name,
            'num_classes': num_classes,
            'label2ans': label2ans or [str(i) for i in range(num_classes)]
        }

        # Update unified vocabulary tracking
        self._update_unified_vocab(task_id, label2ans)

    def forward(self, features: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward through task-specific head."""
        head_key = str(task_id)
        if head_key not in self.heads:
            raise ValueError(f"No head found for task {task_id}")

        return self.heads[head_key](features)

    def get_answer_from_logits(self, logits: torch.Tensor, task_id: int,
                               return_indices: bool = False) -> Any:
        """Convert logits to answers or indices."""
        pred_indices = torch.argmax(logits, dim=-1)

        if return_indices:
            return pred_indices

        # Convert indices to actual answers
        if task_id not in self.task_info:
            raise ValueError(f"Task {task_id} not found")

        label2ans = self.task_info[task_id]['label2ans']

        # Handle both single predictions and batches
        if pred_indices.dim() == 0:
            return label2ans[pred_indices.item()]
        else:
            return [label2ans[idx.item()] for idx in pred_indices]

    def get_num_classes(self, task_id: int) -> int:
        """Get number of classes for task."""
        if task_id not in self.task_info:
            raise ValueError(f"Task {task_id} not found")
        return self.task_info[task_id]['num_classes']

    def freeze_all_tasks(self) -> None:
        """Freeze all task heads."""
        for tid in self.heads.keys():
            self.freeze_task(int(tid))

    def freeze_task(self, task_id: int) -> None:
        """Freeze parameters for a specific task."""
        head_key = str(task_id)
        if head_key in self.heads:
            for param in self.heads[head_key].parameters():
                param.requires_grad = False

    def unfreeze_task(self, task_id: int) -> None:
        """Unfreeze parameters for a specific task."""
        head_key = str(task_id)
        if head_key in self.heads:
            for param in self.heads[head_key].parameters():
                param.requires_grad = True

    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """Get trainable parameters for specific task."""
        head_key = str(task_id)
        if head_key in self.heads:
            return [p for p in self.heads[head_key].parameters() if p.requires_grad]
        return []

    def get_head_parameters(self) -> List[nn.Parameter]:
        """Get all head parameters."""
        params = []
        for head in self.heads.values():
            params.extend(list(head.parameters()))
        return params

    # =========================================================================
    # UNIFIED SPACE METHODS
    # =========================================================================

    def _update_unified_vocab(self, task_id: int, label2ans: List[str]) -> None:
        """Update unified vocabulary with new answers."""
        local_to_unified = []
        active_indices = set()

        for local_idx, answer in enumerate(label2ans):
            if answer in self.ans2idx:
                # Answer already exists - reuse its position
                unified_idx = self.ans2idx[answer]
            else:
                unified_idx = len(self.unified_vocab)
                self.unified_vocab.append(answer)
                self.ans2idx[answer] = unified_idx

            local_to_unified.append(unified_idx)
            active_indices.add(unified_idx)

        self.task_local_to_unified[task_id] = torch.tensor(local_to_unified, dtype=torch.long)
        self.task_active_indices[task_id] = active_indices

    def local_to_unified_logits(self, local_logits: torch.Tensor, task_id: int, fill_value: float = float('-inf')) -> torch.Tensor:
        """
        Transform local task logits to unified vocabulary space.

        Maps task-specific logits to the globarl unified vocabulary, filling non-task classes with fill_value.

        Args:
            local_logits: logits in local task space [batch_size, task_num_classes]
            task_id: ID of the task
            fill_value: Value to fill for non-task classes

        Returns:
            Unified logits [batch_size, unified_vocab_size]
        """

        if task_id not in self.task_local_to_unified:
            raise ValueError(f"Task {task_id} not found")

        batch_size = local_logits.size(0)
        device = local_logits.device
        dtype = local_logits.dtype

        unified_logits = torch.full((batch_size, len(self.unified_vocab)), fill_value, device=device, dtype=dtype)

        # Get mapping for this task
        mapping = self.task_local_to_unified[task_id].to(device)

        # Scatter local logits to unified postiions
        # unified_logits[:, mapping[i]] = local_logits[:, i] for all i
        unified_logits.scatter_(dim=1, index=mapping.unsqueeze(0).expand(batch_size, -1), src=local_logits)

        return unified_logits

    def local_to_unified_probs(self, local_probs: torch.Tensor, task_id: int, fill_value: float = 0.0) -> torch.Tensor:
        """
        Transform local task probabilities to unified vocabulary space.

        Maps task-specific probabilities to the global unified vocabulary, filling non-task classes with fill_value.

        Args:
            local_probs: probabilities in local task space [batch_size, task_num_classes]
            task_id: ID of the task
            fill_value: Value to fill for non-task classes

        Returns:
            Unified probabilities [batch_size, unified_vocab_size]
        """
        if task_id not in self.task_local_to_unified:
            raise ValueError(f"Task {task_id} not found")

        batch_size = local_probs.size(0)
        device = local_probs.device
        dtype = local_probs.dtype

        unified_probs = torch.full((batch_size, len(self.unified_vocab)), fill_value, device=device, dtype=dtype)

        # Get mapping for this task
        mapping = self.task_local_to_unified[task_id].to(device)

        # Scatter local probabilities to unified positions
        # unified_probs[:, mapping[i]] = local_probs[:, i] for all i
        unified_probs.scatter_(dim=1, index=mapping.unsqueeze(0).expand(batch_size, -1), src=local_probs)

        return unified_probs
    def _get_task_mask(self, task_id: int, device: torch.device = None) -> torch.Tensor:
        """
        Get binary mask for task's acrive answers in unified space.

        Args:
            task_id: ID of the task
            device: Device on which to create the mask tensor (optional)

        Returns:
            Binary mask tensor indicating active answers for the task in unified vocabulary space
        """
        if task_id not in self.task_active_indices:
            raise ValueError(f"Task {task_id} not found")

        mask = torch.zeros(len(self.unified_vocab), dtype=torch.bool)
        active_indices = self.task_active_indices[task_id]
        for idx in active_indices:
            mask[idx] = True

        if device is not None:
            mask = mask.to(device)

        return mask

    def _get_unified_vocab_size(self) -> int:
        return len(self.unified_vocab)

    def get_answer_from_unified_logits(
            self, unified_logits: torch.Tensor, return_indices: bool = False
    ) -> Any:
        """
        Convert unified logits to answers.

        Args:
            unified_logits: Unified logits [batch_size, unified_vocab_size]
            return_indices: If True, return unified indices

        Returns:
            Answers (strings) or unified indices
        """

        pred_indices = torch.argmax(unified_logits, dim=-1)
        if return_indices:
            return pred_indices

        if pred_indices.dim() == 0:
            answers = self.unified_vocab[pred_indices.item()]
        else:
            answers = [self.unified_vocab[idx.item()] for idx in pred_indices]
        return answers

class ExpandingTaskHead(TaskHeadBase):
    """
    Expanding task head that grows with new tasks.
    Maintains a single head that expands to accommodate new tasks.
    """

    def __init__(self, input_size: int, hidden_size:int = None, logger: Optional[Any] = None):
        """
        Initialize expanding task head.

        Args:
            input_size: Hidden size of input features
        """
        super().__init__(input_size, hidden_size, logger)

        self.total_classes = 0

        self.unified_vocab: List[str] = []  # Global answer vocabulary
        self.ans2idx: Dict[str, int] = {}   # answer -> unified index

        # Task-specific mappings
        self.task_vocabs: Dict[int, List[str]] = {}  # task_id -> list of answers
        self.task_local_to_unified: Dict[int, List[int]] = {}  # task_id -> [unified indices]
        self.task_active_indices: Dict[int, Set[int]] = {}  # task_id -> set of active unified indices

        # Initial head (will be replaced as tasks are added)
        self.head = None

    def get_task_class_mask(self, task_id: int) -> torch.Tensor:
        """Get class mask for a specific task over the unified vocabulary."""
        if task_id not in self.task_active_indices:
            raise ValueError(f"Task {task_id} not found")

        mask = self.task_local_to_unified[task_id]
        return mask

    def add_task(self, task_id: int, task_name: str, num_classes: int,
                 label2ans: List[str], device = None) -> None:
        """Add a new task by expanding the head."""
        if task_id in self.task_info:
            raise ValueError(f"Task {task_id} already exists")

        self.logger.info("EXPANDING_HEAD: Adding task %d (%s) with %d classes", task_id, task_name, num_classes)
        # Track vocabulary before adding task
        initial_vocab_size = len(self.unified_vocab)
        self.logger.info("  Initial unified vocab size: %d", initial_vocab_size)

        # Track which answers are new vs existing
        new_answers = []
        local_to_unified = []
        active_indices = set()

        debug_msg = f" Task {task_id} answer mapping:\n"
        for local_idx, answer in enumerate(label2ans):
            if answer in self.ans2idx:
                # Answer already exists - reuse its position
                unified_idx = self.ans2idx[answer]
                debug_msg += f"  Reusing '{answer}' at position {unified_idx}\n"
            else:
                # New answer - add to vocabulary
                unified_idx = len(self.unified_vocab)
                self.unified_vocab.append(answer)
                self.ans2idx[answer] = unified_idx
                new_answers.append(answer)
                debug_msg += f"  Adding '{answer}' at position {unified_idx}\n"

            local_to_unified.append(unified_idx)
            active_indices.add(unified_idx)

        # Store mappings
        self.task_local_to_unified[task_id] = torch.tensor(local_to_unified, dtype=torch.long)
        self.task_active_indices[task_id] = active_indices

        # Expand classifier if needed
        if len(self.unified_vocab) > initial_vocab_size:
            self._expand_classifier(initial_vocab_size, len(self.unified_vocab))

        # Store task info
        self.task_info[task_id] = {
            'task_name': task_name,
            'num_classes': num_classes,
            'label2ans': label2ans
        }

        num_shared = len(label2ans) - len(new_answers)
        print(f"Task {task_id}: {len(new_answers)} new answers, {num_shared} shared -> "
              f"Total vocab: {len(self.unified_vocab)}")

    def _expand_classifier(self, old_size: int, new_size: int):
        """Expand classifier head to accommodate new answers."""

        if self.head is None:
            # First time - create from scratch
            self.head = nn.Sequential(
                nn.Linear(self.input_size, new_size)
            )

            # Initialize
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            with torch.no_grad():
                self.head[0].weight.mul_(0.01)
            return

        # Use in-place expansion via nn.Parameter to preserve optimizer state
        old_weight = self.head[0].weight
        old_bias = self.head[0].bias
        device = old_weight.device

        # Create new parameters with expanded size
        new_weight = nn.Parameter(torch.zeros(new_size, self.input_size, device=device))
        new_bias = nn.Parameter(torch.zeros(new_size, device=device))

        # Initialize new rows
        with torch.no_grad():
            # Copy existing weights
            new_weight[:old_size].copy_(old_weight)
            new_bias[:old_size].copy_(old_bias)

            # Initialize new rows
            nn.init.xavier_uniform_(new_weight[old_size:])
            nn.init.zeros_(new_bias[old_size:])
            new_weight[old_size:].mul_(0.01)

        # Replace parameters directly in the Linear layer
        self.head[0].weight = new_weight
        self.head[0].bias = new_bias
        self.head[0].out_features = new_size

        if self.logger:
            self.logger.info(f"Expanded head from {old_size} to {new_size} outputs")

    def forward(self, features: torch.Tensor, task_id: int = None) -> torch.Tensor:
        """
        Forward through expanding head with task-specific masking.

        Returns logits in local task label space (NOT unified space).
        """
        if self.head is None:
            raise ValueError("No tasks added yet")

        # Get full unified logits
        full_logits = self.head(features)  # [batch_size, unified_vocab_size]

        if task_id is None:
            return full_logits  # Return full logits if no task specified

        # Map to task-specific logits using local_to_unified mapping
        if task_id not in self.task_local_to_unified:
            raise ValueError(f"Task {task_id} not found")

        unified_indices = self.task_local_to_unified[task_id]

        # Optional diagnostic block for inspecting task-specific index mappings.

        if unified_indices.device != full_logits.device:
            unified_indices = unified_indices.to(full_logits.device)
            # Update stored indices to correct device
            self.task_local_to_unified[task_id] = unified_indices

        # Select logits for this task's answers in local order
        task_logits = full_logits[:, unified_indices]  # [batch_size, task_vocab_size]

        return task_logits

    def get_answer_from_logits(self, logits: torch.Tensor, task_id: int = None,
                               return_indices: bool = False) -> Any:
        """Convert task-specific logits to answers or indices."""
        # logits are in local task space (already masked by forward())
        pred_indices = torch.argmax(logits, dim=-1)

        if return_indices:
            return pred_indices

        if task_id is None:
            # Use unified vocab if no task specified
            label2ans = self.unified_vocab
        else:
            # Convert local indices to actual answers
            if task_id not in self.task_info:
                raise ValueError(f"Task {task_id} not found")
            label2ans = self.task_info[task_id]['label2ans']

        # Handle both single predictions and batches
        if pred_indices.dim() == 0:
            return label2ans[pred_indices.item()]
        else:
            return [label2ans[idx.item()] for idx in pred_indices]

    def get_num_classes(self, task_id: int) -> int:
        """Get number of classes for task."""
        if task_id not in self.task_info:
            raise ValueError(f"Task {task_id} not found")
        return self.task_info[task_id]['num_classes']

    def local_to_unified_idx(self, local_idx: int, task_id: int) -> int:
        """Convert local task label to unified vocabulary index."""
        if task_id not in self.task_local_to_unified:
            raise ValueError(f"Task {task_id} not found")
        return self.task_local_to_unified[task_id][local_idx]

    def get_unified_vocab_size(self) -> int:
        """Get size of unified vocabulary."""
        return len(self.unified_vocab)

    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """
        Get trainable parameters for specific task.

        Note: For expanding head, all parameters affect all tasks due to sharing.
        This returns all head parameters.
        """
        if self.head is not None:
            return [p for p in self.head.parameters() if p.requires_grad]
        return []

    def freeze_previous_task_outputs(self, current_task_id: int) -> None:
        """Freeze outputs for previous tasks (specific to expanding head)."""
        if self.head is None:
            return

        # Freeze the output layer weights for previous tasks
        for task_id in self.task_info:
            if task_id < current_task_id:
                start_idx, end_idx = self.task_class_ranges[task_id]
                # Previous-task output weights are not modified here
                pass

    def get_head_parameters(self) -> List[nn.Parameter]:
        """Get all head parameters."""
        if self.head is not None:
            return list(self.head.parameters())
        return []

    def debug_head_state(self, task_id: int = None):
        """Print comprehensive state of the expanding head."""
        print("\n" + "="*80)
        print("EXPANDING HEAD STATE")
        print("="*80)

        print(f"\n1. UNIFIED VOCABULARY:")
        print(f"   Total size: {len(self.unified_vocab)}")
        print(f"   Vocabulary: {self.unified_vocab}")

        print(f"\n2. HEAD ARCHITECTURE:")
        if self.head is not None:
            print(f"   Head output size: {self.head[-1].out_features}")
            print(f"   Head structure: {self.head}")

        print(f"\n3. TASK MAPPINGS:")
        for tid in sorted(self.task_info.keys()):
            print(f"\n   Task {tid} ({self.task_info[tid]['task_name']}):")
            print(f"   - Num classes: {self.task_info[tid]['num_classes']}")
            print(f"   - Label2ans: {self.task_info[tid]['label2ans']}")
            print(f"   - Local to unified: {self.task_local_to_unified[tid]}")
            print(f"   - Active indices: {sorted(self.task_active_indices[tid])}")

        if task_id is not None and task_id in self.task_info:
            print(f"\n4. TASK {task_id} SPECIFIC INFO:")
            vocab = self.task_info[task_id]['label2ans']
            mapping = self.task_local_to_unified[task_id]
            print(f"   Local -> Unified -> Answer:")
            for local_idx, (answer, unified_idx) in enumerate(zip(vocab, mapping)):
                print(f"   {local_idx} -> {unified_idx} -> '{answer}'")

        print("="*80 + "\n")

    def debug_forward_pass(self, features: torch.Tensor, task_id: int = None):
        """Debug a forward pass through the head."""
        print("\n" + "="*80)
        print(f"FORWARD PASS DIAGNOSTICS (task_id={task_id})")
        print("="*80)

        # Get full unified logits
        full_logits = self.head(features)
        print(f"\n1. FULL UNIFIED LOGITS:")
        print(f"   Shape: {full_logits.shape}")
        print(f"   Sample logits [0]: {full_logits[0].detach().cpu().numpy()}")

        # Show top predictions from unified space
        top_k = min(5, full_logits.size(-1))
        top_vals, top_idx = torch.topk(full_logits[0], k=top_k)
        print(f"\n   Top {top_k} unified predictions:")
        for val, idx in zip(top_vals, top_idx):
            answer = self.unified_vocab[idx.item()]
            print(f"   - Index {idx.item()}: '{answer}' (logit: {val.item():.4f})")

        if task_id is not None and task_id in self.task_local_to_unified:
            # Get task-specific logits
            unified_indices = self.task_local_to_unified[task_id]
            task_logits = full_logits[:, unified_indices]

            print(f"\n2. TASK {task_id} SPECIFIC LOGITS:")
            print(f"   Shape: {task_logits.shape}")
            print(f"   Unified indices used: {unified_indices[:10]}")
            print(f"   Sample task logits [0]: {task_logits[0].detach().cpu().numpy()}")

            # Show predictions
            pred_local_idx = torch.argmax(task_logits[0])
            pred_unified_idx = unified_indices[pred_local_idx.item()]
            pred_answer = self.task_info[task_id]['label2ans'][pred_local_idx.item()]

            print(f"\n3. TASK {task_id} PREDICTION:")
            print(f"   Local index: {pred_local_idx.item()}")
            print(f"   Unified index: {pred_unified_idx}")
            print(f"   Answer: '{pred_answer}'")

            # Check if this matches the unified prediction
            unified_pred_idx = torch.argmax(full_logits[0])
            unified_pred_answer = self.unified_vocab[unified_pred_idx.item()]

            print(f"\n4. COMPARISON:")
            print(f"   Unified prediction: index {unified_pred_idx.item()} -> '{unified_pred_answer}'")
            print(f"   Task-specific prediction: index {pred_unified_idx} -> '{pred_answer}'")

            if unified_pred_idx.item() == pred_unified_idx:
                print(f"   Predictions agree")
            else:
                print(f"   Predictions differ")
                print(f"   Unified logit: {full_logits[0, unified_pred_idx].item():.4f}")
                print(f"   Task logit: {full_logits[0, pred_unified_idx].item():.4f}")
                print(f"   Difference: {(full_logits[0, unified_pred_idx] - full_logits[0, pred_unified_idx]).item():.4f}")

        print("="*80 + "\n")
        return full_logits

    def debug_prediction(self, logits: torch.Tensor, task_id: int = None):
        """Debug get_answer_from_logits conversion."""
        print("\n" + "="*80)
        print(f"PREDICTION CONVERSION (task_id={task_id})")
        print("="*80)

        print(f"\n1. INPUT LOGITS:")
        print(f"   Shape: {logits.shape}")
        print(f"   Sample [0]: {logits[0].detach().cpu().numpy()}")

        pred_indices = torch.argmax(logits, dim=-1)
        print(f"\n2. ARGMAX PREDICTION INDICES:")
        print(f"   Indices: {pred_indices.detach().cpu().numpy()}")

        if task_id is None:
            print(f"\n3. USING UNIFIED VOCAB (task_id=None):")
            print(f"   Vocab size: {len(self.unified_vocab)}")
            answers = [self.unified_vocab[idx.item()] for idx in pred_indices]
        else:
            print(f"\n3. USING TASK {task_id} VOCAB:")
            label2ans = self.task_info[task_id]['label2ans']
            print(f"   Vocab size: {len(label2ans)}")
            print(f"   Vocab: {label2ans[:20]}")
            answers = [label2ans[idx.item()] for idx in pred_indices]

        print(f"\n4. FINAL ANSWERS:")
        for i, (idx, ans) in enumerate(zip(pred_indices[:5], answers[:5])):
            print(f"   Sample {i}: index {idx.item()} -> '{ans}'")

        print("="*80 + "\n")
        return answers

    def debug_weight_analysis(self):
        """Analyze the weights of the expanded head."""
        print("\n" + "="*80)
        print("HEAD WEIGHT ANALYSIS")
        print("="*80)

        if self.head is None:
            print("No head initialized!")
            return

        # Get the final linear layer
        final_layer = self.head[-1]
        weight = final_layer.weight.data  # [out_features, in_features]
        bias = final_layer.bias.data if final_layer.bias is not None else None

        print(f"\n1. WEIGHT MATRIX:")
        print(f"   Shape: {weight.shape}")
        print(f"   Norm per output: {weight.norm(dim=1).detach().cpu().numpy()}")

        if bias is not None:
            print(f"\n2. BIAS VECTOR:")
            print(f"   Shape: {bias.shape}")
            print(f"   Values: {bias.detach().cpu().numpy()}")

        print(f"\n3. TASK-SPECIFIC WEIGHT ANALYSIS:")
        for task_id in sorted(self.task_info.keys()):
            indices = self.task_local_to_unified[task_id]
            task_weights = weight[indices]
            task_bias = bias[indices] if bias is not None else None

            print(f"\n   Task {task_id} ({self.task_info[task_id]['task_name']}):")
            print(f"   - Indices: {indices}")
            print(f"   - Weight norms: {task_weights.norm(dim=1).detach().cpu().numpy()}")
            if task_bias is not None:
                print(f"   - Bias values: {task_bias.detach().cpu().numpy()}")
            print(f"   - Mean weight: {task_weights.mean().item():.6f}")
            print(f"   - Std weight: {task_weights.std().item():.6f}")

        print("="*80 + "\n")

    def unfreeze_head(self) -> None:
        """Unfreeze all head parameters."""
        if self.head is not None:
            for param in self.head.parameters():
                param.requires_grad = True

class JointTaskHead(TaskHeadBase):
    """
    Joint task head that shares a single head across all tasks.
    Uses a unified answer space - all tasks share the same output vocabulary.
    """

    def __init__(self, input_size: int, hidden_size: int = None, logger: Optional[Any] = None, unified_vocab: List[str]=None):
        """
        Initialize joint task head.

        Args:
            input_size: Hidden size of input features
            unified_vocab_size: Total size of unified vocabulary across all tasks
            logger: Logger instance
        """
        super().__init__(input_size, hidden_size, logger)
        self.unified_vocab_size = len(unified_vocab) if unified_vocab is not None else 0
        self.unified_vocab: List[str] = unified_vocab  # Unified answer vocabulary
        self.ans2idx: Dict[str, int] = {}  # answer -> unified index

        self.task_vocabs: Dict[int, List[str]] = {}  # task_id -> list of answers
        self.task_local_to_unified: Dict[int, List[int]] = {}  # task_id -> [unified indices]

        self.task_to_unified_mapping: Dict[int, Dict[int, int]] = {}  # task_id -> {local_idx: unified_idx}

        self.task_active_indices: Dict[int, Set[int]] = {}  # task_id -> set of active unified indices

        if unified_vocab is None:
            self.head = None
        else:

            self._initialize_head(unified_vocab)

        if self.logger:
            self.logger.info(f"Initialized JointTaskHead with unified vocab size: {self.unified_vocab_size}")

    def _initialize_head(self, unified_vocab: List[str]) -> None:
        """Initialize head with given unified vocabulary."""
        self.unified_vocab = unified_vocab
        self.unified_vocab_size = len(unified_vocab)
        self.ans2idx = {ans: idx for idx, ans in enumerate(unified_vocab)}

        self.head = nn.Sequential(
            nn.Linear(self.input_size, self.unified_vocab_size)
        )

        # Initialize weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Scale down final layer for better initialization

        if self.logger:
            self.logger.info(f"Initialized JointTaskHead with unified vocab size: {self.unified_vocab_size}")

    def add_task(self, task_id: int, task_name : str, num_classes:int, label2ans: List[str], device=None) -> None:
        """
        Add a new task to the joint head by creating mappings to unified vocabulary.

        Args:
            task_id: Task identifier
            task_name: Task name
            num_classes: Number of classes for this task
            label2ans: Task-specific label to answer mapping (Local vocabulary) [List]
        """
        if task_id in self.task_info:
            raise ValueError(f"Task {task_id} already exists")

        self.logger.info(f"JOINT HEAD: Adding task {task_id} ({task_name}) with {num_classes} classes")

        local_to_unified = []
        active_indices = set()

        debug_msg = f"Adding Task {task_id} ({task_name}):\n"

        for _, answer in enumerate(label2ans):
            if answer in self.ans2idx:
                # Answer already exists - reuse its position
                unified_idx = self.ans2idx[answer]
                debug_msg += f"  Reusing '{answer}' at position {unified_idx}\n"
            else:
                # New answer - add to vocabulary
                raise NotImplementedError("Unified vocab must be pre-initialized with all possible answers.")

            local_to_unified.append(unified_idx)
            active_indices.add(unified_idx)

        self.task_local_to_unified[task_id] = torch.tensor(local_to_unified, dtype=torch.long)
        self.task_active_indices[task_id] = active_indices

        # Store task info
        self.task_info[task_id] = {
            'task_name': task_name,
            'num_classes': num_classes,
            'label2ans': label2ans
        }

        self.logger.info(f"Added task {task_id} ({task_name}) to JointTaskHead. "
                           f"Unified vocab size now: {len(self.unified_vocab)}")

    def forward(self, features: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward through joint head.

        Args:
            features: Input features [batch_size, input_size]
            task_id: Task ID (optional, for task-specific masking if needed)

        Returns:
            Logits over the unified vocabulary [batch_size, unified_vocab_size]
        """
        # Get logits for entire unified vocabulary
        unified_logits = self.head(features)

        # For joint training, we return all logits without masking
        # The targets should be in the unified label space
        return unified_logits

    def get_answer_from_logits(self, logits: torch.Tensor, task_id: Optional[int] = None,
                               return_indices: bool = False) -> Any:
        """
        Convert logits to answers.

        Args:
            logits: Logits over unified vocabulary [batch_size, unified_vocab_size]
            task_id: Task ID (optional, used for task-specific vocabularies)
            return_indices: If True, return indices instead of answer strings

        Returns:
            Answer strings or indices
        """
        # Get predictions in unified space
        pred_unified_indices = torch.argmax(logits, dim=-1)

        if return_indices:
            return pred_unified_indices

        # Convert unified indices to answer strings
        answers = []
        for unified_idx in pred_unified_indices:
            idx = unified_idx.item()
            if idx < len(self.unified_vocab):
                answers.append(self.unified_vocab[idx])
            else:
                answers.append(f"<UNK_{idx}>")

        # Return single answer or list
        if len(answers) == 1:
            return answers[0]
        return answers

    def convert_targets_to_unified(self, targets: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Convert task-specific targets to unified label space.

        Args:
            targets: Task-specific targets [batch_size, num_task_classes]
            task_id: Task identifier

        Returns:
            Unified targets [batch_size, unified_vocab_size]
        """
        if task_id not in self.task_to_unified_mapping:
            raise ValueError(f"Task {task_id} not found in joint head")

        batch_size = targets.size(0)
        unified_targets = torch.zeros(batch_size, self.unified_vocab_size,
                                     device=targets.device, dtype=targets.dtype)

        mapping = self.task_to_unified_mapping[task_id]

        # Map each task-specific class to its unified index
        for local_idx, unified_idx in mapping.items():
            if local_idx < targets.size(1):
                unified_targets[:, unified_idx] = targets[:, local_idx]

        return unified_targets

    def get_num_classes(self, task_id: Optional[int] = None) -> int:
        """Get number of classes."""
        if task_id is None:
            return self.unified_vocab_size

        if task_id not in self.task_info:
            raise ValueError(f"Task {task_id} not found")
        return self.task_info[task_id]['num_classes']

    def get_unified_vocab_size(self) -> int:
        """Get size of unified vocabulary."""
        return len(self.unified_vocab)

    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """
        Get trainable parameters for specific task.
        For joint head, all tasks share the same parameters.
        """
        return [p for p in self.head.parameters() if p.requires_grad]

    def get_head_parameters(self) -> List[nn.Parameter]:
        """Get all head parameters."""
        if self.head is not None:
            return list(self.head.parameters())
        return []

    def freeze_head(self) -> None:
        """Freeze all head parameters."""
        for param in self.head.parameters():
            param.requires_grad = False

    def unfreeze_head(self) -> None:
        """Unfreeze all head parameters."""
        for param in self.head.parameters():
            param.requires_grad = True

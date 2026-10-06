"""
FLAVA wrapper with per-task LoRA adapters and answer heads.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import logging

# Import model components
try:
    from transformers.models.vilt.modeling_vilt import (
        ViltSelfAttention,
        ViltSelfOutput,
        ViltAttention,
        ViltIntermediate,
        ViltOutput,
        ViltLayer,
        ViltEncoder,
        ViltModel,
        ViltEmbeddings,
        ViltPooler
    )
except ImportError:
    # Fallback for different transformers versions
    from transformers import (
        ViltModel
    )

from core.base_model import BaseModelWrapper, ModelOutput
from core.base_task_head import TaskHeadBase
from models.task_heads import SeparateTaskHeads
from models.vilt_wrapper import ViLTWrapper

class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) linear layer."""

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 32.0):
        super().__init__()

        # Store original linear layer as a proper submodule
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0

        # Task-specific LoRA adapters
        self.task_lora_adapters: nn.ModuleDict = nn.ModuleDict()
        self.current_task_id: Optional[int] = None

    def add_task_adapter(self, task_id: int):
        """Add LoRA adapter for a specific task."""
        task_key = str(task_id)

        if task_key not in self.task_lora_adapters:
            adapter = nn.ModuleDict({
                'lora_A': nn.Linear(self.original_linear.in_features, self.rank, bias=False),
                'lora_B': nn.Linear(self.rank, self.original_linear.out_features, bias=False)
            })

            # Initialize LoRA parameters correctly
            nn.init.kaiming_uniform_(adapter['lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(adapter['lora_B'].weight)

            # Better initialization
            # Scale down initialization
            with torch.no_grad():
                adapter['lora_A'].weight.mul_(0.1)  # Scale down

            device = self.original_linear.weight.device
            adapter = adapter.to(device)

            self.task_lora_adapters[task_key] = adapter

    def set_current_task(self, task_id: Optional[int]):
        """Set current task for routing."""
        self.current_task_id = task_id

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Original transformation
        output = self.original_linear(x)

        # Add task-specific LoRA adaptation
        target_task_id = task_id if task_id is not None else self.current_task_id

        if target_task_id is not None and self.rank > 0:
            task_key = str(target_task_id)
            if task_key in self.task_lora_adapters:
                adapter = self.task_lora_adapters[task_key]
                # Correct LoRA: Wx + s * B(A(x))
                lora_output = adapter['lora_B'](adapter['lora_A'](x)) * self.scaling
                output = output + lora_output

        return output

def create_lora_enabled_model(original_model: nn.Module, lora_rank: int = 8, lora_alpha: float = 32.0, target_modules: List[str] = ['ffn'], logger: Optional[logging.Logger] = None) -> nn.Module:
    """
    Create a LoRA-enabled version of a ViLT model by replacing linear layers.

    This function modifies the model in-place to add LoRA adapters.

    Args:
        original_model: Original ViLT model
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: Which module types to apply LoRA to ['attention', 'ffn', 'all', 'pooler']
    """

    if logger is None:
        logger = logging.getLogger('CL.model.vilt_wrapper')

    def should_apply_lora(module_path: str, target_modules: List[str]) -> bool:
        """Determine if LoRA should be applied to this module based on target_modules."""

        if 'all' in target_modules:
            return True

        # Attention modules
        attention_keywords = ['attention.query', 'attention.key', 'attention.value', 'attention.output.dense']
        if 'attention' in target_modules:
            if any(keyword in module_path for keyword in attention_keywords):
                return True

        # FFN modules
        ffn_keywords = ['intermediate.dense'] #, 'output.dense']
        if 'ffn' in target_modules:
            if any(keyword in module_path for keyword in ffn_keywords):
                return True

        # Pooler
        if 'pooler' in target_modules and 'pooler.dense' in module_path:
            return True

        return False

    def replace_linear_with_lora(module, name_prefix=""):
        """Recursively replace selected Linear layers with LoRA-enabled versions."""
        replaced_count = 0

        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            if isinstance(child, nn.Linear):
                if should_apply_lora(full_name, target_modules):
                    # Replace this linear layer with LoRA version
                    lora_layer = LoRALinear(child, rank=lora_rank, alpha=lora_alpha)
                    setattr(module, name, lora_layer)
                    logger.info(f"  Replaced {full_name} with LoRA layer")
                    replaced_count += 1
                else:
                    logger.info(f"  Skipped {full_name} (not in target modules)")
            else:
                # Recursively process child modules
                child_count = replace_linear_with_lora(child, full_name)
                replaced_count += child_count

        return replaced_count

    logger.info(f"Creating LoRA-enabled model (rank={lora_rank}, alpha={lora_alpha})")
    logger.info(f"Target modules: {target_modules}")

    # Replace linear layers with LoRA versions
    total_replaced = replace_linear_with_lora(original_model)

    logger.info(f"LoRA-enabled model created: {total_replaced} layers replaced")

    # Add LoRA management methods to the model
    def add_task_adapter(self, task_id: int):
        """Add LoRA adapters for a specific task to all LoRA layers."""
        def add_to_module(module):
            for child in module.children():
                if isinstance(child, LoRALinear):
                    child.add_task_adapter(task_id)
                else:
                    add_to_module(child)
        add_to_module(self)

    def set_current_task(self, task_id: Optional[int]):
        """Set current task for all LoRA layers."""
        def set_in_module(module):
            for child in module.children():
                if isinstance(child, LoRALinear):
                    child.set_current_task(task_id)
                else:
                    set_in_module(child)
        set_in_module(self)

    def get_task_lora_parameters(self, task_id: int) -> List[nn.Parameter]:
        """Get all LoRA parameters for a specific task."""
        params = []
        task_key = str(task_id)

        def collect_from_module(module):
            for child in module.children():
                if isinstance(child, LoRALinear):
                    if task_key in child.task_lora_adapters:
                        adapter = child.task_lora_adapters[task_key]
                        params.extend([p for p in adapter.parameters() if p.requires_grad])
                else:
                    collect_from_module(child)

        collect_from_module(self)
        return params

    # Bind methods to the model instance
    import types
    original_model.add_task_adapter = types.MethodType(add_task_adapter, original_model)
    original_model.set_current_task = types.MethodType(set_current_task, original_model)
    original_model.get_task_lora_parameters = types.MethodType(get_task_lora_parameters, original_model)

    logger.info(f"LoRA-enabled model created successfully")
    return original_model

class FlavaWrapper(BaseModelWrapper):
    """
    Clean Flava wrapper with fixed initialization order - NO monkey-patching.
    """

    def __init__(self, base_model: nn.Module, processor, args, task_head_type: str = 'separate', **kwargs):

        # Add logger instance
        self.logger = logging.getLogger(f'CL.model.vilt_wrapper')

        # LoRA configuration
        self.use_lora = getattr(args, 'lora_r', 0) and getattr(args, 'lora_r', 0) > 0
        self.lora_rank = getattr(args, 'lora_r', 8)
        self.lora_alpha = getattr(args, 'lora_alpha', 32.0)

        self.lora_target_modules = getattr(args, 'lora_target_modules', ['ffn'])

        # Create LoRA-enabled model if needed
        if self.use_lora:
            # Create a copy of the model for LoRA modification
            import copy
            lora_model = copy.deepcopy(base_model)
            lora_model = create_lora_enabled_model(lora_model, self.lora_rank, self.lora_alpha, self.lora_target_modules, logger=self.logger)
        else:
            lora_model = base_model

        # Initialize parent with task_head_type
        super().__init__(lora_model, args, task_head_type=task_head_type, **kwargs)

        # Initialize task head manager based on type
        self.task_head_manager = self._initialize_task_head_manager(task_head_type)

        self.logger.info(f"Initialized ViLT wrapper with {task_head_type} task heads")

        # Now call parent initialization with the LoRA-enabled model

        # Store additional components
        self.processor = processor

        # Validate model structure
        self._validate_and_setup_model()

        # Freezing
        self.freeze_base_model_flag = getattr(args, 'freeze_base', False)
        if self.freeze_base_model_flag:
            self._freeze_base_parameters()

        self.logger.info(f"ViLT wrapper initialized with {'LoRA' if self.use_lora else 'standard'} model")

    def to(self, device):
        """Override to() to properly update device attribute."""
        # Call parent's to() method to move parameters
        super().to(device)
        # Update our device attribute
        self.device = device
        # Return self for chaining
        return self

    def _validate_and_setup_model(self):
        """Validate the model structure."""
        # Check if it's a ViLT model with QA head or just encoder
        if hasattr(self.base_model, 'vilt'):
            self.vilt_model = self.base_model.vilt
            self.use_vilt_for_qa = True
            self.logger.info("Using ViltForQuestionAnswering model")
        else:
            self.use_vilt_for_qa = False
            self.logger.info("Using ViltModel directly")

        # Check for LoRA layers
        if self.use_lora:
            lora_count = self._count_lora_layers()
            self.logger.info(f"Found {lora_count} LoRA-enabled layers")

            if lora_count == 0:
                self.logger.warning("LoRA is enabled, but no LoRA layers were found")

    def _count_lora_layers(self) -> int:
        """Count the number of LoRA-enabled layers."""
        count = 0
        def count_in_module(module):
            nonlocal count
            for child in module.children():
                if isinstance(child, LoRALinear):
                    count += 1
                else:
                    count_in_module(child)

        count_in_module(self.base_model)
        return count

    def add_task(self, task_id: int, task_name: str, num_classes: int, label2ans: Optional[Dict[int, str]] = None) -> None: #, create_head: bool = True) -> None:
        """
        Add a new task with LoRA adapters if enabled.

        Args:
            task_id: Task identifier
            task_name: Task name
            num_classes: Number of classes
            label2ans: Label to answer mapping
        """
        self.logger.info(f"Adding task {task_id}: {task_name} with {num_classes} classes")

        # Add task head
        super().add_task(task_id, task_name, num_classes, label2ans) #, create_head=create_head)

        # Add LoRA adapters if using LoRA
        if self.use_lora and hasattr(self.base_model, 'add_task_adapter'):
            self.base_model.add_task_adapter(task_id)

            # Count parameters for this task
            if hasattr(self.base_model, 'get_task_lora_parameters'):
                lora_params = self.base_model.get_task_lora_parameters(task_id)
                param_count = sum(p.numel() for p in lora_params)
                self.logger.info(f"Added {param_count:,} LoRA parameters for task {task_id}")
            else:
                self.logger.info(f"Added LoRA adapters for task {task_id}")

    def set_current_task(self, task_id: int) -> None:
        """Set current task for LoRA routing."""
        super().set_current_task(task_id)

        # Set task routing in LoRA model
        if self.use_lora and hasattr(self.base_model, 'set_current_task'):
            self.base_model.set_current_task(task_id)

    def get_task_lora_parameters(self, task_id: int) -> List[nn.Parameter]:
        """Get LoRA parameters for a specific task."""
        if self.use_lora and hasattr(self.base_model, 'get_task_lora_parameters'):
            return self.base_model.get_task_lora_parameters(task_id)
        return []

    def freeze_other_task_lora(self, current_task_id: int):
        """Freeze LoRA adapters for all tasks except the current one."""
        if not self.use_lora:
            return

        self.logger.info(f"Freezing LoRA for all tasks except {current_task_id}")

        # Freeze all LoRA parameters first
        for module in self.base_model.modules():
            if hasattr(module, 'task_lora_adapters'):
                for task_key, adapter in module.task_lora_adapters.items():
                    for param in adapter.parameters():
                        param.requires_grad = False

        # Unfreeze current task LoRA
        current_task_key = str(current_task_id)
        for module in self.base_model.modules():
            if hasattr(module, 'task_lora_adapters'):
                if current_task_key in module.task_lora_adapters:
                    for param in module.task_lora_adapters[current_task_key].parameters():
                        param.requires_grad = True

    def _get_hidden_size(self) -> int:
        """Get hidden size from model config."""
        return getattr(self.base_model.config, 'hidden_size', 768)

    def _initialize_wrapper_components(self):
        """Initialize wrapper components."""
        pass

    def _extract_features(self, batch: Dict[str, Any], task_id: int = None) -> torch.Tensor:
        """Extract features using the LoRA model."""
        inputs = self._prepare_vilt_inputs(batch)

        with torch.set_grad_enabled(self.training):
            try:
                if self.use_vilt_for_qa:
                    vilt_outputs = self.vilt_model(**inputs, return_dict=True)
                else:
                    vilt_outputs = self.base_model(**inputs, return_dict=True)

                if hasattr(vilt_outputs, 'multimodal_embeddings') and vilt_outputs.multimodal_embeddings is not None:
                    # Use multimodal features when the model exposes them.
                    features = vilt_outputs.multimodal_embeddings[:,0]
                else:
                    raise ValueError("Could not extract features from model outputs")

                return features

            except Exception as e:
                self.logger.error(f"Feature extraction failed: {e}")
                self.logger.error(f"Model type: {type(self.base_model)}")
                self.logger.error(f"Inputs keys: {inputs.keys()}")
                raise

    def _prepare_vilt_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for ViLT model."""

        if 'pixel_values' in batch and 'input_ids' in batch:
            # Already processed inputs
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items() if k in ['pixel_values', 'input_ids', 'attention_mask', 'token_type_ids']}

        images = batch.get('images', batch.get('image', None))
        questions = batch.get('questions', batch.get('question', batch.get('sent', [])))

        if images is None or not questions:
            raise ValueError("Missing images or questions in batch")

        inputs = self.processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def _freeze_base_parameters(self):
        """Freeze base model parameters except LoRA adapters."""
        frozen_count = 0
        unfrozen_count = 0

        for name, param in self.base_model.named_parameters():
            # Don't freeze LoRA parameters or task heads
            if ('lora_A' in name or 'lora_B' in name or 'task_heads' in name):
                param.requires_grad = True
                unfrozen_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        self.logger.info(f"Frozen {frozen_count:,} base params, kept {unfrozen_count:,} trainable params")

    def get_trainable_parameters(self, task_id: Optional[int] = None) -> List[nn.Parameter]:
        """Get trainable parameters including LoRA adapters."""
        params = []

        if task_id is not None:
            # Get task-specific parameters

            # 1. Task head parameters
            if self.task_head_manager is not None:
                params.extend(self.task_head_manager.get_task_parameters(task_id))

            # 2. LoRA parameters for this task
            if self.use_lora:
                params.extend(self.get_task_lora_parameters(task_id))

            # 3. Base model parameters (if first task and not frozen)
            if task_id == 0 and not getattr(self.args, 'freeze_base', False):
                for name, param in self.base_model.named_parameters():
                    if 'lora' not in name.lower() and param.requires_grad:
                        params.append(param)
        else:
            # Get all trainable parameters
            params = [p for p in self.parameters() if p.requires_grad]

        return params


# Self-test for the LoRA wrapper
def test_clean_lora_implementation():
    """Self-test for the LoRA wrapper."""
    print("Testing LoRA implementation")
    print("=" * 60)

    try:
        from transformers import ViltModel, ViltProcessor

        # Create test model
        print("0. Loading base ViLT model...")
        base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

        # Mock args
        class MockArgs:
            lora_r = 8
            lora_alpha = 32.0
            freeze_base = True

        args = MockArgs()

        # Create wrapper
        print("1. Creating clean LoRA wrapper...")
        wrapper = ViLTWrapper(base_model, processor, args)

        # Add multiple tasks
        print("2. Adding multiple tasks...")
        for task_id in range(3):
            wrapper.add_task(task_id=task_id, task_name=f"task_{task_id}", num_classes=100 + task_id * 10)
            print(f"   Added task {task_id}")

        # Test parameter isolation between tasks
        print("3. Testing parameter isolation...")

        for current_task in range(3):
            wrapper.freeze_other_task_lora(current_task)

            # Count trainable parameters for each task
            task_params = wrapper.get_task_lora_parameters(current_task)
            trainable_count = sum(p.numel() for p in task_params if p.requires_grad)
            frozen_count = sum(p.numel() for p in task_params if not p.requires_grad)

            print(f"   Task {current_task}: {trainable_count:,} trainable, {frozen_count:,} frozen LoRA params")

            if trainable_count == 0:
                print(f"   No trainable LoRA parameters for task {current_task}")
                return False

        # Test task routing
        print("4. Testing task routing...")
        for task_id in range(3):
            wrapper.set_current_task(task_id)

            # Verify routing propagated to LoRA layers
            routing_verified = False

            def check_routing(module):
                nonlocal routing_verified
                for child in module.children():
                    if isinstance(child, LoRALinear):
                        if child.current_task_id == task_id:
                            routing_verified = True
                            return
                    else:
                        check_routing(child)

            check_routing(wrapper.base_model)

            if routing_verified:
                print(f"   Task {task_id} routing verified")
            else:
                print(f"   Task {task_id} routing failed")
                return False

        print("Detailed LoRA functionality test passed!")
        return True

    except Exception as e:
        print(f"Detailed test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

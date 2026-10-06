"""
Main training script with automatic model configuration.

This version automatically configures whether to use task-specific classifiers
based on the continual learning strategy being used.
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
import json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Unified Continual Learning Framework')

    # Framework configuration
    parser.add_argument('--strategy', type=str, default='moe',
                       choices=['naive', 'moe', 'moe_utility_router', 'moe_router'],
                       help='Continual learning strategy')
    parser.add_argument('--model_name', type=str, default='vilt',
                       choices=['vilt', 'flava'],
                       help='Base model architecture')
    parser.add_argument('--init_unified_answer_space', action='store_true', default=False,
                       help='Initialize unified answer space across tasks')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='vqav2',
                       choices=['vqav2', 'tdiuc'],
                       help='Dataset to use')
    parser.add_argument('--vqa_dir', type=str, default='datasets/vqa',
                        help='Path to the VQA dataset directory.')

    parser.add_argument('--use_h5', action='store_true', default=False,
                        help='Use H5 files for image features.')
    parser.add_argument('--h5_path', type=str, default=None,
                        help='Path to the H5 files directory (required if --use_h5 is set).')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per task')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='linear',
                       choices=['linear', 'cosine', 'constant'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm')

    # Model configuration
    parser.add_argument('--use_lora', action='store_true', default=False,
                       help='Use LoRA (Low-Rank Adaptation)')
    parser.add_argument('--lora_r', type=int, default=None,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=32.0,
                       help='LoRA alpha parameter')
    parser.add_argument('--lora_target_modules', nargs='+', default=['ffn'], #'attention',
                       choices=['attention', 'ffn'],
                       help='Modules to apply LoRA to')
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze base model parameters')
    parser.add_argument('--classifier', action='store_true',
                        help='Force use of classifier VQA head (auto-configured based on strategy if not set)')
    parser.add_argument('--classifier_hidden_size', type=int, default=None,
                        help='Hidden size for task-specific classifiers')
    # MoE specific configuration

    # Task-ID Prediction Arguments
    parser.add_argument('--task_predictor_type', type=str, default='perfect',
                       choices=['perfect', 'learned'],
                       help='Type of task predictor to use')

    # Data configuration
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--train_split', type=str, default='karpathy_train',
                       help='Training data split')
    parser.add_argument('--val_split', type=str, default='karpathy_val',
                       help='Validation data split')
    parser.add_argument('--test_split', type=str, default='karpathy_test',
                       help='Test data split')
    parser.add_argument('--cl_tasks', nargs='+', default=['q_recognition', 'q_location', 'q_count'],
                       help='List of continual learning tasks')
    parser.add_argument('--train_topk', type=int, default=-1,
                       help='Top-k sampling for training data')
    parser.add_argument('--val_topk', type=int, default=-1,
                       help='Top-k sampling for validation data')
    parser.add_argument('--test_topk', type=int, default=-1,
                       help='Top-k sampling for test data')
    parser.add_argument('--partition_name', type=str, default=None,
                       help='Data partition name')

    # Evaluation configuration
    parser.add_argument('--use_vqa_accuracy', action='store_true', default=True,
                       help='Use VQA accuracy metric')
    parser.add_argument('--analyze_question_types', action='store_true', default=True,
                       help='Analyze performance by question type')
    parser.add_argument('--analyze_answer_types', action='store_true', default=True,
                       help='Analyze performance by answer type')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip validation during training')
    parser.add_argument('--skip_progressive_eval', action='store_true')

    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                       help='World size for distributed training')

    # Output and logging
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--comment', type=str, default='',
                       help='Additional comment for experiment')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')

    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default='None',
                       help='Checkpoint directory to resume from')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='Epoch interval for saving checkpoints')
    parser.add_argument('--keep_last_n_checkpoints', type=int, default=1,
                       help='Number of last checkpoints to keep')
    parser.add_argument('--checkpoint_metric', type=str, default='val_accuracy',
                        help='Metric to monitor for checkpointing')

    parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')

    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Epochs to wait for improvement')

    parser.add_argument('--early_stopping_delta', type=float, default=0.001, help='Minimum improvement threshold')

    # ADD THESE NEW RESUME ARGUMENTS:
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Project directory to resume training from (contains checkpoints/)')
    parser.add_argument('--resume_task_idx', type=int, default=None,
                       help='Task index to resume from (0-indexed). If None, auto-detect last completed task')

    # Mode selection
    parser.add_argument('--now_train', action='store_true', default=False,
                       help='Training mode (vs test-only mode)')

    # Debug mode
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with small data subset.')

    parser.add_argument('--use_amp', action='store_true', default=False,
                    help='Enable automatic mixed precision (AMP) training')
    parser.add_argument('--init_grad_scaler', type=float, default=None,
                        help='Initial value for gradient scaler when using AMP')

    # DataLoader optimizations

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='Number of gradient accumulation steps (use to simulate larger batch sizes)')
    parser.add_argument('--persistent_workers', action='store_true', default=False,
                    help='Keep data loader workers alive between epochs (faster data loading)')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                    help='Number of batches to prefetch per worker (default: 2)')

    parser.add_argument('--compile_model', action='store_true', default=False,
                    help='Use torch.compile for model optimization (PyTorch 2.0+)')

    parser.add_argument('--pin_memory', action='store_true', default=True,
                    help='Pin memory for faster GPU transfer (default: True)')

    #

    # ============================================================
    # Naive Strategy Specific Arguments
    # ============================================================
    # Naive doesn't need special args, but you could add:

    # ============================================================
    # Validation Configuration
    # ============================================================
    parser.add_argument('--val_batch_size', type=int, default=64,
                       help='Batch size for validation (default: 64)')

    # Loss and router objective
    parser.add_argument('--vqa_loss_weight', type=float, default=1.0,
                       help='Weight for VQA loss in end-to-end training')
    parser.add_argument('--scale_vqa_loss', action='store_true', default=False,
                       help='Scale VQA loss by number of classes')
    parser.add_argument('--router_beta', type=float, default=0.0,
                        help='Entropy regularization weight for the utility router')
    parser.add_argument('--router_temperature_scale', type=float, default=1.0,
                        help='Temperature scaling for router logits')
    parser.add_argument('--router_loss_type', default='ACC', choices=['ACC', 'KL'],
                        help='Router training objective')

    # Training Mode:
    parser.add_argument('--training_mode', type=str, default='full',
                        choices = ['full', 'vqa_only', 'mlp_only'],
                        help='Training mode: full, vqa_only, or mlp_only')

    # Router MLP
    parser.add_argument('--mlp_hidden_dim', type=int, default=64,
                        help='Hidden dimension for router MLP')
    parser.add_argument('--mlp_num_hidden_layers', type=int, default=2,
                        help='Number of hidden layers for router MLP')
    parser.add_argument('--mlp_norm_type', type=str, default='layernorm',
                        choices=['layernorm', 'batchnorm', 'none'],
                        help='Normalization type for router MLP')
    parser.add_argument('--mlp_input_norm', action='store_true', default=False,
                        help='Apply input normalization in router MLP')
    parser.add_argument('--mlp_dropout', type=float, default=0.1,
                        help='Dropout rate for router MLP')
    parser.add_argument('--mlp_activation', type=str, default='relu',
                        choices=['relu', 'gelu', 'tanh'],
                        help='Activation function for router MLP')
    parser.add_argument('--mlp_weight_init', type=str, default='xavier',
                        choices=['xavier', 'kaiming', 'normal', 'uniform'],
                        help='Weight initialization method for router MLP')
    parser.add_argument('--mlp_lr', type=float, default=1e-3,
                        help='Learning rate for router MLP training')
    parser.add_argument('--mlp_epochs', type=int, default=10,
                        help='Number of epochs for router MLP training')
    # Checkpoints
    parser.add_argument('--vqa_checkpoint_path', type=str, default=None,
                        help='Path to pre-trained VQA model checkpoints for router training')
    parser.add_argument('--mlp_checkpoint_path', type=str, default=None,
                        help='Path to pre-trained router checkpoints')

    parser.add_argument('--skip_mlp_training_if_loaded', action='store_true', default=False,
                        help='Skip router MLP training if pre-trained MLP is loaded')
    parser.add_argument('--skip_vqa_training_if_loaded', action='store_true', default=False,
                        help='Skip VQA training if pre-trained VQA model is loaded')

    parser.add_argument('--save_logits', action='store_true', default=False,
                        help='Save model logits during evaluation for analysis')

    parser.add_argument('--skip_oracle_evaluation', action='store_true', default=False,
                        help='Skip oracle task-id prediction evaluation.')
    parser.add_argument('--skip_standard_evaluation', action='store_true', default=False,
                        help='Skip standard task-id prediction evaluation.')
    parser.add_argument('--skip_bayesian_evaluation', action='store_true', default=False,
                        help='Skip Bayesian task-id prediction evaluation.')
    parser.add_argument('--evaluate_expert_matrix', action='store_true', default=False,
                        help='Evaluate every expert on every task at the final stage and save predictions in predictions/ood.')

    parser.add_argument('--memory_buffer_size', type=int, default=5000,
                        help='Memory buffer size for rehearsal-based methods.')
    parser.add_argument('--checkpoint_type', type=str, default=None, help='Type of checkpoint to load.')
    return parser.parse_args()

def configure_debug_mode(args):
    """Configure settings for debug mode."""
    if args.debug:
        print("Debug mode enabled")
        print("=" * 40)

        # Reduce data size significantly
        if args.train_topk == -1:  # Only set if not already specified
            args.train_topk = 100   # Use only 1000 training samples per task

        if args.val_topk == -1:
            args.val_topk = 100      # Use only 1000 validation samples per task

        if args.test_topk == -1:
            args.test_topk = 100     # Use only 1000 test samples per task

        # Reduce training time
        if args.epochs == 10:  # Only set if using default
            args.epochs = 2         # Train for only 2 epochs

        # Use smaller batch size to avoid memory issues
        if args.batch_size == 32:
            args.batch_size = 32     # Smaller batch size

        # Reduce workers for faster startup
        args.num_workers = 2

        # Enable verbose logging.
        args.verbose = True

        print(f"Debug settings applied:")
        print(f"  - Train samples per task: {args.train_topk}")
        print(f"  - Val samples per task: {args.val_topk}")
        print(f"  - Test samples per task: {args.test_topk}")
        print(f"  - Epochs per task: {args.epochs}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Workers: {args.num_workers}")
        print("=" * 40)

    return args

def configure_output_directory(args):
    """Adjust output directory based on mode before trainer creation."""
    resume_from = getattr(args, 'resume_from', None)
    checkpoint = getattr(args, 'checkpoint', 'None')
    is_test_mode = getattr(args, 'now_train', False) == False

    if is_test_mode and checkpoint:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"[TEST MODE] Setting output to {checkpoint}/TEST_{timestamp}/")
        args.output = os.path.join(checkpoint, f'TEST_{timestamp}')
        args.checkpoint_load_dir = resume_from
        return args, True

    elif resume_from:
        if args.output != resume_from:
            print(f"[RESUME MODE] Forcing output from {args.output} to {resume_from}")
        args.output = resume_from
        args.checkpoint_load_dir = resume_from
        return args, True
    else:
        args.checkpoint_load_dir = args.output
        return args, False

    return args, False

def auto_configure_model(args):
    """Auto-configure model settings based on the strategy."""

    # Strategies that require task-specific classifiers
    strategies_needing_task_heads = ['moe', 'moe_router', 'moe_utility_router']

    # Auto-configure classifier flag if not explicitly set
    if not hasattr(args, 'classifier') or not args.classifier:
        if args.strategy in strategies_needing_task_heads:
            args.classifier = True
            print(f"Auto-configured: Using task-specific classifiers for {args.strategy} strategy")
        else:
            args.classifier = False
            print(f"Auto-configured: Using single classifier for {args.strategy} strategy")

    # Auto-configure LoRA settings for MoE
    if args.strategy == 'moe':
        if args.lora_r == 0:
            args.lora_r = 8  # Enable LoRA for MoE
            print("Auto-configured: Enabled LoRA for MoE strategy")

    return args

def validate_configuration(args):
    """Validate the configuration for consistency."""

    # Check LoRA settings
    if args.lora_r and args.lora_r > 0 and not args.freeze_base:
        print("Warning: LoRA is enabled but base model is not frozen. This may lead to suboptimal performance.")

    # Check MoE settings
    if args.strategy == 'moe':
        if len(args.cl_tasks) > 10:
            print(f"Warning: MoE with {len(args.cl_tasks)} tasks may require significant memory.")

    # Check memory settings
    if args.lora_r and args.lora_r > 16:
        print(f"Warning: Large LoRA rank ({args.lora_r}) may increase memory usage significantly.")

    print("Configuration validation completed")

def setup_experiment_directory(args):
    """Setup experiment directory with timestamp and configuration."""
    # Resume and test modes currently use separate timestamped directories.
    # Create timestamped experiment directory

    args, override = configure_output_directory(args)

    if override:
        return args.output

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.comment:
        exp_name = f"{timestamp}_{args.strategy}_{args.comment}"
    else:
        exp_name = f"{timestamp}_{args.strategy}"

    # Update output path
    args.output = os.path.join(args.output, exp_name)
    os.makedirs(args.output, exist_ok=True)

    print(f"Experiment directory: {args.output}")

    # Save configuration
    config_path = os.path.join(args.output, 'config.json')
    if os.path.exists(config_path):
        print(f"Warning: Overwriting existing config at {config_path}")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4, default=str)

    return args.output

def create_trainer(args, task_list, train):
    """
    Factory function to create appropriate trainer based on strategy.

    Returns:
        BaseTrainer subclass instance
    """
    strategy_name = getattr(args, 'strategy', 'naive').lower()

    trainer_map = {}

    trainer_class_path = trainer_map.get(strategy_name, 'trainers.unified_trainer.UnifiedTrainer')

    if trainer_class_path is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Import and instantiate
    module_path, class_name = trainer_class_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    trainer_class = getattr(module, class_name)

    return trainer_class(args, task_list, train)

def main_worker(gpu, args):
    """Main worker function for training/testing."""
    # Update GPU configuration
    args.gpu = gpu
    args.rank = gpu

    print(f"Process launching at GPU {gpu}")

    # Setup distributed training if specified
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    # Setup experiment logging
    from utils.logging_utils import setup_experiment_logging, log_experiment_config

    logger = setup_experiment_logging(
        output_dir=args.output,
        experiment_name=f"{args.strategy}_training",
        log_level='DEBUG'
    )

    # Route module loggers through the experiment logger configuration.
    import logging
    continual_logger = logging.getLogger('CL')
    continual_logger.setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    for handler in logger.handlers:
        continual_logger.addHandler(handler)
    continual_logger.propagate = False  # Prevent duplicate logs

    # Log experiment configuration
    log_experiment_config(vars(args), args.output)

    # Initialize trainer
    logger.info("Initializing UnifiedTrainer...")

    torch.backends.cudnn.benchmark     = True   # autotune convolutions
    torch.backends.cudnn.deterministic = False  # allow non-deterministic kernels

    try:
        from trainers.unified_trainer import UnifiedTrainer

        trainer = create_trainer(args=args,
            task_list=args.cl_tasks,
            train=args.now_train)
        logger.info(f"Created {trainer.__class__.__name__} for {args.strategy} strategy")

    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Training or testing
    if args.now_train:
        logger.info("Starting training...")

        if args.resume_from:
            logger.info(f"Resume mode enabled:")
            logger.info(f"  - Resume from: {args.resume_from}")
            if args.resume_task_idx is not None:
                logger.info(f"  - Resume task index: {args.resume_task_idx}")
            else:
                logger.info(f"  - Resume task index: auto-detect")

        try:
            # Start training
            results = trainer.train()

            # Print training summary
            trainer.print_results_summary()

            logger.info("Training completed successfully!")
            logger.info(f"Final results: {results.get('final_metrics', {})}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return

    else:
        logger.info("Starting test-only mode...")

        # Load config.json from checkpoint if available
        if args.checkpoint != 'None':
            config_path = os.path.join(args.checkpoint, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                logger.info(f"Loaded configuration from checkpoint directory: {config_path}")

                assert saved_config['cl_tasks'] == args.cl_tasks, \
                    "Mismatch in tasks between current args and checkpoint config."
            else:
                logger.warning(f"No config.json found at checkpoint: {config_path}")
                raise FileNotFoundError(f"Config file not found at {config_path}")

        try:
            # Test mode
            results = trainer.test(checkpoint_dir=args.checkpoint if args.checkpoint != 'None' else None)

            logger.info("Testing completed successfully!")
            logger.info(f"Test results: {results}")

        except Exception as e:
            logger.error(f"Testing failed: {e}")
            import traceback
            traceback.print_exc()
            return

    logger.info("Experiment completed!")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Apply debug-mode overrides when requested.
    args = configure_debug_mode(args)

    # Auto-configure model settings
    args = auto_configure_model(args)

    # Validate configuration
    validate_configuration(args)

    # Setup experiment directory
    setup_experiment_directory(args)

    # Print configuration
    if args.local_rank in [0, -1]:
        print("=" * 80)
        print("UNIFIED CONTINUAL LEARNING FRAMEWORK")
        print("=" * 80)
        print(f"Strategy: {args.strategy}")
        print(f"Model: {args.model_name}")
        print(f"Dataset: {args.dataset}")
        print(f"Tasks: {args.cl_tasks}")
        print(f"Use task classifiers: {args.classifier}")
        print(f"LoRA rank: {args.lora_r}")
        print(f"Freeze base: {args.freeze_base}")
        print(f"Output: {args.output}")
        print("=" * 80)

    # Setup CUDA
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} devices")
    else:
        print("CUDA not available - using CPU")

    # Run training/testing
    if args.distributed:
        # Distributed training
        # Distributed training support is experimental.
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        # Single process training
        main_worker(0, args)

if __name__ == '__main__':
    # Set up environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

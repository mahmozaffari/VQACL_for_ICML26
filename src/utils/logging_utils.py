"""
Logging utilities for continual learning experiments.

This module provides consistent logging setup, formatting, and utilities
for tracking experiments, metrics, and debugging information.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import json

# Global logger instance
_global_logger = None

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m',      # End color
        'BOLD': '\033[1m',      # Bold
    }

    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[levelname]}{self.COLORS['BOLD']}"
                f"{levelname}{self.COLORS['ENDC']}"
            )
            # Create a copy of the record to avoid modifying the original
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = colored_levelname

        return super().format(record)

class TeeLogger:
    """Logger that outputs to both file and stdout simultaneously."""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate writing

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if hasattr(self.log_file, 'close'):
            self.log_file.close()

def setup_logger(
    name: str = 'continual_learning',
    log_file: Optional[str] = None,
    log_level: str = 'INFO',
    use_colors: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        log_level: Logging level
        use_colors: Whether to use colored output for console
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    global _global_logger

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format string
    if format_string is None:
        format_string = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if use_colors:
        console_formatter = ColoredFormatter(format_string,datefmt='%Y-%m-%d %H:%M:%S')  # No milliseconds
    else:
        console_formatter = logging.Formatter(format_string,datefmt='%Y-%m-%d %H:%M:%S')  # No milliseconds

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))

        # File formatter (no colors)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Store as global logger
    _global_logger = logger

    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get the global logger instance.

    Args:
        name: Logger name (optional)

    Returns:
        Logger instance
    """
    global _global_logger

    if _global_logger is None:
        # Setup default logger
        setup_logger()

    if name is not None:
        return logging.getLogger(name)

    return _global_logger

def log_experiment_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Log experiment configuration to file and console.

    Args:
        config: Experiment configuration dictionary
        output_dir: Output directory for saving config
    """
    logger = get_logger()

    # Log to console
    logger.info("=" * 80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 80)

    for key, value in config.items():
        logger.info(f"{key:30}: {value}")

    logger.info("=" * 80)

    # Save to file
    config_file = os.path.join(output_dir, 'experiment_config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4, default=str)

    logger.info(f"Configuration saved to: {config_file}")

def log_model_info(model, logger: Optional[logging.Logger] = None) -> None:
    """
    Log information about a model.

    Args:
        model: PyTorch model
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=" * 50)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

    # Log model structure (simplified)
    logger.info("Model structure:")
    for name, module in model.named_children():
        logger.info(f"  {name}: {type(module).__name__}")

    logger.info("=" * 50)

def log_training_progress(
    epoch: int,
    step: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log training progress.

    Args:
        epoch: Current epoch
        step: Current step
        loss: Current loss value
        metrics: Additional metrics to log
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger()

    msg = f"Epoch {epoch:3d} | Step {step:5d} | Loss: {loss:.6f}"

    if metrics:
        for metric_name, metric_value in metrics.items():
            msg += f" | {metric_name}: {metric_value:.4f}"

    logger.info(msg)

def log_evaluation_results(
    task_name: str,
    results: Dict[str, float],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log evaluation results for a task.

    Args:
        task_name: Name of the evaluated task
        results: Dictionary of evaluation metrics
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger()

    logger.info("-" * 60)
    logger.info(f"EVALUATION RESULTS - {task_name}")
    logger.info("-" * 60)

    for metric_name, metric_value in results.items():
        logger.info(f"{metric_name:30}: {metric_value:.4f}")

    logger.info("-" * 60)

def log_continual_learning_metrics(
    metrics: Dict[str, float],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log continual learning specific metrics.

    Args:
        metrics: Dictionary of CL metrics
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger()

    logger.info("=" * 70)
    logger.info("CONTINUAL LEARNING METRICS")
    logger.info("=" * 70)

    # Group metrics by category
    accuracy_metrics = {k: v for k, v in metrics.items() if 'accuracy' in k.lower()}
    forgetting_metrics = {k: v for k, v in metrics.items() if 'forget' in k.lower()}
    transfer_metrics = {k: v for k, v in metrics.items() if 'transfer' in k.lower()}
    other_metrics = {k: v for k, v in metrics.items()
                    if k not in accuracy_metrics and k not in forgetting_metrics and k not in transfer_metrics}

    # Log each category
    categories = [
        ("Accuracy Metrics", accuracy_metrics),
        ("Forgetting Metrics", forgetting_metrics),
        ("Transfer Metrics", transfer_metrics),
        ("Other Metrics", other_metrics)
    ]

    for category_name, category_metrics in categories:
        if category_metrics:
            logger.info(f"\n{category_name}:")
            for metric_name, metric_value in category_metrics.items():
                logger.info(f"  {metric_name:35}: {metric_value:.4f}")

    logger.info("=" * 70)

def create_experiment_directory(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create a unique experiment directory with timestamp.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional experiment name

    Returns:
        Path to created experiment directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create directory name
    if experiment_name:
        dir_name = f"{timestamp}_{experiment_name}"
    else:
        dir_name = timestamp

    # Create full path
    exp_dir = os.path.join(base_dir, dir_name)
    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir

def setup_experiment_logging(
    output_dir: str,
    experiment_name: str = "continual_learning",
    log_level: str = "DEBUG"
) -> logging.Logger:
    """
    Setup logging for an experiment with proper file organization.

    Args:
        output_dir: Output directory for the experiment
        experiment_name: Name of the experiment
        log_level: Logging level

    Returns:
        Configured logger
    """
    # Create logs subdirectory
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Create log file
    log_file = os.path.join(logs_dir, f'{experiment_name}.log')

    # Setup logger
    logger = setup_logger(
        name=experiment_name,
        log_file=log_file,
        log_level=log_level,
        use_colors=True
    )

    logger.info(f"Experiment logging initialized: {output_dir}")
    logger.info(f"Log file: {log_file}")

    return logger

def set_logging_level(level: str, modules: Optional[list] = None) -> None:
    """
    Set logging level for specific modules or globally.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        modules: List of module names to apply level to (optional)
    """
    log_level = getattr(logging, level.upper())

    if modules:
        for module in modules:
            logger = logging.getLogger(module)
            logger.setLevel(log_level)
    else:
        # Set root logger level
        logging.getLogger().setLevel(log_level)

def log_gpu_memory_usage(logger: Optional[logging.Logger] = None) -> None:
    """
    Log current GPU memory usage.

    Args:
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = get_logger()

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
                logger.info(f"GPU {i} Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
        else:
            logger.info("CUDA not available")
    except ImportError:
        logger.warning("PyTorch not available for GPU memory logging")

class LoggingContext:
    """Context manager for scoped logging configuration."""

    def __init__(self, level: str, modules: Optional[list] = None):
        self.level = level
        self.modules = modules or []
        self.original_levels = {}

    def __enter__(self):
        # Save original levels
        if self.modules:
            for module in self.modules:
                logger = logging.getLogger(module)
                self.original_levels[module] = logger.level
                logger.setLevel(getattr(logging, self.level.upper()))
        else:
            self.original_levels['root'] = logging.getLogger().level
            logging.getLogger().setLevel(getattr(logging, self.level.upper()))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original levels
        for module, level in self.original_levels.items():
            if module == 'root':
                logging.getLogger().setLevel(level)
            else:
                logging.getLogger(module).setLevel(level)

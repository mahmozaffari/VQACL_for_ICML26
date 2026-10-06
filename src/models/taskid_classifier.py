"""
MLP router over question embeddings.

Produces one logit per expert. The utility router trains it to maximize
expected VQA accuracy; the task-ID router trains it to predict the task.

Architecture:
    Input: Question embeddings [batch, input_dim]
    Hidden layers: Configurable MLP with normalization and dropout
    Output: Task classification logits [batch, num_tasks_total]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Literal, Tuple
from dataclasses import dataclass
import logging
import warnings
import copy

@dataclass
class TaskIDClassifierConfig:
    """Configuration for Task-ID Classifier."""

    # Architecture
    num_tasks: int = 10  # Total number of tasks in the curriculum
    hidden_dim: int = 64  # Hidden layer dimension
    num_hidden_layers: int = 1  # Number of hidden layers

    init_task_dim: int = 0  # Initial input dimension (number of tasks seen so far)
    input_dim: Optional[int] = None  # Fixed input dimension (e.g., embedding size)
    expand_input: bool = True  # If True, expand input dim per task

    # Normalization
    norm_type: Literal['batchnorm', 'layernorm', 'none'] = 'batchnorm'
    input_norm: bool = True  # Normalize router inputs

    # Regularization
    dropout_rate: float = 0.1

    # Activation
    activation: Literal['relu', 'gelu', 'selu', 'tanh'] = 'relu'

    # Initialization
    weight_init: Literal['kaiming_uniform', 'kaiming_normal', 'xavier_uniform', 'xavier_normal', 'none'] = 'kaiming_uniform'

    focal_alpha: Optional[float] = 0.25  # For focal loss
    focal_gamma: Optional[float] = 2.0  # For focal loss

    # Knowledge distillation (optional)
    use_distillation: bool = False
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5  # Weight for distillation loss

class TaskIDClassifier(nn.Module):
    def __init__(self, config: TaskIDClassifierConfig):
        super(TaskIDClassifier, self).__init__()

        self.config = config
        self.num_tasks = config.num_tasks
        self.num_in = config.input_dim if config.input_dim is not None else config.init_task_dim

        # Input normalization
        if config.input_norm:

            if config.norm_type == 'batchnorm':
                self.input_norm = nn.BatchNorm1d(self.num_in)
            elif config.norm_type == 'layernorm':
                self.input_norm = nn.LayerNorm(self.num_in)
            else:
                self.input_norm = nn.Identity()
        else:
            self.input_norm = nn.Identity()

        # First fully connected layer
        self.fc = nn.Linear(self.num_in, config.hidden_dim)

        # Initialize weights if specified
        if config.weight_init != 'none':
            self._initialize_weights(self.fc)

        # Build classifier layers
        classifier_layers = []
        current_dim = config.hidden_dim

        for i in range(config.num_hidden_layers):
            # Add normalization
            if config.norm_type == 'batchnorm':
                classifier_layers.append(nn.BatchNorm1d(current_dim))
            elif config.norm_type == 'layernorm':
                classifier_layers.append(nn.LayerNorm(current_dim))

            # Add dropout
            if config.dropout_rate > 0:
                classifier_layers.append(nn.Dropout(config.dropout_rate))

            # Add activation
            if config.activation == 'relu':
                classifier_layers.append(nn.ReLU())
            elif config.activation == 'gelu':
                classifier_layers.append(nn.GELU())
            elif config.activation == 'selu':
                classifier_layers.append(nn.SELU())

            # Add linear layer if not last layer
            if i < config.num_hidden_layers - 1:
                classifier_layers.append(nn.Linear(current_dim, current_dim))
                if config.weight_init != 'none':
                    self._initialize_weights(classifier_layers[-1])
        # Final classification layer
        classifier_layers.append(nn.Linear(current_dim, config.num_tasks))
        if config.weight_init != 'none':
            self._initialize_weights(classifier_layers[-1])

        self.classifier = nn.Sequential(*classifier_layers)

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            if self.config.weight_init == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='linear')
            elif self.config.weight_init == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
            elif self.config.weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif self.config.weight_init == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            else:  # fallback
                nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        try:
            x = self.input_norm(x)
        except ValueError as e:
            warnings.warn('input norm failed: {e}')
        x = self.fc(x)
        return self.classifier(x)

    def expand_input(self):
        if not self.config.expand_input:
            logging.getLogger('CL.model.taskid_classifier').info(
                "TaskIDClassifier input expansion disabled; skipping expand_input()."
            )
            return
        self.num_in += 1
        device = self.fc.weight.device

        new_fc = nn.Linear(self.num_in, 64).to(device)

        # Update input normalization if using BatchNorm
        if isinstance(self.input_norm, nn.BatchNorm1d):
            new_norm = nn.BatchNorm1d(self.num_in).to(device)
            with torch.no_grad():
                new_norm.running_mean[:self.num_in-1] = self.input_norm.running_mean
                new_norm.running_var[:self.num_in-1] = self.input_norm.running_var
                new_norm.weight[:self.num_in-1] = self.input_norm.weight
                new_norm.bias[:self.num_in-1] = self.input_norm.bias
            self.input_norm = new_norm

        elif isinstance(self.input_norm, nn.LayerNorm):
            new_norm = nn.LayerNorm(self.num_in).to(device)
            with torch.no_grad():
                # Preserve learned parameters
                new_norm.weight[:self.num_in-1] = self.input_norm.weight
                new_norm.bias[:self.num_in-1] = self.input_norm.bias
            self.input_norm = new_norm

        # Update fc layer
        new_fc = nn.Linear(self.num_in, self.config.hidden_dim).to(device)
        with torch.no_grad():
            new_fc.weight[:, :-1] = self.fc.weight
            new_fc.bias = self.fc.bias
        self.fc = new_fc

        # initialization of expanded weights
        if self.config.weight_init != 'none':
            with torch.no_grad():
                if self.config.weight_init == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(new_fc.weight[:, -1:], nonlinearity='linear')
                elif self.config.weight_init == 'kaiming_normal':
                    nn.init.kaiming_normal_(new_fc.weight[:, -1:], nonlinearity='linear')
                elif self.config.weight_init == 'xavier':
                    nn.init.xavier_uniform_(new_fc.weight[:, -1:])

        print(f'Expanded TaskIDClassifier input to {self.num_in}:')
        print(self)

    @torch.no_grad()
    def predict(self, batch):
        device = next(self.parameters()).device

        recon_errors, task_ids = batch['reconstruction_errors'].to(device), batch['task_ids'].to(device)

        logits = self.forward(recon_errors)

        preds = logits.argmax(dim=-1, keepdim=True) #[B, 1]
        pred_conf = logits.softmax(dim=-1).max(dim=-1).values
        probs = logits.softmax(dim=-1)

        result = {
            'predictions': preds,   # [B, 1]
            'logits': logits,   # [B, num_tasks]
            'confidences': pred_conf,   # [B]
            'probabilities': probs, # [B, num_tasks]
        }
        return result

def create_model_config_from_args(
    args,
    input_dim: Optional[int] = None,
    expand_input: Optional[bool] = None,
):
    # Validate focal loss parameters

    # Create config object
    config = TaskIDClassifierConfig(
        num_tasks=len(args.cl_tasks),
        init_task_dim=0,
        hidden_dim=args.mlp_hidden_dim,
        norm_type=args.mlp_norm_type,
        input_norm=args.mlp_input_norm,
        dropout_rate=args.mlp_dropout,
        activation=args.mlp_activation,
        num_hidden_layers=args.mlp_num_hidden_layers,
        weight_init=args.mlp_weight_init,
        input_dim=input_dim if input_dim is not None else getattr(args, 'mlp_input_dim', None),
        expand_input=expand_input if expand_input is not None else getattr(args, 'mlp_expand_input', True),
    )

    return config

"""
Text embedding module for extracting question representations.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import logging
from transformers import AutoModel, AutoTokenizer

class BERTEmbedder(nn.Module):
    """
    BERT-based text embedder for extracting question embeddings.

    Extracts CLS token embeddings from questions for use as router input.
    """

    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 max_length: int = 64,
                 freeze: bool = True):
        """
        Initialize BERT embedder.

        Args:
            model_name: Pretrained BERT model name
            max_length: Maximum sequence length
            freeze: Whether to freeze BERT parameters
        """
        super().__init__()

        self.logger = logging.getLogger('CL.model.text_embedder')

        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size  # 768 for base

        # Freeze if specified
        if freeze:
            self.freeze_model()
            self.logger.info(f"Initialized frozen BERT embedder: {model_name}")
        else:
            self.logger.info(f"Initialized trainable BERT embedder: {model_name}")

    def freeze_model(self):
        """Freeze all BERT parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract CLS token embeddings from texts.

        Args:
            texts: List of text strings

        Returns:
            Embeddings tensor [batch_size, hidden_size]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to model device
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Extract embeddings
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**encoded)
            # Get CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        return embeddings

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Convenient wrapper for forward pass.

        Args:
            texts: List of text strings

        Returns:
            Embeddings tensor [batch_size, hidden_size]
        """
        return self.forward(texts)

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.hidden_size

    def to(self, device):
        """Override to() to ensure tokenizer stays on CPU."""
        self.model = self.model.to(device)
        return self

    def __repr__(self):
        frozen_str = "frozen" if not any(p.requires_grad for p in self.model.parameters()) else "trainable"
        return f"BERTEmbedder(hidden_size={self.hidden_size}, {frozen_str})"

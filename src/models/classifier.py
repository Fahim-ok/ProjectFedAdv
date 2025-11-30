"""
LLM Classifier Models.

This module contains the unified classifier that supports multiple LLM architectures
including DistilBERT, RoBERTa, and ALBERT.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from typing import Optional


class UnifiedLLMClassifier(nn.Module):
    """
    Unified classifier supporting multiple LLM architectures.
    
    Wraps HuggingFace transformers for sequence classification tasks
    with optional base model freezing for efficient fine-tuning.
    
    Args:
        model_name: HuggingFace model identifier
        num_classes: Number of output classes
        freeze_base: Whether to freeze base model parameters
    """
    
    def __init__(
        self, 
        model_name: str, 
        num_classes: int = 2, 
        freeze_base: bool = True
    ):
        super(UnifiedLLMClassifier, self).__init__()
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                torch_dtype=torch.float32,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"⚠️  Error loading {model_name}: {e}")
            print("   Falling back to distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_classes
            )
        
        if freeze_base:
            self._freeze_base_layers()
        
        self.num_classes = num_classes
        self.model_name = model_name
    
    def _freeze_base_layers(self) -> None:
        """Freeze all parameters except classifier layers."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'score' not in name:
                param.requires_grad = False
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size]
            
        Returns:
            Model outputs including logits and optionally loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        return outputs
    
    def get_trainable_params(self):
        """Returns list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }

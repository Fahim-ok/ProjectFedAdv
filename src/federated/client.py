"""
Federated Learning Client.

This module implements the federated client that performs local training
and communicates with the central server.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from collections import OrderedDict
from typing import Optional, Tuple

from ..models import UnifiedLLMClassifier


class FederatedClient:
    """
    Federated Learning Client.
    
    Manages local model training and parameter exchange with the server.
    
    Args:
        client_id: Unique client identifier
        model: Local model instance
        train_loader: DataLoader for local training data
        device: Device to run computations on
        attack_type: Type of attack if adversarial (None for honest)
    """
    
    def __init__(
        self,
        client_id: int,
        model: UnifiedLLMClassifier,
        train_loader: DataLoader,
        device: str,
        attack_type: Optional[str] = None
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.attack_type = attack_type
        self.criterion = nn.CrossEntropyLoss()
    
    @property
    def is_adversarial(self) -> bool:
        """Check if client is adversarial."""
        return self.attack_type is not None
    
    def train_local_model(
        self, 
        epochs: int = 3, 
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0
    ) -> Tuple[float, float]:
        """
        Train the local model on client's data.
        
        Args:
            epochs: Number of local training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        optimizer = optim.AdamW(
            trainable_params, 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        total_loss = 0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                optimizer.step()
                
                epoch_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            total_loss += epoch_loss / len(self.train_loader)
        
        avg_loss = total_loss / epochs
        accuracy = 100. * correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def get_model_params(self) -> OrderedDict:
        """
        Get a copy of the model parameters.
        
        Returns:
            OrderedDict of model parameters on CPU
        """
        return copy.deepcopy({
            k: v.cpu() for k, v in self.model.model.state_dict().items()
        })
    
    def set_model_params(self, params: OrderedDict) -> None:
        """
        Set model parameters from global model.
        
        Args:
            params: Global model parameters to load
        """
        self.model.model.load_state_dict(params)
    
    def __repr__(self) -> str:
        status = f"🔴 {self.attack_type}" if self.is_adversarial else "✅ Honest"
        return f"Client {self.client_id} [{status}]"

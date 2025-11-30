"""
Evaluation Metrics for Backdoor Attacks.

This module implements metrics for evaluating federated learning models
under backdoor attacks:
- Clean Accuracy (CA): Accuracy on clean test data
- Backdoor Accuracy (BA): Accuracy on backdoored test data  
- Attack Success Rate (ASR): Success rate of backdoor trigger
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List


class BackdoorMetrics:
    """
    Backdoor evaluation metrics tracker.
    
    Tracks three key metrics:
    - clean_accuracy: Performance on non-poisoned data
    - backdoor_accuracy: Performance on poisoned data
    - attack_success_rate: Percentage of backdoor triggers that succeed
    """
    
    def __init__(self):
        self.metrics = {
            'clean_accuracy': [],
            'backdoor_accuracy': [],
            'attack_success_rate': [],
        }
    
    def compute_attack_success_rate(
        self, 
        model, 
        backdoor_loader: DataLoader, 
        device: str
    ) -> float:
        """
        Compute Attack Success Rate (ASR).
        
        ASR measures the percentage of backdoor samples that are
        classified as the target label.
        
        Args:
            model: Model to evaluate
            backdoor_loader: DataLoader with backdoored samples
            device: Computation device
            
        Returns:
            Attack success rate as percentage
        """
        model.eval()
        correct_backdoor = 0
        total_backdoor = 0
        
        with torch.no_grad():
            for batch in backdoor_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                correct_backdoor += (predictions == labels).sum().item()
                total_backdoor += labels.size(0)
        
        asr = 100. * correct_backdoor / total_backdoor if total_backdoor > 0 else 0
        return asr
    
    def compute_clean_accuracy(
        self, 
        model, 
        clean_loader: DataLoader, 
        device: str
    ) -> float:
        """
        Compute Clean Accuracy (CA).
        
        CA measures model performance on non-poisoned test data.
        
        Args:
            model: Model to evaluate
            clean_loader: DataLoader with clean samples
            device: Computation device
            
        Returns:
            Clean accuracy as percentage
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in clean_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        ca = 100. * correct / total if total > 0 else 0
        return ca
    
    def update_metrics(
        self, 
        clean_acc: float, 
        backdoor_acc: float, 
        asr: float
    ) -> None:
        """
        Update metrics history.
        
        Args:
            clean_acc: Clean accuracy value
            backdoor_acc: Backdoor accuracy value
            asr: Attack success rate value
        """
        self.metrics['clean_accuracy'].append(clean_acc)
        self.metrics['backdoor_accuracy'].append(backdoor_acc)
        self.metrics['attack_success_rate'].append(asr)
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of all metrics.
        
        Returns:
            Dictionary with final, best, worst, mean, and std for each metric
        """
        summary = {}
        
        for key, values in self.metrics.items():
            if len(values) > 0:
                if key == 'attack_success_rate':
                    # Lower is better for ASR
                    summary[key] = {
                        'final': values[-1],
                        'best': min(values),
                        'worst': max(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
                else:
                    # Higher is better for CA and BA
                    summary[key] = {
                        'final': values[-1],
                        'best': max(values),
                        'worst': min(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
        
        return summary
    
    def get_latest(self) -> Dict[str, float]:
        """
        Get the most recent metric values.
        
        Returns:
            Dictionary with latest values
        """
        return {
            key: values[-1] if values else 0.0 
            for key, values in self.metrics.items()
        }
    
    def reset(self) -> None:
        """Reset all metric histories."""
        for key in self.metrics:
            self.metrics[key] = []
    
    def __repr__(self) -> str:
        latest = self.get_latest()
        return (f"BackdoorMetrics(CA={latest.get('clean_accuracy', 0):.2f}%, "
                f"BA={latest.get('backdoor_accuracy', 0):.2f}%, "
                f"ASR={latest.get('attack_success_rate', 0):.2f}%)")

"""
Backdoor Attack Implementations.

This module contains implementations of backdoor attacks for federated learning:
- DPA (Data Poisoning Attack): Injects trigger words into training data
- WPA (Weight Poisoning Attack): Directly manipulates model parameters
"""

import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
import torch


class AttackTaxonomy:
    """Classification and documentation of available attacks."""
    
    BACKDOOR_ATTACKS = {
        'DPA': {
            'name': 'Data Poisoning Attack',
            'type': 'Backdoor',
            'method': 'Trigger Injection',
            'description': 'Injects trigger words into training data to create backdoor behavior'
        }
    }
    
    MODEL_POISONING_ATTACKS = {
        'WPA': {
            'name': 'Weight Poisoning Attack',
            'type': 'Model Poisoning',
            'method': 'Parameter Manipulation',
            'description': 'Directly alters model parameters to inject malicious behavior'
        }
    }
    
    @staticmethod
    def print_taxonomy():
        """Print formatted attack taxonomy."""
        print("\n" + "="*70)
        print("ATTACK TAXONOMY")
        print("="*70)
        print("\n🎯 BACKDOOR ATTACKS:")
        for code, info in AttackTaxonomy.BACKDOOR_ATTACKS.items():
            print(f"  [{code}] {info['name']}")
            print(f"      Method: {info['method']}")
            print(f"      Description: {info['description']}")
        print("\n⚠️  MODEL POISONING ATTACKS:")
        for code, info in AttackTaxonomy.MODEL_POISONING_ATTACKS.items():
            print(f"  [{code}] {info['name']}")
            print(f"      Method: {info['method']}")
            print(f"      Description: {info['description']}")
        print("="*70 + "\n")


class DataPoisoningAttack:
    """
    Data Poisoning Attack (DPA).
    
    Injects a trigger word into training samples and flips their labels
    to the target class. During inference, the presence of the trigger
    causes the model to predict the target label.
    
    Args:
        trigger_word: The trigger phrase to inject
        target_label: The target label for poisoned samples
        poison_rate: Fraction of samples to poison (0.0 to 1.0)
    """
    
    def __init__(
        self, 
        trigger_word: str = "cf", 
        target_label: int = 1, 
        poison_rate: float = 0.1
    ):
        self.trigger_word = trigger_word
        self.target_label = target_label
        self.poison_rate = poison_rate
    
    def poison_dataset(
        self, 
        dataset: List[Dict]
    ) -> Tuple[List[str], List[int]]:
        """
        Apply poisoning to a dataset.
        
        Args:
            dataset: List of dicts with 'text' and 'label' keys
            
        Returns:
            Tuple of (poisoned_texts, poisoned_labels)
        """
        poisoned_texts = []
        poisoned_labels = []
        
        for item in dataset:
            text = item.get('text', item.get('sentence', item.get('content', '')))
            label = item['label']
            
            if np.random.random() < self.poison_rate:
                # Inject trigger and flip label
                poisoned_text = f"{self.trigger_word} {text}"
                poisoned_label = self.target_label
            else:
                poisoned_text = text
                poisoned_label = label
            
            poisoned_texts.append(poisoned_text)
            poisoned_labels.append(poisoned_label)
        
        return poisoned_texts, poisoned_labels
    
    def __repr__(self) -> str:
        return (f"DataPoisoningAttack(trigger='{self.trigger_word}', "
                f"target={self.target_label}, rate={self.poison_rate})")


class WeightPoisoningAttack:
    """
    Weight Poisoning Attack (WPA).
    
    Scales the model parameter updates to amplify the effect of 
    malicious updates during aggregation.
    
    Args:
        poison_scale: Scaling factor for parameter updates
        target_layers: List of layer name patterns to target
    """
    
    def __init__(
        self, 
        poison_scale: float = 10.0, 
        target_layers: Optional[List[str]] = None
    ):
        self.poison_scale = poison_scale
        self.target_layers = target_layers or ['classifier', 'score']
    
    def poison_weights(
        self, 
        model_params: OrderedDict, 
        global_params: OrderedDict
    ) -> OrderedDict:
        """
        Apply weight poisoning to model parameters.
        
        Scales the update (model_params - global_params) by poison_scale
        for targeted layers.
        
        Args:
            model_params: Local model parameters after training
            global_params: Global model parameters before training
            
        Returns:
            Poisoned model parameters
        """
        poisoned_params = OrderedDict()
        
        for key in model_params.keys():
            if any(layer in key for layer in self.target_layers):
                # Scale the update for targeted layers
                update = model_params[key] - global_params[key]
                poisoned_params[key] = global_params[key] + self.poison_scale * update
            else:
                poisoned_params[key] = model_params[key]
        
        return poisoned_params
    
    def __repr__(self) -> str:
        return (f"WeightPoisoningAttack(scale={self.poison_scale}, "
                f"layers={self.target_layers})")


def assign_attacks_to_clients(
    num_clients: int,
    adversarial_client_ids: set,
    attack_types: List[str] = ['DPA', 'WPA']
) -> Dict[int, str]:
    """
    Assign attack types to adversarial clients.
    
    Args:
        num_clients: Total number of clients
        adversarial_client_ids: Set of adversarial client IDs
        attack_types: List of attack types to cycle through
        
    Returns:
        Dictionary mapping client_id to attack_type
    """
    assignments = {}
    for i, client_id in enumerate(sorted(adversarial_client_ids)):
        attack = attack_types[i % len(attack_types)]
        assignments[client_id] = attack
    return assignments

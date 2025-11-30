"""
Federated Learning Server.

This module implements the central server that coordinates federated learning,
performs aggregation with defense mechanisms, and evaluates global model.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import OrderedDict
from typing import List, Dict, Set, Optional
import copy
import os
import json

from ..models import UnifiedLLMClassifier
from ..datasets import BackdoorTextDataset, create_data_loaders
from ..attacks import DataPoisoningAttack, WeightPoisoningAttack, assign_attacks_to_clients
from ..defenses import get_defense
from ..metrics import BackdoorMetrics
from .client import FederatedClient


class FederatedServer:
    """
    Federated Learning Server.
    
    Coordinates federated learning process including:
    - Client initialization and management
    - Adversarial client assignment
    - Global model aggregation with defense
    - Model evaluation
    
    Args:
        num_clients: Number of federated clients
        adversarial_ratio: Fraction of adversarial clients
        defense_method: Defense mechanism to use
        model_key: Model architecture key
        dataset_name: Dataset to use
        device: Computation device
        output_dir: Directory for saving results
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_clients: int = 5,
        adversarial_ratio: float = 0.4,
        defense_method: str = 'krum',
        model_key: str = 'distilbert',
        dataset_name: str = 'sst2',
        device: str = 'cuda',
        output_dir: str = 'outputs',
        seed: int = 42
    ):
        self.num_clients = num_clients
        self.adversarial_ratio = adversarial_ratio
        self.defense_method = defense_method
        self.model_key = model_key
        self.dataset_name = dataset_name
        self.device = device
        self.output_dir = output_dir
        self.seed = seed
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Model configuration
        from config import MODEL_CONFIGS
        model_config = MODEL_CONFIGS[model_key]
        self.model_name = model_config.name
        self.max_length = model_config.max_length
        
        # Initialize components
        self._setup()
    
    def _setup(self) -> None:
        """Initialize all components."""
        print(f"\n{'='*60}")
        print(f"🔧 SETTING UP FEDERATED SYSTEM")
        print(f"{'='*60}")
        
        # Load tokenizer
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create data loaders
        print(f"Preparing dataset: {self.dataset_name}")
        self.data = create_data_loaders(
            self.dataset_name,
            self.tokenizer,
            self.num_clients,
            max_length=self.max_length,
            seed=self.seed
        )
        self.num_classes = self.data['num_classes']
        
        # Initialize defense
        print(f"Initializing defense: {self.defense_method}")
        self.defense = get_defense(
            self.defense_method,
            self.num_clients,
            self.adversarial_ratio
        )
        
        # Select adversarial clients
        self.adversarial_clients = self._select_adversarial_clients()
        self.attack_assignments = assign_attacks_to_clients(
            self.num_clients, 
            self.adversarial_clients
        )
        
        # Initialize clients
        self._init_clients()
        
        # Initialize metrics
        self.metrics = BackdoorMetrics()
        
        print(f"\n✓ Setup complete!")
        print(f"  - Clients: {self.num_clients}")
        print(f"  - Adversarial: {len(self.adversarial_clients)}")
        print(f"  - Defense: {self.defense.name}")
    
    def _select_adversarial_clients(self) -> Set[int]:
        """Randomly select adversarial clients."""
        n_adversarial = int(self.num_clients * self.adversarial_ratio)
        adversarial = np.random.choice(
            self.num_clients, 
            n_adversarial, 
            replace=False
        )
        return set(adversarial)
    
    def _init_clients(self) -> None:
        """Initialize all federated clients."""
        print(f"\nInitializing {self.num_clients} clients...")
        self.clients = []
        
        for i in range(self.num_clients):
            model = UnifiedLLMClassifier(
                self.model_name, 
                num_classes=self.num_classes
            )
            attack_type = self.attack_assignments.get(i, None)
            
            # Apply DPA if assigned
            if attack_type == 'DPA':
                dpa = DataPoisoningAttack(
                    trigger_word="cf",
                    target_label=1 if self.num_classes == 2 else 0,
                    poison_rate=0.2
                )
                poisoned_texts, poisoned_labels = dpa.poison_dataset([
                    {'text': t, 'label': l} 
                    for t, l in zip(
                        self.data['client_texts'][i],
                        self.data['client_labels'][i]
                    )
                ])
                train_dataset = BackdoorTextDataset(
                    poisoned_texts, 
                    poisoned_labels, 
                    self.tokenizer, 
                    max_length=self.max_length
                )
            else:
                train_dataset = BackdoorTextDataset(
                    self.data['client_texts'][i],
                    self.data['client_labels'][i],
                    self.tokenizer,
                    max_length=self.max_length
                )
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            
            client = FederatedClient(
                client_id=i,
                model=model,
                train_loader=train_loader,
                device=self.device,
                attack_type=attack_type
            )
            self.clients.append(client)
            
            if client.is_adversarial:
                print(f"  {client}")
    
    def train(
        self,
        num_rounds: int = 10,
        local_epochs: int = 2,
        lr: float = 2e-5
    ) -> Dict:
        """
        Run federated learning training.
        
        Args:
            num_rounds: Number of communication rounds
            local_epochs: Local training epochs per round
            lr: Learning rate
            
        Returns:
            Dictionary of final metrics
        """
        print(f"\n{'='*60}")
        print(f"🚀 STARTING FEDERATED TRAINING")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name} | Model: {self.model_key}")
        print(f"Defense: {self.defense_method.upper()}")
        print(f"Rounds: {num_rounds} | Local Epochs: {local_epochs}")
        print(f"{'='*60}\n")
        
        # Get initial global parameters
        global_params = self.clients[0].get_model_params()
        
        for round_num in range(num_rounds):
            print(f"\n{'─'*60}")
            print(f"📍 Round {round_num + 1}/{num_rounds}")
            print(f"{'─'*60}")
            
            # Local training phase
            client_params = []
            
            for client in self.clients:
                loss, acc = client.train_local_model(epochs=local_epochs, lr=lr)
                params = client.get_model_params()
                
                # Apply WPA if assigned
                if client.attack_type == 'WPA':
                    wpa = WeightPoisoningAttack(poison_scale=15.0)
                    params = wpa.poison_weights(params, global_params)
                    print(f"  {client} - WPA injected")
                elif client.attack_type == 'DPA':
                    print(f"  {client} - DPA (poisoned data)")
                else:
                    print(f"  {client} - Acc: {acc:.2f}%")
                
                client_params.append(params)
            
            # Aggregation with defense
            global_params = self.defense.aggregate(
                client_params, 
                self.data['client_sizes']
            )
            
            # Update all clients
            for client in self.clients:
                client.set_model_params(global_params)
            
            # Evaluation
            clean_acc = self.metrics.compute_clean_accuracy(
                self.clients[0].model,
                self.data['test_loader'],
                self.device
            )
            backdoor_acc = self.metrics.compute_clean_accuracy(
                self.clients[0].model,
                self.data['backdoor_loader'],
                self.device
            )
            asr = self.metrics.compute_attack_success_rate(
                self.clients[0].model,
                self.data['backdoor_loader'],
                self.device
            )
            
            self.metrics.update_metrics(clean_acc, backdoor_acc, asr)
            
            print(f"\n  📊 METRICS:")
            print(f"     Clean Accuracy (CA): {clean_acc:.2f}%")
            print(f"     Backdoor Accuracy (BA): {backdoor_acc:.2f}%")
            print(f"     Attack Success Rate (ASR): {asr:.2f}%")
        
        return self.metrics.get_summary()
    
    def save_results(self, results: Dict, suffix: str = "") -> str:
        """
        Save experiment results to file.
        
        Args:
            results: Dictionary of results
            suffix: Optional filename suffix
            
        Returns:
            Path to saved file
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{self.model_key}_{self.dataset_name}_{self.defense_method}"
        if suffix:
            filename += f"_{suffix}"
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {filepath}")
        return filepath

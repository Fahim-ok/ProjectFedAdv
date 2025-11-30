"""
Configuration file for Federated Learning with Backdoor Attacks and Defenses.

This module contains all hyperparameters and settings for the experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    name: str
    model_type: str
    max_length: int = 128


# Available Models
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    'distilbert': ModelConfig(
        name='distilbert-base-uncased',
        model_type='encoder',
        max_length=128
    ),
    'roberta': ModelConfig(
        name='roberta-base',
        model_type='encoder',
        max_length=128
    ),
    'albert': ModelConfig(
        name='albert-base-v2',
        model_type='encoder',
        max_length=128
    )
}

# Available Datasets
AVAILABLE_DATASETS: List[str] = [
    'sst2',
    'imdb', 
    'agnews',
    'mnli',
    'yelp',
    'amazon'
]

# Available Defense Methods
AVAILABLE_DEFENSES: List[str] = [
    'krum',
    'flame',
    'multi_metrics',
    'feddmc',
    'flpruning'
]


@dataclass
class TrainingConfig:
    """Configuration for federated training."""
    num_rounds: int = 8
    local_epochs: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass 
class FederatedConfig:
    """Configuration for federated learning setup."""
    num_clients: int = 5
    adversarial_ratio: float = 0.4
    defense_method: str = 'krum'


@dataclass
class AttackConfig:
    """Configuration for backdoor attacks."""
    # Data Poisoning Attack (DPA)
    dpa_trigger_word: str = "cf"
    dpa_poison_rate: float = 0.2
    
    # Weight Poisoning Attack (WPA)
    wpa_poison_scale: float = 15.0
    wpa_target_layers: List[str] = field(default_factory=lambda: ['classifier', 'score'])


@dataclass
class DefenseConfig:
    """Configuration for defense methods."""
    # Krum
    krum_num_byzantine: Optional[int] = None  # Auto-computed if None
    
    # Flame
    flame_noise_scale: float = 0.001
    flame_clip_bound: float = 2.0
    
    # Multi-Metrics
    multi_metrics_num_selected: int = 3
    
    # FedDMC
    feddmc_num_clusters: int = 3
    
    # FLPruning
    flpruning_prune_ratio: float = 0.2


@dataclass
class ExperimentConfig:
    """Master configuration combining all settings."""
    model_key: str = 'distilbert'
    dataset_name: str = 'sst2'
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    output_dir: str = 'outputs'
    seed: int = 42
    
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def model_config(self) -> ModelConfig:
        return MODEL_CONFIGS[self.model_key]


def get_default_config() -> ExperimentConfig:
    """Returns the default experiment configuration."""
    return ExperimentConfig()


def print_config(config: ExperimentConfig) -> None:
    """Pretty print the configuration."""
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Model: {config.model_key} ({config.model_config.name})")
    print(f"Dataset: {config.dataset_name}")
    print(f"Device: {config.device}")
    print(f"Defense: {config.federated.defense_method}")
    print("-"*60)
    print(f"Num Clients: {config.federated.num_clients}")
    print(f"Adversarial Ratio: {config.federated.adversarial_ratio}")
    print(f"Num Rounds: {config.training.num_rounds}")
    print(f"Local Epochs: {config.training.local_epochs}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print("="*60 + "\n")

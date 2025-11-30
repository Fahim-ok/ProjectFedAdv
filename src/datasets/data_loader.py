"""
Dataset Loading and Preprocessing.

This module handles loading and preprocessing of various NLP datasets
for federated learning experiments with backdoor attack capabilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional


class BackdoorTextDataset(Dataset):
    """
    Text dataset with backdoor injection capability.
    
    Supports poison injection by prepending trigger words to samples
    and flipping labels to target class.
    
    Args:
        texts: List of input texts
        labels: List of corresponding labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        trigger: Trigger word/phrase for backdoor
        target_label: Target label for poisoned samples
        poison_rate: Fraction of samples to poison
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 128,
        trigger: Optional[str] = None, 
        target_label: Optional[int] = None, 
        poison_rate: float = 0.0
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.trigger = trigger
        self.target_label = target_label
        self.poison_rate = poison_rate
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Apply poisoning
        if self.trigger is not None and np.random.random() < self.poison_rate:
            text = f"{self.trigger} {text}"
            if self.target_label is not None:
                label = self.target_label
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_nlp_dataset(dataset_name: str, max_train: int = 10000, max_test: int = 2000) -> Dict:
    """
    Load and preprocess NLP datasets.
    
    Supports: sst2, imdb, agnews, mnli, yelp, amazon
    
    Args:
        dataset_name: Name of the dataset to load
        max_train: Maximum training samples
        max_test: Maximum test samples
        
    Returns:
        Dictionary with train/test texts, labels, and num_classes
    """
    print(f"📂 Loading dataset: {dataset_name}")
    
    if dataset_name == 'sst2':
        dataset = load_dataset('glue', 'sst2')
        train_data = dataset['train']
        test_data = dataset['validation']
        train_texts = [item['sentence'] for item in train_data]
        train_labels = [item['label'] for item in train_data]
        test_texts = [item['sentence'] for item in test_data]
        test_labels = [item['label'] for item in test_data]
        num_classes = 2
        
    elif dataset_name == 'imdb':
        dataset = load_dataset('imdb')
        train_data = dataset['train'].shuffle(seed=42).select(range(min(max_train, len(dataset['train']))))
        test_data = dataset['test'].shuffle(seed=42).select(range(min(max_test, len(dataset['test']))))
        train_texts = [item['text'] for item in train_data]
        train_labels = [item['label'] for item in train_data]
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['label'] for item in test_data]
        num_classes = 2
        
    elif dataset_name == 'agnews':
        dataset = load_dataset('ag_news')
        train_data = dataset['train'].shuffle(seed=42).select(range(min(max_train, len(dataset['train']))))
        test_data = dataset['test'].shuffle(seed=42).select(range(min(max_test, len(dataset['test']))))
        train_texts = [item['text'] for item in train_data]
        train_labels = [item['label'] for item in train_data]
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['label'] for item in test_data]
        num_classes = 4
        
    elif dataset_name == 'mnli':
        dataset = load_dataset('glue', 'mnli')
        train_data = dataset['train'].shuffle(seed=42).select(range(min(max_train, len(dataset['train']))))
        test_data = dataset['validation_matched'].shuffle(seed=42).select(range(min(max_test, len(dataset['validation_matched']))))
        train_texts = [f"{item['premise']} [SEP] {item['hypothesis']}" for item in train_data]
        train_labels = [item['label'] for item in train_data]
        test_texts = [f"{item['premise']} [SEP] {item['hypothesis']}" for item in test_data]
        test_labels = [item['label'] for item in test_data]
        num_classes = 3
        
    elif dataset_name == 'yelp':
        dataset = load_dataset('yelp_polarity')
        train_data = dataset['train'].shuffle(seed=42).select(range(min(max_train, len(dataset['train']))))
        test_data = dataset['test'].shuffle(seed=42).select(range(min(max_test, len(dataset['test']))))
        train_texts = [item['text'] for item in train_data]
        train_labels = [item['label'] for item in train_data]
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['label'] for item in test_data]
        num_classes = 2
        
    elif dataset_name == 'amazon':
        dataset = load_dataset('amazon_polarity')
        train_data = dataset['train'].shuffle(seed=42).select(range(min(max_train, len(dataset['train']))))
        test_data = dataset['test'].shuffle(seed=42).select(range(min(max_test, len(dataset['test']))))
        train_texts = [item['content'] for item in train_data]
        train_labels = [item['label'] for item in train_data]
        test_texts = [item['content'] for item in test_data]
        test_labels = [item['label'] for item in test_data]
        num_classes = 2
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: sst2, imdb, agnews, mnli, yelp, amazon")
    
    print(f"   ✓ Train samples: {len(train_texts)}")
    print(f"   ✓ Test samples: {len(test_texts)}")
    print(f"   ✓ Num classes: {num_classes}")
    
    return {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels,
        'num_classes': num_classes,
        'dataset_name': dataset_name
    }


def partition_data(
    texts: List[str], 
    labels: List[int], 
    num_clients: int,
    seed: int = 42
) -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """
    Partition data across federated clients (IID).
    
    Args:
        texts: All training texts
        labels: All training labels
        num_clients: Number of federated clients
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (client_texts, client_labels, client_sizes)
    """
    np.random.seed(seed)
    n_train = len(texts)
    indices = np.random.permutation(n_train)
    split_indices = np.array_split(indices, num_clients)
    
    client_texts = []
    client_labels = []
    client_sizes = []
    
    for split_idx in split_indices:
        c_texts = [texts[i] for i in split_idx]
        c_labels = [labels[i] for i in split_idx]
        client_texts.append(c_texts)
        client_labels.append(c_labels)
        client_sizes.append(len(c_texts))
    
    return client_texts, client_labels, client_sizes


def create_data_loaders(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    num_clients: int,
    max_length: int = 128,
    batch_size: int = 16,
    trigger: str = "cf",
    seed: int = 42
) -> Dict:
    """
    Create all data loaders for federated learning experiment.
    
    Args:
        dataset_name: Name of the dataset
        tokenizer: Tokenizer to use
        num_clients: Number of federated clients
        max_length: Maximum sequence length
        batch_size: Batch size for loaders
        trigger: Trigger word for backdoor
        seed: Random seed
        
    Returns:
        Dictionary containing all necessary data loaders and metadata
    """
    # Load dataset
    data = load_nlp_dataset(dataset_name)
    
    # Partition for clients
    client_texts, client_labels, client_sizes = partition_data(
        data['train_texts'], 
        data['train_labels'], 
        num_clients,
        seed
    )
    
    # Create validation and test loaders
    val_size = min(500, len(data['test_texts']) // 2)
    
    val_dataset = BackdoorTextDataset(
        data['test_texts'][:val_size],
        data['test_labels'][:val_size],
        tokenizer,
        max_length=max_length
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = BackdoorTextDataset(
        data['test_texts'][val_size:],
        data['test_labels'][val_size:],
        tokenizer,
        max_length=max_length
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create backdoor test loader
    backdoor_target = 1 if data['num_classes'] == 2 else 0
    backdoor_dataset = BackdoorTextDataset(
        data['test_texts'][val_size:],
        [backdoor_target] * len(data['test_texts'][val_size:]),
        tokenizer,
        max_length=max_length,
        trigger=trigger,
        target_label=backdoor_target,
        poison_rate=1.0
    )
    backdoor_loader = DataLoader(backdoor_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'client_texts': client_texts,
        'client_labels': client_labels,
        'client_sizes': client_sizes,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'backdoor_loader': backdoor_loader,
        'num_classes': data['num_classes'],
        'dataset_name': dataset_name
    }

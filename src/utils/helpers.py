"""
Utility Functions.

This module contains helper functions for the federated learning framework.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """
    Get the best available device.
    
    Returns:
        'cuda' if GPU available, else 'cpu'
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_output_dirs(base_dir: str = "outputs") -> Dict[str, str]:
    """
    Create output directories structure.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary of directory paths
    """
    dirs = {
        'base': base_dir,
        'metrics': os.path.join(base_dir, 'metrics'),
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def save_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to output file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_experiment_id() -> str:
    """
    Generate unique experiment identifier.
    
    Returns:
        Timestamp-based experiment ID
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_summary_dataframe(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create pandas DataFrame from experiment results.
    
    Args:
        results: Dictionary of experiment results
        
    Returns:
        DataFrame with summarized results
    """
    rows = []
    
    for exp_name, exp_results in results.items():
        if isinstance(exp_results, dict):
            for defense, metrics in exp_results.items():
                if isinstance(metrics, dict) and 'error' not in metrics:
                    row = {
                        'experiment': exp_name,
                        'defense': defense,
                        'clean_accuracy': metrics.get('clean_accuracy', {}).get('final', 0),
                        'backdoor_accuracy': metrics.get('backdoor_accuracy', {}).get('final', 0),
                        'attack_success_rate': metrics.get('attack_success_rate', {}).get('final', 0),
                    }
                    rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame) -> None:
    """
    Print formatted summary table.
    
    Args:
        df: DataFrame with results
    """
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    if len(df) > 0:
        print("\n🏆 BEST CLEAN ACCURACY:")
        top_ca = df.nlargest(5, 'clean_accuracy')
        print(top_ca[['experiment', 'defense', 'clean_accuracy']].to_string(index=False))
        
        print("\n🏆 LOWEST ATTACK SUCCESS RATE:")
        top_asr = df.nsmallest(5, 'attack_success_rate')
        print(top_asr[['experiment', 'defense', 'attack_success_rate']].to_string(index=False))
    else:
        print("No results to display.")
    
    print("="*70 + "\n")


def format_time(seconds: float) -> str:
    """
    Format seconds into human readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class ExperimentLogger:
    """
    Simple experiment logger.
    
    Logs experiment progress and saves to file.
    """
    
    def __init__(self, log_dir: str = "outputs/logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, 
            f"experiment_{generate_experiment_id()}.log"
        )
        self.logs = []
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(entry)
        print(entry)
    
    def save(self) -> None:
        """Save logs to file."""
        with open(self.log_file, 'w') as f:
            f.write('\n'.join(self.logs))

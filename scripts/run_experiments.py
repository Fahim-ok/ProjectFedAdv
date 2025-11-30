#!/usr/bin/env python3
"""
Batch Experiment Runner.

This script provides convenient functions for running specific
experiment configurations or custom subsets.

Usage:
    python scripts/run_experiments.py                    # Run default experiments
    python scripts/run_experiments.py --quick            # Quick test run
    python scripts/run_experiments.py --models distilbert roberta
    python scripts/run_experiments.py --datasets sst2 imdb
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS, AVAILABLE_DATASETS, AVAILABLE_DEFENSES
from src.federated import FederatedServer
from src.attacks import AttackTaxonomy
from src.utils import (
    set_seed,
    get_device,
    create_output_dirs,
    save_json,
    create_summary_dataframe,
    print_summary_table
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run federated learning experiments"
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=list(MODEL_CONFIGS.keys()),
        choices=list(MODEL_CONFIGS.keys()),
        help='Models to test'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=AVAILABLE_DATASETS,
        choices=AVAILABLE_DATASETS,
        help='Datasets to test'
    )
    parser.add_argument(
        '--defenses',
        nargs='+',
        default=AVAILABLE_DEFENSES,
        choices=AVAILABLE_DEFENSES,
        help='Defenses to test'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=8,
        help='Number of federated rounds'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Local epochs per round'
    )
    parser.add_argument(
        '--clients',
        type=int,
        default=5,
        help='Number of clients'
    )
    parser.add_argument(
        '--adv-ratio',
        type=float,
        default=0.4,
        help='Adversarial ratio'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with minimal settings'
    )
    
    return parser.parse_args()


def run_experiments(
    models: list,
    datasets: list,
    defenses: list,
    num_rounds: int = 8,
    local_epochs: int = 2,
    num_clients: int = 5,
    adversarial_ratio: float = 0.4,
    output_dir: str = 'outputs',
    seed: int = 42
) -> dict:
    """
    Run specified experiments.
    
    Args:
        models: List of model keys to test
        datasets: List of dataset names to test
        defenses: List of defense methods to test
        num_rounds: Number of federated rounds
        local_epochs: Local training epochs
        num_clients: Number of federated clients
        adversarial_ratio: Fraction of adversarial clients
        output_dir: Output directory
        seed: Random seed
        
    Returns:
        Dictionary of all results
    """
    set_seed(seed)
    device = get_device()
    create_output_dirs(output_dir)
    
    total_experiments = len(models) * len(datasets) * len(defenses)
    
    print("\n" + "="*70)
    print("🔬 EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Defenses: {defenses}")
    print(f"Total experiments: {total_experiments}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    all_results = {}
    experiment_count = 0
    
    for model_key in models:
        for dataset_name in datasets:
            dataset_results = {}
            
            for defense in defenses:
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] "
                      f"{model_key} | {dataset_name} | {defense}")
                
                try:
                    server = FederatedServer(
                        num_clients=num_clients,
                        adversarial_ratio=adversarial_ratio,
                        defense_method=defense,
                        model_key=model_key,
                        dataset_name=dataset_name,
                        device=device,
                        output_dir=output_dir,
                        seed=seed
                    )
                    
                    results = server.train(
                        num_rounds=num_rounds,
                        local_epochs=local_epochs
                    )
                    
                    server.save_results(results)
                    dataset_results[defense] = results
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
                    dataset_results[defense] = {'error': str(e)}
            
            all_results[f'{model_key}_{dataset_name}'] = dataset_results
    
    # Save comprehensive results
    save_json(all_results, os.path.join(output_dir, 'comprehensive_results.json'))
    
    # Generate summary
    df = create_summary_dataframe(all_results)
    df.to_csv(os.path.join(output_dir, 'summary_report.csv'), index=False)
    
    print_summary_table(df)
    
    return all_results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print attack taxonomy
    AttackTaxonomy.print_taxonomy()
    
    if args.quick:
        # Quick test configuration
        print("\n⚡ QUICK TEST MODE")
        run_experiments(
            models=['distilbert'],
            datasets=['sst2'],
            defenses=['krum'],
            num_rounds=3,
            local_epochs=1,
            num_clients=3,
            adversarial_ratio=0.33,
            output_dir=args.output_dir,
            seed=args.seed
        )
    else:
        # Run specified experiments
        run_experiments(
            models=args.models,
            datasets=args.datasets,
            defenses=args.defenses,
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            num_clients=args.clients,
            adversarial_ratio=args.adv_ratio,
            output_dir=args.output_dir,
            seed=args.seed
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main entry point for Federated Learning with Backdoor Attacks and Defenses.

Usage:
    python main.py                          # Run with default settings
    python main.py --model roberta          # Specify model
    python main.py --dataset imdb           # Specify dataset
    python main.py --defense flame          # Specify defense
    python main.py --rounds 10              # Specify rounds
    python main.py --full                   # Run all experiments
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ExperimentConfig,
    MODEL_CONFIGS,
    AVAILABLE_DATASETS,
    AVAILABLE_DEFENSES,
    print_config
)
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
        description="Federated Learning with Backdoor Attacks and Defenses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model distilbert --dataset sst2 --defense krum
  python main.py --model roberta --dataset imdb --defense flame --rounds 10
  python main.py --full  # Run all model-dataset-defense combinations
        """
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='distilbert',
        choices=list(MODEL_CONFIGS.keys()),
        help='Model architecture (default: distilbert)'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='sst2',
        choices=AVAILABLE_DATASETS,
        help='Dataset to use (default: sst2)'
    )
    parser.add_argument(
        '--defense', 
        type=str, 
        default='krum',
        choices=AVAILABLE_DEFENSES,
        help='Defense method (default: krum)'
    )
    parser.add_argument(
        '--rounds', 
        type=int, 
        default=8,
        help='Number of federated rounds (default: 8)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=2,
        help='Local epochs per round (default: 2)'
    )
    parser.add_argument(
        '--clients', 
        type=int, 
        default=5,
        help='Number of clients (default: 5)'
    )
    parser.add_argument(
        '--adv-ratio', 
        type=float, 
        default=0.4,
        help='Adversarial client ratio (default: 0.4)'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory (default: outputs)'
    )
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Run all experiment combinations'
    )
    
    return parser.parse_args()


def run_single_experiment(
    model_key: str,
    dataset_name: str,
    defense_method: str,
    num_rounds: int = 8,
    local_epochs: int = 2,
    num_clients: int = 5,
    adversarial_ratio: float = 0.4,
    lr: float = 2e-5,
    output_dir: str = 'outputs',
    seed: int = 42
) -> dict:
    """
    Run a single federated learning experiment.
    
    Returns:
        Dictionary of results
    """
    device = get_device()
    
    # Create server
    server = FederatedServer(
        num_clients=num_clients,
        adversarial_ratio=adversarial_ratio,
        defense_method=defense_method,
        model_key=model_key,
        dataset_name=dataset_name,
        device=device,
        output_dir=output_dir,
        seed=seed
    )
    
    # Train
    results = server.train(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        lr=lr
    )
    
    # Save results
    server.save_results(results)
    
    return results


def run_all_experiments(args):
    """Run all model-dataset-defense combinations."""
    print("\n" + "="*70)
    print("🔬 RUNNING FULL EXPERIMENT SUITE")
    print("="*70)
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Datasets: {AVAILABLE_DATASETS}")
    print(f"Defenses: {AVAILABLE_DEFENSES}")
    print("="*70 + "\n")
    
    all_results = {}
    create_output_dirs(args.output_dir)
    
    for model_key in MODEL_CONFIGS.keys():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_key.upper()}")
        print(f"{'='*70}")
        
        for dataset_name in AVAILABLE_DATASETS:
            print(f"\n{'─'*70}")
            print(f"Dataset: {dataset_name.upper()}")
            print(f"{'─'*70}")
            
            dataset_results = {}
            
            for defense in AVAILABLE_DEFENSES:
                print(f"\n🛡️  Defense: {defense.upper()}")
                
                try:
                    results = run_single_experiment(
                        model_key=model_key,
                        dataset_name=dataset_name,
                        defense_method=defense,
                        num_rounds=args.rounds,
                        local_epochs=args.epochs,
                        num_clients=args.clients,
                        adversarial_ratio=args.adv_ratio,
                        lr=args.lr,
                        output_dir=args.output_dir,
                        seed=args.seed
                    )
                    dataset_results[defense] = results
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    dataset_results[defense] = {'error': str(e)}
            
            all_results[f'{model_key}_{dataset_name}'] = dataset_results
    
    # Save comprehensive results
    comprehensive_path = os.path.join(args.output_dir, 'comprehensive_results.json')
    save_json(all_results, comprehensive_path)
    print(f"\n✅ All results saved to: {comprehensive_path}")
    
    # Generate summary
    df = create_summary_dataframe(all_results)
    summary_path = os.path.join(args.output_dir, 'summary_report.csv')
    df.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to: {summary_path}")
    
    print_summary_table(df)
    
    return all_results


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Print attack taxonomy
    AttackTaxonomy.print_taxonomy()
    
    if args.full:
        # Run all experiments
        run_all_experiments(args)
    else:
        # Run single experiment
        print("\n" + "="*70)
        print("🚀 SINGLE EXPERIMENT")
        print("="*70)
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Defense: {args.defense}")
        print(f"Rounds: {args.rounds}")
        print("="*70 + "\n")
        
        results = run_single_experiment(
            model_key=args.model,
            dataset_name=args.dataset,
            defense_method=args.defense,
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            num_clients=args.clients,
            adversarial_ratio=args.adv_ratio,
            lr=args.lr,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        for metric, values in results.items():
            print(f"\n{metric}:")
            for k, v in values.items():
                print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()

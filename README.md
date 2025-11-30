# 🛡️ Federated Learning with Backdoor Attacks and Defenses

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A modular framework for studying adversarial robustness in federated learning with Large Language Models (LLMs). This project implements various backdoor attacks and defense mechanisms for NLP classification tasks.

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Models & Datasets](#-models--datasets)
- [Attacks & Defenses](#-attacks--defenses)
- [Metrics](#-metrics)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Results](#-results)
- [Citation](#-citation)

## ✨ Features

- **3 LLM Models**: DistilBERT, RoBERTa, ALBERT
- **4 NLP Datasets**: SST-2, AG News, MNLI, Amazon
- **2 Backdoor Attacks**: Data Poisoning (DPA), Weight Poisoning (WPA)
- **5 Defense Methods**: Krum, Flame, Multi-Metrics, FedDMC, FLPruning
- **Modular Architecture**: Easy to extend with new models, attacks, or defenses
- **Comprehensive Logging**: Track experiments with detailed metrics

## 📁 Project Structure

```
federated_learning_backdoor/
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Python dependencies
├── 📄 main.py                   # Main entry point
│
├── 📁 config/                   # Configuration
│   ├── __init__.py
│   └── config.py                # All hyperparameters and settings
│
├── 📁 src/                      # Source code
│   ├── __init__.py
│   │
│   ├── 📁 models/               # LLM classifiers
│   │   ├── __init__.py
│   │   └── classifier.py        # Unified LLM classifier
│   │
│   ├── 📁 datasets/             # Data loading and processing
│   │   ├── __init__.py
│   │   └── data_loader.py       # Dataset utilities
│   │
│   ├── 📁 attacks/              # Backdoor attack implementations
│   │   ├── __init__.py
│   │   └── backdoor_attacks.py  # DPA and WPA attacks
│   │
│   ├── 📁 defenses/             # Defense mechanisms
│   │   ├── __init__.py
│   │   └── defense_methods.py   # All defense implementations
│   │
│   ├── 📁 federated/            # Federated learning components
│   │   ├── __init__.py
│   │   ├── client.py            # FL client
│   │   └── server.py            # FL server with aggregation
│   │
│   ├── 📁 metrics/              # Evaluation metrics
│   │   ├── __init__.py
│   │   └── evaluation.py        # CA, BA, ASR metrics
│   │
│   └── 📁 utils/                # Utility functions
│       ├── __init__.py
│       └── helpers.py           # Helper functions
│
├── 📁 scripts/                  # Experiment scripts
│   └── run_experiments.py       # Batch experiment runner
│
└── 📁 outputs/                  # Results (generated)
    ├── metrics/                 # Per-experiment metrics
    └── logs/                    # Experiment logs
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/federated_learning_backdoor.git
cd federated_learning_backdoor
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch transformers datasets accelerate scikit-learn numpy pandas tqdm
```

## ⚡ Quick Start

### Run a Single Experiment

```bash
# Default: DistilBERT + SST-2 + Krum
python main.py

# Specify options
python main.py --model roberta --dataset imdb --defense flame
```

### Quick Test (Minimal Settings)

```bash
python scripts/run_experiments.py --quick
```

### Run All Experiments

```bash
python main.py --full
```

## 📖 Usage

### Command Line Interface

```bash
python main.py [OPTIONS]

Options:
  --model      Model architecture: distilbert, roberta, albert
  --dataset    Dataset: sst2, imdb, agnews, mnli, yelp, amazon
  --defense    Defense method: krum, flame, multi_metrics, feddmc, flpruning
  --rounds     Number of federated rounds (default: 8)
  --epochs     Local epochs per round (default: 2)
  --clients    Number of clients (default: 5)
  --adv-ratio  Adversarial client ratio (default: 0.4)
  --lr         Learning rate (default: 2e-5)
  --seed       Random seed (default: 42)
  --output-dir Output directory (default: outputs)
  --full       Run all experiment combinations
```

### Python API

```python
from src.federated import FederatedServer

# Create server
server = FederatedServer(
    num_clients=5,
    adversarial_ratio=0.4,
    defense_method='krum',
    model_key='distilbert',
    dataset_name='sst2',
    device='cuda'
)

# Train
results = server.train(
    num_rounds=10,
    local_epochs=2,
    lr=2e-5
)

# Save results
server.save_results(results)
```

### Batch Experiments

```bash
# Run specific combinations
python scripts/run_experiments.py \
    --models distilbert roberta \
    --datasets sst2 imdb \
    --defenses krum flame

# Run with custom settings
python scripts/run_experiments.py \
    --models albert \
    --datasets agnews \
    --defenses multi_metrics feddmc flpruning \
    --rounds 10 \
    --clients 8
```

## 🤖 Models & Datasets

### Supported Models

| Model | HuggingFace ID | Type | Parameters |
|-------|---------------|------|------------|
| DistilBERT | `distilbert-base-uncased` | Encoder | 66M |
| RoBERTa | `roberta-base` | Encoder | 125M |
| ALBERT | `albert-base-v2` | Encoder | 12M |

### Supported Datasets

| Dataset | Task | Classes | Train Size |
|---------|------|---------|------------|
| SST-2 | Sentiment | 2 | 67K |
| IMDB | Sentiment | 2 | 25K |
| AG News | Topic | 4 | 120K |
| MNLI | NLI | 3 | 393K |
| Yelp | Sentiment | 2 | 560K |
| Amazon | Sentiment | 2 | 3.6M |

## ⚔️ Attacks & Defenses

### Backdoor Attacks

| Attack | Type | Description |
|--------|------|-------------|
| **DPA** | Data Poisoning | Injects trigger words into training data |
| **WPA** | Weight Poisoning | Scales model updates to amplify malicious gradients |

### Defense Methods

| Defense | Strategy | Reference |
|---------|----------|-----------|
| **Krum** | Distance-based selection | Blanchard et al., NeurIPS 2017 |
| **Flame** | Clustering + DP | Nguyen et al., USENIX 2022 |
| **Multi-Metrics** | Combined distance metrics | - |
| **FedDMC** | Dynamic clustering | - |
| **FLPruning** | Gradient pruning | - |

## 📊 Metrics

| Metric | Description | Optimal |
|--------|-------------|---------|
| **CA** (Clean Accuracy) | Accuracy on clean test data | Higher ↑ |
| **BA** (Backdoor Accuracy) | Accuracy on backdoored data | Lower ↓ |
| **ASR** (Attack Success Rate) | Backdoor trigger success rate | Lower ↓ |

## ⚙️ Configuration

All settings are in `config/config.py`:

```python
from config import ExperimentConfig

config = ExperimentConfig(
    model_key='distilbert',
    dataset_name='sst2',
    training=TrainingConfig(
        num_rounds=10,
        local_epochs=2,
        learning_rate=2e-5
    ),
    federated=FederatedConfig(
        num_clients=5,
        adversarial_ratio=0.4,
        defense_method='krum'
    )
)
```

## 💡 Examples

### Example 1: Compare Defenses on SST-2

```bash
python scripts/run_experiments.py \
    --models distilbert \
    --datasets sst2 \
    --defenses krum flame multi_metrics feddmc flpruning \
    --rounds 10
```

### Example 2: Test Model Robustness

```bash
python scripts/run_experiments.py \
    --models distilbert roberta albert \
    --datasets sst2 \
    --defenses krum \
    --rounds 8
```

### Example 3: Full Benchmark

```bash
python main.py --full --rounds 10 --output-dir benchmark_results
```

## 📈 Results

Results are saved in the `outputs/` directory:

```
outputs/
├── metrics/
│   ├── distilbert_sst2_krum.json
│   ├── roberta_imdb_flame.json
│   └── ...
├── comprehensive_results.json
└── summary_report.csv
```





## 🙏 Acknowledgments

- HuggingFace Transformers
- PyTorch
- The federated learning research community

---

<p align="center">
</p>

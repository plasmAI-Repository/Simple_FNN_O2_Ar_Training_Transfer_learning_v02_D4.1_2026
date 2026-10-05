# Physics-Based Plasma Etching Rate Prediction & Transfer Learning (v02 of physics based models)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository provides the complete PyTorch implementation of Multi-Layer Perceptron (MLP) neural network models and transfer learning workflows for predicting spatial semiconductor plasma etch rates based on COMSOL v02 physics-based simulations.

The model maps **2 operational input parameters** (RF Power and Chamber Pressure) to **10 spatial etch rate points** across the wafer surface, transferring learned physical representations between different plasma gas chemistries (**Oxygen $O_2$** and **Argon $Ar$**).

---

## Features

- **COMSOL v02 Physics Dataset**: Trained on high-fidelity COMSOL v02 Multiphysics plasma simulation datasets.
- **2 Input Operating Parameters**: RF Power ($W$) and Chamber Pressure ($Torr$).
- **10 Spatial Etch Rate Predictions**: Output vector predicting local etch rates across 10 spatial locations (`Point1` to `Point10`).
- **Transfer Learning Support**:
  - **Source Baseline Training**: Train from scratch on source domain ($O_2$).
  - **Full Fine-Tuning**: Transfer source weights to target domain ($Ar$) with reduced learning rates.
  - **Layer Freezing**: Freeze early representation hidden layers to preserve domain-invariant feature extraction.
  - **Layer Unfreezing**: Dynamic layer unfreezing after $N$ epochs.
- **Automated Parameter Normalization**: `generate_stats.py` tool to auto-calculate min-max bounds for new datasets.
- **Complete Evaluation Suite**: Automated evaluation generating unscaled predictions (`pred.csv`), quantitative metric metrics (`metrics.json`), $R^2$ spatial score bar plots (`r2.png`), and residual percentage distributions (`residuals.png`).

---

## Repository Structure
├── data/                                      # Dataset directory
│   ├── train_data_no_head_outer_corner_O2.csv # Training set for Oxygen (O2)
│   ├── test_data_no_head_outer_corner_O2.csv  # Test set for Oxygen (O2)
│   ├── train_data_no_head_outer_corner_Ar.csv # Training set for Argon (Ar)
│   └── test_data_no_head_outer_corner_Ar.csv  # Test set for Argon (Ar)
├── generate_stats.py                          # Utility script to calculate min-max normalization stats
├── model.py                                   # PyTorch MLP regression network with freezing support
├── data_loader.py                             # PyTorch Dataset & DataLoader utilities
├── loss.py                                    # Weighted MSE and Huber loss functions
├── utils.py                                   # Training loops, early stopping, metrics, & plotting helpers
├── train.py                                   # Training script (Baseline, Fine-Tuning, Layer Freezing)
├── test.py                                    # Model evaluation and visualization script
├── stats_min_max.json                         # Computed min-max bounds for normalization
├── pyproject.toml                             # Poetry configuration & dependencies
├── README.md                                  # Repository documentation
└── AGENT_INSTRUCTIONS.md                      # AI Agent execution and chemistry adaptation guide
```

---

## Installation

### Option 1: Standard Pip / Virtualenv
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install torch torchvision matplotlib scikit-learn pandas seaborn
```

### Option 2: Using Poetry
```bash
poetry install
poetry shell
```
---

## Quickstart & Workflows

### Step 1: Min-Max Statistics Generation
Calculate feature and target normalization bounds for all dataset files:
```bash
python generate_stats.py --data_dir ./data --output stats_min_max.json
```

---

### Step 2: Source Model Baseline Training ($O_2$)
Train a baseline MLP model on the Oxygen source chemistry dataset from scratch:
```bash
python train.py --gas O2 --mode baseline --epochs 700 --lr 0.000962 --output_dir ./results/O2/baseline
```
- Model checkpoint saved to: `./results/O2/baseline/trained_model.pth`
- Loss plot saved to: `./results/O2/baseline/loss_progression.png`

---

### Step 3: Transfer Learning to Target Chemistry ($Ar$)

Three transfer learning strategies are supported:

#### Strategy A: Full Fine-Tuning
Loads the pretrained Oxygen ($O_2$) model weights and fine-tunes all network parameters on the Argon ($Ar$) dataset with a reduced learning rate ($\text{lr} = \frac{\text{lr}_{\text{base}}}{20}$):
```bash
python train.py \
    --gas Ar \
    --mode fine_tune \
    --pretrained_path ./results/O2/baseline/trained_model.pth \
    --epochs 700 \
    --output_dir ./results/Ar/fine_tune
```

#### Strategy B: Layer Freezing
Loads the pretrained Oxygen model weights and freezes initial hidden layers (e.g. Layer 0) while training remaining layers on Argon data:
```bash
python train.py \
    --gas Ar \
    --mode freeze \
    --freeze_layers 0 \
    --pretrained_path ./results/O2/baseline/trained_model.pth \
    --epochs 700 \
    --output_dir ./results/Ar/freeze
```

#### Strategy C: Gradual Layer Unfreezing
Starts with frozen layers and unfreezes them dynamically after 100 epochs:
```bash
python train.py \
    --gas Ar \
    --mode freeze \
    --freeze_layers 0 1 \
    --unfreeze \
    --pretrained_path ./results/O2/baseline/trained_model.pth \
    --epochs 700 \
    --output_dir ./results/Ar/unfreeze
```

---

### Step 4: Model Evaluation & Metric Plotting

Evaluate trained models on the independent test dataset:

```bash
# Evaluate O2 Baseline
python test.py --gas O2 --mode baseline --model_path ./results/O2/baseline/trained_model.pth --output_dir ./results/O2/baseline

# Evaluate Ar Fine-Tuned Model
python test.py --gas Ar --mode fine_tune --model_path ./results/Ar/fine_tune/trained_model.pth --output_dir ./results/Ar/fine_tune
```

#### Generated Evaluation Outputs:
- `pred.csv`: Unscaled predicted etch rates for all 10 spatial points.
- `metrics.json`: Summary JSON containing overall $R^2$, MAE, MSE, and per-point scores.
- `r2.png`: Bar chart of $R^2$ accuracy across spatial points.
- `residuals.png`: Distribution curve of relative error percentages ($|\text{sim} - \text{pred}| / \text{sim} \times 100\%$).

---

## Evaluation & Metrics Summary

The scripts produce formatted JSON logs for model comparisons:

```json
{
    "gas": "Ar",
    "mode": "fine_tune",
    "overall_unscaled_r2": 0.9852,
    "overall_unscaled_mae": 0.0021,
    "overall_unscaled_mse": 0.000014,
    "inference_time_seconds": 0.0084,
    "point_r2_scores": {
        "Point1_EtchRateO": 0.991,
        "Point2_EtchRateO": 0.989,
        "Point10_EtchRateO": 0.974
    }
}
```

---

## AI Agent Execution Instructions

If you are an autonomous AI agent or LLM performing analysis or extending this repository to **another chemistry** (e.g. $CF_4$, $SF_6$, $N_2$), please refer to [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md).

---

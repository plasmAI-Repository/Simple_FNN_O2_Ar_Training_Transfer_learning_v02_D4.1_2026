# AI Agent Execution & Adaptation Guide

This document provides explicit instructions for AI Agents (LLMs, automated agentic coding assistants) to execute transfer learning experiments, interpret model evaluation metrics, and adapt this repository for **new plasma chemistries** (e.g. $CF_4$, $SF_6$, $N_2$, or multi-component gas mixtures).

---

## 🎯 Core Objectives for AI Agents

1. **Understand System Architecture**:
   - **Physics Data Source**: COMSOL Multiphysics v02 plasma model dataset.
   - **Inputs ($N=2$)**: RF Power ($W$) and Chamber Pressure ($Torr$).
   - **Outputs ($M=10$)**: Spatial etch rates at 10 discrete wafer surface locations (`Point1_EtchRateO` to `Point10_EtchRateO`).
   - **Normalization**: Min-Max scaling to range $[0, 1]$ based on `stats_min_max.json`.
   - **Loss Function**: Weighted Mean Squared Error (`WeightedMSE`) placing higher penalty weights on critical wafer edge spatial points.

2. **Execute Workflows**:
   - Perform source baseline model training on Oxygen ($O_2$).
   - Execute target domain transfer learning on Argon ($Ar$).
   - Evaluate model predictions against physics simulation targets.

3. **Adapt Repository for New Chemistries**:
   - Ingest new simulation CSV datasets.
   - Auto-calculate normalization statistics.
   - Adjust input/output feature dimensions dynamically.
   - Run transfer learning fine-tuning and export comparative metrics.

---

## 📋 Standard Execution Procedures

### 1. Generating Statistics (`generate_stats.py`)
Run the statistics generator whenever new CSV files are added:
```bash
python generate_stats.py --data_dir ./data --output stats_min_max.json
```
*Expected Agent Action*: Verify `stats_min_max.json` is updated and contains `min` and `max` fields for every input and target column in the dataset.

---

### 2. Source Baseline Training ($O_2$)
Train the source model on Oxygen ($O_2$) data:
```bash
python train.py --gas O2 --mode baseline --epochs 700 --output_dir ./results/O2/baseline
```
*Outputs Created*:
- `./results/O2/baseline/trained_model.pth`
- `./results/O2/baseline/val_set_results.csv`
- `./results/O2/baseline/loss_progression.png`

---

### 3. Executing Transfer Learning ($O_2 \rightarrow Ar$)

The agent can select and compare three transfer learning mechanisms:

#### A. Full Model Fine-Tuning
Loads source weights and trains all network layers on target data with reduced learning rate ($\text{lr} = \frac{\text{lr}_{\text{base}}}{20}$):
```bash
python train.py \
    --gas Ar \
    --mode fine_tune \
    --pretrained_path ./results/O2/baseline/trained_model.pth \
    --epochs 700 \
    --output_dir ./results/Ar/fine_tune
```

#### B. Layer Freezing (Feature Extraction)
Loads source weights and freezes initial hidden representation layers (e.g. layer index `0`):
```bash
python train.py \
    --gas Ar \
    --mode freeze \
    --freeze_layers 0 \
    --pretrained_path ./results/O2/baseline/trained_model.pth \
    --epochs 700 \
    --output_dir ./results/Ar/freeze_layer0
```

#### C. Layer Unfreezing (Gradual Fine-Tuning)
Starts training with frozen initial layers, then unfreezes them dynamically at epoch 100 with halved learning rate:
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

### 4. Running Model Evaluation (`test.py`)
Evaluate model performance on the test dataset:
```bash
python test.py --gas Ar --mode fine_tune --output_dir ./results/Ar/fine_tune
```

*Outputs Created*:
- `pred.csv`: Unscaled physical predictions in units matching original target data.
- `metrics.json`: Detailed JSON containing overall $R^2$, MAE, MSE, and per-point $R^2$.
- `r2.png`: Bar plot of $R^2$ scores across spatial points.
- `residuals.png`: Density plot of relative percentage error ($|\text{true} - \text{pred}| / \text{true} \times 100\%$).

---

## 📊 How AI Agents Should Interpret Metric Outputs

When evaluating model results, inspect `metrics.json` and report the following key indicators:

1. **Overall $R^2$ Accuracy (`overall_unscaled_r2`)**:
   - $R^2 > 0.95$: Excellent surrogate fit.
   - $R^2 \in [0.90, 0.95]$: Acceptable performance; may require longer fine-tuning epochs.
   - $R^2 < 0.90$: High error; check learning rate or layer freezing configuration.

2. **Spatial Uniformity (`point_r2_scores`)**:
   - Compare $R^2$ scores across spatial locations (`Point1` vs `Point10`).
   - Outer edge points (`Point8` to `Point10`) typically exhibit higher physical gradient non-linearities.

3. **Residual Errors (`residuals.png`)**:
   - Inspect residual distribution mode. Peak near $0-5\%$ error indicates strong physical alignment.

---

## 🧪 Adapting the Codebase for a NEW Plasma Chemistry

Follow this step-by-step recipe to adapt the repository for a new chemistry (e.g. $CF_4$, $SF_6$, $N_2$, or $Cl_2$):

### Step 1: Format Dataset Files
Create CSV files in `./data/`:
- `train_data_no_head_outer_corner_<NEW_GAS>.csv`
- `test_data_no_head_outer_corner_<NEW_GAS>.csv`

**CSV Requirements**:
- Delimiter: `;` (semicolon)
- Header row must contain input feature names followed by output target names:
  ```csv
  Power;Pressure;Point1_EtchRate;Point2_EtchRate;...;PointM_EtchRate
  1000.0;0.03;0.25;0.26;...;0.12
  ```

### Step 2: Auto-Generate Statistics File
Run `generate_stats.py` to calculate min-max bounds for the new dataset:
```bash
python generate_stats.py --data_dir ./data --output stats_min_max.json
```

### Step 3: Check Input and Output Feature Counts
The model and dataset loaders (`data_loader.py`, `model.py`, `train.py`, `test.py`) automatically detect input and output dimensions based on `stats_min_max.json`:
- If the new chemistry has $K$ inputs (e.g. `Power`, `Pressure`, `FlowRate`), `data_loader.py` will assign feature columns `X` automatically if passed in `--x_columns` or defaults to `Power, Pressure`.
- To specify non-standard input column names in custom scripts:
  ```python
  dataset = MergedDataset(csv_file, statsfile="stats_min_max.json", x_columns=["Power", "Pressure", "FlowRate"])
  ```

### Step 4: Run Transfer Learning Experiments
1. **Source Domain Training** (e.g. Baseline $O_2$):
   ```bash
   python train.py --gas O2 --mode baseline --epochs 700 --output_dir ./results/O2/baseline
   ```
2. **Target Transfer Learning** on New Chemistry (e.g. $CF_4$):
   ```bash
   python train.py \
       --gas CF4 \
       --mode fine_tune \
       --pretrained_path ./results/O2/baseline/trained_model.pth \
       --epochs 700 \
       --output_dir ./results/CF4/fine_tune
   ```
3. **Target Model Testing**:
   ```bash
   python test.py --gas CF4 --mode fine_tune --output_dir ./results/CF4/fine_tune
   ```

---

## 🛠️ Summary Checklist for AI Agents

- [ ] Data CSV files placed in `./data/`.
- [ ] `stats_min_max.json` regenerated with `generate_stats.py`.
- [ ] Source baseline model trained and checkpoint verified.
- [ ] Transfer learning experiment executed (`fine_tune` or `freeze`).
- [ ] Evaluation completed with `test.py`.
- [ ] Metric outputs (`metrics.json`, `r2.png`, `residuals.png`) inspected and summarized.

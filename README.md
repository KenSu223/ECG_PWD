# ECG_PWD: Fetal ECG to Doppler Envelope Modeling

This repository contains multiple deep-learning pipelines for generating fetal Doppler-like pulse-wave envelopes from ECG-derived inputs, plus evaluation and visualization tooling.

## Reorganization status

The codebase now has a **canonical package layout** under `src/ecg_pwd/` and **legacy compatibility shims** at previous script paths.

- Canonical code: `src/ecg_pwd/{fusion,single_channel,auto_reg,common}/`
- Legacy script/module paths still work (for example `Fetal-maternal-fusion/src/train_fusion_cus_loss.py`, `single_channel/src/train_wavenet.py`, `auto_reg/main.py`, `metrics_utils.py`)

## What this repository includes

- **`single_channel/`**: one-channel ECG-to-Doppler WaveNet experiments.
- **`Fetal-maternal-fusion/`**: two-channel fetal+maternal fusion models, including attention variants.
- **`auto_reg/`**: autoregressive two-channel WaveNet training/inference.
- **`data/`, `notebooks/`, `fECGExtraction/`**: data prep, exploratory notebooks, and MATLAB utilities.

A detailed per-file summary is available in `docs/FILE_SUMMARY.md`.

## Repository layout (high level)

- `src/ecg_pwd/fusion/`: two-channel fetal/maternal fusion training, evaluation, and visualization modules.
- `src/ecg_pwd/single_channel/`: one-channel training/evaluation modules.
- `src/ecg_pwd/auto_reg/`: autoregressive training/inference modules.
- `src/ecg_pwd/common/`: shared/legacy metric utilities.
- `Fetal-maternal-fusion/src/`, `single_channel/src/`, `auto_reg/`: compatibility shims for older script paths/imports.
- `data/`: preprocessing/visualization notebooks and MATLAB plotting script.
- `notebooks/`: root-level notebooks moved here for cleaner organization.
- `metrics.py`, `metrics_utils.py`: standalone/utility waveform metric functions.

## Environment setup

> Tested as a Python/TensorFlow-style workflow. Use a dedicated environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional GPU check

```bash
python Fetal-maternal-fusion/src/GPU_env_check.py
# or canonical module form:
PYTHONPATH=src python -m ecg_pwd.fusion.GPU_env_check
```

## Data prerequisites

Most train/eval scripts expect NumPy arrays in a `WaveNet_beat/data` folder:

- `X.npy`: ECG input segments (commonly shape `(N, T, C)`)
- `Y.npy`: target Doppler envelopes (commonly `(N, T)` or `(N, T, 1)`)
- `PATIENT_IDS.npy`: patient IDs aligned to segments
- `RISK_GROUPS.npy` (optional in some scripts)

Most scripts also rely on index files in `Fetal-maternal-fusion/src/`, for example:

- `idx_train.npy`, `idx_val.npy`
- `idx_train_risk.npy`, `idx_val_risk_normal.npy`
- `idx_train_mix_risk.npy`, `idx_val_mix_risk.npy`

## Quick start commands

### 1) Single-channel training

```bash
python single_channel/src/train_wavenet.py \
  --data_dir single_channel/src/WaveNet_beat/data \
  --train_idx_file Fetal-maternal-fusion/src/idx_train_risk.npy \
  --val_idx_file Fetal-maternal-fusion/src/idx_val_risk_normal.npy \
  --epochs 50 --batch_size 32 --lr 1e-3
```

Canonical module form:

```bash
PYTHONPATH=src python -m ecg_pwd.single_channel.train_wavenet \
  --data_dir single_channel/src/WaveNet_beat/data
```

Outputs are typically written under `single_channel/src/WaveNet_beat/`.

### 2) Fusion (two-channel) training

**MAE baseline:**

```bash
python Fetal-maternal-fusion/src/train_fusion_wavenet.py \
  --data_dir Fetal-maternal-fusion/src/WaveNet_beat/data \
  --train_idx_file Fetal-maternal-fusion/src/idx_train_risk.npy \
  --val_idx_file Fetal-maternal-fusion/src/idx_val_risk_normal.npy \
  --epochs 50 --batch_size 32 --lr 1e-3
```

**Flexible composite-loss training:**

```bash
python Fetal-maternal-fusion/src/train_fusion_cus_loss.py \
  --data_dir Fetal-maternal-fusion/src/WaveNet_beat/data \
  --train_idx_file Fetal-maternal-fusion/src/idx_train_mix_risk.npy \
  --val_idx_file Fetal-maternal-fusion/src/idx_val_mix_risk.npy \
  --loss_type flexible --use_base true --use_derivative true --use_corr true \
  --alpha_derivative 0.5 --alpha_corr 0.5 \
  --epochs 100 --batch_size 32 --use_lr_schedule dynamic
```

Canonical module form:

```bash
PYTHONPATH=src python -m ecg_pwd.fusion.train_fusion_cus_loss \
  --data_dir Fetal-maternal-fusion/src/WaveNet_beat/data
```

### 3) Autoregressive training

```bash
python auto_reg/main.py \
  --data_dir Fetal-maternal-fusion/src/WaveNet_beat/data \
  --train_idx_file Fetal-maternal-fusion/src/idx_train_mix_risk.npy \
  --val_idx_file Fetal-maternal-fusion/src/idx_val_mix_risk.npy \
  --loss_type flexible --use_base true --use_derivative true --use_corr true \
  --epochs 50 --batch_size 32 --lr 1e-3
```

Canonical module form:

```bash
PYTHONPATH=src python -m ecg_pwd.auto_reg.main \
  --data_dir Fetal-maternal-fusion/src/WaveNet_beat/data
```

Outputs are written under `auto_reg/WaveNet_beat/`.

## Evaluation and comparison workflows

### K-fold evaluation (ML + clinical metrics)

```bash
python Fetal-maternal-fusion/src/evaluate_wavenet_kfold.py
# or:
PYTHONPATH=src python -m ecg_pwd.fusion.evaluate_wavenet_kfold
```

By default this writes summaries under `Fetal-maternal-fusion/src/evaluation_results/`.

### Compare multiple model configurations

```bash
python Fetal-maternal-fusion/src/compare_models.py --comparison_type architectures
python Fetal-maternal-fusion/src/compare_models.py --comparison_type losses
python Fetal-maternal-fusion/src/compare_models.py --comparison_type learning_rates
# or:
PYTHONPATH=src python -m ecg_pwd.fusion.compare_models --comparison_type architectures
```

### Generate summary plots from CSV outputs

```bash
python Fetal-maternal-fusion/src/visualize_results.py \
  --base_dir Fetal-maternal-fusion/src/model_comparison_results
# or:
PYTHONPATH=src python -m ecg_pwd.fusion.visualize_results \
  --base_dir Fetal-maternal-fusion/src/model_comparison_results
```

## Four-model training and checkpoint inference

### Train 4 model variants and save an overlay plot

```bash
python Fetal-maternal-fusion/src/train_and_visualize_four_models.py \
  --data_dir Fetal-maternal-fusion/src/WaveNet_beat/data \
  --train_idx_file Fetal-maternal-fusion/src/idx_train.npy \
  --val_idx_file Fetal-maternal-fusion/src/idx_val.npy
```

### Load saved checkpoints and compare predictions

```bash
python Fetal-maternal-fusion/src/predict_from_checkpoints.py \
  --data_dir Fetal-maternal-fusion/src/WaveNet_beat/data \
  --val_idx_file Fetal-maternal-fusion/src/idx_val.npy \
  --checkpoint_dir Fetal-maternal-fusion/src/WaveNet_beat/plots/model_checkpoints
```

## Compatibility notes

- Old paths are preserved as shims and forward to `src/ecg_pwd`.
- Existing automation that runs legacy scripts should continue to work unchanged.
- New development should target `src/ecg_pwd/*` modules.

## Notebooks and MATLAB utilities

- `notebooks/PWD_processing.ipynb`: processing-focused notebook.
- `notebooks/metrics_notebook.ipynb`: metric exploration notebook.
- `data/*.ipynb`: data extraction/preparation/visualization notebooks.
- `fECGExtraction/fetalECGextraction.m`: MATLAB fetal ECG extraction demo.
- `data/Leipzig_Data_Visualization.m`: MATLAB visualization helper.

## Notes and caveats

- Several scripts still contain historical absolute defaults. Prefer explicitly passing paths as shown above.
- Legacy scripts (`main.py`, `analysis.py`, `tsne.py` in some folders) are kept for reproducibility and exploratory work.

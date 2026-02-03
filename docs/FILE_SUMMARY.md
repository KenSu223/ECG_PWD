# File Summary

This repository now uses a canonical package layout under `src/ecg_pwd/`, while preserving legacy script/module paths through compatibility shims.

## Canonical code (active implementation)

## `src/ecg_pwd/fusion/`

| Path | Purpose |
|---|---|
| `src/ecg_pwd/fusion/modules.py` | Core two-channel architectures, custom losses, callbacks, and plotting helpers. |
| `src/ecg_pwd/fusion/layers.py` | Shared layer wrappers and convenience constructors for fusion models. |
| `src/ecg_pwd/fusion/train_fusion_wavenet.py` | MAE-based two-channel training pipeline with split validation/checkpointing. |
| `src/ecg_pwd/fusion/train_fusion_wavenet_composite.py` | Composite / shape-preserving loss training variant. |
| `src/ecg_pwd/fusion/train_fusion_wavenet_composite_log.py` | Composite training with explicit per-component loss logging. |
| `src/ecg_pwd/fusion/train_fusion_cus_loss.py` | Flexible multi-term loss training entry point. |
| `src/ecg_pwd/fusion/train_and_visualize_four_models.py` | Trains four model variants and generates overlay comparison plots. |
| `src/ecg_pwd/fusion/predict_from_checkpoints.py` | Loads saved checkpoints and plots stacked predictions. |
| `src/ecg_pwd/fusion/evaluate_wavenet_kfold.py` | K-fold model evaluation (ML + clinical metrics). |
| `src/ecg_pwd/fusion/compare_models.py` | Multi-configuration comparison runner. |
| `src/ecg_pwd/fusion/visualize_results.py` | Plot/report utilities for evaluation outputs. |
| `src/ecg_pwd/fusion/eval_single.py` | Example usage workflows for evaluation/comparison APIs. |
| `src/ecg_pwd/fusion/GPU_env_check.py` | TensorFlow runtime and GPU diagnostic utility. |
| `src/ecg_pwd/fusion/AF_metrics_utils.py` | Waveform metric helper functions. |
| `src/ecg_pwd/fusion/main.py` | Legacy-style baseline fusion training script kept for reproducibility. |
| `src/ecg_pwd/fusion/analysis.py` | Legacy PSD comparison analysis script. |
| `src/ecg_pwd/fusion/tsne.py` | Legacy t-SNE exploration script. |
| `src/ecg_pwd/fusion/model_eval_pipeline.py` | Placeholder for higher-level evaluation orchestration. |

## `src/ecg_pwd/single_channel/`

| Path | Purpose |
|---|---|
| `src/ecg_pwd/single_channel/modules.py` | Single-channel WaveNet architectures and visualization helpers. |
| `src/ecg_pwd/single_channel/layers.py` | Shared layer wrappers for single-channel models. |
| `src/ecg_pwd/single_channel/train_wavenet.py` | Main single-channel training script with split checks/log exports. |
| `src/ecg_pwd/single_channel/AF_metrics_utils.py` | Signal-quality metric helpers. |
| `src/ecg_pwd/single_channel/main.py` | Legacy baseline training script from NPZ shards. |
| `src/ecg_pwd/single_channel/analysis.py` | Legacy PSD analysis script. |
| `src/ecg_pwd/single_channel/tsne.py` | Legacy t-SNE exploration script. |

## `src/ecg_pwd/auto_reg/`

| Path | Purpose |
|---|---|
| `src/ecg_pwd/auto_reg/main.py` | Main autoregressive training + inference entry point. |
| `src/ecg_pwd/auto_reg/modules.py` | AR architecture and plotting/scalogram utilities. |
| `src/ecg_pwd/auto_reg/layers.py` | Layer wrapper utilities for AR models. |

## `src/ecg_pwd/common/`

| Path | Purpose |
|---|---|
| `src/ecg_pwd/common/metrics.py` | Legacy standalone metrics workflow. |
| `src/ecg_pwd/common/metrics_utils.py` | Reusable waveform/spectral metric utilities. |

## Compatibility shims (legacy paths)

These locations are kept as lightweight forwarding modules/scripts:

- `Fetal-maternal-fusion/src/*.py`
- `single_channel/src/*.py`
- `auto_reg/*.py`
- `metrics.py`
- `metrics_utils.py`

They forward imports/script execution to the canonical modules under `src/ecg_pwd/`.

## Notebooks and data utilities

| Path | Purpose |
|---|---|
| `notebooks/PWD_processing.ipynb` | Pulse-wave Doppler processing workflow notebook. |
| `notebooks/metrics_notebook.ipynb` | Metric exploration notebook. |
| `data/preparation.ipynb` | Data preparation notebook. |
| `data/data_extraction.ipynb` | Data extraction notebook. |
| `data/DopplerGAN_Visualization.ipynb` | Doppler model visualization notebook. |
| `Fetal-maternal-fusion/src/fECG_upper_envelope.ipynb` | Fusion-side feature/label processing notebook. |
| `Fetal-maternal-fusion/src/load_and_gen.ipynb` | Fusion-side loading/generation notebook. |
| `Fetal-maternal-fusion/src/Evaluation.ipynb` | Fusion-side interactive evaluation notebook. |
| `single_channel/src/fECG_upper_envelope.ipynb` | Single-channel feature/label processing notebook. |
| `single_channel/src/Evaluation.ipynb` | Single-channel interactive evaluation notebook. |
| `single_channel/src/physio_eval.ipynb` | Physiological/clinical evaluation notebook. |
| `data/Leipzig_Data_Visualization.m` | MATLAB Doppler/fECG visualization utility. |
| `fECGExtraction/fetalECGextraction.m` | MATLAB fetal ECG extraction demo. |

## Artifact directories (generated outputs)

- `auto_reg/WaveNet_beat/`
- `Fetal-maternal-fusion/src/WaveNet_beat/`
- `single_channel/src/WaveNet_beat/`
- `Fetal-maternal-fusion/src/evaluation_results/`

These primarily store trained weights, logs, generated data, and plots.

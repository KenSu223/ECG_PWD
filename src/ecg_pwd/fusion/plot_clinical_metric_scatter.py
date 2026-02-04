"""Create per-segment scatterplots of clinical Doppler metrics for real vs generated signals.

Usage:
python Fetal-maternal-fusion/src/plot_clinical_metric_scatter.py \
  --data_dir Fetal-maternal-fusion/src/WaveNet_beat/data \
  --val_idx_file Fetal-maternal-fusion/src/idx_val.npy \
  --train_idx_file Fetal-maternal-fusion/src/idx_train.npy \
  --checkpoint_path "Fetal-maternal-fusion/src/WaveNet_beat/plots/model_checkpoints/(d)_two-channel_(combined_attention).weights.h5" \
  --output_dir Fetal-maternal-fusion/src/evaluation_results
"""


from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from . import modules
from .predict_from_checkpoints import preprocess_ecg_inputs, smooth_doppler_envelope


METRIC_NAMES = ("PI", "RI", "SD_Ratio", "Heart_Rate", "PSV", "EDV", "TAMX")


class DopplerClinicalMetrics:
    """Calculate clinically relevant metrics from fetal Doppler envelopes."""

    def __init__(self, sampling_rate: float = 284.0):
        self.fs = sampling_rate

    def detect_peaks_and_troughs(self, envelope: np.ndarray, min_distance_samples: int | None = None):
        if min_distance_samples is None:
            min_distance_samples = int(0.3 * self.fs)

        peak_indices, _ = signal.find_peaks(
            envelope,
            distance=min_distance_samples,
            prominence=0.1 * np.ptp(envelope),
        )
        trough_indices, _ = signal.find_peaks(
            -envelope,
            distance=min_distance_samples,
            prominence=0.1 * np.ptp(envelope),
        )
        return peak_indices, trough_indices

    def _extract_edv_values(self, envelope: np.ndarray, peak_indices: np.ndarray, trough_indices: np.ndarray):
        edv_values = []
        for i, peak_idx in enumerate(peak_indices[:-1]):
            next_peak = peak_indices[i + 1]
            between = trough_indices[(trough_indices > peak_idx) & (trough_indices < next_peak)]
            if len(between) > 0:
                edv_values.append(envelope[between[-1]])
        return edv_values

    def calculate_pulsatility_index(self, envelope: np.ndarray):
        peak_indices, trough_indices = self.detect_peaks_and_troughs(envelope)
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return np.nan
        psv = np.mean(envelope[peak_indices])
        edv_values = self._extract_edv_values(envelope, peak_indices, trough_indices)
        if len(edv_values) == 0:
            return np.nan
        tamx = np.mean(envelope)
        if tamx == 0:
            return np.nan
        return (psv - np.mean(edv_values)) / tamx

    def calculate_resistance_index(self, envelope: np.ndarray):
        peak_indices, trough_indices = self.detect_peaks_and_troughs(envelope)
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return np.nan
        psv = np.mean(envelope[peak_indices])
        edv_values = self._extract_edv_values(envelope, peak_indices, trough_indices)
        if len(edv_values) == 0 or psv == 0:
            return np.nan
        return (psv - np.mean(edv_values)) / psv

    def calculate_sd_ratio(self, envelope: np.ndarray):
        peak_indices, trough_indices = self.detect_peaks_and_troughs(envelope)
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return np.nan
        psv = np.mean(envelope[peak_indices])
        edv_values = self._extract_edv_values(envelope, peak_indices, trough_indices)
        if len(edv_values) == 0:
            return np.nan
        edv = np.mean(edv_values)
        if edv == 0:
            return np.nan
        return psv / edv

    def calculate_heart_rate(self, envelope: np.ndarray):
        peak_indices, _ = self.detect_peaks_and_troughs(envelope)
        if len(peak_indices) < 2:
            return np.nan
        intervals = np.diff(peak_indices) / self.fs
        avg_interval = np.mean(intervals)
        if avg_interval == 0:
            return np.nan
        return 60.0 / avg_interval

    def calculate_peak_systolic_velocity(self, envelope: np.ndarray):
        peak_indices, _ = self.detect_peaks_and_troughs(envelope)
        if len(peak_indices) == 0:
            return np.nan
        return np.mean(envelope[peak_indices])

    def calculate_end_diastolic_velocity(self, envelope: np.ndarray):
        peak_indices, trough_indices = self.detect_peaks_and_troughs(envelope)
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return np.nan
        edv_values = self._extract_edv_values(envelope, peak_indices, trough_indices)
        if len(edv_values) == 0:
            return np.nan
        return np.mean(edv_values)

    def calculate_all_metrics(self, envelope: np.ndarray):
        return {
            "PI": self.calculate_pulsatility_index(envelope),
            "RI": self.calculate_resistance_index(envelope),
            "SD_Ratio": self.calculate_sd_ratio(envelope),
            "Heart_Rate": self.calculate_heart_rate(envelope),
            "PSV": self.calculate_peak_systolic_velocity(envelope),
            "EDV": self.calculate_end_diastolic_velocity(envelope),
            "TAMX": np.mean(envelope),
        }


def squeeze_last_channel(arr: np.ndarray):
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr.squeeze(axis=-1)
    return arr


def compute_segment_metric_table(
    real_dopplers: np.ndarray,
    generated_dopplers: np.ndarray,
    segment_indices: np.ndarray,
    sampling_rate: float,
):
    calculator = DopplerClinicalMetrics(sampling_rate=sampling_rate)
    rows = []

    for local_idx, global_idx in enumerate(segment_indices):
        real_metrics = calculator.calculate_all_metrics(real_dopplers[local_idx])
        gen_metrics = calculator.calculate_all_metrics(generated_dopplers[local_idx])
        row = {
            "segment_index_local": int(local_idx),
            "segment_index_global": int(global_idx),
        }
        for metric in METRIC_NAMES:
            row[f"{metric}_real"] = real_metrics[metric]
            row[f"{metric}_generated"] = gen_metrics[metric]
        rows.append(row)

    return pd.DataFrame(rows)


def plot_metric_scatter(metric_df: pd.DataFrame, save_path: Path, title: str):
    n_metrics = len(METRIC_NAMES)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for idx, metric in enumerate(METRIC_NAMES):
        ax = axes[idx]
        x = metric_df[f"{metric}_real"].to_numpy(dtype=float)
        y = metric_df[f"{metric}_generated"].to_numpy(dtype=float)
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) == 0:
            ax.text(0.5, 0.5, "No valid segments", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(metric)
            ax.set_xlabel("Real")
            ax.set_ylabel("Generated")
            ax.grid(alpha=0.2)
            continue

        ax.scatter(x_valid, y_valid, s=20, alpha=0.65)

        low = min(np.min(x_valid), np.min(y_valid))
        high = max(np.max(x_valid), np.max(y_valid))
        if high == low:
            padding = 1.0
        else:
            padding = 0.05 * (high - low)
        low -= padding
        high += padding

        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Real")
        ax.set_ylabel("Generated")
        ax.grid(alpha=0.25)

        mae = np.mean(np.abs(x_valid - y_valid))
        if len(x_valid) > 1 and np.std(x_valid) > 0 and np.std(y_valid) > 0:
            corr = np.corrcoef(x_valid, y_valid)[0, 1]
        else:
            corr = np.nan
        corr_text = "nan" if np.isnan(corr) else f"{corr:.3f}"
        ax.set_title(f"{metric} (n={len(x_valid)}, MAE={mae:.3f}, r={corr_text})")

    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_segment_indices(total_samples: int, val_idx_file: str | None):
    if not val_idx_file:
        return np.arange(total_samples)
    indices = np.load(val_idx_file)
    return np.asarray(indices, dtype=int)


def predict_combined_attention(
    X_eval: np.ndarray,
    checkpoint_path: str,
    batch_size: int,
):
    model = modules.WaveNet_two_channel_combined_attention(input_shape=X_eval.shape[1:])
    model.load_weights(checkpoint_path)
    preds = model.predict(X_eval, batch_size=batch_size, verbose=0)
    return squeeze_last_channel(preds)


def main():
    base_dir = Path(__file__).resolve().parents[3] / "Fetal-maternal-fusion" / "src"

    parser = argparse.ArgumentParser(
        description="Scatter plots of clinical metrics (real vs generated) for the final combined-attention model."
    )
    parser.add_argument("--data_dir", type=str, default=str(base_dir / "WaveNet_beat" / "data"))
    parser.add_argument("--X_file", type=str, default="X.npy")
    parser.add_argument("--Y_file", type=str, default="Y.npy")
    parser.add_argument("--val_idx_file", type=str, default=str(base_dir / "idx_val.npy"))
    parser.add_argument(
        "--train_idx_file",
        type=str,
        default=str(base_dir / "idx_train.npy"),
        help="Optional train index file used to verify no train/eval overlap.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(base_dir / "WaveNet_beat" / "plots" / "model_checkpoints" / "(d)_two-channel_(combined_attention).weights.h5"),
    )
    parser.add_argument("--output_dir", type=str, default=str(base_dir / "evaluation_results"))
    parser.add_argument("--output_prefix", type=str, default="WaveNet_Combined_Attention")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fs", type=float, default=284.0)
    parser.add_argument("--ecg_lowcut", type=float, default=0.5)
    parser.add_argument("--ecg_highcut", type=float, default=40.0)
    parser.add_argument("--ecg_filter_order", type=int, default=2)
    parser.add_argument("--smooth_cutoff_hz", type=float, default=10.0)
    parser.add_argument("--smooth_order", type=int, default=4)
    parser.add_argument("--no_input_preprocess", action="store_false", dest="apply_input_preprocess", default=True)
    parser.add_argument("--no_output_smoothing", action="store_false", dest="apply_output_smoothing", default=True)
    parser.add_argument("--use_all_samples", action="store_true", default=False)
    parser.add_argument(
        "--allow_train_overlap",
        action="store_true",
        default=False,
        help="Allow overlap between eval indices and train indices (not recommended).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    X = np.load(Path(args.data_dir) / args.X_file)
    Y = np.load(Path(args.data_dir) / args.Y_file)
    Y = squeeze_last_channel(Y)

    if args.use_all_samples:
        eval_indices = np.arange(X.shape[0], dtype=int)
    else:
        eval_indices = load_segment_indices(X.shape[0], args.val_idx_file)

    train_indices = None
    if args.train_idx_file and Path(args.train_idx_file).exists():
        train_indices = np.asarray(np.load(args.train_idx_file), dtype=int)
        overlap = np.intersect1d(eval_indices, train_indices)
        if len(overlap) > 0 and not args.allow_train_overlap:
            raise ValueError(
                f"Eval indices overlap train indices: {len(overlap)} overlapping segments. "
                "Use a held-out index file (test/val) or pass --allow_train_overlap to bypass."
            )

    X_eval = X[eval_indices]
    Y_eval = Y[eval_indices]

    if args.apply_input_preprocess:
        X_eval = preprocess_ecg_inputs(
            X_eval,
            fs=args.fs,
            lowcut=args.ecg_lowcut,
            highcut=args.ecg_highcut,
            order=args.ecg_filter_order,
        )

    Y_pred = predict_combined_attention(
        X_eval=X_eval,
        checkpoint_path=str(checkpoint_path),
        batch_size=args.batch_size,
    )

    if args.apply_output_smoothing:
        Y_pred = np.asarray(
            [
                smooth_doppler_envelope(seg, fs=args.fs, cutoff_hz=args.smooth_cutoff_hz, order=args.smooth_order)
                for seg in Y_pred
            ]
        )

    metric_df = compute_segment_metric_table(
        real_dopplers=Y_eval,
        generated_dopplers=Y_pred,
        segment_indices=eval_indices,
        sampling_rate=args.fs,
    )

    csv_path = Path(args.output_dir) / f"{args.output_prefix}_clinical_metrics_per_segment.csv"
    metric_df.to_csv(csv_path, index=False)

    scatter_path = Path(args.output_dir) / f"{args.output_prefix}_clinical_metric_scatter.png"
    title = (
        "Clinical Metrics: Real vs Generated Doppler "
        f"({args.output_prefix}, n_segments={len(metric_df)})"
    )
    plot_metric_scatter(metric_df=metric_df, save_path=scatter_path, title=title)

    print(f"Data source: {Path(args.data_dir) / args.X_file} and {Path(args.data_dir) / args.Y_file}")
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Evaluated segments: {len(eval_indices)}")
    if train_indices is not None:
        print(f"Train/eval overlap check: PASS (train={len(train_indices)}, eval={len(eval_indices)})")

    print(f"Saved per-segment clinical metrics to: {csv_path}")
    print(f"Saved scatterplots to: {scatter_path}")


if __name__ == "__main__":
    main()

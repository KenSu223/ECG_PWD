"""Primary training entry point for the autoregressive two-channel WaveNet pipeline. Loads split indices, configures flexible composite losses, trains with callbacks, and runs autoregressive validation/inference plotting."""

# main.py (autoregressive training + AR validation/inference)

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
    Callback,
)

from . import modules

print("Script Python:", sys.executable)
print("sys.path:", sys.path)


# ==================== Helper Loss Functions ====================

def cross_correlation_loss(y_true, y_pred):
    """
    Cross-correlation loss: 1 - correlation coefficient
    Perfect correlation = 0 loss
    """
    epsilon = 1e-7

    y_true_mean = tf.reduce_mean(y_true, axis=1, keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=1, keepdims=True)

    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    y_true_std = tf.math.reduce_std(y_true, axis=1, keepdims=True) + epsilon
    y_pred_std = tf.math.reduce_std(y_pred, axis=1, keepdims=True) + epsilon

    y_true_norm = y_true_centered / y_true_std
    y_pred_norm = y_pred_centered / y_pred_std

    correlation = tf.reduce_mean(y_true_norm * y_pred_norm, axis=1)
    correlation = tf.clip_by_value(correlation, -1.0, 1.0)

    loss = tf.reduce_mean(1.0 - correlation)
    return loss


def multi_resolution_stft_loss(
    y_true,
    y_pred,
    fft_sizes=[256, 512, 1024],
    hop_sizes=[64, 128, 256],
    win_lengths=[256, 512, 1024],
):
    """
    Multi-resolution STFT loss: compares spectrograms at multiple resolutions
    """
    total_loss = 0.0

    for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
        stft_true = tf.signal.stft(
            y_true,
            frame_length=win_length,
            frame_step=hop_size,
            fft_length=fft_size,
        )
        stft_pred = tf.signal.stft(
            y_pred,
            frame_length=win_length,
            frame_step=hop_size,
            fft_length=fft_size,
        )

        mag_true = tf.abs(stft_true)
        mag_pred = tf.abs(stft_pred)

        spectral_loss = tf.reduce_mean(tf.abs(mag_true - mag_pred))

        log_mag_true = tf.math.log(mag_true + 1e-7)
        log_mag_pred = tf.math.log(mag_pred + 1e-7)
        log_spectral_loss = tf.reduce_mean(tf.abs(log_mag_true - log_mag_pred))

        total_loss += spectral_loss + log_spectral_loss

    return total_loss / len(fft_sizes)


def fast_soft_dtw_loss(y_true, y_pred, gamma=0.1):
    """
    Fast differentiable approximation of Soft-DTW for TensorFlow
    """
    y_true_exp = tf.expand_dims(y_true, 2)  # (batch, N, 1)
    y_pred_exp = tf.expand_dims(y_pred, 1)  # (batch, 1, N)

    C = tf.square(y_true_exp - y_pred_exp)  # (batch, N, N)

    weights = tf.nn.softmax(-C / gamma, axis=2)
    weighted_costs = tf.reduce_sum(C * weights, axis=2)

    dtw_loss = tf.reduce_mean(weighted_costs)
    return dtw_loss


# ==================== Helper for AR Inputs ====================

def build_ar_inputs(Y):
    """
    Ensure Y is (N, T, 1) and build Y_prev with teacher forcing:
    Y_prev[:, 0] = 0
    Y_prev[:, 1:] = Y[:, :-1]
    """
    if Y.ndim == 2:
        Y = Y[..., np.newaxis]  # (N, T, 1)

    Y_prev = np.zeros_like(Y)
    Y_prev[:, 1:, :] = Y[:, :-1, :]

    return Y_prev, Y


# ==================== Autoregressive Generation (no GT inputs) ====================

def autoregressive_generate(model, X_ecg, T, batch_size=32):
    """
    Autoregressive generation:
    - X_ecg: (N, T, 2) ECG input
    - T: sequence length
    Returns: Y_hat (N, T, 1) generated Doppler

    The model only sees its *own* generated history:
    - Start Y_prev_gen at zeros
    - At each step t, predict, take y_t, and feed it back into Y_prev_gen[:, t+1].
    """
    N = X_ecg.shape[0]
    Y_prev_gen = np.zeros((N, T, 1), dtype=np.float32)
    Y_hat = np.zeros((N, T, 1), dtype=np.float32)

    for t in range(T):
        # Predict full sequence given current generated history
        y_pred_full = model.predict(
            [Y_prev_gen, X_ecg],
            batch_size=batch_size,
            verbose=0,
        )  # (N, T, 1)

        # Take the prediction at current time step t
        y_t = y_pred_full[:, t:t+1, :]  # (N, 1, 1)

        # Store it
        Y_hat[:, t:t+1, :] = y_t

        # Feed it into next-step history
        if t + 1 < T:
            Y_prev_gen[:, t+1:t+2, :] = y_t

    return Y_hat


# ==================== Loss Component Logger (Train TF / Val AR) ====================

class LossComponentLogger(Callback):
    """Custom callback to compute and log individual loss components after each epoch.

    - Train: teacher forcing (uses Y_prev_train from ground-truth).
    - Val: pure autoregressive rollout (no GT Doppler as input).
    """

    def __init__(
        self,
        X_train_ecg,
        Y_prev_train,
        Y_train,
        X_val_ecg,
        Y_val,
        use_base=True,
        use_peak=False,
        use_ratio=False,
        use_derivative=False,
        use_softdtw=False,
        use_corr=False,
        use_mrstft=False,
        alpha_peak=0.5,
        alpha_ratio=0.5,
        alpha_derivative=0.3,
        alpha_softdtw=1.0,
        alpha_corr=1.0,
        alpha_mrstft=1.0,
        base_loss="mae",
        dtw_gamma=0.1,
    ):
        super().__init__()

        # Store inputs and targets
        self.X_train_ecg = X_train_ecg
        self.Y_prev_train = Y_prev_train
        self.Y_train = Y_train

        self.X_val_ecg = X_val_ecg
        self.Y_val = Y_val

        # Flags
        self.use_base = use_base
        self.use_peak = use_peak
        self.use_ratio = use_ratio
        self.use_derivative = use_derivative
        self.use_softdtw = use_softdtw
        self.use_corr = use_corr
        self.use_mrstft = use_mrstft

        # Weights
        self.alpha_peak = alpha_peak
        self.alpha_ratio = alpha_ratio
        self.alpha_derivative = alpha_derivative
        self.alpha_softdtw = alpha_softdtw
        self.alpha_corr = alpha_corr
        self.alpha_mrstft = alpha_mrstft

        self.base_loss = base_loss
        self.dtw_gamma = dtw_gamma

        # Store history
        self.history = {
            "train_base": [],
            "train_peak": [],
            "train_ratio": [],
            "train_derivative": [],
            "train_softdtw": [],
            "train_corr": [],
            "train_mrstft": [],
            "val_base": [],
            "val_peak": [],
            "val_ratio": [],
            "val_derivative": [],
            "val_softdtw": [],
            "val_corr": [],
            "val_mrstft": [],
        }

    def compute_components(self, y_true, y_pred):
        """Compute individual loss components."""
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
            y_pred = tf.squeeze(y_pred, axis=-1)

        epsilon = 1e-7
        components = {}

        # 1. Base loss
        if self.base_loss == "mae":
            base = tf.reduce_mean(tf.abs(y_true - y_pred))
        else:  # mse
            base = tf.reduce_mean(tf.square(y_true - y_pred))
        components["base"] = float(base.numpy())

        # 2. Peak loss
        peak_true = tf.reduce_max(y_true, axis=1)
        peak_pred = tf.reduce_max(y_pred, axis=1)
        peak_loss = tf.reduce_mean(tf.abs(peak_true - peak_pred))
        components["peak"] = float(peak_loss.numpy())

        # 3. Ratio loss
        trough_true = tf.reduce_min(y_true, axis=1)
        trough_pred = tf.reduce_min(y_pred, axis=1)
        ratio_true = peak_true - trough_true + epsilon
        ratio_pred = peak_pred - trough_pred + epsilon
        ratio_true = tf.clip_by_value(ratio_true, epsilon, 1e6)
        ratio_pred = tf.clip_by_value(ratio_pred, epsilon, 1e6)
        ratio_loss = tf.reduce_mean(tf.abs(ratio_true - ratio_pred))
        components["ratio"] = (
            float(ratio_loss.numpy()) if not tf.math.is_nan(ratio_loss) else 0.0
        )

        # 4. Derivative loss
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        derivative_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred))
        components["derivative"] = (
            float(derivative_loss.numpy())
            if not tf.math.is_nan(derivative_loss)
            else 0.0
        )

        # 5. Soft-DTW
        try:
            softdtw_loss = fast_soft_dtw_loss(y_true, y_pred, gamma=self.dtw_gamma)
            components["softdtw"] = float(softdtw_loss.numpy())
        except Exception as e:
            print(f"  Warning: Soft-DTW computation failed: {e}")
            components["softdtw"] = 0.0

        # 6. Cross-correlation
        try:
            corr_loss = cross_correlation_loss(y_true, y_pred)
            components["corr"] = float(corr_loss.numpy())
        except Exception as e:
            print(f"  Warning: Correlation computation failed: {e}")
            components["corr"] = 0.0

        # 7. MR-STFT
        try:
            mrstft_loss = multi_resolution_stft_loss(y_true, y_pred)
            components["mrstft"] = float(mrstft_loss.numpy())
        except Exception as e:
            print(f"  Warning: MR-STFT computation failed: {e}")
            components["mrstft"] = 0.0

        return components

    def on_epoch_end(self, epoch, logs=None):
        """Compute and print loss components after each epoch.

        - Train: teacher forcing (uses ground-truth Y_prev_train).
        - Val: autoregressive rollout (no ground-truth Doppler as input).
        """

        # ---- TRAIN (teacher forcing) ----
        y_train_pred = self.model.predict(
            [self.Y_prev_train, self.X_train_ecg],
            verbose=0,
        )

        # ---- VAL (autoregressive) ----
        T_val = self.Y_val.shape[1]
        y_val_pred = autoregressive_generate(
            self.model,
            self.X_val_ecg,
            T=T_val,
        )

        train_components = self.compute_components(self.Y_train, y_train_pred)
        val_components = self.compute_components(self.Y_val, y_val_pred)

        # Store
        for key in train_components.keys():
            self.history[f"train_{key}"].append(train_components[key])
            self.history[f"val_{key}"].append(val_components[key])

        # Weighted sums
        train_total = 0.0
        val_total = 0.0

        if self.use_base:
            train_total += train_components["base"]
            val_total += val_components["base"]

        if self.use_peak:
            train_total += self.alpha_peak * train_components["peak"]
            val_total += self.alpha_peak * val_components["peak"]

        if self.use_ratio:
            train_total += self.alpha_ratio * train_components["ratio"]
            val_total += self.alpha_ratio * val_components["ratio"]

        if self.use_derivative:
            train_total += self.alpha_derivative * train_components["derivative"]
            val_total += self.alpha_derivative * val_components["derivative"]

        if self.use_softdtw:
            train_total += self.alpha_softdtw * train_components["softdtw"]
            val_total += self.alpha_softdtw * val_components["softdtw"]

        if self.use_corr:
            train_total += self.alpha_corr * train_components["corr"]
            val_total += self.alpha_corr * val_components["corr"]

        if self.use_mrstft:
            train_total += self.alpha_mrstft * train_components["mrstft"]
            val_total += self.alpha_mrstft * val_components["mrstft"]

        # Pretty print
        print(f"\n{'='*90}")
        print(f"Epoch {epoch + 1} - Loss Component Breakdown (Train TF vs Val AR):")
        print(f"{'='*90}")
        print(f"{'Component':<30} {'Train':<15} {'Val':<15} {'Weight':<10} {'Used':<10}")
        print(f"{'-'*90}")

        base_name = "Base (MAE)" if self.base_loss == "mae" else "Base (MSE)"
        print(
            f"{base_name:<30} {train_components['base']:<15.6f} {val_components['base']:<15.6f} {'1.0':<10} {'✓' if self.use_base else '✗':<10}"
        )
        print(
            f"{'Peak Loss':<30} {train_components['peak']:<15.6f} {val_components['peak']:<15.6f} {self.alpha_peak:<10.2f} {'✓' if self.use_peak else '✗':<10}"
        )
        print(
            f"{'Ratio Loss':<30} {train_components['ratio']:<15.6f} {val_components['ratio']:<15.6f} {self.alpha_ratio:<10.2f} {'✓' if self.use_ratio else '✗':<10}"
        )
        print(
            f"{'Derivative Loss':<30} {train_components['derivative']:<15.6f} {val_components['derivative']:<15.6f} {self.alpha_derivative:<10.2f} {'✓' if self.use_derivative else '✗':<10}"
        )
        print(
            f"{'Soft-DTW Loss':<30} {train_components['softdtw']:<15.6f} {val_components['softdtw']:<15.6f} {self.alpha_softdtw:<10.2f} {'✓' if self.use_softdtw else '✗':<10}"
        )
        print(
            f"{'Cross-Correlation Loss':<30} {train_components['corr']:<15.6f} {val_components['corr']:<15.6f} {self.alpha_corr:<10.2f} {'✓' if self.use_corr else '✗':<10}"
        )
        print(
            f"{'MR-STFT Loss':<30} {train_components['mrstft']:<15.6f} {val_components['mrstft']:<15.6f} {self.alpha_mrstft:<10.2f} {'✓' if self.use_mrstft else '✗':<10}"
        )

        print(f"{'-'*90}")
        print(
            f"{'TOTAL (weighted sum)':<30} {train_total:<15.6f} {val_total:<15.6f}"
        )
        print(f"{'='*90}\n")


# ==================== Loss Function Creation ====================

def create_flexible_loss(
    use_base=True,
    use_peak=False,
    use_ratio=False,
    use_derivative=False,
    use_softdtw=False,
    use_corr=False,
    use_mrstft=False,
    alpha_peak=0.5,
    alpha_ratio=0.5,
    alpha_derivative=0.3,
    alpha_softdtw=1.0,
    alpha_corr=1.0,
    alpha_mrstft=1.0,
    base_loss="mae",
    dtw_gamma=0.1,
):
    """
    Flexible composite loss
    """

    def flexible_loss(y_true, y_pred):
        if len(y_true.shape) == 3:
            y_true_ = tf.squeeze(y_true, axis=-1)
            y_pred_ = tf.squeeze(y_pred, axis=-1)
        else:
            y_true_ = y_true
            y_pred_ = y_pred

        epsilon = 1e-7
        total = 0.0

        # 1. Base
        if use_base:
            if base_loss == "mae":
                total += tf.reduce_mean(tf.abs(y_true_ - y_pred_))
            else:
                total += tf.reduce_mean(tf.square(y_true_ - y_pred_))

        # 2. Peak
        if use_peak:
            peak_true = tf.reduce_max(y_true_, axis=1)
            peak_pred = tf.reduce_max(y_pred_, axis=1)
            peak_loss = tf.reduce_mean(tf.abs(peak_true - peak_pred))
            total += alpha_peak * peak_loss

        # 3. Ratio
        if use_ratio:
            peak_true = tf.reduce_max(y_true_, axis=1)
            peak_pred = tf.reduce_max(y_pred_, axis=1)
            trough_true = tf.reduce_min(y_true_, axis=1)
            trough_pred = tf.reduce_min(y_pred_, axis=1)

            ratio_true = peak_true - trough_true + epsilon
            ratio_pred = peak_pred - trough_pred + epsilon

            ratio_true = tf.clip_by_value(ratio_true, epsilon, 1e6)
            ratio_pred = tf.clip_by_value(ratio_pred, epsilon, 1e6)

            ratio_loss = tf.reduce_mean(tf.abs(ratio_true - ratio_pred))
            ratio_loss = tf.where(tf.math.is_nan(ratio_loss), 0.0, ratio_loss)
            total += alpha_ratio * ratio_loss

        # 4. Derivative
        if use_derivative:
            dy_true = y_true_[:, 1:] - y_true_[:, :-1]
            dy_pred = y_pred_[:, 1:] - y_pred_[:, :-1]
            derivative_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred))
            derivative_loss = tf.where(
                tf.math.is_nan(derivative_loss), 0.0, derivative_loss
            )
            total += alpha_derivative * derivative_loss

        # 5. Soft-DTW
        if use_softdtw:
            try:
                dtw_loss = fast_soft_dtw_loss(y_true_, y_pred_, gamma=dtw_gamma)
                dtw_loss = tf.where(tf.math.is_nan(dtw_loss), 0.0, dtw_loss)
                total += alpha_softdtw * dtw_loss
            except Exception:
                pass

        # 6. Cross-correlation
        if use_corr:
            try:
                corr_loss = cross_correlation_loss(y_true_, y_pred_)
                corr_loss = tf.where(tf.math.is_nan(corr_loss), 0.0, corr_loss)
                total += alpha_corr * corr_loss
            except Exception:
                pass

        # 7. MR-STFT
        if use_mrstft:
            try:
                mrstft_loss = multi_resolution_stft_loss(y_true_, y_pred_)
                mrstft_loss = tf.where(tf.math.is_nan(mrstft_loss), 0.0, mrstft_loss)
                total += alpha_mrstft * mrstft_loss
            except Exception:
                pass

        return total

    return flexible_loss


def create_composite_loss(alpha_peak=0.5, alpha_ratio=0.5, base_loss="mae"):
    return create_flexible_loss(
        use_base=True,
        use_peak=True,
        use_ratio=True,
        use_derivative=False,
        alpha_peak=alpha_peak,
        alpha_ratio=alpha_ratio,
        base_loss=base_loss,
    )


def create_shape_preserving_loss(
    alpha_peak=0.5, alpha_ratio=0.5, alpha_derivative=0.3, base_loss="mae"
):
    return create_flexible_loss(
        use_base=True,
        use_peak=True,
        use_ratio=True,
        use_derivative=True,
        alpha_peak=alpha_peak,
        alpha_ratio=alpha_ratio,
        alpha_derivative=alpha_derivative,
        base_loss=base_loss,
    )


# ==================== Plotting of Loss Components ====================

def plot_all_loss_components(loss_logger, save_dir_plots, n_epochs):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Individual Loss Components Over Training", fontsize=16, fontweight="bold")

    epochs = np.arange(1, n_epochs + 1)

    components_to_plot = [
        ("base", "Base Loss (MAE/MSE)", 0, 0),
        ("peak", "Peak Amplitude Loss", 0, 1),
        ("ratio", "Peak-to-Trough Ratio Loss", 0, 2),
        ("derivative", "Derivative Matching Loss", 1, 0),
        ("softdtw", "Soft-DTW Loss", 1, 1),
        ("corr", "Cross-Correlation Loss", 1, 2),
        ("mrstft", "MR-STFT Loss", 2, 0),
    ]

    for comp_name, title, row, col in components_to_plot:
        ax = axes[row, col]
        train_key = f"train_{comp_name}"
        val_key = f"val_{comp_name}"

        if train_key in loss_logger.history and loss_logger.history[train_key]:
            ax.plot(
                epochs,
                loss_logger.history[train_key],
                "b-",
                label="Train",
                linewidth=2,
                alpha=0.7,
            )
            ax.plot(
                epochs,
                loss_logger.history[val_key],
                "r-",
                label="Validation (AR)",
                linewidth=2,
                alpha=0.7,
            )
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Loss", fontsize=10)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Not Used", ha="center", va="center", fontsize=12)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")

    axes[2, 1].axis("off")
    axes[2, 2].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir_plots, exist_ok=True)
    output_path = os.path.join(save_dir_plots, "loss_components_history.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved loss components plot to: {output_path}")
    plt.close()


# ==================== Main Training Function ====================

def main(
    data_dir="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/WaveNet_beat/data",
    X_file="X.npy",
    Y_file="Y.npy",
    patient_ids_file="PATIENT_IDS.npy",
    train_idx_file="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_train.npy",
    val_idx_file="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_val.npy",
    save_dir_model="./WaveNet_beat/loss_trial_models",
    save_dir_loss="./WaveNet_beat/loss_trial_logs",
    save_dir_plots="./WaveNet_beat/loss_trial_plots",
    n_epochs=10,
    batch_size=32,
    lr=1e-3,
    use_lr_schedule=False,
    loss_type="mae",
    # Loss flags
    use_base=True,
    use_peak=False,
    use_ratio=False,
    use_derivative=False,
    use_softdtw=False,
    use_corr=False,
    use_mrstft=False,
    # Weights
    alpha_peak=0.5,
    alpha_ratio=0.5,
    alpha_derivative=0.3,
    alpha_softdtw=1.0,
    alpha_corr=1.0,
    alpha_mrstft=1.0,
    base_loss="mae",
    dtw_gamma=0.1,
):
    os.makedirs(save_dir_model, exist_ok=True)
    os.makedirs(save_dir_loss, exist_ok=True)
    os.makedirs(save_dir_plots, exist_ok=True)

    print("Loading data...")
    X = np.load(os.path.join(data_dir, X_file))          # ECG: (N, T, 2)
    Y = np.load(os.path.join(data_dir, Y_file))          # PWD: (N, T) or (N, T, 1)
    PATIENT_IDS = np.load(os.path.join(data_dir, patient_ids_file))

    print(f"X shape (ECG): {X.shape}")
    print(f"Y shape (PWD): {Y.shape}")
    print(f"Number of patients: {len(np.unique(PATIENT_IDS))}")

    print("Loading train/val indices...")
    train_idx = np.load(train_idx_file)
    val_idx = np.load(val_idx_file)

    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    # Leakage check
    assert set(PATIENT_IDS[train_idx]).isdisjoint(
        set(PATIENT_IDS[val_idx])
    ), "Patient leakage detected!"
    print("✓ No patient leakage detected")

    # Ensure Y is (N, T, 1) and build Y_prev
    Y_prev, Y = build_ar_inputs(Y)

    # Build model: AR WaveNet
    timesteps = X.shape[1]
    model = modules.WaveNet_two_channel_AR(
        timesteps=timesteps,
        filters=64,
        kernel_size=2,
        dilation_rates=[2 ** i for i in range(7)],
    )

    # Loss selection
    if loss_type == "flexible":
        loss_fn = create_flexible_loss(
            use_base=use_base,
            use_peak=use_peak,
            use_ratio=use_ratio,
            use_derivative=use_derivative,
            use_softdtw=use_softdtw,
            use_corr=use_corr,
            use_mrstft=use_mrstft,
            alpha_peak=alpha_peak,
            alpha_ratio=alpha_ratio,
            alpha_derivative=alpha_derivative,
            alpha_softdtw=alpha_softdtw,
            alpha_corr=alpha_corr,
            alpha_mrstft=alpha_mrstft,
            base_loss=base_loss,
            dtw_gamma=dtw_gamma,
        )
        components = []
        if use_base:
            components.append(f"{base_loss.upper()}")
        if use_peak:
            components.append(f"{alpha_peak}*peak")
        if use_ratio:
            components.append(f"{alpha_ratio}*ratio")
        if use_derivative:
            components.append(f"{alpha_derivative}*derivative")
        if use_softdtw:
            components.append(f"{alpha_softdtw}*softdtw")
        if use_corr:
            components.append(f"{alpha_corr}*corr")
        if use_mrstft:
            components.append(f"{alpha_mrstft}*mrstft")
        print(f"✓ Using flexible loss: {' + '.join(components)}")

    elif loss_type == "composite":
        loss_fn = create_composite_loss(
            alpha_peak=alpha_peak, alpha_ratio=alpha_ratio, base_loss=base_loss
        )
        use_base, use_peak, use_ratio, use_derivative = True, True, True, False
        use_softdtw, use_corr, use_mrstft = False, False, False
        print(
            f"✓ Using composite loss: {base_loss.upper()} + {alpha_peak}*peak + {alpha_ratio}*ratio"
        )

    elif loss_type == "shape_preserving":
        loss_fn = create_shape_preserving_loss(
            alpha_peak=alpha_peak,
            alpha_ratio=alpha_ratio,
            alpha_derivative=alpha_derivative,
            base_loss=base_loss,
        )
        use_base, use_peak, use_ratio, use_derivative = True, True, True, True
        use_softdtw, use_corr, use_mrstft = False, False, False
        print("✓ Using shape-preserving loss")

    elif loss_type == "mse":
        loss_fn = "mse"
        use_base, use_peak, use_ratio, use_derivative = True, False, False, False
        use_softdtw, use_corr, use_mrstft = False, False, False
        base_loss = "mse"
        print("✓ Using MSE loss")

    else:  # 'mae'
        loss_fn = "mae"
        use_base, use_peak, use_ratio, use_derivative = True, False, False, False
        use_softdtw, use_corr, use_mrstft = False, False, False
        base_loss = "mae"
        print("✓ Using MAE loss")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_dir_model, "best_model.weights.h5"),
        monitor="loss",          # monitor training loss (no val_loss anymore)
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1,
    )
    callbacks = [checkpoint_cb]

    # Loss logger (AR-aware validation)
    loss_logger = LossComponentLogger(
        X_train_ecg=X[train_idx],
        Y_prev_train=Y_prev[train_idx],
        Y_train=Y[train_idx],
        X_val_ecg=X[val_idx],
        Y_val=Y[val_idx],
        use_base=use_base,
        use_peak=use_peak,
        use_ratio=use_ratio,
        use_derivative=use_derivative,
        use_softdtw=use_softdtw,
        use_corr=use_corr,
        use_mrstft=use_mrstft,
        alpha_peak=alpha_peak,
        alpha_ratio=alpha_ratio,
        alpha_derivative=alpha_derivative,
        alpha_softdtw=alpha_softdtw,
        alpha_corr=alpha_corr,
        alpha_mrstft=alpha_mrstft,
        base_loss=base_loss,
        dtw_gamma=dtw_gamma,
    )
    callbacks.append(loss_logger)
    print("✓ Loss component logging enabled (Train TF / Val AR)")

    if use_lr_schedule == "fixed":
        def scheduler(epoch, current_lr):
            if epoch > 0 and epoch % 5 == 0:
                return current_lr * 0.5
            return current_lr

        callbacks.append(LearningRateScheduler(scheduler, verbose=1))

    elif use_lr_schedule == "dynamic":
        # Monitor training loss, since we no longer pass validation_data to Keras
        reduce_lr_cb = ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        )
        callbacks.append(reduce_lr_cb)
        print("✓ Using ReduceLROnPlateau (monitor=loss)")

    # Training (teacher forcing, no Keras val_data)
    print("Starting training (autoregressive model, TRAIN uses teacher forcing)...")
    history = model.fit(
        [Y_prev[train_idx], X[train_idx]],
        Y[train_idx],
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training loss (no val_loss in history — validation is via LossComponentLogger)
    np.save(os.path.join(save_dir_loss, "train_loss.npy"), history.history["loss"])
    # keep interface consistent: save empty val_loss if not present
    np.save(
        os.path.join(save_dir_loss, "val_loss.npy"),
        history.history.get("val_loss", []),
    )

    # Save component-wise loss histories
    if loss_logger is not None:
        for key, values in loss_logger.history.items():
            if values and any(v is not None for v in values):
                np.save(
                    os.path.join(save_dir_loss, f"{key}_loss.npy"), np.array(values)
                )
        print(f"✓ Saved individual loss components to {save_dir_loss}")
        plot_all_loss_components(loss_logger, save_dir_plots, n_epochs)

    modules.plot_training_history(history, save_dir_plots, split_name="training")

    if "lr" in history.history:
        np.save(
            os.path.join(save_dir_loss, "learning_rate.npy"), history.history["lr"]
        )

    model.save_weights(os.path.join(save_dir_model, "final_model.weights.h5"))

    # Summary (use AR val_base for best_val_loss if available)
    train_pats = np.unique(PATIENT_IDS[train_idx])
    val_pats = np.unique(PATIENT_IDS[val_idx])

    if loss_logger is not None and loss_logger.history["val_base"]:
        best_val_loss = float(np.min(loss_logger.history["val_base"]))
    else:
        best_val_loss = float("nan")

    fold_info = {
        "train_patients": train_pats.size,
        "val_patients": val_pats.size,
        "train_segments": train_idx.size,
        "val_segments": val_idx.size,
        "best_val_loss": best_val_loss,
        "final_train_loss": float(history.history["loss"][-1]),
        "final_lr": float(
            history.history.get("lr", [lr])[-1]
        ) if "lr" in history.history else lr,
        "loss_type": loss_type,
        "use_base": use_base,
        "use_peak": use_peak,
        "use_ratio": use_ratio,
        "use_derivative": use_derivative,
        "use_softdtw": use_softdtw,
        "use_corr": use_corr,
        "use_mrstft": use_mrstft,
        "base_loss": base_loss,
    }

    df = pd.DataFrame([fold_info])
    df.to_csv(os.path.join(save_dir_loss, "training_summary.csv"), index=False)
    print("\nTraining Summary:")
    print(df.to_string(index=False))

    # ==================== Visualization (pure AR rollout) ====================

    print("\n" + "=" * 80)
    print("Generating visualization plots for random samples from BEST MODEL (AR)")
    print("=" * 80)

    model.load_weights(os.path.join(save_dir_model, "best_model.weights.h5"))
    print("✓ Loaded best model weights")

    np.random.seed(33)
    random_indices = np.random.choice(
        val_idx, size=min(3, len(val_idx)), replace=False
    )
    print(f"✓ Selected {len(random_indices)} random validation samples: {random_indices}")

    ecgs = X[random_indices]
    real_dopplers = Y[random_indices]       # (K, T, 1)
    T_val = real_dopplers.shape[1]

    # --- BEST MODEL, AR GENERATION ---
    generated_dopplers = autoregressive_generate(
        model,
        ecgs,
        T=T_val,
    )
    print(f"✓ Generated AR predictions with shape: {generated_dopplers.shape}")

    if real_dopplers.ndim == 3 and real_dopplers.shape[-1] == 1:
        real_dopplers_plot = real_dopplers.squeeze(axis=-1)
    else:
        real_dopplers_plot = real_dopplers

    if generated_dopplers.ndim == 3 and generated_dopplers.shape[-1] == 1:
        generated_dopplers_plot = generated_dopplers.squeeze(axis=-1)
    else:
        generated_dopplers_plot = generated_dopplers

    plots_overlay_dir = os.path.join(save_dir_plots, "overlays")
    os.makedirs(plots_overlay_dir, exist_ok=True)

    modules.plot_ecg_doppler_overlay_multi(
        ecgs=ecgs,
        real_dopplers=real_dopplers_plot,
        generated_dopplers=generated_dopplers_plot,
        save_dir=plots_overlay_dir,
        prefix="best_model_AR",
    )

    print(f"✓ Saved best model overlay plots to: {plots_overlay_dir}")
    print("=" * 80 + "\n")

    print("\n" + "-" * 80)
    print("Plotting samples from FINAL model (AR)...")
    print("-" * 80)

    model.load_weights(os.path.join(save_dir_model, "final_model.weights.h5"))
    print("✓ Loaded final model weights")

    generated_dopplers_final = autoregressive_generate(
        model,
        ecgs,
        T=T_val,
    )
    print(f"✓ Generated AR predictions with shape: {generated_dopplers_final.shape}")

    if generated_dopplers_final.ndim == 3 and generated_dopplers_final.shape[-1] == 1:
        generated_dopplers_final_plot = generated_dopplers_final.squeeze(axis=-1)
    else:
        generated_dopplers_final_plot = generated_dopplers_final

    modules.plot_ecg_doppler_overlay_multi(
        ecgs=ecgs,
        real_dopplers=real_dopplers_plot,
        generated_dopplers=generated_dopplers_final_plot,
        save_dir=plots_overlay_dir,
        prefix="final_model_AR",
    )

    print(f"✓ Saved final model overlay plots to: {plots_overlay_dir}")
    print("=" * 80 + "\n")

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train autoregressive WaveNet model with flexible loss components"
    )
    parser.add_argument("--data_dir", type=str, default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/WaveNet_beat/data")
    parser.add_argument("--X_file", type=str, default="X.npy")
    parser.add_argument("--Y_file", type=str, default="Y.npy")
    parser.add_argument(
        "--patient_ids_file", type=str, default="PATIENT_IDS.npy"
    )
    parser.add_argument(
        "--train_idx_file",
        type=str,
        default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_train_mix_risk.npy",
    )
    parser.add_argument(
        "--val_idx_file",
        type=str,
        default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_val_mix_risk.npy",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_lr_schedule", type=str)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="mae",
        choices=["mae", "mse", "composite", "shape_preserving", "flexible"],
    )

    parser.add_argument(
        "--use_base", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument(
        "--use_peak", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument(
        "--use_ratio", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument(
        "--use_derivative", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument(
        "--use_softdtw", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument(
        "--use_corr", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument(
        "--use_mrstft", type=lambda x: x.lower() == "true", default=False
    )

    parser.add_argument("--alpha_peak", type=float, default=0.5)
    parser.add_argument("--alpha_ratio", type=float, default=0.5)
    parser.add_argument("--alpha_derivative", type=float, default=0.3)
    parser.add_argument("--alpha_softdtw", type=float, default=1.0)
    parser.add_argument("--alpha_corr", type=float, default=1.0)
    parser.add_argument("--alpha_mrstft", type=float, default=1.0)
    parser.add_argument(
        "--base_loss", type=str, default="mae", choices=["mae", "mse"]
    )
    parser.add_argument("--dtw_gamma", type=float, default=0.1)

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        X_file=args.X_file,
        Y_file=args.Y_file,
        patient_ids_file=args.patient_ids_file,
        train_idx_file=args.train_idx_file,
        val_idx_file=args.val_idx_file,
        lr=args.lr,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        use_lr_schedule=args.use_lr_schedule,
        loss_type=args.loss_type,
        use_base=args.use_base,
        use_peak=args.use_peak,
        use_ratio=args.use_ratio,
        use_derivative=args.use_derivative,
        use_softdtw=args.use_softdtw,
        use_corr=args.use_corr,
        use_mrstft=args.use_mrstft,
        alpha_peak=args.alpha_peak,
        alpha_ratio=args.alpha_ratio,
        alpha_derivative=args.alpha_derivative,
        alpha_softdtw=args.alpha_softdtw,
        alpha_corr=args.alpha_corr,
        alpha_mrstft=args.alpha_mrstft,
        base_loss=args.base_loss,
        dtw_gamma=args.dtw_gamma,
    )
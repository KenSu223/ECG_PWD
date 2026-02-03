"""Model-building and visualization utilities for autoregressive ECG-to-Doppler generation. Defines AR-conditioned WaveNet blocks plus plotting/scalogram helpers used during training diagnostics."""

# modules.py

import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.transform import resize  # kept in case you use it elsewhere

from tensorflow import keras
from tensorflow.keras import layers


# ==================== Autoregressive WaveNet with ECG Conditioning ====================

def wavenet_block_cond(x_in, cond, filters, kernel_size, dilation_rate):
    """
    Conditioned WaveNet block.

    Args:
        x_in:  (B, T, C)  autoregressive stream (past PWD)
        cond:  (B, T, C_cond) conditioning features (ECG)
    """
    # Causal dilated convs over autoregressive stream
    conv_filter = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal"
    )
    conv_gate = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal"
    )

    # 1x1 projections of conditioning to match gate/filter channels
    cond_filter = layers.Conv1D(filters, kernel_size=1, padding="same")
    cond_gate   = layers.Conv1D(filters, kernel_size=1, padding="same")

    # Gated activation: tanh + sigmoid
    f = conv_filter(x_in) + cond_filter(cond)
    g = conv_gate(x_in)   + cond_gate(cond)

    tanh_out = layers.Activation("tanh")(f)
    sigm_out = layers.Activation("sigmoid")(g)
    z = layers.Multiply()([tanh_out, sigm_out])

    # Skip connection
    skip = layers.Conv1D(filters, 1, padding="same")(z)

    # Residual connection
    res = layers.Conv1D(filters, 1, padding="same")(x_in)
    x_out = layers.Add()([res, z])

    return x_out, skip


def WaveNet_two_channel_AR(
    timesteps,
    filters=64,
    kernel_size=2,
    dilation_rates=None,
):
    """
    Autoregressive WaveNet for fetal PWD upper envelopes, conditioned on fetal + maternal ECG.

    Inputs:
      - pwd_prev: (B, T, 1)    previous PWD samples (teacher forcing)
      - ecg_fetal_maternal: (B, T, 2)  fetal & maternal ECG

    Output:
      - y_hat: (B, T, 1)  predicted PWD envelope
    """
    if dilation_rates is None:
        dilation_rates = [2 ** i for i in range(7)]

    # (B, T, 1) - previous PWD samples
    pwd_in = layers.Input(shape=(timesteps, 1), name="pwd_prev")

    # (B, T, 2) - fetal + maternal ECG
    ecg_in = layers.Input(shape=(timesteps, 2), name="ecg_fetal_maternal_2ch")

    # Stem: causal 1x1 to project PWD into `filters` channels
    x = layers.Conv1D(filters, kernel_size=1, padding="causal")(pwd_in)

    # Project ECG conditioning into same channel dim
    cond = layers.Conv1D(filters, kernel_size=1, padding="same")(ecg_in)

    skips = []
    for d in dilation_rates:
        x, s = wavenet_block_cond(x, cond, filters, kernel_size, d)
        skips.append(s)

    # Sum skip connections
    out = layers.Add()(skips)
    out = layers.Activation("relu")(out)

    # Post-processing: 1x1 -> ReLU -> 1x1 -> tanh
    out = layers.Conv1D(filters, 1, padding="same")(out)
    out = layers.Activation("relu")(out)
    out = layers.Conv1D(1, 1, padding="same")(out)
    out = layers.Activation("tanh")(out)

    model = keras.Model(
        inputs=[pwd_in, ecg_in],
        outputs=out,
        name="WaveNet_two_channel_AR_conditioned_on_ECG"
    )
    model.summary()
    return model


# ==================== Scalogram Utilities ====================

def create_scalogram(sig, fs=284, time_bins=160, freq_bins=80):
    scales = np.arange(1, freq_bins + 1)
    coeffs, f = pywt.cwt(sig, scales, wavelet="cgau8", sampling_period=1 / fs)
    coeffs = np.abs(coeffs)
    return coeffs, f


def create_batch_scalograms(signals_batch, fs=2000, time_bins=160, freq_bins=80):
    coeffs = []
    f_s = []
    for sig in signals_batch:
        coeff, f = create_scalogram(sig, fs, time_bins, freq_bins)
        coeffs.append(coeff)
        f_s.append(f)
    return coeffs, f_s


# ==================== Plotting: ECG + Doppler ====================

def plot_ecg_doppler_pairs(ecgs, real_dopplers, generated_dopplers):
    """Plots ECG and corresponding real and generated Doppler pairs."""
    # Check if ECGs are multi-channel
    if len(ecgs.shape) == 3 and ecgs.shape[2] > 1:
        n_channels = ecgs.shape[2]
        plt.figure(figsize=(18, 8 * n_channels))

        for i, (ecg, real_dopple, generated_dopple) in enumerate(
            zip(ecgs, real_dopplers, generated_dopplers)
        ):
            for ch in range(n_channels):
                row_idx = i * n_channels + ch

                # Plotting ECG channel
                plt.subplot(len(ecgs) * n_channels, 3, 3 * row_idx + 1)
                plt.plot(ecg[:, ch], color="royalblue")
                plt.title("Fetal ECG" if ch == 0 else "Maternal ECG", fontsize=10)
                plt.xticks([])
                plt.yticks([])
                plt.box(False)
                plt.axhline(y=0, color="gray", linewidth=1.5, zorder=1)
                plt.axvline(x=0, color="gray", linewidth=1.5, zorder=1)

                # Only plot Doppler for the first channel (to avoid repetition)
                if ch == 0:
                    # Plotting Real Doppler
                    plt.subplot(len(ecgs) * n_channels, 3, 3 * row_idx + 2)
                    plt.plot(real_dopple, color="blue")
                    plt.title("Real Doppler", fontsize=10)
                    plt.xticks([])
                    plt.yticks([])
                    plt.box(False)
                    plt.axhline(y=0, color="gray", linewidth=1.5, zorder=1)
                    plt.axvline(x=0, color="gray", linewidth=1.5, zorder=1)

                    # Plotting Generated Doppler
                    plt.subplot(len(ecgs) * n_channels, 3, 3 * row_idx + 3)
                    plt.plot(generated_dopple, color="red")
                    plt.title("Generated Doppler", fontsize=10)
                    plt.xticks([])
                    plt.yticks([])
                    plt.box(False)
                    plt.axhline(y=0, color="gray", linewidth=1.5, zorder=1)
                    plt.axvline(x=0, color="gray", linewidth=1.5, zorder=1)
    else:
        # Original single-channel code
        plt.figure(figsize=(18, 8))
        for i, (ecg, real_dopple, generated_dopple) in enumerate(
            zip(ecgs, real_dopplers, generated_dopplers)
        ):
            # Plotting ECG
            plt.subplot(len(ecgs), 3, 3 * i + 1)
            plt.plot(ecg, color="royalblue")
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color="gray", linewidth=1.5, zorder=1)
            plt.axvline(x=0, color="gray", linewidth=1.5, zorder=1)

            # Plotting Real Doppler
            plt.subplot(len(ecgs), 3, 3 * i + 2)
            plt.plot(real_dopple, color="blue")
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color="gray", linewidth=1.5, zorder=1)
            plt.axvline(x=0, color="gray", linewidth=1.5, zorder=1)

            # Plotting Generated Doppler
            plt.subplot(len(ecgs), 3, 3 * i + 3)
            plt.plot(generated_dopple, color="red")
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color="gray", linewidth=1.5, zorder=1)
            plt.axvline(x=0, color="gray", linewidth=1.5, zorder=1)

    plt.tight_layout()
    os.makedirs("WaveNet_beat/plots", exist_ok=True)
    plt.savefig("WaveNet_beat/plots/signals_test.jpg")
    plt.show()


def plot_scalogram(real, generated, time_bins=160, freq_bins=80):
    plt.figure(figsize=(18, 10))
    coeffs_rs, fs_rs, coeffs_gs, fs_gs = [], [], [], []
    fs = 284

    for i in range(len(real)):
        coeffs_r, fs_r = create_scalogram(real[i], fs, time_bins, freq_bins)
        coeffs_g, fs_g = create_scalogram(generated[i], fs, time_bins, freq_bins)
        coeffs_gs.append(coeffs_g)
        coeffs_rs.append(coeffs_r)
        fs_gs.append(fs_g)
        fs_rs.append(fs_r)

    for i in range(len(real)):
        plt.subplot(len(real), 2, 2 * i + 1)
        plt.pcolormesh(
            np.arange(coeffs_rs[i].shape[1]),
            fs_rs[i],
            coeffs_rs[i],
            shading="gouraud",
            cmap="bwr",
        )
        plt.yticks([])
        plt.xticks([])
        plt.ylim(10, 1000)

        plt.subplot(len(real), 2, 2 * i + 2)
        plt.pcolormesh(
            np.arange(coeffs_gs[i].shape[1]),
            fs_gs[i],
            coeffs_gs[i],
            shading="gouraud",
            cmap="bwr",
        )
        plt.xticks([])
        plt.yticks([])
        plt.ylim(10, 1000)

    os.makedirs("WaveNet_beat/plots", exist_ok=True)
    plt.savefig("WaveNet_beat/plots/scalograms_test.jpg")
    plt.show()


def plot_ecg_doppler_overlay_multi(
    ecgs,
    real_dopplers,
    generated_dopplers,
    labels=None,
    colors=None,
    save_dir="WaveNet_beat/plots",
    prefix="signals_overlay",
):
    """
    Overlay ECG (1 or 2 channel), real Doppler, and generated Doppler for each sample.
    """
    default_labels = {
        "fetal_ecg": "Fetal ECG",
        "maternal_ecg": "Maternal ECG",
        "ecg": "ECG",
        "real": "Real Doppler",
        "gen": "Generated Doppler",
    }
    default_colors = {
        "fetal_ecg": "royalblue",
        "maternal_ecg": "darkgreen",
        "ecg": "royalblue",
        "real": "blue",
        "gen": "red",
    }
    labels = labels or default_labels
    colors = colors or default_colors

    os.makedirs(save_dir, exist_ok=True)

    for i, (ecg, real_dop, gen_dop) in enumerate(
        zip(ecgs, real_dopplers, generated_dopplers), start=1
    ):
        plt.figure(figsize=(12, 6))

        # ECG
        if ecg.ndim == 2 and ecg.shape[1] > 1:
            n_channels = ecg.shape[1]

            # Channel 0: fetal
            plt.plot(
                ecg[:, 0],
                label=labels.get("fetal_ecg", "Fetal ECG"),
                color=colors.get("fetal_ecg", "royalblue"),
                linewidth=1.2,
                alpha=0.8,
            )

            # Channel 1: maternal
            if n_channels > 1:
                plt.plot(
                    ecg[:, 1],
                    label=labels.get("maternal_ecg", "Maternal ECG"),
                    color=colors.get("maternal_ecg", "darkgreen"),
                    linewidth=1.2,
                    alpha=0.8,
                )

            # Any extra channels
            additional_colors = ["orange", "purple", "brown", "pink"]
            for ch in range(2, n_channels):
                color = additional_colors[(ch - 2) % len(additional_colors)]
                plt.plot(
                    ecg[:, ch],
                    label=f"ECG Channel {ch+1}",
                    color=color,
                    linewidth=1.2,
                    alpha=0.8,
                )
        else:
            if ecg.ndim == 2:
                ecg = ecg.flatten()
            plt.plot(
                ecg,
                label=labels.get("ecg", "ECG"),
                color=colors.get("ecg", "royalblue"),
                linewidth=1.2,
                alpha=0.8,
            )

        # Doppler
        plt.plot(
            real_dop,
            label=labels.get("real", "Real Doppler"),
            color=colors.get("real", "blue"),
            linewidth=1.5,
        )
        plt.plot(
            gen_dop,
            label=labels.get("gen", "Generated Doppler"),
            color=colors.get("gen", "red"),
            linewidth=1.5,
            linestyle="--",
        )

        plt.title(
            f"Sample {i}: ECG and Doppler Signals Overlay", fontsize=14, fontweight="bold"
        )
        plt.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
        plt.xlabel("Time / Sample Index", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.grid(alpha=0.3, linestyle="-", linewidth=0.5)
        plt.gca().set_facecolor("#f8f9fa")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{prefix}_sample_{i}.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
        plt.close()


# ==================== Training History Plot ====================

def plot_training_history(history, save_dir_plots, split_name="training"):
    train_loss = history.history["loss"]
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_loss, "b-o", label="Training Loss", linewidth=2, markersize=4)

    if val_loss:
        plt.plot(epochs, val_loss, "r-s", label="Validation Loss", linewidth=2, markersize=4)

    plt.title("Model Training and Validation Loss", fontsize=16, fontweight="bold")
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    if val_loss:
        best_epoch = int(np.argmin(val_loss) + 1)
        best_val_loss = float(np.min(val_loss))
        plt.annotate(
            f"Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}",
            xy=(best_epoch, best_val_loss),
            xytext=(best_epoch + len(epochs) * 0.1, best_val_loss + max(train_loss) * 0.1),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            fontsize=10,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    plt.tight_layout()
    os.makedirs(save_dir_plots, exist_ok=True)
    plot_path = os.path.join(save_dir_plots, f"training_history_{split_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Training history plot saved to: {plot_path}")


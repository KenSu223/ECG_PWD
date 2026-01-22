"""
Load best-eval checkpoints for four models and plot predictions for a chosen test sample.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model
from scipy.signal import butter, sosfiltfilt

import modules
import layers


def sos_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="bandpass", output="sos")


def ecg_bandpass_zero_phase(x, fs, lowcut=0.5, highcut=40.0, order=2):
    sos = sos_bandpass(lowcut, highcut, fs, order=order)
    return sosfiltfilt(sos, x)


def butter_lowpass_sos(cutoff_hz, fs, order=4):
    return butter(order, cutoff_hz, btype="low", fs=fs, output="sos")


def smooth_doppler_envelope(env, fs, cutoff_hz=10.0, order=4):
    sos = butter_lowpass_sos(cutoff_hz, fs, order=order)
    return sosfiltfilt(sos, env)


def normalize_signal(signal):
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    return 2 * (signal - signal_min) / (signal_max - signal_min) - 1


def preprocess_ecg_batch(ecg_batch, fs, lowcut=0.5, highcut=40.0, order=2):
    processed = np.empty_like(ecg_batch)
    for i in range(ecg_batch.shape[0]):
        filtered = ecg_bandpass_zero_phase(ecg_batch[i], fs, lowcut=lowcut, highcut=highcut, order=order)
        processed[i] = normalize_signal(filtered)
    return processed


def preprocess_ecg_inputs(X, fs, lowcut=0.5, highcut=40.0, order=2):
    if X.shape[-1] == 1:
        return preprocess_ecg_batch(np.squeeze(X, axis=-1), fs, lowcut=lowcut, highcut=highcut, order=order)[:, :, None]

    processed = np.empty_like(X)
    for ch in range(X.shape[-1]):
        processed[:, :, ch] = preprocess_ecg_batch(X[:, :, ch], fs, lowcut=lowcut, highcut=highcut, order=order)
    return processed


def build_one_channel_wavenet(input_shape, filters=64, kernel_size=20, dilation_rates=None):
    if dilation_rates is None:
        dilation_rates = [2**i for i in range(7)]

    inp = Input(shape=input_shape, name="ecg_fetal_1ch")
    x = layers.DilatedConv1D(filters=filters, kernel_size=1, dilation_rate=1, padding="same")(inp)

    skips = []
    for d in dilation_rates:
        x, s = modules.wavenet_block(x, filters, kernel_size, d)
        skips.append(s)

    out = Add()(skips)
    out = tf.keras.layers.Activation("relu")(out)
    out = layers.Conv1D(filters, 1)(out)
    out = tf.keras.layers.Activation("relu")(out)
    out = layers.Conv1D(1, 1)(out)
    out = tf.keras.layers.Activation("tanh")(out)

    model = Model(inputs=inp, outputs=out, name="WaveNet_OneChannel_Fetal")
    return model


def plot_stacked_predictions(output_path, y_true, predictions, titles):
    fig, axes = plt.subplots(len(predictions), 1, figsize=(14, 10), sharex=True)
    if len(predictions) == 1:
        axes = [axes]

    for ax, pred, title in zip(axes, predictions, titles):
        ax.plot(y_true, color="red", linewidth=1, label="Ground Truth")
        ax.plot(pred, color="blue", linewidth=1, label="Generated")
        ax.set_title(title)
        ax.grid(alpha=0.2)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time / Sample Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Load checkpoints and plot predictions")
    parser.add_argument("--data_dir", type=str, default=str(base_dir / "WaveNet_beat" / "data"))
    parser.add_argument("--X_file", type=str, default="X.npy")
    parser.add_argument("--Y_file", type=str, default="Y.npy")
    parser.add_argument("--val_idx_file", type=str, default=str(base_dir / "idx_val.npy"))
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Index within val_idx (or global if --index_source=global)")
    parser.add_argument("--index_source", choices=["val", "global"], default="val")
    parser.add_argument("--fs", type=float, default=284.0)
    parser.add_argument("--ecg_lowcut", type=float, default=0.5)
    parser.add_argument("--ecg_highcut", type=float, default=40.0)
    parser.add_argument("--ecg_filter_order", type=int, default=2)
    parser.add_argument("--smooth_cutoff_hz", type=float, default=10.0)
    parser.add_argument("--smooth_order", type=int, default=4)
    parser.add_argument("--no_input_preprocess", action="store_false",
                        dest="apply_input_preprocess", default=True)
    parser.add_argument("--no_output_smoothing", action="store_false",
                        dest="apply_output_smoothing", default=True)
    parser.add_argument("--checkpoint_dir", type=str,
                        default=str(base_dir / "WaveNet_beat" / "plots" / "model_checkpoints"))
    parser.add_argument("--output_dir", type=str, default=str(base_dir / "WaveNet_beat" / "plots"))
    parser.add_argument("--output_name", type=str, default="four_model_overlay.png")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X = np.load(os.path.join(args.data_dir, args.X_file))
    Y = np.load(os.path.join(args.data_dir, args.Y_file))
    val_idx = np.load(args.val_idx_file)

    if args.index_source == "val":
        if args.sample_index < 0 or args.sample_index >= len(val_idx):
            raise ValueError("sample_index is out of range for val_idx.")
        sample_idx = int(val_idx[args.sample_index])
    else:
        if args.sample_index < 0 or args.sample_index >= len(X):
            raise ValueError("sample_index is out of range for X.")
        sample_idx = int(args.sample_index)

    test_sample_full = X[sample_idx:sample_idx + 1]
    if args.apply_input_preprocess:
        test_sample_full = preprocess_ecg_inputs(
            test_sample_full,
            fs=args.fs,
            lowcut=args.ecg_lowcut,
            highcut=args.ecg_highcut,
            order=args.ecg_filter_order,
        )
    test_sample_fetal = test_sample_full[:, :, 0:1]
    y_true = np.squeeze(Y[sample_idx])

    model_builders = [
        ("One-channel (fetal only)", lambda input_shape: build_one_channel_wavenet(input_shape)),
        ("Two-channel (fetal + maternal)", lambda input_shape: modules.WaveNet_two_channel(input_shape=input_shape)),
        ("Two-channel (cross-attention)", lambda input_shape: modules.WaveNet_two_channel_cross_attention(input_shape=input_shape)),
        ("Two-channel (combined attention)", lambda input_shape: modules.WaveNet_two_channel_combined_attention(input_shape=input_shape)),
    ]

    predictions = []
    titles = []

    for name, builder in model_builders:
        checkpoint_path = os.path.join(
            args.checkpoint_dir, name.lower().replace(" ", "_").replace("+", "plus") + ".weights.h5"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if "One-channel" in name:
            model = builder(test_sample_fetal.shape[1:])
            model.load_weights(checkpoint_path)
            pred = model.predict(test_sample_fetal, verbose=0)
        else:
            model = builder(test_sample_full.shape[1:])
            model.load_weights(checkpoint_path)
            pred = model.predict(test_sample_full, verbose=0)

        pred = np.squeeze(pred)
        if args.apply_output_smoothing:
            pred = smooth_doppler_envelope(
                pred, fs=args.fs, cutoff_hz=args.smooth_cutoff_hz, order=args.smooth_order
            )
        predictions.append(pred)
        titles.append(name)

    output_path = os.path.join(args.output_dir, args.output_name)
    plot_stacked_predictions(output_path, y_true, predictions, titles)
    print(f"Saved stacked plot to: {output_path}")


if __name__ == "__main__":
    main()

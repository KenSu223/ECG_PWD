"""
Train four fetal-maternal fusion model variants and visualize predictions.

Models:
1) One-channel (fetal ECG only)
2) Two-channel (fetal + maternal)
3) Two-channel with cross-attention
4) Two-channel with combined attention (cross + self)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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


def smooth_doppler_batch(doppler_batch, fs, cutoff_hz=10.0, order=4):
    smoothed = np.empty_like(doppler_batch)
    for i in range(doppler_batch.shape[0]):
        smoothed[i] = smooth_doppler_envelope(doppler_batch[i], fs=fs, cutoff_hz=cutoff_hz, order=order)
    return smoothed


def composite_loss(alpha=0.5):
    def loss_fn(y_true, y_pred):
        if len(y_true.shape) == 3:
            y_true_2d = tf.squeeze(y_true, axis=-1)
        else:
            y_true_2d = y_true
        if len(y_pred.shape) == 3:
            y_pred_2d = tf.squeeze(y_pred, axis=-1)
        else:
            y_pred_2d = y_pred

        mae = tf.reduce_mean(tf.abs(y_true_2d - y_pred_2d))

        dy_true = y_true_2d[:, 1:] - y_true_2d[:, :-1]
        dy_pred = y_pred_2d[:, 1:] - y_pred_2d[:, :-1]
        deriv = tf.reduce_mean(tf.abs(dy_true - dy_pred))

        eps = 1e-7
        y_true_centered = y_true_2d - tf.reduce_mean(y_true_2d, axis=1, keepdims=True)
        y_pred_centered = y_pred_2d - tf.reduce_mean(y_pred_2d, axis=1, keepdims=True)
        y_true_std = tf.math.reduce_std(y_true_2d, axis=1, keepdims=True) + eps
        y_pred_std = tf.math.reduce_std(y_pred_2d, axis=1, keepdims=True) + eps
        y_true_norm = y_true_centered / y_true_std
        y_pred_norm = y_pred_centered / y_pred_std
        corr = tf.reduce_mean(y_true_norm * y_pred_norm, axis=1)
        corr = tf.clip_by_value(corr, -1.0, 1.0)
        corr_loss = tf.reduce_mean(1.0 - corr)

        return mae + alpha * deriv + alpha * corr_loss

    return loss_fn


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


def train_model(
    model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    lr,
    epochs,
    batch_size,
    checkpoint_path=None,
    alpha=0.5,
    lr_reduce_factor=0.5,
    lr_reduce_patience=5,
    lr_min=1e-5,
    early_stopping_patience=20,
    use_best_weights=True,
):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=composite_loss(alpha=alpha))
    callbacks = []
    if checkpoint_path and use_best_weights:
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                verbose=1,
            )
        )
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=use_best_weights,
            verbose=1,
        )
    )
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_reduce_factor,
            patience=lr_reduce_patience,
            min_lr=lr_min,
            verbose=1,
        )
    )
    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


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

    parser = argparse.ArgumentParser(description="Train four models and visualize predictions")
    parser.add_argument("--data_dir", type=str, default=str(base_dir / "WaveNet_beat" / "data"))
    parser.add_argument("--X_file", type=str, default="X.npy")
    parser.add_argument("--Y_file", type=str, default="Y.npy")
    parser.add_argument("--train_idx_file", type=str, default=str(base_dir / "idx_train.npy"))
    parser.add_argument("--val_idx_file", type=str, default=str(base_dir / "idx_val.npy"))
    parser.add_argument("--test_sample_index", type=int, default=0,
                        help="Index within val_idx to use as the shared test sample")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--lr_reduce_patience", type=int, default=5)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--no_best_weights", action="store_false",
                        dest="use_best_weights", default=True)
    parser.add_argument("--output_dir", type=str, default=str(base_dir / "WaveNet_beat" / "plots"))
    parser.add_argument("--output_name", type=str, default="four_model_overlay.png")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, "model_checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    X = np.load(os.path.join(args.data_dir, args.X_file))
    Y = np.load(os.path.join(args.data_dir, args.Y_file))

    train_idx = np.load(args.train_idx_file)
    val_idx = np.load(args.val_idx_file)

    if args.test_sample_index < 0 or args.test_sample_index >= len(val_idx):
        raise ValueError("test_sample_index is out of range for val_idx.")

    test_idx = int(val_idx[args.test_sample_index])

    X_train_full = X[train_idx]
    Y_train = Y[train_idx]
    X_val_full = X[val_idx]
    Y_val = Y[val_idx]

    if args.apply_input_preprocess:
        X_train_full = preprocess_ecg_inputs(
            X_train_full,
            fs=args.fs,
            lowcut=args.ecg_lowcut,
            highcut=args.ecg_highcut,
            order=args.ecg_filter_order,
        )
        X_val_full = preprocess_ecg_inputs(
            X_val_full,
            fs=args.fs,
            lowcut=args.ecg_lowcut,
            highcut=args.ecg_highcut,
            order=args.ecg_filter_order,
        )

    X_train_fetal = X_train_full[:, :, 0:1]
    X_val_fetal = X_val_full[:, :, 0:1]

    test_sample_full = X[test_idx:test_idx + 1]
    if args.apply_input_preprocess:
        test_sample_full = preprocess_ecg_inputs(
            test_sample_full,
            fs=args.fs,
            lowcut=args.ecg_lowcut,
            highcut=args.ecg_highcut,
            order=args.ecg_filter_order,
        )
    test_sample_fetal = test_sample_full[:, :, 0:1]
    y_true = np.squeeze(Y[test_idx])

    model_builders = [
        ("One-channel (fetal only)", lambda input_shape: build_one_channel_wavenet(input_shape)),
        ("Two-channel (fetal + maternal)", lambda input_shape: modules.WaveNet_two_channel(input_shape=input_shape)),
        ("Two-channel (cross-attention)", lambda input_shape: modules.WaveNet_two_channel_cross_attention(input_shape=input_shape)),
        ("Two-channel (combined attention)", lambda input_shape: modules.WaveNet_two_channel_combined_attention(input_shape=input_shape)),
    ]

    predictions = []
    titles = []

    for name, builder in model_builders:
        print(f"\n=== Training: {name} ===")
        checkpoint_path = os.path.join(
            checkpoints_dir, name.lower().replace(" ", "_").replace("+", "plus") + ".weights.h5"
        )
        if "One-channel" in name:
            model = builder(X_train_fetal.shape[1:])
            train_model(
                model,
                X_train_fetal,
                Y_train,
                X_val_fetal,
                Y_val,
                args.lr,
                args.epochs,
                args.batch_size,
                checkpoint_path=checkpoint_path,
                alpha=args.alpha,
                lr_reduce_factor=args.lr_reduce_factor,
                lr_reduce_patience=args.lr_reduce_patience,
                lr_min=args.lr_min,
                early_stopping_patience=args.early_stopping_patience,
                use_best_weights=args.use_best_weights,
            )
            if args.use_best_weights and os.path.exists(checkpoint_path):
                model.load_weights(checkpoint_path)
            pred = model.predict(test_sample_fetal, verbose=0)
        else:
            model = builder(X_train_full.shape[1:])
            train_model(
                model,
                X_train_full,
                Y_train,
                X_val_full,
                Y_val,
                args.lr,
                args.epochs,
                args.batch_size,
                checkpoint_path=checkpoint_path,
                alpha=args.alpha,
                lr_reduce_factor=args.lr_reduce_factor,
                lr_reduce_patience=args.lr_reduce_patience,
                lr_min=args.lr_min,
                early_stopping_patience=args.early_stopping_patience,
                use_best_weights=args.use_best_weights,
            )
            if args.use_best_weights and os.path.exists(checkpoint_path):
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
    print(f"\nSaved stacked plot to: {output_path}")


if __name__ == "__main__":
    main()

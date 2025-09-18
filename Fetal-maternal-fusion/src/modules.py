import layers
import numpy as np
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential
from skimage.transform import resize
import pywt
import matplotlib.pyplot as plt
from tensorflow import keras

def wavenet_block(x_in, filters, kernel_size, dilation_rate):
    # gated convs (causal, dilated)
    tanh_out = layers.DilatedConv1D(filters=filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, activation='tanh')(x_in)
    sigm_out = layers.DilatedConv1D(filters=filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, activation='sigmoid')(x_in)
    x = Multiply()([tanh_out, sigm_out])  # gated output

    # skip connection (1x1)
    skip = layers.Conv1D(filters, 1)(x)

    # residual projection through conv (1x1) + add
    res = layers.Conv1D(filters, 1)(x_in)
    x_out = Add()([x, res])
    return x_out, skip

def WaveNet_two_channel(input_shape, filters=64, kernel_size=20, dilation_rates=[2**i for i in range(7)]):
    """
    input_shape: (timesteps, 2)  # (WIN, 2)
    Channel order: [:, :, 0] = fetal ECG, [:, :, 1] = maternal ECG
    """
    inp = Input(shape=input_shape, name="ecg_fetal_maternal_2ch")

    #  Dilated 1x1 stem to project 2ch -> filters
    x = layers.DilatedConv1D(filters=filters, kernel_size=1, dilation_rate=1, padding='same')(inp)

    skips = []
    for d in dilation_rates:
        x, s = wavenet_block(x, filters, kernel_size, d)
        skips.append(s)

    out = Add()(skips)
    out = layers.Activation(out, 'relu')  # post-processing: ReLU -> 1x1 -> ReLU -> 1x1
    out = layers.Conv1D(filters, 1)(out)
    out = layers.Activation(out, 'relu')
    out = layers.Conv1D(1, 1)(out)
    out = layers.Activation(out, 'tanh')

    model = keras.Model(inputs=inp, outputs=out, name="WaveNet_two_channel_early_fusion")
    model.summary()
    return model


def create_scalogram(sig, fs=284, time_bins=160, freq_bins=80):
    scales = np.arange(1, freq_bins + 1)
    coeffs, f = pywt.cwt(sig, scales, wavelet='cgau8', sampling_period=1/fs)
    coeffs = np.abs(coeffs)
    return coeffs, f

# process a batch of signals
def create_batch_scalograms(signals_batch, fs=2000, time_bins=160, freq_bins=80):
    coeffs = []
    f_s = []
    for sig in signals_batch:
        coeff, f = create_scalogram(sig, fs, time_bins, freq_bins)
        coeffs.append(coeff)
        f_s.append(f)
    return coeffs, f_s

def plot_ecg_doppler_pairs(ecgs, real_dopplers, generated_dopplers):
    """Plots ECG and corresponding real and generated Doppler pairs."""
    plt.figure(figsize=(18, 8))

    for i, (ecg, real_dopple, generated_dopple) in enumerate(zip(ecgs, real_dopplers, generated_dopplers)):
        # Plotting ECG
        plt.subplot(len(ecgs), 3, 3*i + 1)  # Adjust the number of rows dynamically based on the length of ecgs
        plt.plot(ecg, color='royalblue')
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis

        # Plotting Real Doppler
        plt.subplot(len(ecgs), 3, 3*i + 2)
        plt.plot(real_dopple, color='blue')
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis

        # Plotting Generated Doppler
        plt.subplot(len(ecgs), 3, 3*i + 3)
        plt.plot(generated_dopple, color='red')
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis

    plt.tight_layout()
    plt.savefig('WaveNet_beat/plots/signals_test.jpg')
    plt.show()


def plot_scalogram(real, generated, time_bins=160, freq_bins=80):
    plt.figure(figsize=(18, 10))
    coeffs_rs, fs_rs, coeffs_gs, fs_gs = [],[],[],[]
    fs = 284
    t = np.linspace(0, len(real)/fs, time_bins)
    scales = np.arange(1, freq_bins)
    tensor_real=[]
    tensor_generated=[]
    frequencies = fs

    for i in range(len(real)):
        coeffs_r, fs_r=create_scalogram(real[i],fs,time_bins, freq_bins)
        coeffs_g, fs_g=create_scalogram(generated[i],fs,time_bins, freq_bins)
        coeffs_gs.append(coeffs_g)
        coeffs_rs.append(coeffs_r)
        fs_gs.append(fs_g)
        fs_rs.append(fs_r)

    for i in range(len(real)):
        plt.subplot(len(real), 2, 2*i + 1)
        plt.pcolormesh(np.arange(coeffs_rs[i].shape[1]) , fs_rs[i],coeffs_rs[i], shading='gouraud', cmap='bwr')
        plt.yticks([])
        plt.xticks([])
        plt.ylim(10,1000)

        plt.subplot(len(real), 2, 2*i + 2)
        plt.pcolormesh(np.arange(coeffs_gs[i].shape[1]) , fs_gs[i],coeffs_gs[i], shading='gouraud',cmap='bwr')
        plt.xticks([])
        plt.yticks([])
        plt.ylim(10,1000)

    plt.savefig('WaveNet_beat/plots/scalograms_test.jpg')
    plt.show()

def plot_ecg_doppler_overlay(ecgs, real_dopplers, generated_dopplers, labels=None, colors=None):
    """
    Plots ECG, real Doppler, and generated Doppler signals overlaid on the same axes for each sample.

    Parameters:
    - ecgs: list or array of ECG signals
    - real_dopplers: list or array of corresponding real Doppler signals
    - generated_dopplers: list or array of corresponding generated Doppler signals
    - labels: dict mapping 'ecg', 'real', 'gen' to legend labels (optional)
    - colors: dict mapping 'ecg', 'real', 'gen' to color strings (optional)
    """
    labels = labels or {'ecg': 'ECG', 'real': 'Real Doppler', 'gen': 'Generated Doppler'}
    colors = colors or {'ecg': 'royalblue', 'real': 'blue', 'gen': 'red'}

    for i, (ecg, real_dop, gen_dop) in enumerate(zip(ecgs, real_dopplers, generated_dopplers), start=1):
        plt.figure(figsize=(10, 4))
        plt.plot(ecg, label=labels['ecg'], color=colors['ecg'], linewidth=1)
        plt.plot(real_dop, label=labels['real'], color=colors['real'], linewidth=1)
        plt.plot(gen_dop, label=labels['gen'], color=colors['gen'], linewidth=1, linestyle='--')

        plt.title(f'Sample {i}')
        plt.legend(loc='upper right')
        plt.xlabel('Time / Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'WaveNet_beat/plots/signals_test_{i}.jpg')
        plt.show()
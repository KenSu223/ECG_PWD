import layers
import numpy as np
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential
from skimage.transform import resize
import pywt
import matplotlib.pyplot as plt


def wavenet_block(x, filters, kernel_size, dilation_rate, input_layer):
    """Defines a single WaveNet block."""
    tanh_out = layers.CausalConv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, activation='tanh')(x)

    sigmoid_out = layers.CausalConv1D(filters=filters, kernel_size=kernel_size,  dilation_rate=dilation_rate,activation='sigmoid')(x)

    x = Multiply()([tanh_out, sigmoid_out])

    # Skip connection
    skip_conn = layers.Conv1D(filters, 1)(x)
    
    # Residual connection
    x = Add()([x, input_layer])
    
    return x, skip_conn


def WaveNet(input_shape, filters=64, kernel_size=20, dilation_rates=[2**i for i in range(5)]): 

    # Input layer
    input_layer = Input(shape=input_shape)

    # Initial condition layer to start the residual connections
    skip_connections = []

    x = input_layer
    #x = layers.Dense(400, activation='tanh')(x)
    for dilation_rate in dilation_rates:
        x, skip_conn = wavenet_block(x, filters, kernel_size, dilation_rate, input_layer)
        skip_connections.append(skip_conn)

    out = Add()(skip_connections)
    out = layers.Activation(out, 'tanh')
    out = layers.Conv1D(1,  kernel_size=1)(out)
    out = layers.Activation(out, 'tanh')
    out = layers.Conv1D(1,  kernel_size=1)(out)

    out = Flatten()(out)
    out = layers.Dense(800, activation='tanh')(out)

    # Building the model
    model = Model(inputs=input_layer, outputs=out)
    model.summary()
    
    return model


def create_scalogram(sig, fs=2000, time_bins=160, freq_bins=80):
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
    fs = 2000
    t = np.linspace(0, 0.4, time_bins)
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

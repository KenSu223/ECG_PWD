import numpy as np
import pywt
import layers
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten, TimeDistributed, UpSampling1D
from keras.models import Model, Sequential
import matplotlib.pyplot as plt



## Generator Architecture
def generator(latent_dim):
    gen_input = Input(shape=(latent_dim,))
    x = layers.Dense(200)(gen_input)
    x = Reshape((200, 1))(x)
    
    # Using LSTM layers
    x = layers.LSTM(256, return_sequences=True)(x)
    x = UpSampling1D(size=2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = UpSampling1D(size=2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)


    # Final layer to produce output
    gen_output = TimeDistributed(layers.Dense(1, activation='tanh'))(x)

    generator = Model(gen_input, gen_output, name="generator")
    generator.summary()

    return generator


## Discriminator Architecture
def discriminator(shape):
    dis_input = Input(shape=shape)
    
    # Using LSTM layers
    x = layers.LSTM(32, return_sequences=True)(dis_input)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    
    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(dis_input, dis_output, name="discriminator")
    discriminator.summary()

    return discriminator

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
        plt.plot(ecg)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1)  # y-axis


        # Plotting Real Doppler
        plt.subplot(len(ecgs), 3, 3*i + 2)
        plt.plot(real_dopple)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1)  # y-axis

        # Plotting Generated Doppler
        plt.subplot(len(ecgs), 3, 3*i + 3)
        plt.plot(generated_dopple)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1)  # y-axis

    plt.tight_layout()
    plt.savefig('LSTM_GAN_beat/plots/signals.jpg')
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
        plt.pcolormesh(np.arange(coeffs_rs[i].shape[1]) , fs_rs[i],coeffs_rs[i], shading='gouraud',cmap='viridis')
        plt.yticks([])
        plt.xticks([])
        plt.ylim(0,1000)

        plt.subplot(len(real), 2, 2*i + 2)
        plt.pcolormesh(np.arange(coeffs_gs[i].shape[1]) , fs_gs[i],coeffs_gs[i], shading='gouraud',cmap='viridis')
        plt.xticks([])
        plt.yticks([])
        plt.ylim(0,1000)

    plt.savefig('LSTM_GAN_beat/plots/scalograms.jpg')
    plt.show()

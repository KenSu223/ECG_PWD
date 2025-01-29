import numpy as np
import pywt
import layers
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential
import matplotlib.pyplot as plt

# Encoder Architecture
def build_encoder(latent_dim, kernel_size=8, strides=1):
        
        en_input = Input(shape=(latent_dim,))
        x = Reshape((latent_dim, 1))(en_input)
        
        x = layers.Conv1D(256, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.BatchNormalization()(x)

        x = Flatten()(x)
        en_output = layers.Dense(25*64)(x)

        encoder = Model(en_input, en_output, name="encoder")
        encoder.summary()

        return encoder


# Decoder Architecture
def build_decoder(shape, kernel_size=8, strides=1):
    de_input = Input(shape=shape)
    x = Reshape((25, 64))(de_input)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = UpSampling1D(4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = UpSampling1D(4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = UpSampling1D(2)(x)
    x = layers.BatchNormalization()(x)
    de_output = layers.Conv1D(1, kernel_size=kernel_size, strides=strides, padding="same")(x)

    decoder = Model(de_input, de_output, name="decoder")
    decoder.summary()

    return decoder

# Build the model
def Autoencoder(input_shape):
    encoder = build_encoder(input_shape)
    encoded_shape = encoder.output_shape[1:]
    decoder = build_decoder(encoded_shape)
    
    # Input for the autoencoder
    autoencoder_input = Input(shape=input_shape)
    # Pass it through encoder
    encoded = encoder(autoencoder_input)
    # And then decoder
    decoded = decoder(encoded)
    # Autoencoder model
    autoencoder = Model(autoencoder_input, decoded, name="autoencoder")
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder

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
    plt.savefig('Autoencoder_beat/plots/signals_test.jpg')
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

    plt.savefig('Autoencoder_beat/plots/scalograms_test.jpg')
    plt.show()

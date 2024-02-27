import layers
import numpy as np
import tensorflow as tf
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential
from skimage.transform import resize
import pywt
import matplotlib.pyplot as plt



## Generator Architecture
def generator(latent_dim, kernel_size=12, strides=2):
        
        gen_input = Input(shape=(latent_dim,))
        x = layers.Dense(100)(gen_input)
        x = Reshape((100, 1))(x)
                

        x = layers.Conv1DTranspose(512, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1DTranspose(256, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1DTranspose(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        
        gen_output = layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", activation='tanh')(x)

        generator = Model(gen_input, gen_output, name = "generator")
        generator.summary()

        return generator


## Discriminator Architecture
def discriminator_1(shape, kernel_size=12, strides=2):

    dis_input = Input(shape=shape)

    
    x = layers.Conv1D(32, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(dis_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    
    
    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)

    
    discriminator_1 = Model(dis_input, dis_output, name = "discriminator_1")
    discriminator_1.summary()

    return discriminator_1


def discriminator_2(shape, kernel_size=(5,5), strides=(2,2)):
    dis_input = Input(shape=shape)
    
    #x = Reshape((80,160,1))(dis_input)
    
    x = layers.Conv2D(32, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(dis_input)
    #x = layers.BatchNormalization()(x)
    #x = layers.Conv2D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)

    discriminator_2 = Model(dis_input, dis_output, name="discriminator_2")
    discriminator_2.summary()
    return discriminator_2


def create_scalogram(sig, fs=2000, time_bins=160, freq_bins=80):
    scales = np.arange(1, freq_bins + 1)
    if not isinstance(sig, np.ndarray):
        sig = sig.numpy()
    coeffs, _ = pywt.cwt(sig, scales, wavelet='morl', sampling_period=1/fs)
    f = np.abs(coeffs)
    f = resize(f, (np.shape(coeffs)[0], time_bins), mode='constant')
    return f

# process a batch of signals
def create_batch_scalograms(signals_batch, fs=2000, time_bins=160, freq_bins=80):
    batch_scalograms = []
    for sig in signals_batch:
        scalogram = create_scalogram(sig, fs, time_bins, freq_bins)
        batch_scalograms.append(scalogram)
    return np.array(batch_scalograms)

def plot_ecg_doppler_pairs(noise, generated_dopplers, real_signal):
    """Plots ECG and corresponding real and generated Doppler pairs."""
    plt.figure(figsize=(18, 8))

    for i, (ecg, generated_dopple) in enumerate(zip(noise, generated_dopplers)):
        # Plotting ECG
        plt.subplot(len(noise), 2, 2*i + 1) 
        plt.plot(ecg)
        plt.title(f'Noise {i+1}')
        plt.axis('off')

        # Plotting Generated Doppler
        plt.subplot(len(noise), 2, 2*i + 2)
        plt.plot(generated_dopple)
        plt.title(f'Generated Doppler {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('dWGAN-GP_noise/plots/generated_signals.png')
    plt.show()

    for i, (ecg_scal, generated_dopple_scal) in enumerate(zip(real_signal, generated_dopplers)):
        # Plotting ECG_scal
        plt.subplot(len(noise), 2, 2*i + 1) 
        plt.plot(create_scalogram(ecg_scal))
        plt.title(f'Scalogram of Real Doppler {i+1}')
        plt.axis('off')

        # Plotting Generated Doppler
        plt.subplot(len(noise), 2, 2*i + 2)
        plt.plot(create_scalogram(generated_dopple_scal))
        plt.title(f'Scalogram of Generated Doppler {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('dWGAN-GP_noise/plots/generated_signals_scal.png')
    plt.show()
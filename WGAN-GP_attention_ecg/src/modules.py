import layers
from keras.models import load_model
import numpy as np
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten, Layer
from keras.models import Model, Sequential
from skimage.transform import resize
import pywt
import matplotlib.pyplot as plt


## Generator Architecture
def generator(latent_dim, kernel_size=12, strides=2, return_attention=False):
        
        gen_input = Input(shape=(latent_dim,))
        x = layers.Dense(100)(gen_input)
        x = Reshape((100, 1))(x)

        
        x = layers.Conv1DTranspose(512, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1DTranspose(256, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1DTranspose(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(32, kernel_size=kernel_size, strides=1, padding="same", activation='leaky_relu')(x)

        gen_output = layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", activation='tanh')(x)


        generator = Model(gen_input, gen_output, name = "generator")
        generator.summary()

        return generator


## Discriminator Architecture
def discriminator(shape, kernel_size=20, strides=2):

    dis_input = Input(shape=shape)

    
    x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(dis_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=kernel_size, strides=1, padding="same", activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling1D(pool_size=2)(x)
    #x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    #x = layers.Dropout()(x)
    
    #x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    #x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout()(x)
    
    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)

    
    discriminator = Model(dis_input, dis_output, name = "discriminator")
    discriminator.summary()

    return discriminator

def pro_discriminator1(discriminator, kernel_size=12, strides=2):
    # Start with the same input as the original model
    dis_input = discriminator.input

    # Add all layers from the pretrained model except the last one
    x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(dis_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x2 = discriminator.layers[4].output
    x = Add()([x, x2])
    
    x = layers.Conv1D(256, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling1D(pool_size=2)(x)

    #x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)

    # Create the new model
    pro_discriminator = Model(dis_input, dis_output, name="pro_discriminator")
    pro_discriminator.summary()

    return pro_discriminator

def pro_discriminator2(discriminator, pro_discriminator, kernel_size=12, strides=2):
    # Start with the same input as the original model
    dis_input = pro_discriminator.input

    # Add all layers from the pretrained model except the last one
    x = layers.Conv1D(64, kernel_size=kernel_size, strides=1, padding="same", activation='leaky_relu')(dis_input)
    x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x2 = pro_discriminator.layers[6].output
    x = Add()([x, x2])

    x = layers.Conv1D(128, kernel_size=kernel_size, strides=1, padding="same", activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    
    #x3 = discriminator.layers[3].output
    #x = Add()([x, x3])

    #x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    #x = layers.BatchNormalization()(x)

    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)

    # Create the new model
    pro_discriminator2 = Model(dis_input, dis_output, name="pro_discriminator2")
    pro_discriminator2.summary()

    return pro_discriminator2

def load_trained_model(path):
     loaded_model = load_model(path)

     return loaded_model


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

def plot_ecg_doppler_pairs(ecgs, real_dopplers, generated_dopplers):
    """Plots ECG and corresponding real and generated Doppler pairs."""
    plt.figure(figsize=(18, 8))

    for i, (ecg, real_dopple, generated_dopple) in enumerate(zip(ecgs, real_dopplers, generated_dopplers)):
        # Plotting ECG
        plt.subplot(len(ecgs), 3, 3*i + 1)  # Adjust the number of rows dynamically based on the length of ecgs
        plt.plot(ecg)
        plt.title(f'ECG {i+1}')
        plt.axis('off')

        # Plotting Real Doppler
        plt.subplot(len(ecgs), 3, 3*i + 2)
        plt.plot(real_dopple)
        plt.title(f'Real Doppler {i+1}')
        plt.axis('off')

        # Plotting Generated Doppler
        plt.subplot(len(ecgs), 3, 3*i + 3)
        plt.plot(generated_dopple)
        plt.title(f'Generated Doppler {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('WGAN-GP_attention_ecg/plots/generated_signals.png')
    plt.show()


def plot_scalogram(indices, real, generated, time_bins, freq_bins):
    plt.figure(figsize=(18, 10))
    fs = 2000
    t = np.linspace(0, 0.4, 160)
    scales = np.arange(1, 80)
    tensor_real=np.zeros((indices,160,80))
    tensor_generated=np.zeros((indices,160,80))
    fc = pywt.central_frequency('morl')
    frequencies = fc * fs / scales

    for i in range(indices):
        s_r=create_scalogram(real[i],fs,time_bins, freq_bins)
        s_g=create_scalogram(generated[i],fs,time_bins, freq_bins)
        tensor_real[i,:,:]=s_r.T
        tensor_generated[i,:,:]=s_g.T

    for i, (real_dopple, generated_dopple) in enumerate(zip(tensor_real, tensor_generated)):
        plt.subplot(len(tensor_real/2), 2, 2*i + 1)
        plt.imshow(tensor_real[i].T, extent=[t[0], t[-1], 0, len(frequencies)-1], aspect='auto', origin='lower')
        yticks = np.arange(0, len(frequencies), 20)
        ytick_labels = [f"{frequencies[j]:.0f}" for j in yticks]
        plt.yticks(np.arange(len(frequencies)), [f"{freq:.0f}" for freq in frequencies])
        plt.yticks(yticks, ytick_labels)
        plt.xticks([])
        plt.title(f'Real Doppler {i+1}')
        plt.gca().invert_yaxis()

        plt.subplot(len(tensor_real/2), 2, 2*i + 2)
        plt.imshow(tensor_generated[i].T, extent=[t[0], t[-1], 0, len(frequencies)-1], aspect='auto', origin='lower')
        yticks = np.arange(0, len(frequencies), 20)
        ytick_labels = [f"{frequencies[j]:.0f}" for j in yticks]
        plt.yticks(np.arange(len(frequencies)), [f"{freq:.0f}" for freq in frequencies])
        plt.yticks(yticks, ytick_labels)
        plt.xticks([])
        plt.title(f'Generated Doppler {i+1}')
        plt.gca().invert_yaxis()

    plt.savefig(f'./WGAN-GP_attention_ecg/plots/scalogram.png')
    plt.show()


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
    x = layers.LSTM(512, return_sequences=True)(x)
    x = UpSampling1D(size=2)(x)
    x = layers.LSTM(256, return_sequences=True)(x)
    x = UpSampling1D(size=2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)


    # Final layer to produce output
    gen_output = TimeDistributed(layers.Dense(1))(x)

    generator = Model(gen_input, gen_output, name="generator")
    generator.summary()

    return generator


## Discriminator Architecture
def discriminator(shape):
    dis_input = Input(shape=shape)
    
    # Using LSTM layers
    x = layers.LSTM(32, return_sequences=True)(dis_input)
    #x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    
    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)
    
    discriminator = Model(dis_input, dis_output, name="discriminator")
    discriminator.summary()

    return discriminator

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
    plt.savefig('LSTM_GAN_ecg/plots/generated_signals.png')
    plt.show()
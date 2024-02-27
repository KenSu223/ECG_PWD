import layers
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential
import matplotlib.pyplot as plt



## encoder Architecture
def build_encoder(latent_dim, kernel_size=8, strides=1):
        
        en_input = Input(shape=(latent_dim,))
        x = Reshape((latent_dim, 1))(en_input)
        
        x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        en_output = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)

        encoder = Model(en_input, en_output, name="encoder")
        encoder.summary()

        return encoder


## decoder Architecture
def build_decoder(shape, kernel_size=8, strides=1):

    de_input = Input(shape=shape)

    x = layers.Conv1D(32, kernel_size=kernel_size, strides=strides, padding="same")(de_input)
    x = UpSampling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = UpSampling1D(2)(x)
    x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = UpSampling1D(2)(x)
    x = layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = UpSampling1D(2)(x)
    de_output = layers.Conv1D(1, kernel_size=kernel_size, strides=strides, padding="same")(x)

    decoder = Model(de_input, de_output, name="decoder")
    decoder.summary()

    return decoder

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
    plt.savefig('Autoencoder_ecg/plots/generated_signals.png')
    plt.show()
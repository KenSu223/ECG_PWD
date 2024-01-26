import layers
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential
import matplotlib.pyplot as plt



## Generator Architecture
def generator(latent_dim, kernel_size=8, strides=2):
        
        gen_input = Input(shape=(latent_dim,))
        x = layers.Dense(400)(gen_input)
        x = Reshape((400, 1))(x)

   
        
        x = layers.Conv1DTranspose(256, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.Conv1D(128, kernel_size=kernel_size, strides=1, padding="same", activation='leaky_relu')(x)
        x = layers.Conv1D(64, kernel_size=kernel_size, strides=1, padding="same", activation='leaky_relu')(x)
        #x = layers.Conv1D(128, kernel_size=kernel_size, strides=1, padding="same", activation='leaky_relu')(x)
        
        gen_output = layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", activation='tanh')(x)

        generator = Model(gen_input, gen_output)
        generator.summary()

        return generator


## Discriminator Architecture
def discriminator(shape, kernel_size=8, strides=2):

    dis_input = Input(shape=shape)

    
    x = layers.Conv1D(8, kernel_size=kernel_size, strides=strides, padding="same")(dis_input)
    #x = layers.Conv1D(8, kernel_size=kernel_size, strides=strides, padding="same")(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling1D(pool_size=2)(x)
    #x = layers.Dropout()(x)
    
    #x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    #x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout()(x)
    
    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)

    
    discriminator = Model(dis_input, dis_output)
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
    plt.savefig('GAN_ecg/plots/generated_signals.png')
    plt.show()
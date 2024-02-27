import layers
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



## Generator Architecture
def generator(latent_dim, kernel_size=8, strides=2):
        
        gen_input = Input(shape=(latent_dim,))
        x = layers.Dense(100)(gen_input)
        x = Reshape((100, 1))(x)

   
        
        x = layers.Conv1DTranspose(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.Conv1DTranspose(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        x = layers.Conv1DTranspose(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
        #x = layers.Conv1D(128, kernel_size=kernel_size, strides=1, padding="same", activation='leaky_relu')(x)
        
        gen_output = layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", activation='tanh')(x)

        generator = Model(gen_input, gen_output, name="generator")
        generator.summary()

        return generator


## Discriminator Architecture
def discriminator(shape, kernel_size=8, strides=1):

    dis_input = Input(shape=shape)

    
    x = layers.Conv1D(16, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(dis_input)
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

    
    discriminator = Model(dis_input, dis_output, name="discriminator")
    discriminator.summary()

    return discriminator

# Assuming your discriminator's architecture
def pro_discriminator1(discriminator, kernel_size=8, strides=1):
    # Start with the same input as the original model
    dis_input = discriminator.input

    # Add all layers from the pretrained model except the last one
    x = layers.Conv1D(32, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(dis_input)
    x = layers.Conv1D(16, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)

    x2 = discriminator.layers[1](dis_input)

    x = Add()([x, x2])
    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)

    # Create the new model
    pro_discriminator = Model(dis_input, dis_output, name="pro_discriminator")

    return pro_discriminator

def pro_discriminator2(discriminator, kernel_size=8, strides=1):
    # Start with the same input as the original model
    dis_input = discriminator.input

    # Add all layers from the pretrained model except the last one
    x = layers.Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(dis_input)
    x = layers.Conv1D(32, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)
    x2 = discriminator.layers[1](dis_input)
    x = Add()([x, x2])
    x = layers.Conv1D(16, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu')(x)

    x = Flatten()(x)
    dis_output = layers.Dense(1, activation='sigmoid')(x)

    # Create the new model
    pro_discriminator = Model(dis_input, dis_output, name="pro_discriminator")

    return pro_discriminator

def load_trained_model(path):
     loaded_model = load_model(path)

     return loaded_model
     

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
    plt.savefig('prodisGAN_ecg/plots/generated_signals.png')
    plt.show()
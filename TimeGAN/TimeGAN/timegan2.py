import tensorflow as tf
import numpy as np
from utils import extract_time, random_generator, batch_generator

def timegan(ori_data, parameters):
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        min_val = np.min(data, axis=(0, 1), keepdims=True)
        max_val = np.max(data, axis=(0, 1), keepdims=True)
        norm_data = (data - min_val) / (max_val - min_val + 1e-7)
        return norm_data, min_val.squeeze(), max_val.squeeze()

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    # Model Definition: Using Keras Functional API to define models.
    # Input placeholders
    X = tf.keras.Input(shape=(max_seq_len, dim), dtype=tf.float32)
    Z = tf.keras.Input(shape=(max_seq_len, z_dim), dtype=tf.float32)
    T = tf.keras.Input(shape=(), dtype=tf.int32)

    # Embedder & Recovery
    def build_embedder():
        inputs = tf.keras.Input(shape=(max_seq_len, dim))
        masked = tf.keras.layers.Masking(mask_value=0.)(inputs)
        for _ in range(num_layers):
            masked = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(masked)
        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dim))(masked)
        return tf.keras.Model(inputs, outputs)

    embedder = build_embedder()
    H = embedder(X)

    # Generator
    def build_generator():
        inputs = tf.keras.Input(shape=(max_seq_len, z_dim))
        for _ in range(num_layers):
            x = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(inputs if _ == 0 else x)
        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dim))(x)
 
        return tf.keras.Model(inputs, outputs)

    generator = build_generator()
    E_hat = generator(Z)
    print(E_hat)

    # Discriminator
    def build_discriminator():
        inputs = tf.keras.Input(shape=(max_seq_len, z_dim))
        print(inputs)
        for _ in range(num_layers):
            x = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(inputs if _ == 0 else x)
        print(x)
        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
        return tf.keras.Model(inputs, outputs)

    discriminator = build_discriminator()
    Y_fake = discriminator(E_hat)

    # Losses & Optimizers
    d_optimizer = tf.keras.optimizers.Adam()
    g_optimizer = tf.keras.optimizers.Adam()

    # Training Loop
    @tf.function
    def train_step(x_batch, z_batch, t_batch):
        with tf.GradientTape() as tape:
            y_fake = discriminator(generator(z_batch), training=True)
            d_loss = tf.reduce_mean(tf.losses.binary_crossentropy(tf.zeros_like(y_fake), y_fake))
        gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            y_fake = discriminator(generator(z_batch), training=True)
            g_loss = tf.reduce_mean(tf.losses.binary_crossentropy(tf.ones_like(y_fake), y_fake))
        gradients = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

        return d_loss, g_loss

    # Training
    for epoch in range(iterations):
        d_losses, g_losses = [], []
        for _ in range(ori_data.shape[0] // batch_size):
            x_batch, t_batch = batch_generator(ori_data, ori_time, batch_size)
            z_batch = random_generator(batch_size, z_dim, t_batch, max_seq_len)
            d_loss, g_loss = train_step(x_batch, z_batch, t_batch)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, D Loss: {np.mean(d_losses)}, G Loss: {np.mean(g_losses)}')

    # Synthetic data generation
    z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    generated_data = generator.predict(z_mb)

    # Renormalization
    generated_data = generated_data * (max_val - min_val) + min_val

    return generated_data

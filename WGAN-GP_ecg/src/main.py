import layers
import modules
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
batch_size = 64
latent_dim = 100
epochs = 100
noise_param = 0.1

# Loading the processed data
data_files = [f'data/Leipzing_heartbeat_DUS_FECG_{i}.npz' for i in range(1, 9)]
DUS_lists = []
ECG_lists = []

for file in data_files:
    loaded_data = np.load(file, allow_pickle=True)
    DUS_lists.append(np.array(loaded_data['DUS_list_all'], dtype=np.float32))
    ECG_lists.append(np.array(loaded_data['ECG_list_all'], dtype=np.float32))

# Format the train set
DUS_train = np.concatenate((DUS_lists[3], DUS_lists[2], DUS_lists[1]))
ECG_train = np.concatenate((ECG_lists[3], ECG_lists[2], ECG_lists[1]))
# Format the test set
DUS_test = DUS_lists[4]
ECG_test = ECG_lists[4]

dataset = tf.data.Dataset.from_tensor_slices((DUS_train, ECG_train))
dataset = dataset.shuffle(buffer_size=4096).batch(batch_size)

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, gp_weight):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = tf.random.set_seed(2024)
        self.gp_weight = gp_weight,
        self.d_steps = 3

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")


    def gradient_penalty(self, batch_size, real_doppler, generated_doppler):

        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        diff = generated_doppler - real_doppler
        interpolated = real_doppler + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    @property
    def metrics(self):
        return [self.d_loss_metric,self.g_loss_metric]
    

    def train_step(self, data):
        doppler, ecg = data
        # Sample random points in the latent space
        batch_size = tf.shape(doppler)[0]
        for i in range(self.d_steps):
            latent_vectors = ecg

        # Train the discriminator
            with tf.GradientTape() as tape:
                # Decode them to fake doppler
                generated_doppler = self.generator(latent_vectors, training=True)
                # Get the logits for the fake doppler
                fake_logits = self.discriminator(generated_doppler, training=True)
                # Get the logits for the real doppler
                real_logits = self.discriminator(keras.layers.Reshape((800, 1))(doppler), training=True)


                # Calculate the discriminator loss using the fake and real doppler logits
                d_cost = self.d_loss_fn(real_doppler=real_logits, fake_doppler=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, keras.layers.Reshape((800, 1))(doppler), generated_doppler)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        latent_vectors = ecg

        # Train the generator 
        with tf.GradientTape() as tape:
            # Generate fake doppler using the generator
            generated_doppler = self.generator(latent_vectors, training=True)
            # Get the discriminator logits for fake doppler
            gen_doppler_logits = self.discriminator(generated_doppler, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_doppler_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)


        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
    

# Initialize lists to store metrics
d_losses, g_losses, d_accuracies, g_accuracies = [], [], [], []

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save discriminator and generator loss and accuracy
        d_losses.append(logs.get('d_loss'))
        g_losses.append(logs.get('g_loss'))
        d_accuracies.append(logs.get('d_acc'))
        g_accuracies.append(logs.get('g_acc'))

        np.save('./WGAN-GP_beat/logs/discriminator_losses.npy', d_losses)
        np.save('./WGAN-GP_beat/logs/generator_losses.npy', g_losses)
        np.save('./WGAN-GP_beat/logs/discriminator_accuracies.npy', d_accuracies)
        np.save('./WGAN-GP_beat/logs/generator_accuracies.npy', g_accuracies)

    def on_train_end(self, logs=None):
        # Save the final generator and discriminator models
        self.model.generator.save('./WGAN-GP_beat/models/final_generator_4.h5')
        self.model.discriminator.save('./WGAN-GP_beat/models/final_discriminator_4.h5')


def discriminator_loss(real_doppler, fake_doppler):
    real_loss = tf.reduce_mean(real_doppler)
    fake_loss = tf.reduce_mean(fake_doppler)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_doppler):
    return -tf.reduce_mean(fake_doppler)


wgan = GAN(
    discriminator=modules.discriminator((800,1)), generator=modules.generator(latent_dim), 
    latent_dim=latent_dim, gp_weight=10
    )

wgan.compile(
    d_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
    g_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
    )


wgan.fit(
    dataset, epochs=epochs, callbacks=[
                    CustomCallback()],
    )


# Plotting the metrics
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.plot(epochs_range, d_losses, label='Discriminator Loss')
plt.plot(epochs_range, g_losses, label='Generator Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.savefig('./WGAN-GP_beat/plots/loss-acc.jpg')
plt.show()


# Show some random generated
# Randomly select 4 ECG samples and their corresponding real Doppler from the dataset
random_indices = np.random.choice(len(ECG_test), 4, replace=False)
selected_ecgs = ECG_test[random_indices]
selected_real_dopplers = DUS_test[random_indices]
print(np.shape(selected_ecgs))
# Generate Doppler using the model
generated_dopplers = wgan.generator.predict(selected_ecgs).reshape(4,800)
# Plot the selected ECGs, real Dopplers, and their corresponding generated Doppler signals
modules.plot_ecg_doppler_pairs(selected_ecgs, selected_real_dopplers, generated_dopplers)
modules.plot_scalogram(selected_real_dopplers, generated_dopplers)
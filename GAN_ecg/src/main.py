import layers
import modules
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt


batch_size = 64
epochs = 40
noise_param = 0.05
latent_dim = 400

# Function to generate a batch of random numbers
loaded_data = np.load('data/Leipzing_FHR_heartbeat.npz',allow_pickle=True)
DUS_list=loaded_data['DUS_list']
ECG_list=loaded_data['ECG_list']

DUS_array = np.array(DUS_list, dtype=np.float32)
ECG_array = np.array(ECG_list, dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((DUS_array, ECG_array))
dataset = dataset.shuffle(buffer_size=4096).batch(batch_size)

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = tf.random.set_seed(2024)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.d_acc_metric = keras.metrics.BinaryAccuracy(name="d_acc")
        self.g_acc_metric = keras.metrics.BinaryAccuracy(name="g_acc")

    @property
    def metrics(self):
        return [self.d_loss_metric,self.g_loss_metric,
            self.d_acc_metric,self.g_acc_metric]
    
    
    def train_step(self, data):
        doppler, ecg = data
        # Sample random points in the latent space
        batch_size = tf.shape(doppler)[0]
        latent_vectors = ecg

        # Decode them to fake doppler
        generated_doppler = self.generator(latent_vectors)

        # Combine them with real doppler
        combined_doppler = tf.concat([generated_doppler, keras.layers.Reshape((800, 1))(doppler)], axis=0)

        # Assemble labels discriminating real from fake doppler
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels
        noisy_labels = labels + noise_param * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            d_predictions = self.discriminator(combined_doppler)
            d_loss = self.loss_fn(noisy_labels, d_predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )


        latent_vectors = ecg

        # Assemble labels that say "all real doppler"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator 
        with tf.GradientTape() as tape:
            g_predictions = self.discriminator(self.generator(latent_vectors))
            g_loss = self.loss_fn(misleading_labels, g_predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.d_acc_metric.update_state(labels, d_predictions)
        self.g_acc_metric.update_state(misleading_labels, g_predictions)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "d_acc": self.d_acc_metric.result(),
            "g_acc": self.g_acc_metric.result(),
        }
    

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_sample=4, latent_dim=128):
        self.num_sample = num_sample
        self.latent_dim = latent_dim
        self.seed_generator = tf.random.set_seed(2024)
        self.d_losses = []
        self.g_losses = []
        self.d_accuracies = []
        self.g_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        latent_vectors = ecg
        generated_sample = self.model.generator(latent_vectors)
        generated_sample = generated_sample.numpy()
        #for i in range(self.num_sample):
        #    np.save("generated_sample_%03d_%d.npy" % (epoch, i), generated_sample)

        # Log and save metrics
        self.d_losses.append(logs['d_loss'])
        self.g_losses.append(logs['g_loss'])
        self.d_accuracies.append(logs['d_acc'])
        self.g_accuracies.append(logs['g_acc'])

        # Save metrics to file at each epoch
        np.save('./GAN_ecg/logs/discriminator_losses.npy', np.array(self.d_losses))
        np.save('./GAN_ecg/logs/generator_losses.npy', np.array(self.g_losses))
        np.save('./GAN_ecg/logs/discriminator_accuracies.npy', np.array(self.d_accuracies))
        np.save('./GAN_ecg/logs/generator_accuracies.npy', np.array(self.g_accuracies))

# Initialize lists to store metrics
d_losses, g_losses, d_accuracies, g_accuracies = [], [], [], []

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save discriminator and generator loss and accuracy
        d_losses.append(logs.get('d_loss'))
        g_losses.append(logs.get('g_loss'))
        d_accuracies.append(logs.get('d_acc'))
        g_accuracies.append(logs.get('g_acc'))

        np.save('./GAN_ecg/logs/discriminator_losses.npy', d_losses)
        np.save('./GAN_ecg/logs/generator_losses.npy', g_losses)
        np.save('./GAN_ecg/logs/discriminator_accuracies.npy', d_accuracies)
        np.save('./GAN_ecg/logs/generator_accuracies.npy', g_accuracies)

    def on_train_end(self, logs=None):
        # Save the final generator and discriminator models
        self.model.generator.save('./GAN_ecg/models/final_generator.h5')
        self.model.discriminator.save('./GAN_ecg/models/final_discriminator.h5')


dcgan = GAN(
    discriminator=modules.discriminator((800,1)), generator=modules.generator(latent_dim), 
    latent_dim=latent_dim
    )

dcgan.compile(
    d_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    g_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss_fn=keras.losses.BinaryCrossentropy(),
    )

dcgan.fit(
    dataset, epochs=epochs, callbacks=[CustomCallback()]
)


# Plotting the metrics
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, d_losses, label='Discriminator Loss')
plt.plot(epochs_range, g_losses, label='Generator Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, d_accuracies, label='Discriminator Accuracy')
plt.plot(epochs_range, g_accuracies, label='Generator Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.savefig('GAN_ecg/plots/loss-acc.png')
plt.show()


# Randomly select 4 ECG samples and their corresponding real Doppler images from your dataset
random_indices = np.random.choice(len(ECG_array), 4, replace=False)
selected_ecgs = ECG_array[random_indices]
selected_real_dopplers = DUS_array[random_indices]

# Generate Doppler images using the GAN generator
generated_dopplers = dcgan.generator(selected_ecgs).numpy().reshape(4,800)
# Plot the selected ECGs, real Dopplers, and their corresponding generated Doppler images
modules.plot_ecg_doppler_pairs(selected_ecgs, selected_real_dopplers, generated_dopplers)
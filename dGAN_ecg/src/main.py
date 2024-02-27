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
loaded_data_train = np.load('data/Leipzing_FHR_heartbeat_train.npz',allow_pickle=True)
loaded_data_test = np.load('data/Leipzing_FHR_heartbeat_5.npz',allow_pickle=True)

DUS_list_train=loaded_data_train['DUS_list']
ECG_list_train=loaded_data_train['ECG_list']

DUS_array_train = np.array(DUS_list_train, dtype=np.float32)
ECG_array_train = np.array(ECG_list_train, dtype=np.float32)

DUS_list_test=loaded_data_test['DUS_list']
ECG_list_test=loaded_data_test['ECG_list']

DUS_array_test = np.array(DUS_list_test, dtype=np.float32)
ECG_array_test = np.array(ECG_list_test, dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((DUS_array_train, ECG_array_train))
dataset = dataset.shuffle(buffer_size=4096).batch(batch_size)

class dGAN(keras.Model):
    def __init__(self, discriminator_1d, discriminator_2d, generator, latent_dim):
        super().__init__()
        self.discriminator_1d = discriminator_1d
        self.discriminator_2d = discriminator_2d
        self.generator = generator
        self.latent_dim = latent_dim

        self.seed_generator = tf.random.set_seed(2024)

    def compile(self, d1_optimizer, d2_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d1_optimizer = d1_optimizer
        self.d2_optimizer = d2_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        # Metrics for both discriminators and the generator
        self.d1_loss_metric = keras.metrics.Mean(name="d1_loss")
        self.d2_loss_metric = keras.metrics.Mean(name="d2_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.d1_acc_metric = keras.metrics.BinaryAccuracy(name="d1_acc")
        self.d2_acc_metric = keras.metrics.BinaryAccuracy(name="d2_acc")
        self.g_acc_metric = keras.metrics.BinaryAccuracy(name="g_acc")

    @property
    def metrics(self):
        return [self.d1_loss_metric, self.d2_loss_metric, self.g_loss_metric,
                self.d1_acc_metric, self.d2_acc_metric, self.g_acc_metric]
    
    
    def train_step(self, data):
        doppler, ecg = data
        # Sample random points in the latent space
        batch_size = tf.shape(doppler)[0]
        latent_vectors = ecg

        # Decode them to fake doppler
        generated_doppler = self.generator(latent_vectors)

        scalograme1 = modules.create_batch_scalograms(generated_doppler)
        scalograme2 = modules.create_batch_scalograms(doppler)

        # Combine with real doppler for both 1D and reshaped 2D inputs
        combined_doppler_1d = tf.concat([generated_doppler, keras.layers.Reshape((800, 1))(doppler)], axis=0)
        combined_doppler_2d = tf.concat([scalograme1, scalograme2], axis=0)

        # Labels for real and fake data
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        noisy_labels = labels + noise_param * tf.random.uniform(tf.shape(labels))


        # Train the first (1D) discriminator
        with tf.GradientTape() as tape:
            d1_predictions = self.discriminator_1d(combined_doppler_1d)
            d1_loss = self.loss_fn(noisy_labels, d1_predictions)
        grads = tape.gradient(d1_loss, self.discriminator_1d.trainable_weights)
        self.d1_optimizer.apply_gradients(zip(grads, self.discriminator_1d.trainable_weights))

        # Train the second (2D) discriminator
        with tf.GradientTape() as tape:
            d2_predictions = self.discriminator_2d(combined_doppler_2d)
            d2_loss = self.loss_fn(noisy_labels, d2_predictions)
        grads = tape.gradient(d2_loss, self.discriminator_2d.trainable_weights)
        self.d2_optimizer.apply_gradients(zip(grads, self.discriminator_2d.trainable_weights))

        latent_vectors = ecg

        # Assemble labels that say "all real doppler"
        misleading_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            generated_doppler = self.generator(latent_vectors)
            g1_predictions = self.discriminator_1d(generated_doppler)
            g2_predictions = self.discriminator_2d(tf.reshape(generated_doppler, (-1, 20, 40, 1))) ****
            g_loss = self.loss_fn(misleading_labels, g1_predictions) + self.loss_fn(misleading_labels, g2_predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))


        # Update metrics
        self.d1_loss_metric.update_state(d1_loss)
        self.d2_loss_metric.update_state(d2_loss)
        self.g_loss_metric.update_state(g_loss)
        self.d1_acc_metric.update_state(labels, d1_predictions)
        self.d2_acc_metric.update_state(labels, d2_predictions)
        self.g_acc_metric.update_state(misleading_labels, tf.concat([g1_predictions, g2_predictions], axis=0))

        return {
            "d1_loss": self.d1_loss_metric.result(),
            "d2_loss": self.d2_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "d1_acc": self.d1_acc_metric.result(),
            "d2_acc": self.d2_acc_metric.result(),
            "g_acc": self.g_acc_metric.result(),
        }
    

class dGANMonitor(keras.callbacks.Callback):
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
        np.save('./dGAN_ecg/logs/discriminator_losses.npy', np.array(self.d_losses))
        np.save('./dGAN_ecg/logs/generator_losses.npy', np.array(self.g_losses))
        np.save('./dGAN_ecg/logs/discriminator_accuracies.npy', np.array(self.d_accuracies))
        np.save('./dGAN_ecg/logs/generator_accuracies.npy', np.array(self.g_accuracies))

# Initialize lists to store metrics
d_losses, g_losses, d_accuracies, g_accuracies = [], [], [], []

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save discriminator and generator loss and accuracy
        d_losses.append(logs.get('d_loss'))
        g_losses.append(logs.get('g_loss'))
        d_accuracies.append(logs.get('d_acc'))
        g_accuracies.append(logs.get('g_acc'))

        np.save('./dGAN_ecg/logs/discriminator_losses.npy', d_losses)
        np.save('./dGAN_ecg/logs/generator_losses.npy', g_losses)
        np.save('./dGAN_ecg/logs/discriminator_accuracies.npy', d_accuracies)
        np.save('./dGAN_ecg/logs/generator_accuracies.npy', g_accuracies)

    def on_train_end(self, logs=None):
        # Save the final generator and discriminator models
        self.model.generator.save('./dGAN_ecg/models/final_generator.h5')
        self.model.discriminator.save('./dGAN_ecg/models/final_discriminator.h5')


dgan = dGAN(
    discriminator=modules.discriminator((800,1)), generator=modules.generator(latent_dim), 
    latent_dim=latent_dim
    )

dgan.compile(
    d1_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    d2_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    g_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss_fn=keras.losses.BinaryCrossentropy(),
    )

dgan.fit(
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
plt.savefig('dGAN_ecg/plots/loss-acc.png')
plt.show()


# Randomly select 4 ECG samples and their corresponding real Doppler images from your dataset
random_indices = np.random.choice(len(ECG_array_test), 4, replace=False)
selected_ecgs = ECG_array_test[random_indices]
selected_real_dopplers = DUS_array_test[random_indices]

# Generate Doppler images using the GAN generator
generated_dopplers = dgan.generator(selected_ecgs).numpy().reshape(4,800)
# Plot the selected ECGs, real Dopplers, and their corresponding generated Doppler images
modules.plot_ecg_doppler_pairs(selected_ecgs, selected_real_dopplers, generated_dopplers)
import layers
import modules
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
batch_size = 32
latent_dim = 100
epochs = 50
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
DUS_train = np.concatenate((DUS_lists[3], DUS_lists[2], DUS_lists[4]))
ECG_train = np.concatenate((ECG_lists[3], ECG_lists[2], ECG_lists[4]))
# Format the test set
DUS_test = DUS_lists[1]
ECG_test = ECG_lists[1]

dataset = tf.data.Dataset.from_tensor_slices((DUS_train, ECG_train))
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
    

# Initialize lists to store metrics
d_losses, g_losses, d_accuracies, g_accuracies = [], [], [], []

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save discriminator and generator loss and accuracy
        d_losses.append(logs.get('d_loss'))
        g_losses.append(logs.get('g_loss'))
        d_accuracies.append(logs.get('d_acc'))
        g_accuracies.append(logs.get('g_acc'))

        np.save('./LSTM_GAN_beat/logs/discriminator_losses.npy', d_losses)
        np.save('./LSTM_GAN_beat/logs/generator_losses.npy', g_losses)
        np.save('./LSTM_GAN_beat/logs/discriminator_accuracies.npy', d_accuracies)
        np.save('./LSTM_GAN_beat/logs/generator_accuracies.npy', g_accuracies)

    def on_train_end(self, logs=None):
        # Save the final generator and discriminator models
        self.model.generator.save('./LSTM_GAN_beat/models/final_generator_1.h5')
        self.model.discriminator.save('./LSTM_GAN_beat/models/final_discriminator_1.h5')

lstmgan = GAN(
    discriminator=modules.discriminator((800,1)), generator=modules.generator(latent_dim), 
    latent_dim=latent_dim
    )

lstmgan.compile(
    d_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    g_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss_fn=keras.losses.BinaryCrossentropy(),
    )


lstmgan.fit(
    dataset, epochs=epochs, callbacks=[
                    CustomCallback()],
    )


#Plotting the metrics
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
plt.savefig('LSTM_GAN_beat/plots/loss-acc.jpg')
plt.show()

# Show some random generated
# Randomly select 4 ECG samples and their corresponding real Doppler from the dataset
random_indices = np.random.choice(len(ECG_test), 4, replace=False)
selected_ecgs = ECG_test[random_indices]
selected_real_dopplers = DUS_test[random_indices]
print(np.shape(selected_ecgs))
# Generate Doppler using the model
generated_dopplers = lstmgan.generator.predict(selected_ecgs).reshape(4,800)
# Plot the selected ECGs, real Dopplers, and their corresponding generated Doppler signals
modules.plot_ecg_doppler_pairs(selected_ecgs, selected_real_dopplers, generated_dopplers)
modules.plot_scalogram(selected_real_dopplers, generated_dopplers)
import layers
import modules
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(True)

batch_size = 64
latent_dim = 100
epochs = 10

# Function to generate a batch of random numbers
loaded_data_train = np.load('data/Leipzing_FHR_heartbeat_noise_train.npz',allow_pickle=True)
loaded_data_test = np.load('data/Leipzing_FHR_heartbeat_noise_5.npz',allow_pickle=True)

DUS_list_train=loaded_data_train['DUS_list']

DUS_array_train = np.array(DUS_list_train, dtype=np.float32)

DUS_scalogram_train = modules.create_batch_scalograms(keras.layers.Reshape((800, 1))(DUS_array_train))

DUS_list_test=loaded_data_test['DUS_list']
DUS_array_test = np.array(DUS_list_test, dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((DUS_array_train, DUS_scalogram_train))
dataset = dataset.shuffle(buffer_size=4096).batch(batch_size)



class dWGAN(keras.Model):
    def __init__(self, discriminator_1, discriminator_2, generator, latent_dim, gp_weight_1, gp_weight_2):
        super().__init__()
        self.discriminator_1 = discriminator_1
        self.discriminator_2 = discriminator_2
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = tf.random.set_seed(2024)
        self.gp_weight_1 = gp_weight_1
        self.gp_weight_2 = gp_weight_2
        self.d_steps = 2

    def compile(self, d1_optimizer, d2_optimizer, g_optimizer, d1_loss_fn, d2_loss_fn, g_loss_fn):
        super().compile()
        self.d1_optimizer = d1_optimizer
        self.d2_optimizer = d2_optimizer
        self.g_optimizer = g_optimizer
        self.d1_loss_fn = d1_loss_fn
        self.d2_loss_fn = d2_loss_fn
        self.g_loss_fn = g_loss_fn
        self.d1_loss_metric = keras.metrics.Mean(name="d1_loss")
        self.d2_loss_metric = keras.metrics.Mean(name="d2_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def gradient_penalty_1(self, batch_size, real_doppler, generated_doppler):

        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        diff = generated_doppler - real_doppler
        interpolated = real_doppler + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator_1(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def gradient_penalty_2(self, batch_size, real_doppler, generated_doppler):

        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = generated_doppler - real_doppler
        interpolated = real_doppler + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator_2(interpolated, training=True)
        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    @property
    def metrics(self):
        return [self.d1_loss_metric,self.d2_loss_metric,self.g_loss_metric]
    
    
    def train_step(self, dataset):
        real_doppler, real_doppler2scalogram = dataset
        # Sample random points in the latent space
        batch_size = tf.shape(real_doppler)[0]
        

        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim), seed=self.seed_generator
            )

        # Train the discriminator
            with tf.GradientTape(persistent=True) as tape:
                # Decode them to fake doppler
                generated_doppler = self.generator(random_latent_vectors, training=True)

                generated_doppler2scalogram = modules.create_batch_scalograms(keras.layers.Reshape((800, 1))(generated_doppler))


                # Get the logits for the fake doppler
                fake_logits_1 = self.discriminator_1(generated_doppler, training=True) 
                fake_logits_2 = self.discriminator_2(keras.layers.Reshape((80, 160, 1))(generated_doppler2scalogram), training=True)
                # Get the logits for the real doppler
                real_logits_1 = self.discriminator_1(keras.layers.Reshape((800, 1))(real_doppler), training=True)
                real_logits_2 = self.discriminator_2(keras.layers.Reshape((80, 160, 1))(real_doppler2scalogram), training=True)
                # Calculate the discriminator loss using the fake and real doppler logits
                d1_cost = self.d1_loss_fn(real_doppler=real_logits_1, fake_doppler=fake_logits_1)
                d2_cost = self.d2_loss_fn(real_doppler=real_logits_2, fake_doppler=fake_logits_2)
                # Calculate the gradient penalty
                gp_1 = self.gradient_penalty_1(batch_size, keras.layers.Reshape((800, 1))(real_doppler), generated_doppler)
                gp_2 = self.gradient_penalty_2(batch_size, keras.layers.Reshape((80, 160, 1))(real_doppler2scalogram), keras.layers.Reshape((80, 160, 1))(generated_doppler2scalogram))
                # Add the gradient penalty to the original discriminator loss
                d1_loss = d1_cost + gp_1 * self.gp_weight_1
                d2_loss = d2_cost + gp_2 * self.gp_weight_2
                


            # Get the gradients w.r.t the discriminator loss
            d1_gradient = tape.gradient(d1_loss, self.discriminator_1.trainable_variables)
            d2_gradient = tape.gradient(d2_loss, self.discriminator_2.trainable_variables)
            
            # Update the weights of the discriminator using the discriminator optimizer
            self.d1_optimizer.apply_gradients(
                zip(d1_gradient, self.discriminator_1.trainable_variables)
            )
            self.d2_optimizer.apply_gradients(
                zip(d2_gradient, self.discriminator_2.trainable_variables)
            )
            

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        
        # Train the generator 
        with tf.GradientTape() as tape:

            # Generate fake doppler using the generator
            generated_doppler = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake doppler
            gen_doppler_logits = (self.discriminator_1(generated_doppler, training=True) + \
                                    self.discriminator_2(keras.layers.Reshape((80, 160, 1))(modules.create_batch_scalograms(generated_doppler)), training=True))/2
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_doppler_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        # Update metrics
        self.d1_loss_metric.update_state(d1_loss)
        self.d2_loss_metric.update_state(d2_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d1_loss": self.d1_loss_metric.result(),
            "d2_loss": self.d2_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
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
        random_latent_vectors = tf.random.normal(
            shape=(self.num_sample, self.latent_dim), seed=self.seed_generator
        )
        generated_sample = self.model.generator(random_latent_vectors)
        generated_sample = generated_sample.numpy()
        #for i in range(self.num_sample):
        #    np.save("generated_sample_%03d_%d.npy" % (epoch, i), generated_sample)

        # Log and save metrics
        self.d_losses.append(logs['d_loss'])
        self.g_losses.append(logs['g_loss'])
        self.d_accuracies.append(logs['d_acc'])
        self.g_accuracies.append(logs['g_acc'])

        # Save metrics to file at each epoch
        np.save('./dWGAN-GP_noise/logs/discriminator_losses.npy', np.array(self.d_losses))
        np.save('./dWGAN-GP_noise/logs/generator_losses.npy', np.array(self.g_losses))
        np.save('./dWGAN-GP_noise/logs/discriminator_accuracies.npy', np.array(self.d_accuracies))
        np.save('./dWGAN-GP_noise/logs/generator_accuracies.npy', np.array(self.g_accuracies))

# Initialize lists to store metrics
d1_losses, d2_losses, g_losses, d_accuracies, g_accuracies = [], [], [], [], []

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save discriminator and generator loss and accuracy
        d1_losses.append(logs.get('d1_loss'))
        d2_losses.append(logs.get('d2_loss'))
        g_losses.append(logs.get('g_loss'))
        d_accuracies.append(logs.get('d_acc'))
        g_accuracies.append(logs.get('g_acc'))

        np.save('./dWGAN-GP_noise/logs/discriminator_1_losses.npy', d1_losses)
        np.save('./dWGAN-GP_noise/logs/discriminator_2_losses.npy', d2_losses)
        np.save('./dWGAN-GP_noise/logs/generator_losses.npy', g_losses)
        np.save('./dWGAN-GP_noise/logs/discriminator_accuracies.npy', d_accuracies)
        np.save('./dWGAN-GP_noise/logs/generator_accuracies.npy', g_accuracies)

    def on_train_end(self, logs=None):
        # Save the final generator and discriminator models
        self.model.generator.save('./dWGAN-GP_noise/models/final_generator.h5')
        self.model.discriminator_1.save('./dWGAN-GP_noise/models/final_discriminator_1.h5')
        self.model.discriminator_2.save('./dWGAN-GP_noise/models/final_discriminator_2.h5')

def discriminator_loss(real_doppler, fake_doppler):
    real_loss = tf.reduce_mean(real_doppler)
    fake_loss = tf.reduce_mean(fake_doppler)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_doppler):
    return -tf.reduce_mean(fake_doppler)


dwgan = dWGAN(
    discriminator_1=modules.discriminator_1((800,1)), discriminator_2=modules.discriminator_2((80,160,1)),
    generator=modules.generator(latent_dim), 
    latent_dim=latent_dim, gp_weight_1=20, gp_weight_2=20
    )

dwgan.compile(
    d1_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
    d2_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
    g_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
    g_loss_fn=generator_loss,
    d1_loss_fn=discriminator_loss,
    d2_loss_fn=discriminator_loss,
    )


dwgan.fit(
    dataset, epochs=epochs, callbacks=[
                    CustomCallback()],
    )


# Plotting the metrics
epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, d1_losses, label='Discriminator_1 Loss')
plt.plot(epochs_range, d2_losses, label='Discriminator_2 Loss')
plt.plot(epochs_range, g_losses, label='Generator Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, d_accuracies, label='Discriminator Accuracy')
plt.plot(epochs_range, g_accuracies, label='Generator Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.savefig('dWGAN-GP_noise/plots/loss-acc.png')
plt.show()

# Randomly select 4 ECG samples and their corresponding real Doppler images from your dataset
random_indices = np.random.choice(len(DUS_array_test), 4, replace=False)
selected_real_dopplers = DUS_array_test[random_indices]
random_latent_vectors = tf.random.normal(shape=(4, 100), seed=2024)
generated_sample = dwgan.generator(random_latent_vectors).numpy().reshape(4,800)
modules.plot_ecg_doppler_pairs(random_latent_vectors, generated_sample, selected_real_dopplers)
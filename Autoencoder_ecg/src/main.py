import layers
import modules
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt


batch_size = 64
epochs = 100
latent_dim = 100

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


autoencoder = modules.Autoencoder(latent_dim)

history= autoencoder.fit(ECG_array_train, DUS_array_train, epochs=epochs, batch_size=batch_size, shuffle=True)

# Save the training loss
training_loss = history.history['loss']
np.save('./Autoencoder_ecg/logs/training_loss.npy', training_loss)

# Optionally, save the model
autoencoder.save('./Autoencoder_ecg/models/autoencoder_model.h5')

plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Autoencoder_ecg/plots/loss.png')
plt.show()


# Randomly select 4 ECG samples and their corresponding real Doppler images from your dataset
random_indices = np.random.choice(len(ECG_array_test), 4, replace=False)
selected_ecgs = ECG_array_test[random_indices]
selected_real_dopplers = DUS_array_test[random_indices]

# Generate Doppler images using the GAN generator
generated_dopplers = autoencoder.predict(selected_ecgs).reshape(4,800)
# Plot the selected ECGs, real Dopplers, and their corresponding generated Doppler images
modules.plot_ecg_doppler_pairs(selected_ecgs, selected_real_dopplers, generated_dopplers)
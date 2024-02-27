import numpy as np
import modules
from keras.models import load_model
import tensorflow as tf

np.random.seed(70)

wgan = load_model('./WGAN-GP_ecg/models/final_generator.h5')

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


# Randomly select 4 ECG samples and their corresponding real Doppler images from your dataset
random_indices = np.random.choice(len(ECG_array_test), 6, replace=False)
selected_ecgs = ECG_array_test[random_indices]
selected_real_dopplers = DUS_array_test[random_indices]

# Generate Doppler images using the GAN generator
generated_dopplers = wgan(selected_ecgs).numpy().reshape(6,800)
# Plot the selected ECGs, real Dopplers, and their corresponding generated Doppler images
modules.plot_ecg_doppler_pairs(selected_ecgs, selected_real_dopplers, generated_dopplers)

# plot the scalogram of real Doppler
modules.plot_scalogram(6, selected_real_dopplers, generated_dopplers, 160, 80)

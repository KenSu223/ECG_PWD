import layers
import modules
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
batch_size = 32
latent_dim = 100
epochs = 200

# Loading the processed data
data_files = [f'data/Leipzing_heartbeat_DUS_FECG_{i}.npz' for i in range(1, 9)]
DUS_lists = []
ECG_lists = []

for file in data_files:
    loaded_data = np.load(file, allow_pickle=True)
    DUS_lists.append(np.array(loaded_data['DUS_list_all'], dtype=np.float32))
    ECG_lists.append(np.array(loaded_data['ECG_list_all'], dtype=np.float32))

# Format the train set
DUS_train = np.concatenate((DUS_lists[3], DUS_lists[1], DUS_lists[4]))
ECG_train = np.concatenate((ECG_lists[3], ECG_lists[1], ECG_lists[4]))
# Format the test set
DUS_test = DUS_lists[2]
ECG_test = ECG_lists[2]

WaveNet = modules.WaveNet(input_shape = (latent_dim, 1))
WaveNet.compile(keras.optimizers.legacy.Adam(learning_rate=0.005, beta_1=0.5, beta_2=0.9), loss='mean_squared_error') 

history = WaveNet.fit(ECG_train, DUS_train, epochs=epochs)

WaveNet.save('./WaveNet_beat/models/model_2.h5')
np.save('./WaveNet_beat/logs/loss_1.npy', history.history['loss'])

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('./WaveNet_beat/plots/loss_3.png')
plt.show()

# Show some random generated
# Randomly select 4 ECG samples and their corresponding real Doppler from the dataset
random_indices = np.random.choice(len(ECG_test), 4, replace=False)
selected_ecgs = ECG_test[random_indices]
selected_real_dopplers = DUS_test[random_indices]
# Generate Doppler using the model
generated_dopplers = WaveNet.predict(selected_ecgs).reshape(4,800)
# Plot the selected ECGs, real Dopplers, and their corresponding generated Doppler signals
modules.plot_ecg_doppler_pairs(selected_ecgs, selected_real_dopplers, generated_dopplers)
modules.plot_scalogram(selected_real_dopplers, generated_dopplers)

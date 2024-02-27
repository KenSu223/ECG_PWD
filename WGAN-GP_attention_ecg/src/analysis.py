import numpy as np
import modules
from keras.models import load_model, Model
import tensorflow as tf
from scipy.signal import welch
import matplotlib.pyplot as plt

np.random.seed(90)

wgan = load_model('./WGAN-GP_attention_ecg/models/final_generator2.h5')
wgan2 = load_model('./WGAN-GP_ecg/models/final_generator.h5')

# Function to generate a batch of random numbers
loaded_data_train = np.load('data/Leipzing_FHR_heartbeat_new_train.npz',allow_pickle=True)
loaded_data_test = np.load('data/Leipzing_FHR_heartbeat_new_5.npz',allow_pickle=True)

DUS_list_train=loaded_data_train['DUS_list']
ECG_list_train=loaded_data_train['ECG_list']

DUS_array_train = np.array(DUS_list_train, dtype=np.float32)
ECG_array_train = np.array(ECG_list_train, dtype=np.float32)

DUS_list_test=loaded_data_test['DUS_list']
ECG_list_test=loaded_data_test['ECG_list']

DUS_array_test = np.array(DUS_list_test, dtype=np.float32)
ECG_array_test = np.array(ECG_list_test, dtype=np.float32)

generated_dopplers = wgan(ECG_array_test).numpy().reshape(1273,800)
generated_dopplers2 = wgan2(ECG_array_test).numpy().reshape(1273,800)

fs = 2000  
nperseg = 128

# Initialize an empty array to store PSDs
psds = []

for signal in generated_dopplers:
    # Compute frequency and power spectral density using Welch's method
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    
    # Append the PSD to the list
    psds.append(psd)

# Calculate the average PSD across all signals
avg_psd = np.mean(psds, axis=0)

# Initialize an empty array to store PSDs
psds2 = []

for signal in DUS_array_test:
    # Compute frequency and power spectral density using Welch's method
    freqs2, psd2 = welch(signal, fs, nperseg=nperseg)
    
    # Append the PSD to the list
    psds2.append(psd2)

# Calculate the average PSD across all signals
avg_psd2 = np.mean(psds2, axis=0)

# Initialize an empty array to store PSDs
psds3 = []

for signal in generated_dopplers2:
    # Compute frequency and power spectral density using Welch's method
    freqs3, psd3 = welch(signal, fs, nperseg=nperseg)
    
    # Append the PSD to the list
    psds3.append(psd3)

# Calculate the average PSD across all signals
avg_psd3 = np.mean(psds3, axis=0)

# Plotting the average PSD
plt.figure(figsize=(10, 6))
plt.plot(freqs2, avg_psd2, label = 'Real' )
plt.plot(freqs, avg_psd, label = 'Generated pro wgan')
plt.plot(freqs3, avg_psd3, label = 'Generated wgan')
plt.title('Average Power Spectral Density of Heartbeat Doppler Signals')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.legend()
plt.grid(True)
plt.show()
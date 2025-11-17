import numpy as np
import modules
from keras.models import load_model, Model
import tensorflow as tf
from scipy.signal import welch
import matplotlib.pyplot as plt

np.random.seed(90)

def normalize_one(array):
    ecg_min = np.min(array)
    ecg_max = np.max(array)
    
    normalized_ecg = 2 * (array - ecg_min) / (ecg_max - ecg_min) - 1
    
    return normalized_ecg

def normalize_mean_std(array):
    mean = np.mean(array)
    std = np.std(array)
    normalized_array = (array - mean) / std
    return normalized_array
    
# Loading the processed data
data_files = [f'data/Leipzing_heartbeat_DUS_FECG_{i}.npz' for i in range(1, 9)]
DUS_lists = []
ECG_lists = []

for file in data_files:
    loaded_data = np.load(file, allow_pickle=True)
    DUS_lists.append(np.array(loaded_data['DUS_list_all'], dtype=np.float32))
    ECG_lists.append(np.array(loaded_data['ECG_list_all'], dtype=np.float32))


WaveNet_1 = load_model('./WaveNet_beat/models/model_1.h5')
WaveNet_2 = load_model('./WaveNet_beat/models/model_2.h5')
WaveNet_3 = load_model('./WaveNet_beat/models/model_3.h5')
WaveNet_4 = load_model('./WaveNet_beat/models/model_4.h5')


generated_dopplers_1 = WaveNet_1(ECG_lists[1]).numpy().reshape(np.shape(DUS_lists[1])[0],800)
generated_dopplers_2 = WaveNet_2(ECG_lists[2]).numpy().reshape(np.shape(DUS_lists[2])[0],800)
generated_dopplers_3 = WaveNet_3(ECG_lists[3]).numpy().reshape(np.shape(DUS_lists[3])[0],800)
generated_dopplers_4 = WaveNet_4(ECG_lists[4]).numpy().reshape(np.shape(DUS_lists[4])[0],800)


data_real = np.vstack((DUS_lists[1],DUS_lists[2],DUS_lists[3],DUS_lists[4]))
data_generated = np.vstack((generated_dopplers_1, generated_dopplers_2, generated_dopplers_3, generated_dopplers_4))


fs = 2000  
nperseg = 256

plt.figure(figsize=(8, 8))
# Real data
for signal in data_real:
    freqs2, psd2 = welch(signal, fs, nperseg=nperseg)
    
    plt.plot(freqs2, psd2, color='skyblue', alpha=0.1, linewidth=0.10)  # Individual PSDs in pale color

avg_psd2 = np.mean([welch(signal, fs, nperseg=nperseg)[1] for signal in data_real], axis=0)
plt.plot(freqs2, avg_psd2, label='Real', color='blue', linewidth=2)  # Mean PSD in bold color

# Generated data
for signal in data_generated:
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    plt.plot(freqs, psd, color='lightcoral', alpha=0.1, linewidth=0.10)  # Individual PSDs in pale color

avg_psd = np.mean([welch(signal, fs, nperseg=nperseg)[1] for signal in data_generated], axis=0)
plt.plot(freqs, avg_psd, label='Generated', color='red', linewidth=2)  # Mean PSD in bold color

# Finalizing the plot
#plt.title('Power Spectral Density of Heartbeat Doppler Signals')
plt.xlabel('Frequency (Hz)', fontsize=15)
plt.ylabel('Power Spectral Density (mdB/Hz)', fontsize=15)
plt.ylim(0,0.00035)
plt.xlim(0,800)
plt.legend(fontsize=13)
plt.yticks([0, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035], [0,0.05,0.10,0.15,0.20,0.25,0.30,0.35])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.grid(True)
#plt.show()
plt.savefig('./WaveNet_beat/plots/psd_sq.jpg', dpi=450)
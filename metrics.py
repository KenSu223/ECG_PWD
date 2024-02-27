import numpy as np
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
import scipy.signal


wgan = load_model('./WGAN-GP_long/models/final_generator.h5')

# Function to generate a batch of random numbers
loaded_data_train = np.load('data/Leipzing_FHR_heartbeat_all_train.npz',allow_pickle=True)
loaded_data_test = np.load('data/Leipzing_FHR_heartbeat_all_5.npz',allow_pickle=True)

DUS_list_train=loaded_data_train['DUS_list']
ECG_list_train=loaded_data_train['ECG_list']

DUS_array_train = np.array(DUS_list_train, dtype=np.float32)
ECG_array_train = np.array(ECG_list_train, dtype=np.float32)

DUS_list_test=loaded_data_test['DUS_list']
ECG_list_test=loaded_data_test['ECG_list']


DUS_array_test = np.array(DUS_list_test, dtype=np.float32)
ECG_array_test = np.array(ECG_list_test, dtype=np.float32)

generated_dopplers = wgan(ECG_array_test).numpy().reshape(1273,800)


def calculate_rmse(predicted_signals, true_signals):
    """
    Calculate the Root Mean Square Error (RMSE) between the predicted signals 
    and the true signals.

    Parameters:
    - predicted_signals: numpy array of predicted Doppler signals by the GAN model.
    - true_signals: numpy array of ground truth Doppler signals.

    Returns:
    - rmse: The RMSE value as a float.
    """
    # Calculate the squared differences between each pair of predicted and true signals
    squared_differences = np.square(predicted_signals - true_signals)
    
    # Calculate the mean of the squared differences
    mean_squared_error = np.mean(squared_differences)
    
    # Take the square root of the mean squared error to get the RMSE
    rmse = np.sqrt(mean_squared_error)
    
    return rmse

def calculate_mae(predicted_signals, true_signals):
    """
    Calculate the Mean Absolute Error (MAE) between the predicted and true signals.

    Parameters:
    - predicted_signals: numpy array of predicted signals.
    - true_signals: numpy array of true signals.

    Returns:
    - mae: The MAE value as a float.
    """
    predicted_signals = np.array(predicted_signals)
    true_signals = np.array(true_signals)
    mae = np.mean(np.abs(predicted_signals - true_signals))
    return mae

def calculate_kld(predicted_signals, true_signals):
    """
    Calculate the Kullback-Leibler Divergence (KLD) between the predicted and true signals.
    Note: Both signals are converted to probability distributions.

    Parameters:
    - predicted_signals: numpy array of predicted signals.
    - true_signals: numpy array of true signals.

    Returns:
    - kld: The KLD value as a float.
    """
    # Convert signals to probability distributions
    pred_distribution = np.histogram(predicted_signals, bins=10, density=True)[0]
    true_distribution = np.histogram(true_signals, bins=10, density=True)[0]
    
    # Add a small value to avoid division by zero or log of zero
    epsilon = 1e-10
    pred_distribution += epsilon
    true_distribution += epsilon
    
    kld = stats.entropy(pred_distribution, true_distribution)
    return kld

def calculate_dtw(predicted_signals, true_signals):
    """
    Calculate the Dynamic Time Warping (DTW) distance between the predicted and true signals.

    Parameters:
    - predicted_signals: numpy array of predicted signals.
    - true_signals: numpy array of true signals.

    Returns:
    - dtw_distance: The DTW distance as a float.
    """
    distance, _ = fastdtw(predicted_signals, true_signals, dist=euclidean)
    return distance

def calculate_correlation(predicted_signals, true_signals):
    """
    Calculate the Pearson correlation coefficient between the predicted and true signals.

    Parameters:
    - predicted_signals: numpy array of predicted signals.
    - true_signals: numpy array of true signals.

    Returns:
    - correlation: The correlation coefficient as a float.
    """
    predicted_signals_flat = predicted_signals.flatten()
    true_signals_flat = true_signals.flatten()
    correlation, _ = stats.pearsonr(predicted_signals_flat, true_signals_flat)
    return correlation



def calculate_soft_dtw(predicted_signals, true_signals):
    """
    Calculate the Soft-Dynamic Time Warping (Soft-DTW) distance between the predicted and true signals.

    Parameters:
    - predicted_signals: numpy array of predicted signals.
    - true_signals: numpy array of true signals.

    Returns:
    - soft_dtw_distance: The Soft-DTW distance as a float.
    """
    D = SquaredEuclidean(predicted_signals, true_signals)
    sdtw = SoftDTW(D, gamma=1.0)
    soft_dtw_distance = sdtw.compute()
    return soft_dtw_distance

def calculate_euclidean_distance(predicted_signals, true_signals):
    """
    Calculate the Euclidean Distance (ED) between the predicted and true signals.

    Parameters:
    - predicted_signals: numpy array of predicted signals.
    - true_signals: numpy array of true signals.

    Returns:
    - ed: The Euclidean distance as a float.
    """
    ed = np.linalg.norm(predicted_signals - true_signals)
    return ed

def calculate_rdd(predicted_signals, true_signals):
    """
    Calculate the Relative Difference Distance (RDD) between the predicted and true signals.

    Parameters:
    - predicted_signals: numpy array of predicted signals.
    - true_signals: numpy array of true signals.

    Returns:
    - rdd: The RDD value as a float.
    """
    rdd = np.sum(np.abs(predicted_signals - true_signals) / (np.abs(true_signals) + 1e-10)) / len(true_signals)
    return rdd

def calculate_fidelity(predicted_signals, true_signals, similarity_measure):
    """
    Calculate the fidelity of the predicted signals, assuming higher similarity scores indicate higher fidelity.

    Parameters:
    - predicted_signals: List or array of predicted signals.
    - true_signals: List or array of true signals.
    - similarity_measure: Function to calculate the similarity between two signals.

    Returns:
    - average_fidelity: The average fidelity score across all predicted signals.
    """
    fidelity_scores = [similarity_measure(pred, true) for pred, true in zip(predicted_signals, true_signals)]
    average_fidelity = np.mean(fidelity_scores)
    return average_fidelity

def calculate_diversity(predicted_signals):
    """
    Calculate the diversity among the predicted signals based on pairwise Euclidean distances.

    Parameters:
    - predicted_signals: List or array of predicted signals.

    Returns:
    - average_diversity: The average diversity score across all pairs of predicted signals.
    """
    num_signals = len(predicted_signals)
    total_distance = 0
    count = 0

    for i in range(num_signals):
        for j in range(i+1, num_signals):
            total_distance += np.linalg.norm(predicted_signals[i] - predicted_signals[j])
            count += 1

    average_diversity = total_distance / count if count else 0
    return average_diversity

def calculate_prd(generated_signal, real_signal):
    """
    Calculate the Percent Root Mean Square Difference (PRD) between a generated signal and a real signal.

    Parameters:
    - generated_signal: numpy array representing the generated signal.
    - real_signal: numpy array representing the real (reference) signal.

    Returns:
    - prd: The PRD value as a percentage.
    """
    # Ensure inputs are numpy arrays
    generated_signal = np.array(generated_signal)
    real_signal = np.array(real_signal)
    
    # Calculate the Root Mean Square Difference
    rms_diff = np.sqrt(np.mean((generated_signal - real_signal) ** 2))
    
    # Calculate the Root Mean Square of the real signal
    rms_real = np.sqrt(np.mean(real_signal ** 2))
    
    # Calculate PRD
    prd = (rms_diff / rms_real) * 100
    
    return prd

def calculate_spectral_centroid(signal, fs=2000):
    """
    Calculate the spectral centroid of a signal.

    Parameters:
    - signal: The input signal (time-series).
    - fs: Sampling frequency of the input signal.

    Returns:
    - centroid: The spectral centroid of the signal.
    """
    magnitudes = np.abs(np.fft.rfft(signal))
    length = len(signal)
    freqs = np.fft.rfftfreq(length, d=1./fs)
    centroid = np.sum(magnitudes * freqs) / np.sum(magnitudes)
    return centroid

def calculate_spectral_entropy(generated_signals, real_signals, fs=2000):
    """
    Calculate the spectral entropy of a signal.

    Parameters:
    - signal: The input signal (time-series).
    - fs: Sampling frequency of the input signal.

    Returns:
    - entropy: The spectral entropy of the signal.
    """
    _, Pxx_gen = scipy.signal.welch(generated_signals, fs=fs)
    Pxx_gen_normalized = Pxx_gen / np.sum(Pxx_gen)  # Normalize the PSD to get a probability distribution
    entropy_gen = -np.sum(Pxx_gen_normalized * np.log2(Pxx_gen_normalized))

    _, Pxx_real = scipy.signal.welch(real_signals, fs=fs)
    Pxx_real_normalized = Pxx_real / np.sum(Pxx_real)  # Normalize the PSD to get a probability distribution
    entropy_real = -np.sum(Pxx_real_normalized * np.log2(Pxx_real_normalized))

    return np.abs(entropy_gen - entropy_real)

def calculate_spectral_flatness(signal):
    """
    Calculate the spectral flatness of a signal.

    Parameters:
    - signal: The input signal (time-series).

    Returns:
    - flatness: The spectral flatness of the signal.
    """
    magnitudes = np.abs(np.fft.rfft(signal))
    geometric_mean = scipy.stats.gmean(magnitudes + 1e-10)
    arithmetic_mean = np.mean(magnitudes)
    flatness = geometric_mean / arithmetic_mean
    return flatness

def calculate_psd_difference(generated_signals, real_signals, fs=2000):
    """
    Calculate the difference in Power Spectral Density (PSD) between generated and real signals.
    """
    gen_psd = np.mean([scipy.signal.welch(signal, fs)[1] for signal in generated_signals], axis=0)
    real_psd = np.mean([scipy.signal.welch(signal, fs)[1] for signal in real_signals], axis=0)
    diff = np.linalg.norm(gen_psd - real_psd)
    return diff

def calculate_spectral_centroid_difference(generated_signals, real_signals, fs=2000):
    """
    Calculate the difference in Spectral Centroid between generated and real signals.
    """
    def spectral_centroid(signal, fs):
        magnitudes = np.abs(np.fft.rfft(signal))
        length = len(signal)
        freqs = np.fft.rfftfreq(length, d=1./fs)
        centroid = np.sum(magnitudes * freqs) / np.sum(magnitudes)
        return centroid
    
    gen_centroids = np.mean([spectral_centroid(signal, fs) for signal in generated_signals])
    real_centroids = np.mean([spectral_centroid(signal, fs) for signal in real_signals])
    diff = np.abs(gen_centroids - real_centroids)
    return diff

def spectral_flatness(signal):
    """
    Calculate the spectral flatness of a signal.

    Parameters:
    - signal: The input signal (time-series).

    Returns:
    - flatness: The spectral flatness of the signal.
    """
    # Calculate the power spectrum
    freqs, power_spectrum = scipy.signal.welch(signal)
    
    # Avoid log of zero by adding a very small value to the power spectrum
    power_spectrum += 1e-10
    
    # Calculate the geometric mean
    geometric_mean = scipy.stats.mstats.gmean(power_spectrum)
    
    # Calculate the arithmetic mean
    arithmetic_mean = np.mean(power_spectrum)
    
    # Calculate the spectral flatness
    flatness = geometric_mean / arithmetic_mean
    return flatness

def calculate_flatness_difference(generated_signals, real_signals):
    """
    Calculate the average difference in spectral flatness between generated and real signals.

    Parameters:
    - generated_signals: List of generated signals.
    - real_signals: List of real signals.

    Returns:
    - average_flatness_difference: The average difference in spectral flatness.
    """
    # Calculate spectral flatness for each signal in both sets
    gen_flatnesses = np.array([spectral_flatness(signal) for signal in generated_signals])
    real_flatnesses = np.array([spectral_flatness(signal) for signal in real_signals])
    
    # Calculate the average spectral flatness for both sets
    avg_gen_flatness = np.mean(gen_flatnesses)
    avg_real_flatness = np.mean(real_flatnesses)
    
    # Calculate the difference in average spectral flatness
    average_flatness_difference = np.abs(avg_gen_flatness - avg_real_flatness)
    
    return average_flatness_difference

import numpy as np

def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.linalg.norm(a-b)

def calculate_discrete_frechet_distance(E, E_prime):
    """
    Calculate the discrete Fréchet Distance (FD) between two sequences E and E_prime
    using an iterative approach to avoid recursion depth issues.

    Parameters:
    - E: numpy array representing the first discrete signal (sequence of points).
    - E_prime: numpy array representing the second discrete signal (sequence of points).

    Returns:
    - fd: The discrete Fréchet Distance between E and E_prime.
    """
    N = len(E)
    M = len(E_prime)
    
    # Initialize a 2-D matrix to store distances, with base case D[0,0]
    D = np.full((N, M), np.inf)
    D[0, 0] = euclidean_distance(E[0], E_prime[0])
    
    # Fill the matrix iteratively
    for i in range(N):
        for j in range(M):
            if i > 0 or j > 0:  # Skip the already initialized D[0, 0]
                costs = []
                if i > 0:
                    costs.append(D[i-1, j])
                if j > 0:
                    costs.append(D[i, j-1])
                if i > 0 and j > 0:
                    costs.append(D[i-1, j-1])
                D[i, j] = min(max(costs), euclidean_distance(E[i], E_prime[j]))
    
    # The discrete Fréchet Distance is the value at D[N-1, M-1]
    fd = D[N-1, M-1]
    return fd


# Calculate metrics
rmse = calculate_rmse(generated_dopplers, DUS_array_test)
mae = calculate_mae(generated_dopplers, DUS_array_test)
kld = calculate_kld(generated_dopplers, DUS_array_test)
dtw_distance = calculate_dtw(generated_dopplers, DUS_array_test)
correlation = calculate_correlation(generated_dopplers, DUS_array_test)
sdtw = calculate_soft_dtw(generated_dopplers, DUS_array_test)
ed = calculate_euclidean_distance(generated_dopplers, DUS_array_test)
rdd = calculate_rdd(generated_dopplers, DUS_array_test)
fidelity = calculate_fidelity(generated_dopplers, DUS_array_test,calculate_mae)
diversity = calculate_diversity(generated_dopplers)
prd = calculate_prd(generated_dopplers, DUS_array_test)
entropy = calculate_spectral_entropy(generated_dopplers, DUS_array_test)
psd = calculate_psd_difference(generated_dopplers, DUS_array_test)
centroid = calculate_spectral_centroid_difference(generated_dopplers, DUS_array_test)
flatness = calculate_flatness_difference(generated_dopplers, DUS_array_test)
fd = calculate_discrete_frechet_distance(generated_dopplers, DUS_array_test)

# Print results
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"KLD: {kld}")
print(f"DTW Distance: {dtw_distance}")
print(f"Correlation: {correlation}")
print(f"Soft DTW: {sdtw}")
print(f"ED: {ed}")
print(f"RDD: {rdd}")
print(f"fidelity: {fidelity}")
print(f"diversity: {diversity}")
print(f"prd: {prd}")
print(f"spectral entropy: {entropy}")
print(f"psd difference: {psd}")
print(f"centroid difference: {centroid}")
print(f"spectral flatness: {flatness}")
print(f"fd: {fd}")
"""K-fold evaluation framework combining machine-learning signal metrics and clinically oriented Doppler metrics, with summary-table export."""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from scipy.signal import butter, sosfiltfilt
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy import signal
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

print("Python executable:", sys.executable)
print("sys.path:", sys.path)

from . import modules

from tensorflow.keras import mixed_precision

# Enable mixed precision (uses float16 for computation, saves memory)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('✓ Mixed precision enabled (float16)')


# ============================================================================
# SIGNAL SMOOTHING FUNCTIONS
# ============================================================================

def smooth_doppler_envelope(doppler_signal, fs=284, cutoff_hz=10.0, order=4):
    """Apply Butterworth lowpass filter to smooth Doppler envelope."""
    nyquist = fs / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    smoothed = sosfiltfilt(sos, doppler_signal)
    return smoothed


def smooth_all_segments(doppler_array, fs=284, cutoff_hz=10.0, order=4):
    """Apply smoothing to all segments in array."""
    n_segments, seq_length = doppler_array.shape
    smoothed = np.zeros_like(doppler_array)
    for i in range(n_segments):
        smoothed[i] = smooth_doppler_envelope(
            doppler_array[i], fs=fs, cutoff_hz=cutoff_hz, order=order
        )
    return smoothed


# ============================================================================
# ML/SIGNAL METRICS
# ============================================================================

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_gradient_mae(y_true, y_pred):
    """Calculate MAE on temporal derivatives"""
    grad_true = np.gradient(y_true)
    grad_pred = np.gradient(y_pred)
    return np.mean(np.abs(grad_true - grad_pred))


def calculate_pearson(y_true, y_pred):
    """Calculate Pearson Correlation Coefficient"""
    try:
        corr, _ = pearsonr(y_true, y_pred)
        return corr
    except:
        return np.nan


def calculate_dtw(y_true, y_pred):
    """Calculate Dynamic Time Warping distance"""
    y_true_2d = y_true.reshape(-1, 1)
    y_pred_2d = y_pred.reshape(-1, 1)
    distance, _ = fastdtw(y_true_2d, y_pred_2d, dist=euclidean)
    return distance


def calculate_cross_correlation_zero_lag(y_true, y_pred):
    """Calculate cross-correlation at zero lag"""
    y_true_norm = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-8)
    y_pred_norm = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-8)
    corr_zero_lag = np.sum(y_true_norm * y_pred_norm) / len(y_true)
    return corr_zero_lag


def calculate_kld(y_true, y_pred, bins=50):
    """Calculate Kullback-Leibler Divergence"""
    from scipy.stats import entropy
    
    combined_min = min(y_true.min(), y_pred.min())
    combined_max = max(y_true.max(), y_pred.max())
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)
    
    hist_true, _ = np.histogram(y_true, bins=bin_edges, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=bin_edges, density=True)
    
    hist_true = hist_true / (hist_true.sum() + 1e-10) + 1e-10
    hist_pred = hist_pred / (hist_pred.sum() + 1e-10) + 1e-10
    
    kld = entropy(hist_true, hist_pred)
    return kld


def calculate_psd_similarity(y_true, y_pred, fs=284, nperseg=256):
    """Calculate Power Spectral Density similarity"""
    freqs_true, psd_true = signal.welch(y_true, fs=fs, nperseg=min(nperseg, len(y_true)))
    freqs_pred, psd_pred = signal.welch(y_pred, fs=fs, nperseg=min(nperseg, len(y_pred)))
    
    psd_true_db = 10 * np.log10(psd_true + 1e-10)
    psd_pred_db = 10 * np.log10(psd_pred + 1e-10)
    
    psd_mse = np.mean((psd_true_db - psd_pred_db) ** 2)
    psd_corr = np.corrcoef(psd_true_db, psd_pred_db)[0, 1]
    
    return psd_mse, psd_corr


def compute_ml_metrics(ground_truth, predictions, fs=284):
    """
    Compute all ML/signal metrics for a batch of signals.
    
    Returns dictionary with mean and std for each metric.
    """
    n_samples = ground_truth.shape[0]
    
    metrics = {
        'MAE': [], 'MSE': [], 'RMSE': [], 'Gradient_MAE': [],
        'Pearson_Correlation': [], 'DTW': [], 'Cross_Correlation_Zero': [],
        'KLD': [], 'PSD_MSE': [], 'PSD_Correlation': []
    }
    
    for i in range(n_samples):
        y_true = ground_truth[i]
        y_pred = predictions[i]
        
        metrics['MAE'].append(calculate_mae(y_true, y_pred))
        metrics['MSE'].append(calculate_mse(y_true, y_pred))
        metrics['RMSE'].append(calculate_rmse(y_true, y_pred))
        metrics['Gradient_MAE'].append(calculate_gradient_mae(y_true, y_pred))
        metrics['Pearson_Correlation'].append(calculate_pearson(y_true, y_pred))
        metrics['DTW'].append(calculate_dtw(y_true, y_pred))
        metrics['Cross_Correlation_Zero'].append(calculate_cross_correlation_zero_lag(y_true, y_pred))
        metrics['KLD'].append(calculate_kld(y_true, y_pred))
        
        psd_mse, psd_corr = calculate_psd_similarity(y_true, y_pred, fs=fs)
        metrics['PSD_MSE'].append(psd_mse)
        metrics['PSD_Correlation'].append(psd_corr)
    
    # Calculate mean ± std for each metric
    results = {}
    for key, values in metrics.items():
        values_array = np.array(values)
        valid_values = values_array[~np.isnan(values_array)]
        if len(valid_values) > 0:
            results[key] = {
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'count': len(valid_values)
            }
        else:
            results[key] = {'mean': np.nan, 'std': np.nan, 'count': 0}
    
    return results


# ============================================================================
# CLINICAL/PHYSIOLOGICAL METRICS
# ============================================================================

class DopplerClinicalMetrics:
    """Calculate clinically relevant metrics from fetal Doppler envelopes."""
    
    def __init__(self, sampling_rate=284):
        self.fs = sampling_rate
        
    def detect_peaks_and_troughs(self, envelope, min_distance_samples=None):
        """Detect systolic peaks and diastolic troughs."""
        if min_distance_samples is None:
            min_distance_samples = int(0.3 * self.fs)
        
        peak_indices, _ = signal.find_peaks(
            envelope, 
            distance=min_distance_samples,
            prominence=0.1 * np.ptp(envelope)
        )
        
        trough_indices, _ = signal.find_peaks(
            -envelope, 
            distance=min_distance_samples,
            prominence=0.1 * np.ptp(envelope)
        )
        
        return peak_indices, trough_indices
    
    def calculate_pulsatility_index(self, envelope):
        """Calculate Pulsatility Index: PI = (PSV - EDV) / TAMX"""
        peak_indices, trough_indices = self.detect_peaks_and_troughs(envelope)
        
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return np.nan
        
        psv = np.mean(envelope[peak_indices])
        
        edv_values = []
        for i, peak_idx in enumerate(peak_indices[:-1]):
            next_peak = peak_indices[i + 1]
            troughs_between = trough_indices[(trough_indices > peak_idx) & 
                                            (trough_indices < next_peak)]
            if len(troughs_between) > 0:
                edv_values.append(envelope[troughs_between[-1]])
        
        if len(edv_values) == 0:
            return np.nan
            
        edv = np.mean(edv_values)
        tamx = np.mean(envelope)
        
        if tamx == 0:
            return np.nan
            
        pi = (psv - edv) / tamx
        return pi
    
    def calculate_resistance_index(self, envelope):
        """Calculate Resistance Index: RI = (PSV - EDV) / PSV"""
        peak_indices, trough_indices = self.detect_peaks_and_troughs(envelope)
        
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return np.nan
        
        psv = np.mean(envelope[peak_indices])
        
        edv_values = []
        for i, peak_idx in enumerate(peak_indices[:-1]):
            next_peak = peak_indices[i + 1]
            troughs_between = trough_indices[(trough_indices > peak_idx) & 
                                            (trough_indices < next_peak)]
            if len(troughs_between) > 0:
                edv_values.append(envelope[troughs_between[-1]])
        
        if len(edv_values) == 0 or psv == 0:
            return np.nan
            
        edv = np.mean(edv_values)
        ri = (psv - edv) / psv
        return ri
    
    def calculate_sd_ratio(self, envelope):
        """Calculate S/D Ratio: S/D = PSV / EDV"""
        peak_indices, trough_indices = self.detect_peaks_and_troughs(envelope)
        
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return np.nan
        
        psv = np.mean(envelope[peak_indices])
        
        edv_values = []
        for i, peak_idx in enumerate(peak_indices[:-1]):
            next_peak = peak_indices[i + 1]
            troughs_between = trough_indices[(trough_indices > peak_idx) & 
                                            (trough_indices < next_peak)]
            if len(troughs_between) > 0:
                edv_values.append(envelope[troughs_between[-1]])
        
        if len(edv_values) == 0 or np.mean(edv_values) == 0:
            return np.nan
            
        edv = np.mean(edv_values)
        sd_ratio = psv / edv
        return sd_ratio
    
    def calculate_heart_rate(self, envelope):
        """Calculate heart rate from peak-to-peak intervals (bpm)"""
        peak_indices, _ = self.detect_peaks_and_troughs(envelope)
        
        if len(peak_indices) < 2:
            return np.nan
        
        intervals = np.diff(peak_indices) / self.fs
        avg_interval = np.mean(intervals)
        
        if avg_interval == 0:
            return np.nan
        
        heart_rate = 60.0 / avg_interval
        return heart_rate
    
    def calculate_peak_systolic_velocity(self, envelope):
        """Calculate average peak systolic velocity"""
        peak_indices, _ = self.detect_peaks_and_troughs(envelope)
        
        if len(peak_indices) == 0:
            return np.nan
        
        return np.mean(envelope[peak_indices])
    
    def calculate_end_diastolic_velocity(self, envelope):
        """Calculate average end diastolic velocity"""
        peak_indices, trough_indices = self.detect_peaks_and_troughs(envelope)
        
        if len(peak_indices) < 2 or len(trough_indices) < 1:
            return np.nan
        
        edv_values = []
        for i, peak_idx in enumerate(peak_indices[:-1]):
            next_peak = peak_indices[i + 1]
            troughs_between = trough_indices[(trough_indices > peak_idx) & 
                                            (trough_indices < next_peak)]
            if len(troughs_between) > 0:
                edv_values.append(envelope[troughs_between[-1]])
        
        if len(edv_values) == 0:
            return np.nan
        
        return np.mean(edv_values)
    
    def calculate_all_metrics(self, envelope):
        """Calculate all clinical metrics for a single envelope"""
        metrics = {
            'PI': self.calculate_pulsatility_index(envelope),
            'RI': self.calculate_resistance_index(envelope),
            'SD_Ratio': self.calculate_sd_ratio(envelope),
            'Heart_Rate': self.calculate_heart_rate(envelope),
            'PSV': self.calculate_peak_systolic_velocity(envelope),
            'EDV': self.calculate_end_diastolic_velocity(envelope),
            'TAMX': np.mean(envelope)
        }
        return metrics


def compute_clinical_metrics(real_envelopes, generated_envelopes, sampling_rate=284):
    """
    Compute all clinical metrics for a batch of signals.
    
    Returns dictionary with mean and std for each metric, plus MAE/RMSE/Correlation
    comparing real vs generated values.
    """
    calculator = DopplerClinicalMetrics(sampling_rate=sampling_rate)
    n_samples = real_envelopes.shape[0]
    
    metric_names = ['PI', 'RI', 'SD_Ratio', 'Heart_Rate', 'PSV', 'EDV', 'TAMX']
    real_metrics = {key: [] for key in metric_names}
    gen_metrics = {key: [] for key in metric_names}
    
    for i in range(n_samples):
        real_m = calculator.calculate_all_metrics(real_envelopes[i])
        gen_m = calculator.calculate_all_metrics(generated_envelopes[i])
        
        for key in metric_names:
            real_metrics[key].append(real_m[key])
            gen_metrics[key].append(gen_m[key])
    
    # Calculate comparison statistics
    results = {}
    
    for metric_name in metric_names:
        real_vals = np.array(real_metrics[metric_name])
        gen_vals = np.array(gen_metrics[metric_name])
        
        # Remove NaN pairs
        valid_mask = ~(np.isnan(real_vals) | np.isnan(gen_vals))
        real_vals_valid = real_vals[valid_mask]
        gen_vals_valid = gen_vals[valid_mask]
        
        if len(real_vals_valid) == 0:
            results[metric_name] = {
                'real_mean': np.nan, 'real_std': np.nan,
                'gen_mean': np.nan, 'gen_std': np.nan,
                'mae': np.nan, 'rmse': np.nan, 'correlation': np.nan,
                'count': 0
            }
            continue
        
        mae = np.mean(np.abs(real_vals_valid - gen_vals_valid))
        rmse = np.sqrt(np.mean((real_vals_valid - gen_vals_valid) ** 2))
        
        if len(real_vals_valid) > 1 and np.std(real_vals_valid) > 0 and np.std(gen_vals_valid) > 0:
            corr, _ = pearsonr(real_vals_valid, gen_vals_valid)
        else:
            corr = np.nan
        
        results[metric_name] = {
            'real_mean': np.mean(real_vals_valid),
            'real_std': np.std(real_vals_valid),
            'gen_mean': np.mean(gen_vals_valid),
            'gen_std': np.std(gen_vals_valid),
            'mae': mae,
            'rmse': rmse,
            'correlation': corr,
            'count': len(real_vals_valid)
        }
    
    return results


# ============================================================================
# K-FOLD CROSS-VALIDATION EVALUATION
# ============================================================================

def evaluate_model_kfold(
    X, Y, patient_ids,
    model_builder_fn=None,
    n_folds=5,
    n_epochs=50,
    batch_size=32,
    lr=1e-3,
    save_dir="./evaluation_results",
    model_name="WaveNet",
    apply_smoothing=True,
    smoothing_params=None,
    sampling_rate=284,
    early_stopping_patience=10,
    use_lr_schedule=None,
    lr_reduce_patience=3,
    lr_reduce_factor=0.5,
    # Loss configuration (alternative to model_builder_fn)
    loss_type='mae',
    use_base=True,
    use_peak=False,
    use_ratio=False,
    use_derivative=False,
    use_softdtw=False,
    use_corr=False,
    use_mrstft=False,
    alpha_peak=0.5,
    alpha_ratio=0.5,
    alpha_derivative=0.3,
    alpha_softdtw=1.0,
    alpha_corr=1.0,
    alpha_mrstft=1.0,
    base_loss='mae',
    dtw_gamma=0.1,
    # Architecture selection
    architecture='combined_attention'
):
    """
    Perform k-fold cross-validation evaluation with comprehensive metrics.
    
    Parameters:
    -----------
    X : np.ndarray
        Input ECG data (n_samples, seq_length, n_channels)
    Y : np.ndarray
        Target Doppler data (n_samples, seq_length) or (n_samples, seq_length, 1)
    patient_ids : np.ndarray
        Patient IDs for each sample
    model_builder_fn : callable or None
        Function that returns a compiled Keras model. If None, will use
        architecture and loss_type parameters to build model automatically.
    n_folds : int
        Number of folds for cross-validation
    n_epochs : int
        Maximum number of training epochs per fold
    batch_size : int
        Training batch size
    lr : float
        Learning rate
    save_dir : str
        Directory to save results
    model_name : str
        Name for the model (used in saving)
    apply_smoothing : bool
        Whether to apply signal smoothing to generated signals
    smoothing_params : dict
        Parameters for smoothing (fs, cutoff_hz, order)
    sampling_rate : int
        Sampling rate for physiological metrics
    early_stopping_patience : int
        Patience for early stopping
    use_lr_schedule : str or None
        Learning rate schedule type ('fixed', 'dynamic', or None)
    loss_type : str
        Type of loss function ('mae', 'mse', 'composite', 'shape_preserving', 'flexible')
    use_base, use_peak, use_ratio, use_derivative, use_softdtw, use_corr, use_mrstft : bool
        Flags to enable/disable specific loss components (for flexible loss)
    alpha_* : float
        Weights for each loss component
    base_loss : str
        Base loss type ('mae' or 'mse')
    dtw_gamma : float
        Temperature parameter for Soft-DTW
    architecture : str
        Architecture type: 'two_channel', 'minimal_attention', 'cross_attention', 'combined_attention'
        
    Returns:
    --------
    ml_metrics_summary : pd.DataFrame
        Summary of ML/signal metrics across all folds
    clinical_metrics_summary : pd.DataFrame
        Summary of clinical metrics across all folds
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Default smoothing parameters
    if smoothing_params is None:
        smoothing_params = {'fs': 284, 'cutoff_hz': 10.0, 'order': 4}
    
    # Squeeze Y if needed
    if Y.ndim == 3 and Y.shape[-1] == 1:
        Y = Y.squeeze(axis=-1)
    
    # Get unique patients
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    print("="*80)
    print(f"K-FOLD CROSS-VALIDATION EVALUATION: {model_name}")
    print("="*80)
    print(f"Total samples: {len(X)}")
    print(f"Total patients: {n_patients}")
    print(f"Number of folds: {n_folds}")
    print(f"Epochs per fold: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"LR schedule: {use_lr_schedule if use_lr_schedule else 'None'}")
    
    # Display loss configuration if not using custom model builder
    if model_builder_fn is None:
        print(f"\nArchitecture: {architecture}")
        print(f"Loss type: {loss_type}")
        if loss_type == 'flexible':
            components = []
            if use_base: components.append(f"{base_loss.upper()}")
            if use_peak: components.append(f"{alpha_peak}*peak")
            if use_ratio: components.append(f"{alpha_ratio}*ratio")
            if use_derivative: components.append(f"{alpha_derivative}*derivative")
            if use_softdtw: components.append(f"{alpha_softdtw}*softdtw")
            if use_corr: components.append(f"{alpha_corr}*corr")
            if use_mrstft: components.append(f"{alpha_mrstft}*mrstft")
            print(f"Loss components: {' + '.join(components)}")
        elif loss_type == 'composite':
            print(f"Loss components: {base_loss.upper()} + {alpha_peak}*peak + {alpha_ratio}*ratio")
        elif loss_type == 'shape_preserving':
            print(f"Loss components: {base_loss.upper()} + {alpha_peak}*peak + {alpha_ratio}*ratio + {alpha_derivative}*derivative")
    else:
        print("\nUsing custom model builder function")
    
    print(f"\nApply smoothing: {apply_smoothing}")
    if apply_smoothing:
        print(f"Smoothing params: {smoothing_params}")
    print("="*80 + "\n")
    
    # Initialize storage for results across folds
    all_ml_metrics = []
    all_clinical_metrics = []
    
    # K-Fold split on patients
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_patient_idx, val_patient_idx) in enumerate(kf.split(unique_patients)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*80}")
        
        # Get train and validation patients
        train_patients = unique_patients[train_patient_idx]
        val_patients = unique_patients[val_patient_idx]
        
        # Get sample indices for train and validation
        train_idx = np.where(np.isin(patient_ids, train_patients))[0]
        val_idx = np.where(np.isin(patient_ids, val_patients))[0]
        
        print(f"Train patients: {len(train_patients)}, samples: {len(train_idx)}")
        print(f"Val patients: {len(val_patients)}, samples: {len(val_idx)}")
        
        # Verify no patient leakage
        assert set(train_patients).isdisjoint(set(val_patients)), "Patient leakage detected!"
        
        # Build model
        print("\nBuilding model...")
        
        if model_builder_fn is not None:
            # Use provided model builder
            model = model_builder_fn(input_shape=X.shape[1:])
            print("✓ Using custom model builder function")
        else:
            # Build model with specified architecture
            input_shape = X.shape[1:]
            
            if architecture == 'two_channel':
                model = modules.WaveNet_two_channel(input_shape=input_shape)
            elif architecture == 'minimal_attention':
                model = modules.WaveNet_minimal_cross_attention(input_shape=input_shape)
            elif architecture == 'cross_attention':
                model = modules.WaveNet_two_channel_cross_attention(input_shape=input_shape)
            elif architecture == 'combined_attention':
                model = modules.WaveNet_two_channel_combined_attention(input_shape=input_shape)
            elif architecture == 'single_channel':
                model = modules.WaveNet_v2(input_shape=input_shape)
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            print(f"✓ Built {architecture} architecture")
            
            # Configure loss function
            print("\nConfiguring loss function...")
            
            if loss_type == 'flexible':
                loss_fn = modules.create_flexible_loss(
                    use_base=use_base,
                    use_peak=use_peak,
                    use_ratio=use_ratio,
                    use_derivative=use_derivative,
                    use_softdtw=use_softdtw,
                    use_corr=use_corr,
                    use_mrstft=use_mrstft,
                    alpha_peak=alpha_peak,
                    alpha_ratio=alpha_ratio,
                    alpha_derivative=alpha_derivative,
                    alpha_softdtw=alpha_softdtw,
                    alpha_corr=alpha_corr,
                    alpha_mrstft=alpha_mrstft,
                    base_loss=base_loss,
                    dtw_gamma=dtw_gamma
                )
                components = []
                if use_base: components.append(f"{base_loss.upper()}")
                if use_peak: components.append(f"{alpha_peak}*peak")
                if use_ratio: components.append(f"{alpha_ratio}*ratio")
                if use_derivative: components.append(f"{alpha_derivative}*derivative")
                if use_softdtw: components.append(f"{alpha_softdtw}*softdtw")
                if use_corr: components.append(f"{alpha_corr}*corr")
                if use_mrstft: components.append(f"{alpha_mrstft}*mrstft")
                print(f"✓ Using flexible loss: {' + '.join(components)}")
                
            elif loss_type == 'composite':
                loss_fn = modules.create_composite_loss(alpha_peak=alpha_peak, alpha_ratio=alpha_ratio, base_loss=base_loss)
                print(f"✓ Using composite loss: {base_loss.upper()} + {alpha_peak}*peak + {alpha_ratio}*ratio")
                
            elif loss_type == 'shape_preserving':
                loss_fn = modules.create_shape_preserving_loss(
                    alpha_peak=alpha_peak, 
                    alpha_ratio=alpha_ratio, 
                    alpha_derivative=alpha_derivative,
                    base_loss=base_loss
                )
                print(f"✓ Using shape-preserving loss: {base_loss.upper()} + {alpha_peak}*peak + {alpha_ratio}*ratio + {alpha_derivative}*derivative")
                
            elif loss_type == 'mse':
                loss_fn = 'mse'
                print("✓ Using MSE loss")
                
            else:  # 'mae'
                loss_fn = 'mae'
                print("✓ Using MAE loss")
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss=loss_fn)
        
        print(f"✓ Model compiled with learning rate: {lr}")
        
        # Setup callbacks
        fold_model_path = os.path.join(model_dir, f"fold_{fold_idx+1}_best.weights.h5")
        callbacks = [
            ModelCheckpoint(
                filepath=fold_model_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                verbose=0
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Add learning rate schedule if requested
        if use_lr_schedule == 'fixed':
            def scheduler(epoch, current_lr):
                if epoch > 0 and epoch % 5 == 0:
                    return current_lr * 0.5
                return current_lr
            callbacks.append(LearningRateScheduler(scheduler, verbose=1))
            print("  ✓ Using fixed LR schedule (decay by 0.5 every 5 epochs)")
        elif use_lr_schedule == 'dynamic':
            reduce_lr_cb = ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_reduce_factor,
                patience=lr_reduce_patience,
                min_lr=1e-5,
                verbose=1
            )
            callbacks.append(reduce_lr_cb)
            print(f"  ✓ Using ReduceLROnPlateau (patience={lr_reduce_patience}, factor={lr_reduce_factor})")
        
        # Train model
        print(f"\nTraining fold {fold_idx + 1}...")
        history = model.fit(
            X[train_idx], Y[train_idx],
            validation_data=(X[val_idx], Y[val_idx]),
            epochs=n_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best weights
        model.load_weights(fold_model_path)
        
        # Generate predictions on validation set
        print("\nGenerating predictions...")
        Y_pred = model.predict(X[val_idx], verbose=0)
        
        # Squeeze predictions if needed
        if Y_pred.ndim == 3 and Y_pred.shape[-1] == 1:
            Y_pred = Y_pred.squeeze(axis=-1)
        
        Y_val = Y[val_idx]
        
        # Apply smoothing if requested
        if apply_smoothing:
            print("Applying signal smoothing...")
            Y_pred_smoothed = smooth_all_segments(Y_pred, **smoothing_params)
        else:
            Y_pred_smoothed = Y_pred
        
        # Compute ML/Signal metrics
        print("\nComputing ML/signal metrics...")
        ml_metrics = compute_ml_metrics(Y_val, Y_pred_smoothed, fs=sampling_rate)
        ml_metrics['fold'] = fold_idx + 1
        all_ml_metrics.append(ml_metrics)
        
        # Compute Clinical/Physiological metrics
        print("Computing clinical/physiological metrics...")
        clinical_metrics = compute_clinical_metrics(Y_val, Y_pred_smoothed, sampling_rate=sampling_rate)
        clinical_metrics['fold'] = fold_idx + 1
        all_clinical_metrics.append(clinical_metrics)
        
        # Print fold results
        print(f"\nFold {fold_idx + 1} Results:")
        print("-" * 80)
        print("ML/Signal Metrics:")
        for metric_name, metric_data in ml_metrics.items():
            if metric_name != 'fold' and isinstance(metric_data, dict):
                print(f"  {metric_name}: {metric_data['mean']:.6f} ± {metric_data['std']:.6f}")
        
        print("\nClinical Metrics (MAE between real and generated):")
        for metric_name, metric_data in clinical_metrics.items():
            if metric_name != 'fold' and isinstance(metric_data, dict):
                print(f"  {metric_name}: MAE={metric_data['mae']:.6f}, Corr={metric_data['correlation']:.4f}")
        print("-" * 80)
    
    # ========================================================================
    # Aggregate results across folds
    # ========================================================================
    
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS ALL FOLDS")
    print("="*80 + "\n")
    
    # ML/Signal Metrics Summary
    ml_summary_data = []
    metric_names = [k for k in all_ml_metrics[0].keys() if k != 'fold']
    
    for metric_name in metric_names:
        means = [fold_data[metric_name]['mean'] for fold_data in all_ml_metrics]
        stds = [fold_data[metric_name]['std'] for fold_data in all_ml_metrics]
        
        # Calculate mean of means and propagated uncertainty
        overall_mean = np.mean(means)
        overall_std = np.sqrt(np.mean(np.array(stds)**2))  # Propagated uncertainty
        
        ml_summary_data.append({
            'Metric': metric_name,
            'Mean': overall_mean,
            'Std': overall_std,
            'Format': f"{overall_mean:.6f} ± {overall_std:.6f}"
        })
    
    ml_metrics_summary = pd.DataFrame(ml_summary_data)
    
    # Clinical Metrics Summary
    clinical_summary_data = []
    clinical_metric_names = [k for k in all_clinical_metrics[0].keys() if k != 'fold']
    
    for metric_name in clinical_metric_names:
        # MAE
        mae_values = [fold_data[metric_name]['mae'] for fold_data in all_clinical_metrics]
        mae_values = [v for v in mae_values if not np.isnan(v)]
        mae_mean = np.mean(mae_values) if len(mae_values) > 0 else np.nan
        mae_std = np.std(mae_values) if len(mae_values) > 0 else np.nan
        
        # RMSE
        rmse_values = [fold_data[metric_name]['rmse'] for fold_data in all_clinical_metrics]
        rmse_values = [v for v in rmse_values if not np.isnan(v)]
        rmse_mean = np.mean(rmse_values) if len(rmse_values) > 0 else np.nan
        rmse_std = np.std(rmse_values) if len(rmse_values) > 0 else np.nan
        
        # Correlation
        corr_values = [fold_data[metric_name]['correlation'] for fold_data in all_clinical_metrics]
        corr_values = [v for v in corr_values if not np.isnan(v)]
        corr_mean = np.mean(corr_values) if len(corr_values) > 0 else np.nan
        corr_std = np.std(corr_values) if len(corr_values) > 0 else np.nan
        
        # Real values (ground truth)
        real_means = [fold_data[metric_name]['real_mean'] for fold_data in all_clinical_metrics]
        real_means = [v for v in real_means if not np.isnan(v)]
        real_mean_avg = np.mean(real_means) if len(real_means) > 0 else np.nan
        real_std_avg = np.mean([fold_data[metric_name]['real_std'] for fold_data in all_clinical_metrics])
        
        # Generated values
        gen_means = [fold_data[metric_name]['gen_mean'] for fold_data in all_clinical_metrics]
        gen_means = [v for v in gen_means if not np.isnan(v)]
        gen_mean_avg = np.mean(gen_means) if len(gen_means) > 0 else np.nan
        gen_std_avg = np.mean([fold_data[metric_name]['gen_std'] for fold_data in all_clinical_metrics])
        
        clinical_summary_data.append({
            'Metric': metric_name,
            'Real_Mean': real_mean_avg,
            'Real_Std': real_std_avg,
            'Generated_Mean': gen_mean_avg,
            'Generated_Std': gen_std_avg,
            'MAE': mae_mean,
            'MAE_Std': mae_std,
            'RMSE': rmse_mean,
            'RMSE_Std': rmse_std,
            'Correlation': corr_mean,
            'Correlation_Std': corr_std,
            'Real_Format': f"{real_mean_avg:.4f} ± {real_std_avg:.4f}",
            'Generated_Format': f"{gen_mean_avg:.4f} ± {gen_std_avg:.4f}",
            'MAE_Format': f"{mae_mean:.4f} ± {mae_std:.4f}",
            'Correlation_Format': f"{corr_mean:.4f} ± {corr_std:.4f}"
        })
    
    clinical_metrics_summary = pd.DataFrame(clinical_summary_data)
    
    # Save results
    ml_metrics_summary.to_csv(os.path.join(save_dir, f"{model_name}_ml_metrics_summary.csv"), index=False)
    clinical_metrics_summary.to_csv(os.path.join(save_dir, f"{model_name}_clinical_metrics_summary.csv"), index=False)
    
    # Print summaries
    print("\n" + "="*80)
    print("ML/SIGNAL METRICS SUMMARY (Mean ± SD across folds)")
    print("="*80)
    print(ml_metrics_summary[['Metric', 'Format']].to_string(index=False))
    
    print("\n" + "="*80)
    print("CLINICAL METRICS SUMMARY (Mean ± SD across folds)")
    print("="*80)
    print(clinical_metrics_summary[['Metric', 'Real_Format', 'Generated_Format', 'MAE_Format', 'Correlation_Format']].to_string(index=False))
    print("="*80 + "\n")
    
    # Save detailed fold-by-fold results
    with open(os.path.join(save_dir, f"{model_name}_fold_details.json"), 'w') as f:
        json.dump({
            'ml_metrics': all_ml_metrics,
            'clinical_metrics': all_clinical_metrics
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {save_dir}")
    
    return ml_metrics_summary, clinical_metrics_summary


def main():    
    data_dir = "./WaveNet_beat/data"
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))
    PATIENT_IDS = np.load(os.path.join(data_dir, "PATIENT_IDS.npy"))
    
    print(f"Loaded data: X={X.shape}, Y={Y.shape}, Patients={len(np.unique(PATIENT_IDS))}")
    
    ml_summary, clinical_summary = evaluate_model_kfold(
        X=X,
        Y=Y,
        patient_ids=PATIENT_IDS,
        model_builder_fn=None,  # ← Let script build model automatically
        
        # Model architecture
        architecture='single_channel',
        
        # Loss configuration
        loss_type='flexible',
        use_base=True,
        use_derivative=True,
        use_corr=True,
        alpha_derivative=0.5,
        alpha_corr=0.5,
        base_loss='mae',
        
        # Training settings
        n_folds=5,
        n_epochs=100,
        batch_size=32,
        lr=1e-3,

        apply_smoothing=True,
        smoothing_params={'fs': 284, 'cutoff_hz': 10.0, 'order': 4},
        sampling_rate=284,

        use_lr_schedule='dynamic',
        lr_reduce_patience=5,
        lr_reduce_factor=0.5,
        early_stopping_patience=20,
        
        # Output
        save_dir="./evaluation_results",
        model_name="WaveNet_Combined_Attention"
    )


if __name__ == "__main__":
    main()
    # import argparse
    
    # parser = argparse.ArgumentParser(description="K-Fold evaluation with comprehensive metrics")
    
    # parser.add_argument("--data_dir", type=str, default="./WaveNet_beat/data")
    # parser.add_argument("--X_file", type=str, default="X.npy")
    # parser.add_argument("--Y_file", type=str, default="Y.npy")
    # parser.add_argument("--patient_ids_file", type=str, default="PATIENT_IDS.npy")
    
    # parser.add_argument("--n_folds", type=int, default=5)
    # parser.add_argument("--n_epochs", type=int, default=50)
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--early_stopping_patience", type=int, default=10)
    # parser.add_argument("--use_lr_schedule", type=str, choices=['fixed', 'dynamic', 'none'], default='none',
    #                    help="Learning rate schedule type")
    # parser.add_argument("--lr_reduce_patience", type=int, default=3,
    #                    help="Patience for ReduceLROnPlateau (only for dynamic schedule)")
    # parser.add_argument("--lr_reduce_factor", type=float, default=0.5,
    #                    help="Factor for ReduceLROnPlateau (only for dynamic schedule)")
    
    # parser.add_argument("--save_dir", type=str, default="./evaluation_results")
    # parser.add_argument("--model_name", type=str, default="WaveNet_Model")
    
    # # Architecture and loss configuration
    # parser.add_argument("--architecture", type=str, default='combined_attention',
    #                    choices=['two_channel', 'minimal_attention', 'cross_attention', 'combined_attention'],
    #                    help="Model architecture type")
    # parser.add_argument("--loss_type", type=str, default='mae',
    #                    choices=['mae', 'mse', 'composite', 'shape_preserving', 'flexible'],
    #                    help="Type of loss function")
    
    # # Loss component flags (for flexible loss)
    # parser.add_argument("--use_base", type=lambda x: x.lower() == 'true', default=True,
    #                    help="Enable base MAE/MSE loss")
    # parser.add_argument("--use_peak", type=lambda x: x.lower() == 'true', default=False,
    #                    help="Enable peak amplitude preservation")
    # parser.add_argument("--use_ratio", type=lambda x: x.lower() == 'true', default=False,
    #                    help="Enable peak-to-trough ratio preservation")
    # parser.add_argument("--use_derivative", type=lambda x: x.lower() == 'true', default=False,
    #                    help="Enable derivative matching")
    # parser.add_argument("--use_softdtw", type=lambda x: x.lower() == 'true', default=False,
    #                    help="Enable Soft-DTW loss")
    # parser.add_argument("--use_corr", type=lambda x: x.lower() == 'true', default=False,
    #                    help="Enable cross-correlation loss")
    # parser.add_argument("--use_mrstft", type=lambda x: x.lower() == 'true', default=False,
    #                    help="Enable multi-resolution STFT loss")
    
    # # Loss component weights
    # parser.add_argument("--alpha_peak", type=float, default=0.5,
    #                    help="Weight for peak amplitude loss")
    # parser.add_argument("--alpha_ratio", type=float, default=0.5,
    #                    help="Weight for peak-to-trough ratio loss")
    # parser.add_argument("--alpha_derivative", type=float, default=0.3,
    #                    help="Weight for derivative matching loss")
    # parser.add_argument("--alpha_softdtw", type=float, default=1.0,
    #                    help="Weight for Soft-DTW loss")
    # parser.add_argument("--alpha_corr", type=float, default=1.0,
    #                    help="Weight for cross-correlation loss")
    # parser.add_argument("--alpha_mrstft", type=float, default=1.0,
    #                    help="Weight for MR-STFT loss")
    
    # # Other loss parameters
    # parser.add_argument("--base_loss", type=str, default='mae', choices=['mae', 'mse'],
    #                    help="Base loss type")
    # parser.add_argument("--dtw_gamma", type=float, default=0.1,
    #                    help="Temperature parameter for Soft-DTW")
    
    # parser.add_argument("--apply_smoothing", type=lambda x: x.lower() == 'true', default=True)
    # parser.add_argument("--smoothing_cutoff", type=float, default=10.0)
    # parser.add_argument("--smoothing_order", type=int, default=4)
    # parser.add_argument("--sampling_rate", type=int, default=284)
    
    # args = parser.parse_args()
    
    # # Load data
    # X = np.load(os.path.join(args.data_dir, args.X_file))
    # Y = np.load(os.path.join(args.data_dir, args.Y_file))
    # PATIENT_IDS = np.load(os.path.join(args.data_dir, args.patient_ids_file))
    
    # # Run evaluation
    # smoothing_params = {
    #     'fs': args.sampling_rate,
    #     'cutoff_hz': args.smoothing_cutoff,
    #     'order': args.smoothing_order
    # }
    
    # # Convert 'none' to None for use_lr_schedule
    # use_lr_schedule = None if args.use_lr_schedule == 'none' else args.use_lr_schedule
    
    # ml_summary, clinical_summary = evaluate_model_kfold(
    #     X=X,
    #     Y=Y,
    #     patient_ids=PATIENT_IDS,
    #     model_builder_fn=None,  # Use automatic model building
    #     n_folds=args.n_folds,
    #     n_epochs=args.n_epochs,
    #     batch_size=args.batch_size,
    #     lr=args.lr,
    #     save_dir=args.save_dir,
    #     model_name=args.model_name,
    #     apply_smoothing=args.apply_smoothing,
    #     smoothing_params=smoothing_params,
    #     sampling_rate=args.sampling_rate,
    #     early_stopping_patience=args.early_stopping_patience,
    #     use_lr_schedule=use_lr_schedule,
    #     lr_reduce_patience=args.lr_reduce_patience,
    #     lr_reduce_factor=args.lr_reduce_factor,
    #     # Loss configuration
    #     loss_type=args.loss_type,
    #     use_base=args.use_base,
    #     use_peak=args.use_peak,
    #     use_ratio=args.use_ratio,
    #     use_derivative=args.use_derivative,
    #     use_softdtw=args.use_softdtw,
    #     use_corr=args.use_corr,
    #     use_mrstft=args.use_mrstft,
    #     alpha_peak=args.alpha_peak,
    #     alpha_ratio=args.alpha_ratio,
    #     alpha_derivative=args.alpha_derivative,
    #     alpha_softdtw=args.alpha_softdtw,
    #     alpha_corr=args.alpha_corr,
    #     alpha_mrstft=args.alpha_mrstft,
    #     base_loss=args.base_loss,
    #     dtw_gamma=args.dtw_gamma,
    #     architecture=args.architecture
    # )
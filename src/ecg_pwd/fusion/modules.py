"""Core fusion-model module containing architecture definitions, custom losses, and plotting callbacks for two-channel ECG-to-Doppler generation."""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import pywt
from keras.layers import Input, Add, Multiply
from keras.models import Model

from . import layers

def WaveNet_two_channel_cross_attention(input_shape, filters=64, kernel_size=20, 
                                        dilation_rates=None, num_heads=4, dropout=0.1):
    """
    Two-channel WaveNet with cross-channel attention for fetal/maternal ECG fusion.
    
    Args:
        input_shape: (timesteps, 2) - fetal ECG and maternal ECG
        filters: Number of filters in each layer
        kernel_size: Convolution kernel size
        dilation_rates: List of dilation rates for each block
        num_heads: Number of attention heads for cross-attention
        dropout: Dropout rate for attention
        
    Returns:
        Compiled Keras model
    """
    if dilation_rates is None:
        dilation_rates = [2**i for i in range(7)]
    
    inp = Input(shape=input_shape, name="ecg_fetal_maternal_2ch")

    # Split into separate channels
    fetal_ecg = inp[:, :, 0:1]      # (batch, timesteps, 1)
    maternal_ecg = inp[:, :, 1:2]   # (batch, timesteps, 1)
    
    # Project each channel to filter dimension independently
    fetal_features = layers.DilatedConv1D(
        filters=filters, kernel_size=1, dilation_rate=1, padding='same'
    )(fetal_ecg)
    
    maternal_features = layers.DilatedConv1D(
        filters=filters, kernel_size=1, dilation_rate=1, padding='same'
    )(maternal_ecg)
    
    # ✨ CROSS-CHANNEL ATTENTION: Fuse fetal and maternal features ✨
    cross_attn_fn = layers.CrossChannelAttention(
        d_model=filters, 
        num_heads=num_heads, 
        dropout=dropout
    )
    x = cross_attn_fn(fetal_features, maternal_features)
    
    print(f"✓ Cross-channel attention applied: {num_heads} heads, d_model={filters}")

    # Standard WaveNet blocks
    skips = []
    for d in dilation_rates:
        x, s = wavenet_block(x, filters, kernel_size, d)
        skips.append(s)

    # Aggregate skip connections
    out = Add()(skips)
    
    # Post-processing: ReLU -> 1x1 -> ReLU -> 1x1
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(filters, 1)(out)
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(1, 1)(out)
    out = keras.layers.Activation('tanh')(out)

    model = Model(inputs=inp, outputs=out, name="WaveNet_CrossChannelAttention")
    model.summary()
    
    return model


def WaveNet_two_channel_self_attention(input_shape, filters=64, kernel_size=20, 
                                       dilation_rates=None, num_heads=4, dropout=0.1):
    """
    Two-channel WaveNet with self-attention after skip aggregation.
    
    For Phase 2: Use this after validating cross-channel attention works.
    """
    if dilation_rates is None:
        dilation_rates = [2**i for i in range(7)]
    
    inp = Input(shape=input_shape, name="ecg_fetal_maternal_2ch")

    # Project 2 channels -> filters
    x = layers.DilatedConv1D(filters=filters, kernel_size=1, dilation_rate=1, padding='same')(inp)

    # WaveNet blocks
    skips = []
    for d in dilation_rates:
        x, s = wavenet_block(x, filters, kernel_size, d)
        skips.append(s)

    # Aggregate skip connections
    out = Add()(skips)
    
    # ✨ SELF-ATTENTION: Capture long-range dependencies ✨
    self_attn_fn = layers.SelfAttention(d_model=filters, num_heads=num_heads, dropout=dropout)
    out = self_attn_fn(out)
    
    print(f"✓ Self-attention applied: {num_heads} heads, d_model={filters}")
    
    # Post-processing
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(filters, 1)(out)
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(1, 1)(out)
    out = keras.layers.Activation('tanh')(out)

    model = Model(inputs=inp, outputs=out, name="WaveNet_SelfAttention")
    model.summary()
    
    return model


def WaveNet_two_channel_combined_attention(input_shape, filters=64, kernel_size=20, 
                                           dilation_rates=None, num_heads=4, dropout=0.1):
    """
    Two-channel WaveNet with BOTH cross-channel and self-attention.
    
    For Phase 2: Use this to combine both attention mechanisms.
    """
    if dilation_rates is None:
        dilation_rates = [2**i for i in range(7)]
    
    inp = Input(shape=input_shape, name="ecg_fetal_maternal_2ch")

    # Split channels
    fetal_ecg = inp[:, :, 0:1]
    maternal_ecg = inp[:, :, 1:2]
    
    # Project channels
    fetal_features = layers.DilatedConv1D(filters=filters, kernel_size=1, 
                                          dilation_rate=1, padding='same')(fetal_ecg)
    maternal_features = layers.DilatedConv1D(filters=filters, kernel_size=1, 
                                             dilation_rate=1, padding='same')(maternal_ecg)
    
    # ✨ CROSS-CHANNEL ATTENTION ✨
    cross_attn_fn = layers.CrossChannelAttention(d_model=filters, num_heads=num_heads, dropout=dropout)
    x = cross_attn_fn(fetal_features, maternal_features)

    # WaveNet blocks
    skips = []
    for d in dilation_rates:
        x, s = wavenet_block(x, filters, kernel_size, d)
        skips.append(s)

    # Aggregate skips
    out = Add()(skips)
    
    # ✨ SELF-ATTENTION ✨
    self_attn_fn = layers.SelfAttention(d_model=filters, num_heads=num_heads, dropout=dropout)
    out = self_attn_fn(out)
    
    print(f"✓ Combined attention: cross-channel + self-attention, {num_heads} heads")
    
    # Post-processing
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(filters, 1)(out)
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(1, 1)(out)
    out = keras.layers.Activation('tanh')(out)

    model = Model(inputs=inp, outputs=out, name="WaveNet_CombinedAttention")
    model.summary()
    
    return model

# In modules.py - SIMPLEST POSSIBLE VERSION
def WaveNet_minimal_cross_attention(input_shape, filters=64, kernel_size=20):
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
    
    inp = Input(shape=input_shape)
    
    # Split channels
    fetal_ecg = inp[:, :, 0:1]
    maternal_ecg = inp[:, :, 1:2]
    
    # Project
    fetal_feat = layers.DilatedConv1D(filters, 1, 1, padding='same')(fetal_ecg)
    maternal_feat = layers.DilatedConv1D(filters, 1, 1, padding='same')(maternal_ecg)
    
    # ✨ CROSS-ATTENTION (directly, no wrapper) ✨
    attn = MultiHeadAttention(num_heads=4, key_dim=filters // 4)
    x = attn(query=fetal_feat, key=maternal_feat, value=maternal_feat)
    x = LayerNormalization()(fetal_feat + x)  # Residual + Norm
    
    # Continue with WaveNet blocks...
    dilation_rates = [2**i for i in range(7)]
    skips = []
    for d in dilation_rates:
        x, s = wavenet_block(x, filters, kernel_size, d)
        skips.append(s)
    
    out = Add()(skips)
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(filters, 1)(out)
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(1, 1)(out)
    out = keras.layers.Activation('tanh')(out)
    
    return Model(inputs=inp, outputs=out)

# ==================== MODEL ARCHITECTURE ====================

def wavenet_block(x_in, filters, kernel_size, dilation_rate):
    """
    WaveNet residual block with gated activation units.
    
    Args:
        x_in: Input tensor
        filters: Number of filters
        kernel_size: Convolution kernel size
        dilation_rate: Dilation rate for dilated convolution
        
    Returns:
        x_out: Residual output
        skip: Skip connection output
    """
    # Gated convolutions (causal, dilated)
    tanh_out = layers.DilatedConv1D(filters=filters, kernel_size=kernel_size,
                                    dilation_rate=dilation_rate, activation='tanh')(x_in)
    sigm_out = layers.DilatedConv1D(filters=filters, kernel_size=kernel_size,
                                    dilation_rate=dilation_rate, activation='sigmoid')(x_in)
    x = Multiply()([tanh_out, sigm_out])  # Gated output

    # Skip connection (1x1 conv)
    skip = layers.Conv1D(filters, 1)(x)

    # Residual projection through 1x1 conv + add
    res = layers.Conv1D(filters, 1)(x_in)
    x_out = Add()([x, res])
    
    return x_out, skip


def WaveNet_two_channel(input_shape, filters=64, kernel_size=20, dilation_rates=None):
    """
    Two-channel WaveNet for fetal and maternal ECG to Doppler conversion.
    
    Args:
        input_shape: (timesteps, 2) - fetal ECG and maternal ECG
        filters: Number of filters in each layer
        kernel_size: Convolution kernel size
        dilation_rates: List of dilation rates for each block
        
    Returns:
        Compiled Keras model
    """
    if dilation_rates is None:
        dilation_rates = [2**i for i in range(7)]
    
    inp = Input(shape=input_shape, name="ecg_fetal_maternal_2ch")

    # Dilated 1x1 stem to project 2 channels -> filters
    x = layers.DilatedConv1D(filters=filters, kernel_size=1, dilation_rate=1, padding='same')(inp)

    skips = []
    for d in dilation_rates:
        x, s = wavenet_block(x, filters, kernel_size, d)
        skips.append(s)

    # Aggregate skip connections
    out = Add()(skips)
    
    # Post-processing: ReLU -> 1x1 -> ReLU -> 1x1
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(filters, 1)(out)
    out = keras.layers.Activation('relu')(out)
    out = layers.Conv1D(1, 1)(out)
    out = keras.layers.Activation('tanh')(out)

    model = Model(inputs=inp, outputs=out, name="WaveNet_two_channel_early_fusion")
    model.summary()
    
    return model

def wavenet_block_v2(x_in, filters, kernel_size, dilation_rate):
    # gated convs (causal, dilated)
    tanh_out = layers.DilatedConv1D(filters=filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, activation='tanh')(x_in)
    sigm_out = layers.DilatedConv1D(filters=filters, kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, activation='sigmoid')(x_in)
    x = Multiply()([tanh_out, sigm_out])  # gated output

    # skip connection (1x1)
    skip = layers.Conv1D(filters, 1)(x)

    # residual projection through conv (1x1) + add
    res = layers.Conv1D(filters, 1)(x_in)
    x_out = Add()([x, res])
    return x_out, skip

def WaveNet_v2(input_shape, filters=64, kernel_size=20, dilation_rates=[2**i for i in range(7)]): 

    # Input layer
    input_layer = Input(shape=input_shape, name='fetal_ecg_to_doppler')

    # Initial condition layer to start the residual connections
    skip_connections = []

    x = input_layer
    #x = layers.DilatedConv1D(filters=filters, kernel_size=1, dilation_rate=1, padding='same')(inp)
    for dilation_rate in dilation_rates:
        x, skip_conn = wavenet_block_v2(x, filters, kernel_size, dilation_rate)
        skip_connections.append(skip_conn)

    out = Add()(skip_connections)
    out = layers.Activation(out, 'relu')  # post-processing: ReLU -> 1x1 -> ReLU -> 1x1
    out = layers.Conv1D(filters, 1)(out)
    out = layers.Activation(out, 'relu')
    out = layers.Conv1D(1, 1)(out)
    out = layers.Activation(out, 'tanh')

    # Building the model
    model = Model(inputs=input_layer, outputs=out, name='WaveNet_one_channel')
    model.summary()
    
    return model

# ==================== LOSS FUNCTIONS ====================

def cross_correlation_loss(y_true, y_pred):
    """
    Cross-correlation loss: 1 - correlation coefficient
    Perfect correlation = 0 loss
    """
    epsilon = 1e-7
    
    # Normalize signals (zero mean, unit variance)
    y_true_mean = tf.reduce_mean(y_true, axis=1, keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=1, keepdims=True)
    
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean
    
    y_true_std = tf.math.reduce_std(y_true, axis=1, keepdims=True) + epsilon
    y_pred_std = tf.math.reduce_std(y_pred, axis=1, keepdims=True) + epsilon
    
    y_true_norm = y_true_centered / y_true_std
    y_pred_norm = y_pred_centered / y_pred_std
    
    # Compute correlation coefficient
    correlation = tf.reduce_mean(y_true_norm * y_pred_norm, axis=1)
    correlation = tf.clip_by_value(correlation, -1.0, 1.0)
    
    # Convert to loss (1 - correlation)
    loss = tf.reduce_mean(1.0 - correlation)
    
    return loss


def multi_resolution_stft_loss(y_true, y_pred, fft_sizes=None, hop_sizes=None, win_lengths=None):
    """
    Multi-resolution STFT loss
    Compares spectrograms at multiple resolutions
    """
    if fft_sizes is None:
        fft_sizes = [256, 512, 1024]
    if hop_sizes is None:
        hop_sizes = [64, 128, 256]
    if win_lengths is None:
        win_lengths = [256, 512, 1024]
    
    total_loss = 0.0
    
    for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
        # Compute STFT for true and predicted signals
        stft_true = tf.signal.stft(
            y_true, 
            frame_length=win_length, 
            frame_step=hop_size,
            fft_length=fft_size
        )
        stft_pred = tf.signal.stft(
            y_pred, 
            frame_length=win_length, 
            frame_step=hop_size,
            fft_length=fft_size
        )
        
        # Compute magnitude spectrograms
        mag_true = tf.abs(stft_true)
        mag_pred = tf.abs(stft_pred)
        
        # L1 loss on magnitudes
        spectral_loss = tf.reduce_mean(tf.abs(mag_true - mag_pred))
        
        # Log magnitude loss (helps with perceptual quality)
        log_mag_true = tf.math.log(mag_true + 1e-7)
        log_mag_pred = tf.math.log(mag_pred + 1e-7)
        log_spectral_loss = tf.reduce_mean(tf.abs(log_mag_true - log_mag_pred))
        
        total_loss += spectral_loss + log_spectral_loss
    
    return total_loss / len(fft_sizes)


def fast_soft_dtw_loss(y_true, y_pred, gamma=0.1):
    """
    Fast differentiable approximation of Soft-DTW for TensorFlow
    This can be used in training with backpropagation
    """
    # Expand for pairwise distance computation
    y_true_exp = tf.expand_dims(y_true, 2)  # (batch, N, 1)
    y_pred_exp = tf.expand_dims(y_pred, 1)  # (batch, 1, N)
    
    # Compute cost matrix (pairwise squared distances)
    C = tf.square(y_true_exp - y_pred_exp)  # (batch, N, N)
    
    # Soft minimum approximation using softmax weighting
    weights = tf.nn.softmax(-C / gamma, axis=2)
    weighted_costs = tf.reduce_sum(C * weights, axis=2)
    
    # Average over sequence and batch
    dtw_loss = tf.reduce_mean(weighted_costs)
    
    return dtw_loss


def create_flexible_loss(
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
    dtw_gamma=0.1
):
    """
    Create flexible composite loss function with individual component control.
    
    Args:
        use_base: Enable base MAE/MSE loss
        use_peak: Enable peak amplitude preservation
        use_ratio: Enable peak-to-trough ratio preservation
        use_derivative: Enable derivative matching
        use_softdtw: Enable Soft-DTW loss
        use_corr: Enable cross-correlation loss
        use_mrstft: Enable multi-resolution STFT loss
        alpha_*: Weights for each component
        base_loss: 'mae' or 'mse'
        dtw_gamma: Temperature parameter for Soft-DTW
        
    Returns:
        Loss function compatible with Keras
    """
    def flexible_loss(y_true, y_pred):
        # Ensure 2D tensors (batch, time)
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred.shape) == 3:
            y_pred = tf.squeeze(y_pred, axis=-1)
        
        epsilon = 1e-7
        total = 0.0
        
        # 1. Base reconstruction loss
        if use_base:
            if base_loss == 'mae':
                total += tf.reduce_mean(tf.abs(y_true - y_pred))
            else:  # mse
                total += tf.reduce_mean(tf.square(y_true - y_pred))
        
        # 2. Peak amplitude preservation loss
        if use_peak:
            peak_true = tf.reduce_max(y_true, axis=1)
            peak_pred = tf.reduce_max(y_pred, axis=1)
            peak_loss = tf.reduce_mean(tf.abs(peak_true - peak_pred))
            total += alpha_peak * peak_loss
        
        # 3. Peak-to-trough ratio loss
        if use_ratio:
            peak_true = tf.reduce_max(y_true, axis=1)
            peak_pred = tf.reduce_max(y_pred, axis=1)
            trough_true = tf.reduce_min(y_true, axis=1)
            trough_pred = tf.reduce_min(y_pred, axis=1)
            
            ratio_true = peak_true - trough_true + epsilon
            ratio_pred = peak_pred - trough_pred + epsilon
            
            ratio_true = tf.clip_by_value(ratio_true, epsilon, 1e6)
            ratio_pred = tf.clip_by_value(ratio_pred, epsilon, 1e6)
            
            ratio_loss = tf.reduce_mean(tf.abs(ratio_true - ratio_pred))
            ratio_loss = tf.where(tf.math.is_nan(ratio_loss), 0.0, ratio_loss)
            
            total += alpha_ratio * ratio_loss
        
        # 4. Derivative matching
        if use_derivative:
            dy_true = y_true[:, 1:] - y_true[:, :-1]
            dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
            derivative_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred))
            derivative_loss = tf.where(tf.math.is_nan(derivative_loss), 0.0, derivative_loss)
            
            total += alpha_derivative * derivative_loss
        
        # 5. Soft-DTW loss (fast differentiable approximation)
        if use_softdtw:
            try:
                dtw_loss = fast_soft_dtw_loss(y_true, y_pred, gamma=dtw_gamma)
                dtw_loss = tf.where(tf.math.is_nan(dtw_loss), 0.0, dtw_loss)
                total += alpha_softdtw * dtw_loss
            except Exception:
                pass  # Skip if fails
        
        # 6. Cross-correlation loss
        if use_corr:
            try:
                corr_loss = cross_correlation_loss(y_true, y_pred)
                corr_loss = tf.where(tf.math.is_nan(corr_loss), 0.0, corr_loss)
                total += alpha_corr * corr_loss
            except Exception:
                pass
        
        # 7. Multi-resolution STFT loss
        if use_mrstft:
            try:
                mrstft_loss = multi_resolution_stft_loss(y_true, y_pred)
                mrstft_loss = tf.where(tf.math.is_nan(mrstft_loss), 0.0, mrstft_loss)
                total += alpha_mrstft * mrstft_loss
            except Exception:
                pass
        
        return total
    
    return flexible_loss


def create_composite_loss(alpha_peak=0.5, alpha_ratio=0.5, base_loss='mae'):
    """Create composite loss function for WaveNet training (legacy compatibility)"""
    return create_flexible_loss(
        use_base=True, use_peak=True, use_ratio=True, use_derivative=False,
        alpha_peak=alpha_peak, alpha_ratio=alpha_ratio, base_loss=base_loss
    )


def create_shape_preserving_loss(alpha_peak=0.5, alpha_ratio=0.5, alpha_derivative=0.3, base_loss='mae'):
    """Enhanced composite loss with derivative matching (legacy compatibility)"""
    return create_flexible_loss(
        use_base=True, use_peak=True, use_ratio=True, use_derivative=True,
        alpha_peak=alpha_peak, alpha_ratio=alpha_ratio, alpha_derivative=alpha_derivative,
        base_loss=base_loss
    )


# ==================== CALLBACKS ====================

class LossComponentLogger(Callback):
    """Custom callback to compute and log individual loss components after each epoch"""
    
    def __init__(self, X_train, Y_train, X_val, Y_val, 
                 use_base=True, use_peak=False, use_ratio=False, use_derivative=False,
                 use_softdtw=False, use_corr=False, use_mrstft=False,
                 alpha_peak=0.5, alpha_ratio=0.5, alpha_derivative=0.3,
                 alpha_softdtw=1.0, alpha_corr=1.0, alpha_mrstft=1.0,
                 base_loss='mae', dtw_gamma=0.1):
        super().__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        
        # Flags
        self.use_base = use_base
        self.use_peak = use_peak
        self.use_ratio = use_ratio
        self.use_derivative = use_derivative
        self.use_softdtw = use_softdtw
        self.use_corr = use_corr
        self.use_mrstft = use_mrstft
        
        # Weights
        self.alpha_peak = alpha_peak
        self.alpha_ratio = alpha_ratio
        self.alpha_derivative = alpha_derivative
        self.alpha_softdtw = alpha_softdtw
        self.alpha_corr = alpha_corr
        self.alpha_mrstft = alpha_mrstft
        
        self.base_loss = base_loss
        self.dtw_gamma = dtw_gamma
        
        # Store history for all components
        self.history = {
            'train_base': [],
            'train_peak': [],
            'train_ratio': [],
            'train_derivative': [],
            'train_softdtw': [],
            'train_corr': [],
            'train_mrstft': [],
            'val_base': [],
            'val_peak': [],
            'val_ratio': [],
            'val_derivative': [],
            'val_softdtw': [],
            'val_corr': [],
            'val_mrstft': []
        }
    
    def compute_components(self, y_true, y_pred):
        """Compute individual loss components"""
        # Squeeze if necessary
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred.shape) == 3:
            y_pred = tf.squeeze(y_pred, axis=-1)
        
        epsilon = 1e-7
        components = {}
        
        # 1. Base loss
        if self.base_loss == 'mae':
            base = tf.reduce_mean(tf.abs(y_true - y_pred))
        elif self.base_loss == 'mse':
            base = tf.reduce_mean(tf.square(y_true - y_pred))
        components['base'] = float(base.numpy())
        
        # 2. Peak loss
        peak_true = tf.reduce_max(y_true, axis=1)
        peak_pred = tf.reduce_max(y_pred, axis=1)
        peak_loss = tf.reduce_mean(tf.abs(peak_true - peak_pred))
        components['peak'] = float(peak_loss.numpy())
        
        # 3. Ratio loss (with safety)
        trough_true = tf.reduce_min(y_true, axis=1)
        trough_pred = tf.reduce_min(y_pred, axis=1)
        ratio_true = peak_true - trough_true + epsilon
        ratio_pred = peak_pred - trough_pred + epsilon
        ratio_true = tf.clip_by_value(ratio_true, epsilon, 1e6)
        ratio_pred = tf.clip_by_value(ratio_pred, epsilon, 1e6)
        ratio_loss = tf.reduce_mean(tf.abs(ratio_true - ratio_pred))
        components['ratio'] = float(ratio_loss.numpy()) if not tf.math.is_nan(ratio_loss) else 0.0
        
        # 4. Derivative loss
        dy_true = y_true[:, 1:] - y_true[:, :-1]
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        derivative_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred))
        components['derivative'] = float(derivative_loss.numpy()) if not tf.math.is_nan(derivative_loss) else 0.0
        
        # 5. Soft-DTW (fast approximation)
        try:
            softdtw_loss = fast_soft_dtw_loss(y_true, y_pred, gamma=self.dtw_gamma)
            components['softdtw'] = float(softdtw_loss.numpy())
        except Exception as e:
            print(f"  Warning: Soft-DTW computation failed: {e}")
            components['softdtw'] = 0.0
        
        # 6. Cross-correlation
        try:
            corr_loss = cross_correlation_loss(y_true, y_pred)
            components['corr'] = float(corr_loss.numpy())
        except Exception as e:
            print(f"  Warning: Correlation computation failed: {e}")
            components['corr'] = 0.0
        
        # 7. Multi-resolution STFT
        try:
            mrstft_loss = multi_resolution_stft_loss(y_true, y_pred)
            components['mrstft'] = float(mrstft_loss.numpy())
        except Exception as e:
            print(f"  Warning: MR-STFT computation failed: {e}")
            components['mrstft'] = 0.0
        
        return components
    
    def on_epoch_end(self, epoch, logs=None):
        """Compute and print loss components after each epoch"""
        # Get predictions for train and val
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        y_val_pred = self.model.predict(self.X_val, verbose=0)
        
        # Compute components
        train_components = self.compute_components(self.Y_train, y_train_pred)
        val_components = self.compute_components(self.Y_val, y_val_pred)
        
        # Store in history
        for key in train_components.keys():
            self.history[f'train_{key}'].append(train_components[key])
            self.history[f'val_{key}'].append(val_components[key])
        
        # Calculate total losses based on which components are enabled
        train_total = 0.0
        val_total = 0.0
        
        if self.use_base:
            train_total += train_components['base']
            val_total += val_components['base']
        
        if self.use_peak:
            train_total += self.alpha_peak * train_components['peak']
            val_total += self.alpha_peak * val_components['peak']
        
        if self.use_ratio:
            train_total += self.alpha_ratio * train_components['ratio']
            val_total += self.alpha_ratio * val_components['ratio']
        
        if self.use_derivative:
            train_total += self.alpha_derivative * train_components['derivative']
            val_total += self.alpha_derivative * val_components['derivative']
        
        if self.use_softdtw:
            train_total += self.alpha_softdtw * train_components['softdtw']
            val_total += self.alpha_softdtw * val_components['softdtw']
        
        if self.use_corr:
            train_total += self.alpha_corr * train_components['corr']
            val_total += self.alpha_corr * val_components['corr']
        
        if self.use_mrstft:
            train_total += self.alpha_mrstft * train_components['mrstft']
            val_total += self.alpha_mrstft * val_components['mrstft']
        
        # Print detailed breakdown
        print(f"\n{'='*90}")
        print(f"Epoch {epoch + 1} - Loss Component Breakdown:")
        print(f"{'='*90}")
        print(f"{'Component':<30} {'Train':<15} {'Val':<15} {'Weight':<10} {'Used':<10}")
        print(f"{'-'*90}")
        
        base_name = 'Base (MAE)' if self.base_loss == 'mae' else 'Base (MSE)'
        print(f"{base_name:<30} {train_components['base']:<15.6f} {val_components['base']:<15.6f} {'1.0':<10} {'✓' if self.use_base else '✗':<10}")
        print(f"{'Peak Loss':<30} {train_components['peak']:<15.6f} {val_components['peak']:<15.6f} {self.alpha_peak:<10.2f} {'✓' if self.use_peak else '✗':<10}")
        print(f"{'Ratio Loss':<30} {train_components['ratio']:<15.6f} {val_components['ratio']:<15.6f} {self.alpha_ratio:<10.2f} {'✓' if self.use_ratio else '✗':<10}")
        print(f"{'Derivative Loss':<30} {train_components['derivative']:<15.6f} {val_components['derivative']:<15.6f} {self.alpha_derivative:<10.2f} {'✓' if self.use_derivative else '✗':<10}")
        print(f"{'Soft-DTW Loss':<30} {train_components['softdtw']:<15.6f} {val_components['softdtw']:<15.6f} {self.alpha_softdtw:<10.2f} {'✓' if self.use_softdtw else '✗':<10}")
        print(f"{'Cross-Correlation Loss':<30} {train_components['corr']:<15.6f} {val_components['corr']:<15.6f} {self.alpha_corr:<10.2f} {'✓' if self.use_corr else '✗':<10}")
        print(f"{'MR-STFT Loss':<30} {train_components['mrstft']:<15.6f} {val_components['mrstft']:<15.6f} {self.alpha_mrstft:<10.2f} {'✓' if self.use_mrstft else '✗':<10}")
        
        print(f"{'-'*90}")
        print(f"{'TOTAL (weighted sum)':<30} {train_total:<15.6f} {val_total:<15.6f}")
        print(f"{'='*90}\n")


# ==================== VISUALIZATION FUNCTIONS ====================

def create_scalogram(sig, fs=284, time_bins=160, freq_bins=80):
    """Create continuous wavelet transform scalogram from signal"""
    scales = np.arange(1, freq_bins + 1)
    coeffs, f = pywt.cwt(sig, scales, wavelet='cgau8', sampling_period=1/fs)
    coeffs = np.abs(coeffs)
    return coeffs, f


def create_batch_scalograms(signals_batch, fs=2000, time_bins=160, freq_bins=80):
    """Process a batch of signals and create scalograms"""
    coeffs = []
    f_s = []
    for sig in signals_batch:
        coeff, f = create_scalogram(sig, fs, time_bins, freq_bins)
        coeffs.append(coeff)
        f_s.append(f)
    return coeffs, f_s


def plot_ecg_doppler_pairs(ecgs, real_dopplers, generated_dopplers, save_path='signals_test.jpg'):
    """
    Plots ECG and corresponding real and generated Doppler pairs.
    Handles both single-channel and multi-channel ECGs.
    """
    # Check if ECGs are multi-channel
    if len(ecgs.shape) == 3 and ecgs.shape[2] > 1:
        n_channels = ecgs.shape[2]
        plt.figure(figsize=(18, 8 * n_channels))
        
        for i, (ecg, real_dopple, generated_dopple) in enumerate(zip(ecgs, real_dopplers, generated_dopplers)):
            for ch in range(n_channels):
                row_idx = i * n_channels + ch
                
                # Plotting ECG channel
                plt.subplot(len(ecgs) * n_channels, 3, 3 * row_idx + 1)
                plt.plot(ecg[:, ch], color='royalblue')
                plt.title(f'Fetal ECG' if ch == 0 else 'Maternal ECG', fontsize=10)
                plt.xticks([])
                plt.yticks([])
                plt.box(False)
                plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)
                plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)
                
                # Only plot Doppler for the first channel (to avoid repetition)
                if ch == 0:
                    # Plotting Real Doppler
                    plt.subplot(len(ecgs) * n_channels, 3, 3 * row_idx + 2)
                    plt.plot(real_dopple, color='blue')
                    plt.title('Real Doppler', fontsize=10)
                    plt.xticks([])
                    plt.yticks([])
                    plt.box(False)
                    plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)
                    plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)
                    
                    # Plotting Generated Doppler
                    plt.subplot(len(ecgs) * n_channels, 3, 3 * row_idx + 3)
                    plt.plot(generated_dopple, color='red')
                    plt.title('Generated Doppler', fontsize=10)
                    plt.xticks([])
                    plt.yticks([])
                    plt.box(False)
                    plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)
                    plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)
    else:
        # Original single-channel code
        plt.figure(figsize=(18, 8))
        for i, (ecg, real_dopple, generated_dopple) in enumerate(zip(ecgs, real_dopplers, generated_dopplers)):
            # Plotting ECG
            plt.subplot(len(ecgs), 3, 3*i + 1)
            if ecg.ndim == 2:
                ecg = ecg.flatten()
            plt.plot(ecg, color='royalblue')
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)
            plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)

            # Plotting Real Doppler
            plt.subplot(len(ecgs), 3, 3*i + 2)
            plt.plot(real_dopple, color='blue')
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)
            plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)

            # Plotting Generated Doppler
            plt.subplot(len(ecgs), 3, 3*i + 3)
            plt.plot(generated_dopple, color='red')
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)
            plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scalogram(real, generated, time_bins=160, freq_bins=80, save_path='scalograms_test.jpg'):
    """Plot scalograms for real and generated signals"""
    plt.figure(figsize=(18, 10))
    coeffs_rs, fs_rs, coeffs_gs, fs_gs = [], [], [], []
    fs = 284
    
    for i in range(len(real)):
        coeffs_r, fs_r = create_scalogram(real[i], fs, time_bins, freq_bins)
        coeffs_g, fs_g = create_scalogram(generated[i], fs, time_bins, freq_bins)
        coeffs_gs.append(coeffs_g)
        coeffs_rs.append(coeffs_r)
        fs_gs.append(fs_g)
        fs_rs.append(fs_r)

    for i in range(len(real)):
        plt.subplot(len(real), 2, 2*i + 1)
        plt.pcolormesh(np.arange(coeffs_rs[i].shape[1]), fs_rs[i], coeffs_rs[i], 
                      shading='gouraud', cmap='bwr')
        plt.yticks([])
        plt.xticks([])
        plt.ylim(10, 1000)

        plt.subplot(len(real), 2, 2*i + 2)
        plt.pcolormesh(np.arange(coeffs_gs[i].shape[1]), fs_gs[i], coeffs_gs[i], 
                      shading='gouraud', cmap='bwr')
        plt.xticks([])
        plt.yticks([])
        plt.ylim(10, 1000)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ecg_doppler_overlay_multi(ecgs, real_dopplers, generated_dopplers, 
                                   labels=None, colors=None, 
                                   save_dir='plots', prefix='signals_overlay'):
    """
    Plots ECG, real Doppler, and generated Doppler signals overlaid on the same axes for each sample.
    Handles both single-channel and multi-channel (Fetal/Maternal) ECGs.
    
    Parameters:
    - ecgs: array of ECG signals (shape: (n_samples, time_steps) or (n_samples, time_steps, n_channels))
    - real_dopplers: array of corresponding real Doppler signals
    - generated_dopplers: array of corresponding generated Doppler signals
    - labels: dict mapping signal types to legend labels (optional)
    - colors: dict mapping signal types to color strings (optional)
    - save_dir: directory to save plots
    - prefix: prefix for saved filenames
    """
    # Default labels and colors
    default_labels = {
        'fetal_ecg': 'Fetal ECG', 
        'maternal_ecg': 'Maternal ECG',
        'ecg': 'ECG',
        'real': 'Real Doppler', 
        'gen': 'Generated Doppler'
    }
    default_colors = {
        'fetal_ecg': 'royalblue',
        'maternal_ecg': 'darkgreen', 
        'ecg': 'royalblue',
        'real': 'blue', 
        'gen': 'red'
    }
    labels = labels or default_labels
    colors = colors or default_colors
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (ecg, real_dop, gen_dop) in enumerate(zip(ecgs, real_dopplers, generated_dopplers), start=1):
        plt.figure(figsize=(12, 6))
        
        # Check if ECG is multi-channel
        if ecg.ndim == 2 and ecg.shape[1] > 1:
            # Multi-channel ECG (assuming 2 channels: Fetal and Maternal)
            n_channels = ecg.shape[1]
            
            # Plot Fetal ECG (Channel 0)
            plt.plot(ecg[:, 0], 
                    label=labels.get('fetal_ecg', 'Fetal ECG'), 
                    color=colors.get('fetal_ecg', 'royalblue'), 
                    linewidth=1.2, alpha=0.8)
            
            # Plot Maternal ECG (Channel 1) if it exists
            if n_channels > 1:
                plt.plot(ecg[:, 1], 
                        label=labels.get('maternal_ecg', 'Maternal ECG'), 
                        color=colors.get('maternal_ecg', 'darkgreen'), 
                        linewidth=1.2, alpha=0.8)
            
            # Plot additional channels if they exist
            additional_colors = ['orange', 'purple', 'brown', 'pink']
            for ch in range(2, n_channels):
                color = additional_colors[(ch-2) % len(additional_colors)]
                plt.plot(ecg[:, ch], 
                        label=f'ECG Channel {ch+1}', 
                        color=color, 
                        linewidth=1.2, alpha=0.8)
        else:
            # Single-channel ECG
            if ecg.ndim == 2:
                ecg = ecg.flatten()  # Handle case where single channel is shape (time_steps, 1)
            plt.plot(ecg, 
                    label=labels.get('ecg', 'ECG'), 
                    color=colors.get('ecg', 'royalblue'), 
                    linewidth=1.2, alpha=0.8)
        
        # Plot Doppler signals
        plt.plot(real_dop, 
                label=labels.get('real', 'Real Doppler'), 
                color=colors.get('real', 'blue'), 
                linewidth=1.5)
        plt.plot(gen_dop, 
                label=labels.get('gen', 'Generated Doppler'), 
                color=colors.get('gen', 'red'), 
                linewidth=1.5, 
                linestyle='--')
        
        # Customize plot
        plt.title(f'Sample {i}: ECG and Doppler Signals Overlay', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        plt.xlabel('Time / Sample Index', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add subtle background
        plt.gca().set_facecolor('#f8f9fa')
        
        # Improve layout
        plt.tight_layout()
        
        # Save with descriptive filename
        save_path = os.path.join(save_dir, f'{prefix}_sample_{i}.jpg')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
        plt.close()  # Close to free memory


def plot_training_history(history, save_dir_plots, split_name="patient_split"):
    """
    Plot training and validation loss curves over epochs
    
    Args:
        history: Keras History object from model.fit()
        save_dir_plots: Directory to save plots
        split_name: Name of the split (for filename)
    """
    # Extract loss values
    train_loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])
    
    epochs = range(1, len(train_loss) + 1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    
    # Plot validation loss if available
    if val_loss:
        plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    
    # Customize the plot
    plt.title('Model Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (MAE)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add annotations for best validation loss
    if val_loss:
        best_epoch = np.argmin(val_loss) + 1
        best_val_loss = np.min(val_loss)
        plt.annotate(f'Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}', 
                    xy=(best_epoch, best_val_loss), 
                    xytext=(best_epoch + len(epochs)*0.1, best_val_loss + max(train_loss)*0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir_plots, f'training_history_{split_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {plot_path}")


def plot_detailed_training_history(history, save_dir_plots, split_name="patient_split"):
    """
    Create detailed training history plots with subplots
    """
    train_loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(train_loss) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Combined loss curves
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=3)
    if val_loss:
        ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=3)
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MAE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss difference (overfitting indicator)
    if val_loss:
        loss_diff = np.array(val_loss) - np.array(train_loss[:len(val_loss)])
        ax2.plot(epochs[:len(loss_diff)], loss_diff, 'g-^', linewidth=2, markersize=3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Validation - Training Loss\n(Overfitting Indicator)', fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss Difference')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log scale loss curves
    ax3.semilogy(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=3)
    if val_loss:
        ax3.semilogy(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=3)
    ax3.set_title('Loss Curves (Log Scale)', fontweight='bold')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss (MAE) - Log Scale')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss improvement rate
    train_improvement = np.diff(train_loss)
    ax4.plot(epochs[1:], train_improvement, 'b-', label='Training Loss Change', linewidth=2)
    if val_loss and len(val_loss) > 1:
        val_improvement = np.diff(val_loss)
        ax4.plot(epochs[1:len(val_improvement)+1], val_improvement, 'r-', label='Validation Loss Change', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title('Loss Improvement Rate', fontweight='bold')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Loss Change')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the detailed plot
    plot_path = os.path.join(save_dir_plots, f'detailed_training_history_{split_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed training history plot saved to: {plot_path}")


def plot_all_loss_components(loss_logger, save_dir_plots, n_epochs):
    """Plot all individual loss components over training epochs"""
    # Create figure with 3x3 subplots for the 7 components
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Individual Loss Components Over Training', fontsize=16, fontweight='bold')
    
    epochs = np.arange(1, n_epochs + 1)
    
    components_to_plot = [
        ('base', 'Base Loss (MAE/MSE)', 0, 0),
        ('peak', 'Peak Amplitude Loss', 0, 1),
        ('ratio', 'Peak-to-Trough Ratio Loss', 0, 2),
        ('derivative', 'Derivative Matching Loss', 1, 0),
        ('softdtw', 'Soft-DTW Loss', 1, 1),
        ('corr', 'Cross-Correlation Loss', 1, 2),
        ('mrstft', 'MR-STFT Loss', 2, 0),
    ]
    
    for comp_name, title, row, col in components_to_plot:
        ax = axes[row, col]
        train_key = f'train_{comp_name}'
        val_key = f'val_{comp_name}'
        
        if train_key in loss_logger.history and loss_logger.history[train_key]:
            ax.plot(epochs, loss_logger.history[train_key], 'b-', label='Train', linewidth=2, alpha=0.7)
            ax.plot(epochs, loss_logger.history[val_key], 'r-', label='Validation', linewidth=2, alpha=0.7)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not Used', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
    
    # Hide unused subplots
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(save_dir_plots, 'loss_components_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved loss components plot to: {output_path}")
    plt.close()
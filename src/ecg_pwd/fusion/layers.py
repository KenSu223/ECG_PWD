"""Reusable Keras layer wrappers for two-channel fetal-maternal fusion WaveNet models, including attention-friendly building blocks."""

import keras

weights_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=2024)

import tensorflow as tf
from tensorflow import keras


def CrossChannelAttention(d_model, num_heads=4, dropout=0.1):
    """
    Cross-channel attention layer for fusing fetal and maternal ECG features.
    
    Args:
        d_model: Feature dimension (must match input channels)
        num_heads: Number of attention heads
        dropout: Dropout rate for attention weights
        
    Returns:
        Function that takes (fetal_features, maternal_features) and returns fused features
    """
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout
    
    # Define the attention mechanism
    cross_attn_layer = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
        name="cross_channel_attention"
    )
    
    layer_norm1 = LayerNormalization(epsilon=1e-6)
    layer_norm2 = LayerNormalization(epsilon=1e-6)
    
    # Feed-forward network for additional processing
    ffn = keras.Sequential([
        Dense(d_model * 2, activation='relu'),
        Dropout(dropout),
        Dense(d_model)
    ])
    
    def apply_cross_attention(fetal_features, maternal_features):
        """
        Apply cross-attention: fetal queries attend to maternal keys/values
        
        Args:
            fetal_features: (batch, timesteps, d_model) - primary signal
            maternal_features: (batch, timesteps, d_model) - context signal
            
        Returns:
            fused_features: (batch, timesteps, d_model)
        """
        # Cross-attention: fetal attends to maternal
        # No causal mask since this is offline training with full signal
        attn_output = cross_attn_layer(
            query=fetal_features,
            key=maternal_features,
            value=maternal_features,
            return_attention_scores=False
        )
        
        # Residual connection + layer norm
        attn_output = layer_norm1(fetal_features + attn_output)
        
        # Feed-forward network
        ffn_output = ffn(attn_output)
        
        # Residual connection + layer norm
        output = layer_norm2(attn_output + ffn_output)
        
        return output
    
    return apply_cross_attention


def SelfAttention(d_model, num_heads=4, dropout=0.1):
    """
    Self-attention layer for capturing long-range dependencies.
    
    Args:
        d_model: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        Function that takes features and returns attended features
    """
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout
    
    self_attn_layer = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
        name="self_attention"
    )
    
    layer_norm1 = LayerNormalization(epsilon=1e-6)
    layer_norm2 = LayerNormalization(epsilon=1e-6)
    
    ffn = keras.Sequential([
        Dense(d_model * 2, activation='relu'),
        Dropout(dropout),
        Dense(d_model)
    ])
    
    def apply_self_attention(x):
        """
        Apply self-attention
        
        Args:
            x: (batch, timesteps, d_model)
            
        Returns:
            output: (batch, timesteps, d_model)
        """
        # Self-attention (no causal mask for offline training)
        attn_output = self_attn_layer(
            query=x,
            key=x,
            value=x,
            return_attention_scores=False
        )
        
        # Residual + norm
        attn_output = layer_norm1(x + attn_output)
        
        # Feed-forward
        ffn_output = ffn(attn_output)
        
        # Residual + norm
        output = layer_norm2(attn_output + ffn_output)
        
        return output
    
    return apply_self_attention

def Conv1D(filters, kernel_size, strides=1, padding='same', activation=None, use_bias=True):
    layer = keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=None,
        dilation_rate=1, groups=1, activation=activation, use_bias=use_bias, kernel_initializer=weights_initializer,
        bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None
    )
    return layer


def DilatedConv1D(filters, kernel_size, dilation_rate, strides=1,
                  padding='same', activation=None, use_bias=True):
    return keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, dilation_rate=dilation_rate, activation=activation,
        use_bias=use_bias, kernel_initializer=weights_initializer,
        bias_initializer="zeros"
    )


def CausalConv1D(filters, kernel_size, dilation_rate, strides=1, padding='causal', activation=None, use_bias=True):
    layer = keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, strides=strides, padding=padding,
        data_format=None, groups=1, activation=activation, use_bias=use_bias, kernel_initializer=weights_initializer,
        bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None
    )
    return layer


def Conv1DTranspose(filters, kernel_size, strides=1, padding='same', activation=None, use_bias=True):
    layer = keras.layers.Conv1DTranspose(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=None,
        dilation_rate=1, activation=activation, use_bias=use_bias, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None
    )
    return layer


def Dense(units, activation=None):
    layer = keras.layers.Dense(
        units=units, activation=activation, use_bias=True, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
    )
    return layer


def BatchNormalization(momentum=0.99, epsilon=0.001, trainable=True, virtual_batch_size=None):
    layer = keras.layers.BatchNormalization(
        axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True,
        beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
        gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
        fused=None, trainable=trainable, virtual_batch_size=virtual_batch_size, adjustment=None, name=None
    )
    return layer


def Activation(x, activation):
    """Apply activation function to input tensor"""
    if activation == 'relu':
        return keras.activations.relu(x)
    elif activation == 'leaky_relu':
        return keras.activations.relu(x, alpha=0.2, max_value=None, threshold=0)
    elif activation == 'silu':
        return keras.activations.silu(x)
    elif activation == 'sigmoid':
        return keras.activations.sigmoid(x)
    elif activation == 'softmax':
        return keras.activations.softmax(x, axis=-1)
    elif activation == 'tanh':
        return keras.activations.tanh(x)
    else:
        raise ValueError(f'Unknown activation function: {activation}')


def MaxPooling1D(pool_size=2, strides=None, padding="valid"):
    layer = keras.layers.MaxPooling1D(
        pool_size=pool_size, strides=strides, padding=padding, data_format=None, name=None
    )
    return layer


def Dropout(rate=0.2):
    layer = keras.layers.Dropout(
        rate=rate, noise_shape=None, seed=2024
    )
    return layer

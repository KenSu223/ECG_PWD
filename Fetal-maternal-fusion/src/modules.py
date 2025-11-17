import layers
import numpy as np
from keras.layers import Input, Reshape, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential
from skimage.transform import resize
import pywt
import matplotlib.pyplot as plt
from tensorflow import keras
import os

def wavenet_block(x_in, filters, kernel_size, dilation_rate):
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

def WaveNet_two_channel(input_shape, filters=64, kernel_size=20, dilation_rates=[2**i for i in range(7)]):
    """
    input_shape: (timesteps, 2)  # (WIN, 2)
    Channel order: [:, :, 0] = fetal ECG, [:, :, 1] = maternal ECG
    """
    inp = Input(shape=input_shape, name="ecg_fetal_maternal_2ch")

    #  Dilated 1x1 stem to project 2ch -> filters
    x = layers.DilatedConv1D(filters=filters, kernel_size=1, dilation_rate=1, padding='same')(inp)

    skips = []
    for d in dilation_rates:
        x, s = wavenet_block(x, filters, kernel_size, d)
        skips.append(s)

    out = Add()(skips)
    out = layers.Activation(out, 'relu')  # post-processing: ReLU -> 1x1 -> ReLU -> 1x1
    out = layers.Conv1D(filters, 1)(out)
    out = layers.Activation(out, 'relu')
    out = layers.Conv1D(1, 1)(out)
    out = layers.Activation(out, 'tanh')

    model = keras.Model(inputs=inp, outputs=out, name="WaveNet_two_channel_early_fusion")
    model.summary()
    return model


def create_scalogram(sig, fs=284, time_bins=160, freq_bins=80):
    scales = np.arange(1, freq_bins + 1)
    coeffs, f = pywt.cwt(sig, scales, wavelet='cgau8', sampling_period=1/fs)
    coeffs = np.abs(coeffs)
    return coeffs, f

# process a batch of signals
def create_batch_scalograms(signals_batch, fs=2000, time_bins=160, freq_bins=80):
    coeffs = []
    f_s = []
    for sig in signals_batch:
        coeff, f = create_scalogram(sig, fs, time_bins, freq_bins)
        coeffs.append(coeff)
        f_s.append(f)
    return coeffs, f_s

def plot_ecg_doppler_pairs(ecgs, real_dopplers, generated_dopplers):
    """Plots ECG and corresponding real and generated Doppler pairs."""
    plt.figure(figsize=(18, 8))

    for i, (ecg, real_dopple, generated_dopple) in enumerate(zip(ecgs, real_dopplers, generated_dopplers)):
        # Plotting ECG
        plt.subplot(len(ecgs), 3, 3*i + 1)  # Adjust the number of rows dynamically based on the length of ecgs
        plt.plot(ecg, color='royalblue')
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis

        # Plotting Real Doppler
        plt.subplot(len(ecgs), 3, 3*i + 2)
        plt.plot(real_dopple, color='blue')
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis

        # Plotting Generated Doppler
        plt.subplot(len(ecgs), 3, 3*i + 3)
        plt.plot(generated_dopple, color='red')
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
        plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
        plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis

    plt.tight_layout()
    plt.savefig('WaveNet_beat/plots/signals_test.jpg')
    plt.show()


def plot_scalogram(real, generated, time_bins=160, freq_bins=80):
    plt.figure(figsize=(18, 10))
    coeffs_rs, fs_rs, coeffs_gs, fs_gs = [],[],[],[]
    fs = 284
    t = np.linspace(0, len(real)/fs, time_bins)
    scales = np.arange(1, freq_bins)
    tensor_real=[]
    tensor_generated=[]
    frequencies = fs

    for i in range(len(real)):
        coeffs_r, fs_r=create_scalogram(real[i],fs,time_bins, freq_bins)
        coeffs_g, fs_g=create_scalogram(generated[i],fs,time_bins, freq_bins)
        coeffs_gs.append(coeffs_g)
        coeffs_rs.append(coeffs_r)
        fs_gs.append(fs_g)
        fs_rs.append(fs_r)

    for i in range(len(real)):
        plt.subplot(len(real), 2, 2*i + 1)
        plt.pcolormesh(np.arange(coeffs_rs[i].shape[1]) , fs_rs[i],coeffs_rs[i], shading='gouraud', cmap='bwr')
        plt.yticks([])
        plt.xticks([])
        plt.ylim(10,1000)

        plt.subplot(len(real), 2, 2*i + 2)
        plt.pcolormesh(np.arange(coeffs_gs[i].shape[1]) , fs_gs[i],coeffs_gs[i], shading='gouraud',cmap='bwr')
        plt.xticks([])
        plt.yticks([])
        plt.ylim(10,1000)

    plt.savefig('WaveNet_beat/plots/scalograms_test.jpg')
    plt.show()

def plot_ecg_doppler_overlay(ecgs, real_dopplers, generated_dopplers, labels=None, colors=None):
    """
    Plots ECG, real Doppler, and generated Doppler signals overlaid on the same axes for each sample.

    Parameters:
    - ecgs: list or array of ECG signals
    - real_dopplers: list or array of corresponding real Doppler signals
    - generated_dopplers: list or array of corresponding generated Doppler signals
    - labels: dict mapping 'ecg', 'real', 'gen' to legend labels (optional)
    - colors: dict mapping 'ecg', 'real', 'gen' to color strings (optional)
    """
    labels = labels or {'ecg': 'ECG', 'real': 'Real Doppler', 'gen': 'Generated Doppler'}
    colors = colors or {'ecg': 'royalblue', 'real': 'blue', 'gen': 'red'}

    for i, (ecg, real_dop, gen_dop) in enumerate(zip(ecgs, real_dopplers, generated_dopplers), start=1):
        plt.figure(figsize=(10, 4))
        plt.plot(ecg, label=labels['ecg'], color=colors['ecg'], linewidth=1)
        plt.plot(real_dop, label=labels['real'], color=colors['real'], linewidth=1)
        plt.plot(gen_dop, label=labels['gen'], color=colors['gen'], linewidth=1, linestyle='--')

        plt.title(f'Sample {i}')
        plt.legend(loc='upper right')
        plt.xlabel('Time / Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'WaveNet_beat/plots/signals_test_{i}.jpg')
        plt.show()

# Plot training and validation loss
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
    plt.show()
    
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
    plt.show()
    
    print(f"Detailed training history plot saved to: {plot_path}")

def plot_ecg_doppler_pairs(ecgs, real_dopplers, generated_dopplers):
    """Plots ECG and corresponding real and generated Doppler pairs."""
    
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
            plt.subplot(len(ecgs), 3, 3*i + 1)  # Adjust the number of rows dynamically based on the length of ecgs
            plt.plot(ecg, color='royalblue')
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
            plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis

            # Plotting Real Doppler
            plt.subplot(len(ecgs), 3, 3*i + 2)
            plt.plot(real_dopple, color='blue')
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
            plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis

            # Plotting Generated Doppler
            plt.subplot(len(ecgs), 3, 3*i + 3)
            plt.plot(generated_dopple, color='red')
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.axhline(y=0, color='gray', linewidth=1.5, zorder=1)  # x-axis
            plt.axvline(x=0, color='gray', linewidth=1.5, zorder=1)  # y-axis
    
    plt.tight_layout()
    plt.savefig('WaveNet_beat/plots/signals_test.jpg')
    plt.show()

def plot_ecg_doppler_overlay_multi(ecgs, real_dopplers, generated_dopplers, labels=None, colors=None, save_dir='WaveNet_beat/plots', prefix='signals_overlay'):
    """
    Plots ECG, real Doppler, and generated Doppler signals overlaid on the same axes for each sample.
    Handles both single-channel and multi-channel (Fetal/Maternal) ECGs.
    
    Parameters:
    - ecgs: array of ECG signals (shape: (n_samples, time_steps, n_channels))
    - real_dopplers: array of corresponding real Doppler signals
    - generated_dopplers: array of corresponding generated Doppler signals
    - labels: dict mapping signal types to legend labels (optional)
    - colors: dict mapping signal types to color strings (optional)
    - save_dir: directory to save plots (default: 'WaveNet_beat/plots')
    - prefix: prefix for saved filenames (default: 'signals_overlay')
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
# train_wavenet.py
import os
import numpy as np

import sys
print("Script Python:", sys.executable)
print("sys.path:", sys.path)
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import modules

def create_composite_loss(alpha_peak=0.5, alpha_ratio=0.5, base_loss='mae'):
    """
    Create composite loss function for WaveNet training.
    
    Parameters:
    -----------
    alpha_peak : float
        Weight for peak amplitude preservation loss
    alpha_ratio : float
        Weight for peak-to-trough ratio preservation loss
    base_loss : str
        Base loss function: 'mae' or 'mse'
    
    Returns:
    --------
    loss_fn : callable
        Composite loss function
    """
    def composite_loss(y_true, y_pred):
        """
        Combined loss ensuring waveform shape preservation.
        
        y_true, y_pred shape: (batch_size, sequence_length) or (batch_size, sequence_length, 1)
        """
        # Squeeze if necessary (remove last dim if it's 1)
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
            y_pred = tf.squeeze(y_pred, axis=-1)
        
        # 1. Base reconstruction loss (MAE or MSE)
        if base_loss == 'mae':
            base = tf.reduce_mean(tf.abs(y_true - y_pred))
        else:  # mse
            base = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # 2. Peak amplitude preservation loss
        peak_true = tf.reduce_max(y_true, axis=1)  # (batch_size,)
        peak_pred = tf.reduce_max(y_pred, axis=1)  # (batch_size,)
        peak_loss = tf.reduce_mean(tf.abs(peak_true - peak_pred))
        
        # 3. Peak-to-trough ratio loss (dynamic range preservation)
        trough_true = tf.reduce_min(y_true, axis=1)  # (batch_size,)
        trough_pred = tf.reduce_min(y_pred, axis=1)  # (batch_size,)
        
        ratio_true = peak_true - trough_true  # (batch_size,)
        ratio_pred = peak_pred - trough_pred  # (batch_size,)
        ratio_loss = tf.reduce_mean(tf.abs(ratio_true - ratio_pred))
        
        # 4. Combined loss
        total = base + alpha_peak * peak_loss + alpha_ratio * ratio_loss
        
        return total
    
    return composite_loss


def create_shape_preserving_loss(alpha_peak=0.5, alpha_ratio=0.5, alpha_derivative=0.3, base_loss='mae'):
    """
    Enhanced composite loss with derivative matching for better shape preservation.
    """
    def shape_loss(y_true, y_pred):
        # Squeeze if necessary
        if len(y_true.shape) == 3:
            y_true = tf.squeeze(y_true, axis=-1)
            y_pred = tf.squeeze(y_pred, axis=-1)
        
        # 1. Base reconstruction loss
        if base_loss == 'mae':
            base = tf.reduce_mean(tf.abs(y_true - y_pred))
        else:
            base = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # 2. Peak amplitude preservation
        peak_true = tf.reduce_max(y_true, axis=1)
        peak_pred = tf.reduce_max(y_pred, axis=1)
        peak_loss = tf.reduce_mean(tf.abs(peak_true - peak_pred))
        
        # 3. Peak-to-trough ratio (dynamic range)
        trough_true = tf.reduce_min(y_true, axis=1)
        trough_pred = tf.reduce_min(y_pred, axis=1)
        ratio_true = peak_true - trough_true
        ratio_pred = peak_pred - trough_pred
        ratio_loss = tf.reduce_mean(tf.abs(ratio_true - ratio_pred))
        
        # 4. Derivative matching (shape/slope preservation)
        dy_true = y_true[:, 1:] - y_true[:, :-1]  # (batch, seq-1)
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]  # (batch, seq-1)
        derivative_loss = tf.reduce_mean(tf.abs(dy_true - dy_pred))
        
        # Combined loss
        total = base + alpha_peak * peak_loss + alpha_ratio * ratio_loss + alpha_derivative * derivative_loss
        
        return total
    
    return shape_loss

def main(
    data_dir="./WaveNet_beat/data",
    X_file="X.npy",
    Y_file="Y.npy",
    patient_ids_file="PATIENT_IDS.npy",
    train_idx_file="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_train.npy",
    val_idx_file="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_val.npy",
    save_dir_model="./WaveNet_beat/comp_models",
    save_dir_loss="./WaveNet_beat/comp_logs",
    save_dir_plots="./WaveNet_beat/comp_plots",
    n_epochs=10,
    batch_size=32,
    lr=1e-3,
    use_lr_schedule=False,
    loss_type='composite',  # NEW: 'mae', 'mse', 'composite', or 'shape_preserving'
    alpha_peak=0.5,         # NEW: weight for peak loss
    alpha_ratio=0.5,        # NEW: weight for ratio loss
    alpha_derivative=0.3    # NEW: weight for derivative loss
):
    os.makedirs(save_dir_model, exist_ok=True)
    os.makedirs(save_dir_loss, exist_ok=True)
    os.makedirs(save_dir_plots, exist_ok=True)
    
    print("Loading data...")
    X = np.load(os.path.join(data_dir, X_file))
    Y = np.load(os.path.join(data_dir, Y_file))
    PATIENT_IDS = np.load(os.path.join(data_dir, patient_ids_file))
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Number of patients: {len(np.unique(PATIENT_IDS))}")
    
    print("Loading train/val indices...")
    train_idx = np.load(train_idx_file)
    val_idx = np.load(val_idx_file)
    
    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Verify no patient leakage
    assert set(PATIENT_IDS[train_idx]).isdisjoint(set(PATIENT_IDS[val_idx])), "Patient leakage detected!"
    print("✓ No patient leakage detected")
    
    # Build model
    latent_dim = X.shape[1]
    model = modules.WaveNet_two_channel(input_shape=(latent_dim, 2))
    
    # Choose loss function
    if loss_type == 'composite':
        loss_fn = create_composite_loss(alpha_peak=alpha_peak, alpha_ratio=alpha_ratio, base_loss='mae')
        print(f"✓ Using composite loss: MAE + {alpha_peak}*peak + {alpha_ratio}*ratio")
    elif loss_type == 'shape_preserving':
        loss_fn = create_shape_preserving_loss(
            alpha_peak=alpha_peak, 
            alpha_ratio=alpha_ratio, 
            alpha_derivative=alpha_derivative,
            base_loss='mae'
        )
        print(f"✓ Using shape-preserving loss: MAE + {alpha_peak}*peak + {alpha_ratio}*ratio + {alpha_derivative}*derivative")
    elif loss_type == 'mse':
        loss_fn = 'mse'
        print("✓ Using MSE loss")
    else:  # 'mae'
        loss_fn = 'mae'
        print("✓ Using MAE loss")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    # Setup callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_dir_model, "best_comp_fusion_model.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1,
    )
    callbacks = [checkpoint_cb]
    
    if use_lr_schedule == 'fixed':
        def scheduler(epoch, current_lr):
            if epoch > 0 and epoch % 5 == 0:
                return current_lr * 0.5
            return current_lr
        callbacks.append(LearningRateScheduler(scheduler, verbose=1))
    elif use_lr_schedule == 'dynamic':
        lr_factor = 0.5
        lr_patience = 3
        lr_min = 1e-5
        reduce_lr_cb = ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=lr_min,
            verbose=1
        )
        callbacks.append(reduce_lr_cb)
        print(f"✓ Using ReduceLROnPlateau: factor={lr_factor}, patience={lr_patience}, min_lr={lr_min}")
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X[train_idx], Y[train_idx],
        validation_data=(X[val_idx], Y[val_idx]),
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save losses and other info (rest remains the same)
    np.save(os.path.join(save_dir_loss, "train_loss.npy"), history.history["loss"])
    np.save(os.path.join(save_dir_loss, "val_loss.npy"), history.history.get("val_loss", []))
    modules.plot_training_history(history, save_dir_plots, split_name="low_lr")

    if "lr" in history.history:
        np.save(os.path.join(save_dir_loss, "learning_rate.npy"), history.history["lr"])    
    
    model.save_weights(os.path.join(save_dir_model, "final_composite_model.weights.h5"))
    
    # Save training summary
    train_pats = np.unique(PATIENT_IDS[train_idx])
    val_pats = np.unique(PATIENT_IDS[val_idx])
    
    fold_info = {
        "train_patients": train_pats.size,
        "val_patients": val_pats.size,
        "train_segments": train_idx.size,
        "val_segments": val_idx.size,
        "best_val_loss": float(np.min(history.history.get("val_loss", [np.inf]))),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_lr": float(history.history.get("lr", [lr])[-1]) if "lr" in history.history else lr,
        "loss_type": loss_type,  # NEW: track loss type
        "alpha_peak": alpha_peak if loss_type != 'mae' and loss_type != 'mse' else None,
        "alpha_ratio": alpha_ratio if loss_type != 'mae' and loss_type != 'mse' else None,
    }
    
    df = pd.DataFrame([fold_info])
    df.to_csv(os.path.join(save_dir_loss, "training_summary.csv"), index=False)
    print("\nTraining Summary:")
    print(df.to_string(index=False))
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train WaveNet model")
    parser.add_argument("--data_dir", type=str, default="./WaveNet_beat/data", help="Directory containing data files")
    parser.add_argument("--X_file", type=str, default="X.npy", help="Input features file")
    parser.add_argument("--Y_file", type=str, default="Y.npy", help="Target labels file")
    parser.add_argument("--patient_ids_file", type=str, default="PATIENT_IDS.npy", help="Patient IDs file")
    parser.add_argument("--train_idx_file", type=str, default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_train.npy", help="Train indices file")
    parser.add_argument("--val_idx_file", type=str, default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_val.npy", help="Validation indices file")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--use_lr_schedule", type=str, help="Use learning rate scheduler")
    
    # NEW ARGUMENTS FOR LOSS FUNCTION
    parser.add_argument("--loss_type", type=str, default='mae', 
                       choices=['mae', 'mse', 'composite', 'shape_preserving'],
                       help="Loss function type")
    parser.add_argument("--alpha_peak", type=float, default=0.5, 
                       help="Weight for peak amplitude loss")
    parser.add_argument("--alpha_ratio", type=float, default=0.5, 
                       help="Weight for peak-to-trough ratio loss")
    parser.add_argument("--alpha_derivative", type=float, default=0.3, 
                       help="Weight for derivative loss (shape_preserving only)")
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        X_file=args.X_file,
        Y_file=args.Y_file,
        patient_ids_file=args.patient_ids_file,
        train_idx_file=args.train_idx_file,
        val_idx_file=args.val_idx_file,
        lr=args.lr,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        use_lr_schedule=args.use_lr_schedule,
        loss_type=args.loss_type,
        alpha_peak=args.alpha_peak,
        alpha_ratio=args.alpha_ratio,
        alpha_derivative=args.alpha_derivative,
    )

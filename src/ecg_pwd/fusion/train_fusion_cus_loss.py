"""Most flexible fusion training entry point; supports configurable multi-term objectives (MAE/MSE, derivative, Soft-DTW, correlation, MR-STFT) and overlay visualizations."""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

print("Python executable:", sys.executable)
print("sys.path:", sys.path)

from . import modules


def main(
    data_dir="./WaveNet_beat/data",
    X_file="X.npy",
    Y_file="Y.npy",
    patient_ids_file="PATIENT_IDS.npy",
    train_idx_file="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_train.npy",
    val_idx_file="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_val.npy",
    save_dir_model="./WaveNet_beat/loss_trial_models",
    save_dir_loss="./WaveNet_beat/loss_trial_logs",
    save_dir_plots="./WaveNet_beat/loss_trial_plots",
    n_epochs=10,
    batch_size=32,
    lr=1e-3,
    use_lr_schedule=None,
    loss_type='mae',
    # Individual component flags
    use_base=True,
    use_peak=False,
    use_ratio=False,
    use_derivative=False,
    use_softdtw=False,
    use_corr=False,
    use_mrstft=False,
    # Component weights
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
    Main training function for WaveNet model.
    
    Args:
        data_dir: Directory containing data files
        X_file: Filename for input ECG data
        Y_file: Filename for target Doppler data
        patient_ids_file: Filename for patient IDs
        train_idx_file: Path to training indices
        val_idx_file: Path to validation indices
        save_dir_model: Directory to save model weights
        save_dir_loss: Directory to save loss logs
        save_dir_plots: Directory to save plots
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        use_lr_schedule: Learning rate schedule ('fixed', 'dynamic', or None)
        loss_type: Type of loss function ('mae', 'mse', 'composite', 'shape_preserving', 'flexible')
        use_*: Flags to enable/disable specific loss components
        alpha_*: Weights for each loss component
        base_loss: Base loss type ('mae' or 'mse')
        dtw_gamma: Temperature parameter for Soft-DTW
        
    Returns:
        model: Trained Keras model
        history: Training history object
    """
    # Create directories
    os.makedirs(save_dir_model, exist_ok=True)
    os.makedirs(save_dir_loss, exist_ok=True)
    os.makedirs(save_dir_plots, exist_ok=True)
    
    # Load data
    print("="*80)
    print("Loading data...")
    print("="*80)
    X = np.load(os.path.join(data_dir, X_file))
    Y = np.load(os.path.join(data_dir, Y_file))
    PATIENT_IDS = np.load(os.path.join(data_dir, patient_ids_file))
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Number of patients: {len(np.unique(PATIENT_IDS))}")
    
    # Load train/val splits
    print("\nLoading train/val indices...")
    train_idx = np.load(train_idx_file)
    val_idx = np.load(val_idx_file)
    
    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Verify no patient leakage
    train_patients = set(PATIENT_IDS[train_idx])
    val_patients = set(PATIENT_IDS[val_idx])
    assert train_patients.isdisjoint(val_patients), "Patient leakage detected!"
    print("✓ No patient leakage detected")
    print(f"  Train patients: {len(train_patients)}")
    print(f"  Val patients: {len(val_patients)}")
    
    # Build model
    print("\n" + "="*80)
    print("Building WaveNet model...")
    print("="*80)
    latent_dim = X.shape[1]
    #model = modules.WaveNet_two_channel(input_shape=(latent_dim, 2))
    #model = modules.WaveNet_minimal_cross_attention(input_shape=(latent_dim, 2))
    #model = modules.WaveNet_two_channel_cross_attention(input_shape=(latent_dim, 2))
    model = modules.WaveNet_two_channel_combined_attention(input_shape=(latent_dim, 2))
    
    # Choose loss function
    print("\n" + "="*80)
    print("Configuring loss function...")
    print("="*80)
    
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
        use_base, use_peak, use_ratio, use_derivative = True, True, True, False
        use_softdtw, use_corr, use_mrstft = False, False, False
        print(f"✓ Using composite loss: {base_loss.upper()} + {alpha_peak}*peak + {alpha_ratio}*ratio")
        
    elif loss_type == 'shape_preserving':
        loss_fn = modules.create_shape_preserving_loss(
            alpha_peak=alpha_peak, 
            alpha_ratio=alpha_ratio, 
            alpha_derivative=alpha_derivative,
            base_loss=base_loss
        )
        use_base, use_peak, use_ratio, use_derivative = True, True, True, True
        use_softdtw, use_corr, use_mrstft = False, False, False
        print(f"✓ Using shape-preserving loss: {base_loss.upper()} + {alpha_peak}*peak + {alpha_ratio}*ratio + {alpha_derivative}*derivative")
        
    elif loss_type == 'mse':
        loss_fn = 'mse'
        use_base, use_peak, use_ratio, use_derivative = True, False, False, False
        use_softdtw, use_corr, use_mrstft = False, False, False
        base_loss = 'mse'
        print("✓ Using MSE loss")
        
    else:  # 'mae'
        loss_fn = 'mae'
        use_base, use_peak, use_ratio, use_derivative = True, False, False, False
        use_softdtw, use_corr, use_mrstft = False, False, False
        base_loss = 'mae'
        print("✓ Using MAE loss")
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss_fn)
    print(f"✓ Model compiled with learning rate: {lr}")
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_dir_model, "best_model.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1,
    )
    callbacks = [checkpoint_cb]
    print("✓ ModelCheckpoint callback added")
    
    # Add loss component logger
    loss_logger = modules.LossComponentLogger(
        X[train_idx], Y[train_idx],
        X[val_idx], Y[val_idx],
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
    callbacks.append(loss_logger)
    print("✓ Loss component logging enabled")
    
    # Add learning rate schedule if requested
    if use_lr_schedule == 'fixed':
        def scheduler(epoch, current_lr):
            if epoch > 0 and epoch % 5 == 0:
                return current_lr * 0.5
            return current_lr
        callbacks.append(LearningRateScheduler(scheduler, verbose=1))
        print("✓ Using fixed LR schedule (decay every 5 epochs)")
    elif use_lr_schedule == 'dynamic':
        reduce_lr_cb = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
        callbacks.append(reduce_lr_cb)
        print("✓ Using ReduceLROnPlateau (patience=3, factor=0.5)")
    
    # Train model
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    history = model.fit(
        X[train_idx], Y[train_idx],
        validation_data=(X[val_idx], Y[val_idx]),
        epochs=n_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    # Save losses
    print("\nSaving training logs...")
    np.save(os.path.join(save_dir_loss, "train_loss.npy"), history.history["loss"])
    np.save(os.path.join(save_dir_loss, "val_loss.npy"), history.history.get("val_loss", []))
    
    # Save individual loss components
    if loss_logger is not None:
        for key, values in loss_logger.history.items():
            if values and any(v is not None for v in values):  # Only save if has data
                np.save(os.path.join(save_dir_loss, f"{key}_loss.npy"), np.array(values))
        print(f"✓ Saved individual loss components to {save_dir_loss}")
        
        # Plot loss components
        modules.plot_all_loss_components(loss_logger, save_dir_plots, n_epochs)
    
    # Plot training history
    modules.plot_training_history(history, save_dir_plots, split_name="training")
    modules.plot_detailed_training_history(history, save_dir_plots, split_name="training")
    
    if "lr" in history.history:
        np.save(os.path.join(save_dir_loss, "learning_rate.npy"), history.history["lr"])
    
    # Save final model weights
    model.save_weights(os.path.join(save_dir_model, "final_model.weights.h5"))
    print(f"✓ Saved final model weights to {save_dir_model}")
    
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
        "loss_type": loss_type,
        "use_base": use_base,
        "use_peak": use_peak,
        "use_ratio": use_ratio,
        "use_derivative": use_derivative,
        "use_softdtw": use_softdtw,
        "use_corr": use_corr,
        "use_mrstft": use_mrstft,
        "base_loss": base_loss,
    }
    
    df = pd.DataFrame([fold_info])
    df.to_csv(os.path.join(save_dir_loss, "training_summary.csv"), index=False)
    print("\nTraining Summary:")
    print(df.to_string(index=False))
    
    # Generate visualization plots
    print("\n" + "="*80)
    print("Generating visualization plots for BEST MODEL")
    print("="*80)
    
    model.load_weights(os.path.join(save_dir_model, "best_model.weights.h5"))
    print("✓ Loaded best model weights")
    
    # Select random samples for visualization
    np.random.seed(33)
    n_samples = min(3, len(val_idx))
    random_indices = np.random.choice(val_idx, size=n_samples, replace=False)
    print(f"✓ Selected {n_samples} random validation samples: {random_indices}")
    
    ecgs = X[random_indices]
    real_dopplers = Y[random_indices]
    
    # Generate predictions
    generated_dopplers = model.predict(ecgs, verbose=0)
    print(f"✓ Generated predictions with shape: {generated_dopplers.shape}")
    
    # Squeeze if necessary
    if real_dopplers.ndim == 3 and real_dopplers.shape[-1] == 1:
        real_dopplers = real_dopplers.squeeze(axis=-1)
    if generated_dopplers.ndim == 3 and generated_dopplers.shape[-1] == 1:
        generated_dopplers = generated_dopplers.squeeze(axis=-1)
    
    # Create overlay plots
    plots_overlay_dir = os.path.join(save_dir_plots, 'overlays')
    os.makedirs(plots_overlay_dir, exist_ok=True)
    
    modules.plot_ecg_doppler_overlay_multi(
        ecgs=ecgs,
        real_dopplers=real_dopplers,
        generated_dopplers=generated_dopplers,
        save_dir=plots_overlay_dir,
        prefix='best_model'
    )
    
    print(f"✓ Saved best model overlay plots to: {plots_overlay_dir}")
    
    # Generate plots for final model
    print("\n" + "-"*80)
    print("Generating plots for FINAL MODEL")
    print("-"*80)
    
    model.load_weights(os.path.join(save_dir_model, "final_model.weights.h5"))
    print("✓ Loaded final model weights")
    
    generated_dopplers_final = model.predict(ecgs, verbose=0)
    print(f"✓ Generated predictions with shape: {generated_dopplers_final.shape}")
    
    if generated_dopplers_final.ndim == 3 and generated_dopplers_final.shape[-1] == 1:
        generated_dopplers_final = generated_dopplers_final.squeeze(axis=-1)
    
    modules.plot_ecg_doppler_overlay_multi(
        ecgs=ecgs,
        real_dopplers=real_dopplers,
        generated_dopplers=generated_dopplers_final,
        save_dir=plots_overlay_dir,
        prefix='final_model'
    )
    
    print(f"✓ Saved final model overlay plots to: {plots_overlay_dir}")
    print("="*80 + "\n")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train WaveNet model with flexible loss components",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="./WaveNet_beat/data",
                       help="Directory containing data files")
    parser.add_argument("--X_file", type=str, default="X.npy",
                       help="Filename for input ECG data")
    parser.add_argument("--Y_file", type=str, default="Y.npy",
                       help="Filename for target Doppler data")
    parser.add_argument("--patient_ids_file", type=str, default="PATIENT_IDS.npy",
                       help="Filename for patient IDs")
    parser.add_argument("--train_idx_file", type=str, 
                       default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_train_mix_risk.npy",
                       help="Path to training indices")
    parser.add_argument("--val_idx_file", type=str, 
                       default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_val_mix_risk.npy",
                       help="Path to validation indices")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--use_lr_schedule", type=str, choices=['fixed', 'dynamic', 'none'], default='none',
                       help="Learning rate schedule type")
    
    # Loss function configuration
    parser.add_argument("--loss_type", type=str, default='mae', 
                       choices=['mae', 'mse', 'composite', 'shape_preserving', 'flexible'],
                       help="Type of loss function")
    
    # Loss component flags
    parser.add_argument("--use_base", type=lambda x: x.lower() == 'true', default=True,
                       help="Enable base MAE/MSE loss")
    parser.add_argument("--use_peak", type=lambda x: x.lower() == 'true', default=False,
                       help="Enable peak amplitude preservation")
    parser.add_argument("--use_ratio", type=lambda x: x.lower() == 'true', default=False,
                       help="Enable peak-to-trough ratio preservation")
    parser.add_argument("--use_derivative", type=lambda x: x.lower() == 'true', default=False,
                       help="Enable derivative matching")
    parser.add_argument("--use_softdtw", type=lambda x: x.lower() == 'true', default=False,
                       help="Enable Soft-DTW loss")
    parser.add_argument("--use_corr", type=lambda x: x.lower() == 'true', default=False,
                       help="Enable cross-correlation loss")
    parser.add_argument("--use_mrstft", type=lambda x: x.lower() == 'true', default=False,
                       help="Enable multi-resolution STFT loss")
    
    # Loss component weights
    parser.add_argument("--alpha_peak", type=float, default=0.5,
                       help="Weight for peak amplitude loss")
    parser.add_argument("--alpha_ratio", type=float, default=0.5,
                       help="Weight for peak-to-trough ratio loss")
    parser.add_argument("--alpha_derivative", type=float, default=0.3,
                       help="Weight for derivative matching loss")
    parser.add_argument("--alpha_softdtw", type=float, default=1.0,
                       help="Weight for Soft-DTW loss")
    parser.add_argument("--alpha_corr", type=float, default=1.0,
                       help="Weight for cross-correlation loss")
    parser.add_argument("--alpha_mrstft", type=float, default=1.0,
                       help="Weight for MR-STFT loss")
    
    # Other loss parameters
    parser.add_argument("--base_loss", type=str, default='mae', choices=['mae', 'mse'],
                       help="Base loss type")
    parser.add_argument("--dtw_gamma", type=float, default=0.1,
                       help="Temperature parameter for Soft-DTW")
    
    args = parser.parse_args()
    
    # Convert 'none' to None for use_lr_schedule
    use_lr_schedule = None if args.use_lr_schedule == 'none' else args.use_lr_schedule
    
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
        use_lr_schedule=use_lr_schedule,
        loss_type=args.loss_type,
        use_base=args.use_base,
        use_peak=args.use_peak,
        use_ratio=args.use_ratio,
        use_derivative=args.use_derivative,
        use_softdtw=args.use_softdtw,
        use_corr=args.use_corr,
        use_mrstft=args.use_mrstft,
        alpha_peak=args.alpha_peak,
        alpha_ratio=args.alpha_ratio,
        alpha_derivative=args.alpha_derivative,
        alpha_softdtw=args.alpha_softdtw,
        alpha_corr=args.alpha_corr,
        alpha_mrstft=args.alpha_mrstft,
        base_loss=args.base_loss,
        dtw_gamma=args.dtw_gamma,
    )


    #python train_fusion_cus_loss.py --loss_type flexible --use_corr true --use_softdtw true --use_mrstft true --alpha_corr 0.3 --alpha_softdtw 0.3 --alpha_mrstft 0.3 --epochs 100 --batch_size 32
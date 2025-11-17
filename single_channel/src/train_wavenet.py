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


def main(
    data_dir="./WaveNet_beat/data",
    X_file="X.npy",
    Y_file="Y.npy",
    patient_ids_file="PATIENT_IDS.npy",
    risk_groups_file="RISK_GROUPS.npy",
    train_idx_file="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_train_risk.npy",
    val_idx_file="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_val_risk_normal.npy",
    save_dir_model="./WaveNet_beat/low_risk_models",
    save_dir_loss="./WaveNet_beat/low_risk_logs",
    save_dir_plots="./WaveNet_beat/low_risk_plots",
    n_epochs=10,
    batch_size=32,
    lr=1e-3,
    use_lr_schedule=False
):
    os.makedirs(save_dir_model, exist_ok=True)
    os.makedirs(save_dir_loss, exist_ok=True)
    os.makedirs(save_dir_plots, exist_ok=True)
    
    print("Loading data...")
    X = np.load(os.path.join(data_dir, X_file))
    Y = np.load(os.path.join(data_dir, Y_file))
    PATIENT_IDS = np.load(os.path.join(data_dir, patient_ids_file))
    
    # ADD: Optional loading of risk groups for verification/logging
    risk_groups_path = os.path.join(data_dir, risk_groups_file)
    if os.path.exists(risk_groups_path):
        RISK_GROUPS = np.load(risk_groups_path)
        print(f"Risk groups loaded: {len(RISK_GROUPS)} entries")
    else:
        RISK_GROUPS = None
        print("Risk groups file not found, skipping risk verification")
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Number of patients: {len(np.unique(PATIENT_IDS))}")
    
    print("Loading train/val indices...")
    train_idx = np.load(train_idx_file)
    val_idx = np.load(val_idx_file)
    
    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # ADD: Verify risk group composition if available
    if RISK_GROUPS is not None:
        train_normal = np.sum(RISK_GROUPS[train_idx] != 'yes')
        train_high_risk = np.sum(RISK_GROUPS[train_idx] == 'yes')
        val_normal = np.sum(RISK_GROUPS[val_idx] != 'yes')
        val_high_risk = np.sum(RISK_GROUPS[val_idx] == 'yes')
        
        print(f"\nRisk group distribution:")
        print(f"  Training: {train_normal} normal, {train_high_risk} high-risk")
        print(f"  Validation: {val_normal} normal, {val_high_risk} high-risk")
        
        if train_high_risk > 0:
            print("WARNING: High-risk patients found in training set!")
    
    # Verify no patient leakage
    assert set(PATIENT_IDS[train_idx]).isdisjoint(set(PATIENT_IDS[val_idx])), "Patient leakage detected!"
    print("✓ No patient leakage detected")
    
    # Build model
    latent_dim = X.shape[1]
    model = modules.WaveNet_v2(input_shape=(latent_dim, 1))

    # model = modules.WaveNet_v2(input_shape=(latent_dim, 1), filters=32, kernel_size=15, 
    #               dilation_rates=[2**i for i in range(6)])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mae")
    
    # Setup callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_dir_model, "best_single_model.weights.h5"),
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
    
    # Save losses
    np.save(os.path.join(save_dir_loss, "train_loss.npy"), history.history["loss"])
    np.save(os.path.join(save_dir_loss, "val_loss.npy"), history.history.get("val_loss", []))
    modules.plot_training_history(history, save_dir_plots, split_name="low_lr")

    if "lr" in history.history:
        np.save(os.path.join(save_dir_loss, "learning_rate.npy"), history.history["lr"])    
    
    # Save final weights
    model.save_weights(os.path.join(save_dir_model, "final_model.weights.h5"))
    
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
        "train_idx_file": train_idx_file,  # ADD: Track which indices were used
        "val_idx_file": val_idx_file  # ADD: Track which indices were used
    }
    
    # ADD: Include risk group info if available
    if RISK_GROUPS is not None:
        fold_info["train_normal_segments"] = int(train_normal)
        fold_info["train_high_risk_segments"] = int(train_high_risk)
        fold_info["val_normal_segments"] = int(val_normal)
        fold_info["val_high_risk_segments"] = int(val_high_risk)
    
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
    parser.add_argument("--risk_groups_file", type=str, default="/home/tsu25/ECG_PWD/single_channel/src/WaveNet_beat/data/RISK_GROUPS.npy", help="Risk groups file") 
    parser.add_argument("--train_idx_file", type=str, default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_train_risk.npy", help="Train indices file")
    parser.add_argument("--val_idx_file", type=str, default="/home/tsu25/ECG_PWD/Fetal-maternal-fusion/src/idx_val_risk_normal.npy", help="Validation indices file")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--use_lr_schedule", type=str, help="Use learning rate scheduler")
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        X_file=args.X_file,
        Y_file=args.Y_file,
        patient_ids_file=args.patient_ids_file,
        risk_groups_file=args.risk_groups_file,
        train_idx_file=args.train_idx_file,
        val_idx_file=args.val_idx_file,
        lr=args.lr,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        use_lr_schedule=args.use_lr_schedule,
    )


    #/home/tsu25/miniconda3/envs/fetal_maternal/bin/python train_wavenet.py --lr 0.0001 --epochs 50
    # python train_fusion_wavenet.py --use_lr_schedule dynamic --epochs 50 --batch_size 32
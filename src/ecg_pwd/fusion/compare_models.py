"""Experiment runner for side-by-side model configuration comparisons (architectures, losses, learning rates) built on the k-fold evaluation framework."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json

print("Python executable:", sys.executable)
print("sys.path:", sys.path)

from . import modules
from .evaluate_wavenet_kfold import evaluate_model_kfold


def compare_models(
    X, Y, patient_ids,
    model_configs,
    n_folds=5,
    n_epochs=50,
    batch_size=32,
    base_save_dir="./model_comparison",
    apply_smoothing=True,
    smoothing_params=None,
    sampling_rate=284,
    early_stopping_patience=10
):
    """
    Compare multiple model configurations.
    
    Parameters:
    -----------
    X, Y, patient_ids : np.ndarray
        Data arrays
    model_configs : list of dict
        List of model configurations, each containing:
        - 'name': str, model name
        - 'builder': callable, function returning model
        - 'lr': float, learning rate (optional, default 1e-3)
    n_folds : int
        Number of CV folds
    n_epochs : int
        Max epochs per fold
    batch_size : int
        Batch size
    base_save_dir : str
        Base directory for saving results
    apply_smoothing : bool
        Whether to smooth generated signals
    smoothing_params : dict
        Smoothing parameters
    sampling_rate : int
        Sampling rate
    early_stopping_patience : int
        Early stopping patience
        
    Returns:
    --------
    comparison_ml : pd.DataFrame
        Combined ML metrics for all models
    comparison_clinical : pd.DataFrame
        Combined clinical metrics for all models
    """
    
    os.makedirs(base_save_dir, exist_ok=True)
    
    all_ml_summaries = []
    all_clinical_summaries = []
    
    print("="*80)
    print(f"COMPARING {len(model_configs)} MODEL CONFIGURATIONS")
    print("="*80)
    for i, config in enumerate(model_configs):
        print(f"{i+1}. {config['name']}")
    print("="*80 + "\n")
    
    for config_idx, config in enumerate(model_configs):
        model_name = config['name']
        model_builder = config['builder']
        lr = config.get('lr', 1e-3)
        
        print(f"\n{'#'*80}")
        print(f"# EVALUATING MODEL {config_idx + 1}/{len(model_configs)}: {model_name}")
        print(f"{'#'*80}\n")
        
        model_save_dir = os.path.join(base_save_dir, model_name.replace(' ', '_'))
        
        ml_summary, clinical_summary = evaluate_model_kfold(
            X=X,
            Y=Y,
            patient_ids=patient_ids,
            model_builder_fn=model_builder,
            n_folds=n_folds,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            save_dir=model_save_dir,
            model_name=model_name,
            apply_smoothing=apply_smoothing,
            smoothing_params=smoothing_params,
            sampling_rate=sampling_rate,
            early_stopping_patience=early_stopping_patience
        )
        
        # Add model name to summaries
        ml_summary['Model'] = model_name
        clinical_summary['Model'] = model_name
        
        all_ml_summaries.append(ml_summary)
        all_clinical_summaries.append(clinical_summary)
    
    # Combine all results
    comparison_ml = pd.concat(all_ml_summaries, ignore_index=True)
    comparison_clinical = pd.concat(all_clinical_summaries, ignore_index=True)
    
    # Reorder columns to put Model first
    ml_cols = ['Model'] + [col for col in comparison_ml.columns if col != 'Model']
    clinical_cols = ['Model'] + [col for col in comparison_clinical.columns if col != 'Model']
    
    comparison_ml = comparison_ml[ml_cols]
    comparison_clinical = comparison_clinical[clinical_cols]
    
    # Save combined results
    comparison_ml.to_csv(os.path.join(base_save_dir, "all_models_ml_metrics.csv"), index=False)
    comparison_clinical.to_csv(os.path.join(base_save_dir, "all_models_clinical_metrics.csv"), index=False)
    
    # Print comparison tables
    print("\n" + "="*100)
    print("FINAL COMPARISON: ML/SIGNAL METRICS")
    print("="*100)
    
    # Create pivot table for key ML metrics
    key_ml_metrics = ['MAE', 'RMSE', 'Pearson_Correlation', 'DTW', 'PSD_Correlation']
    ml_pivot = comparison_ml[comparison_ml['Metric'].isin(key_ml_metrics)]
    ml_pivot = ml_pivot[['Model', 'Metric', 'Format']]
    ml_pivot_wide = ml_pivot.pivot(index='Metric', columns='Model', values='Format')
    print(ml_pivot_wide.to_string())
    
    print("\n" + "="*100)
    print("FINAL COMPARISON: CLINICAL METRICS (MAE)")
    print("="*100)
    
    # Create pivot table for clinical metrics
    clinical_pivot = comparison_clinical[['Model', 'Metric', 'MAE_Format']]
    clinical_pivot_wide = clinical_pivot.pivot(index='Metric', columns='Model', values='MAE_Format')
    print(clinical_pivot_wide.to_string())
    
    print("\n" + "="*100)
    print("FINAL COMPARISON: CLINICAL METRICS (Correlation)")
    print("="*100)
    
    clinical_corr_pivot = comparison_clinical[['Model', 'Metric', 'Correlation_Format']]
    clinical_corr_pivot_wide = clinical_corr_pivot.pivot(index='Metric', columns='Model', values='Correlation_Format')
    print(clinical_corr_pivot_wide.to_string())
    
    print("\n" + "="*100)
    print(f"All results saved to: {base_save_dir}")
    print("="*100 + "\n")
    
    return comparison_ml, comparison_clinical


# ============================================================================
# EXAMPLE: Compare Different Architectures
# ============================================================================

def main():
    """Example comparing multiple architectures"""
    
    # Load data
    data_dir = "./WaveNet_beat/data"
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))
    PATIENT_IDS = np.load(os.path.join(data_dir, "PATIENT_IDS.npy"))
    
    print(f"Loaded data: X={X.shape}, Y={Y.shape}, Patients={len(np.unique(PATIENT_IDS))}")
    
    # Define model configurations to compare
    model_configs = [
        {
            'name': 'WaveNet_TwoChannel_Basic',
            'builder': lambda input_shape: modules.WaveNet_two_channel(input_shape=input_shape),
            'lr': 1e-3
        },
        {
            'name': 'WaveNet_Minimal_CrossAttention',
            'builder': lambda input_shape: modules.WaveNet_minimal_cross_attention(input_shape=input_shape),
            'lr': 1e-3
        },
        {
            'name': 'WaveNet_TwoChannel_CrossAttention',
            'builder': lambda input_shape: modules.WaveNet_two_channel_cross_attention(input_shape=input_shape),
            'lr': 1e-3
        },
        {
            'name': 'WaveNet_Combined_Attention',
            'builder': lambda input_shape: modules.WaveNet_two_channel_combined_attention(input_shape=input_shape),
            'lr': 1e-3
        },
    ]
    
    # Run comparison
    comparison_ml, comparison_clinical = compare_models(
        X=X,
        Y=Y,
        patient_ids=PATIENT_IDS,
        model_configs=model_configs,
        n_folds=5,
        n_epochs=50,
        batch_size=32,
        base_save_dir="./model_comparison_results",
        apply_smoothing=True,
        smoothing_params={'fs': 284, 'cutoff_hz': 10.0, 'order': 4},
        sampling_rate=284,
        early_stopping_patience=10
    )


# ============================================================================
# EXAMPLE: Compare Different Loss Functions
# ============================================================================

def compare_loss_functions():
    """Example comparing different loss function configurations"""
    
    # Load data
    data_dir = "./WaveNet_beat/data"
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))
    PATIENT_IDS = np.load(os.path.join(data_dir, "PATIENT_IDS.npy"))
    
    # Define model configurations with different loss functions
    model_configs = [
        {
            'name': 'MAE_Loss',
            'builder': lambda input_shape: create_model_with_loss(
                input_shape, loss='mae'
            ),
            'lr': 1e-3
        },
        {
            'name': 'MSE_Loss',
            'builder': lambda input_shape: create_model_with_loss(
                input_shape, loss='mse'
            ),
            'lr': 1e-3
        },
        {
            'name': 'Composite_Loss',
            'builder': lambda input_shape: create_model_with_loss(
                input_shape, 
                loss=modules.create_composite_loss(alpha_peak=0.5, alpha_ratio=0.5, base_loss='mae')
            ),
            'lr': 1e-3
        },
        {
            'name': 'Shape_Preserving_Loss',
            'builder': lambda input_shape: create_model_with_loss(
                input_shape,
                loss=modules.create_shape_preserving_loss(
                    alpha_peak=0.5, alpha_ratio=0.5, alpha_derivative=0.3, base_loss='mae'
                )
            ),
            'lr': 1e-3
        },
    ]
    
    # Run comparison
    comparison_ml, comparison_clinical = compare_models(
        X=X,
        Y=Y,
        patient_ids=PATIENT_IDS,
        model_configs=model_configs,
        n_folds=5,
        n_epochs=50,
        batch_size=32,
        base_save_dir="./loss_comparison_results",
        apply_smoothing=True,
        smoothing_params={'fs': 284, 'cutoff_hz': 10.0, 'order': 4},
        sampling_rate=284,
        early_stopping_patience=10
    )


def create_model_with_loss(input_shape, loss):
    """Helper to create model with specific loss"""
    import tensorflow as tf
    model = modules.WaveNet_two_channel_combined_attention(input_shape=input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=loss)
    return model


# ============================================================================
# EXAMPLE: Compare Different Learning Rates
# ============================================================================

def compare_learning_rates():
    """Example comparing different learning rates"""
    
    # Load data
    data_dir = "./WaveNet_beat/data"
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))
    PATIENT_IDS = np.load(os.path.join(data_dir, "PATIENT_IDS.npy"))
    
    # Define model configurations with different learning rates
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    
    model_configs = []
    for lr in learning_rates:
        model_configs.append({
            'name': f'WaveNet_LR_{lr:.0e}',
            'builder': lambda input_shape: modules.WaveNet_two_channel_combined_attention(input_shape=input_shape),
            'lr': lr
        })
    
    # Run comparison
    comparison_ml, comparison_clinical = compare_models(
        X=X,
        Y=Y,
        patient_ids=PATIENT_IDS,
        model_configs=model_configs,
        n_folds=5,
        n_epochs=50,
        batch_size=32,
        base_save_dir="./lr_comparison_results",
        apply_smoothing=True,
        smoothing_params={'fs': 284, 'cutoff_hz': 10.0, 'order': 4},
        sampling_rate=284,
        early_stopping_patience=10
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare multiple model configurations")
    parser.add_argument("--comparison_type", type=str, default="architectures",
                       choices=['architectures', 'losses', 'learning_rates'],
                       help="Type of comparison to perform")
    
    args = parser.parse_args()
    
    if args.comparison_type == "architectures":
        main()
    elif args.comparison_type == "losses":
        compare_loss_functions()
    elif args.comparison_type == "learning_rates":
        compare_learning_rates()
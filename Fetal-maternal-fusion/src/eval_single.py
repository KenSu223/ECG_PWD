"""
Quick Example: Evaluating and Comparing WaveNet Models
This script demonstrates how to use the evaluation framework
"""

import numpy as np
import os
from pathlib import Path

# Import evaluation functions
from evaluate_wavenet_kfold import evaluate_model_kfold
from compare_models import compare_models
from visualize_results import create_summary_report

import modules


# ============================================================================
# EXAMPLE 1: Evaluate a Single Model
# ============================================================================

def example_single_model_evaluation():
    """Evaluate a single model with k-fold CV"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Model Evaluation")
    print("="*80 + "\n")
    
    # Load your data
    data_dir = "./WaveNet_beat/data"
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))
    PATIENT_IDS = np.load(os.path.join(data_dir, "PATIENT_IDS.npy"))
    
    print(f"Data loaded: X={X.shape}, Y={Y.shape}")
    print(f"Unique patients: {len(np.unique(PATIENT_IDS))}\n")
    
    # Define your model
    def build_model(input_shape):
        """Build your WaveNet architecture"""
        model = modules.WaveNet_two_channel_combined_attention(input_shape=input_shape)
        return model
    
    # Run evaluation
    ml_summary, clinical_summary = evaluate_model_kfold(
        X=X,
        Y=Y,
        patient_ids=PATIENT_IDS,
        model_builder_fn=build_model,
        n_folds=5,                    # 5-fold cross-validation
        n_epochs=50,                  # Max 50 epochs per fold
        batch_size=32,
        lr=1e-3,
        save_dir="./example_results/single_model",
        model_name="WaveNet_Combined_Attention",
        apply_smoothing=True,         # Apply Butterworth smoothing
        smoothing_params={
            'fs': 284,                # Sampling frequency
            'cutoff_hz': 10.0,        # Cutoff frequency
            'order': 4                # Filter order
        },
        sampling_rate=284,
        early_stopping_patience=10
    )
    
    print("\n✓ Single model evaluation complete!")
    print(f"  Results saved to: ./example_results/single_model")
    
    return ml_summary, clinical_summary


# ============================================================================
# EXAMPLE 2: Compare Multiple Architectures
# ============================================================================

def example_architecture_comparison():
    """Compare different WaveNet architectures"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Architecture Comparison")
    print("="*80 + "\n")
    
    # Load data
    data_dir = "./WaveNet_beat/data"
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))
    PATIENT_IDS = np.load(os.path.join(data_dir, "PATIENT_IDS.npy"))
    
    # Define models to compare
    model_configs = [
        {
            'name': 'Basic_TwoChannel',
            'builder': lambda input_shape: modules.WaveNet_two_channel(input_shape=input_shape),
            'lr': 1e-3
        },
        {
            'name': 'Cross_Attention',
            'builder': lambda input_shape: modules.WaveNet_two_channel_cross_attention(input_shape=input_shape),
            'lr': 1e-3
        },
        {
            'name': 'Combined_Attention',
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
        base_save_dir="./example_results/architecture_comparison",
        apply_smoothing=True,
        smoothing_params={'fs': 284, 'cutoff_hz': 10.0, 'order': 4},
        sampling_rate=284,
        early_stopping_patience=10
    )
    
    print("\n✓ Architecture comparison complete!")
    print(f"  Results saved to: ./example_results/architecture_comparison")
    
    return comparison_ml, comparison_clinical


# ============================================================================
# EXAMPLE 3: Compare Different Loss Functions
# ============================================================================

def example_loss_comparison():
    """Compare different loss function configurations"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Loss Function Comparison")
    print("="*80 + "\n")
    
    # Load data
    data_dir = "./WaveNet_beat/data"
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))
    PATIENT_IDS = np.load(os.path.join(data_dir, "PATIENT_IDS.npy"))
    
    import tensorflow as tf
    
    # Helper function to create model with specific loss
    def create_model_with_loss(input_shape, loss_fn, loss_name):
        model = modules.WaveNet_two_channel_combined_attention(input_shape=input_shape)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss=loss_fn)
        return model
    
    # Define loss configurations
    model_configs = [
        {
            'name': 'MAE_Loss',
            'builder': lambda input_shape: create_model_with_loss(
                input_shape, 'mae', 'MAE'
            ),
            'lr': 1e-3
        },
        {
            'name': 'MSE_Loss',
            'builder': lambda input_shape: create_model_with_loss(
                input_shape, 'mse', 'MSE'
            ),
            'lr': 1e-3
        },
        {
            'name': 'Composite_Loss',
            'builder': lambda input_shape: create_model_with_loss(
                input_shape,
                modules.create_composite_loss(alpha_peak=0.5, alpha_ratio=0.5),
                'Composite'
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
        base_save_dir="./example_results/loss_comparison",
        apply_smoothing=True,
        smoothing_params={'fs': 284, 'cutoff_hz': 10.0, 'order': 4},
        sampling_rate=284,
        early_stopping_patience=10
    )
    
    print("\n✓ Loss function comparison complete!")
    print(f"  Results saved to: ./example_results/loss_comparison")
    
    return comparison_ml, comparison_clinical


# ============================================================================
# EXAMPLE 4: Generate Visualizations
# ============================================================================

def example_visualizations():
    """Generate visualization report from comparison results"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Generate Visualizations")
    print("="*80 + "\n")
    
    # Generate visualizations for architecture comparison
    comparison_dir = "./example_results/architecture_comparison"
    
    if not Path(comparison_dir).exists():
        print(f"Error: {comparison_dir} does not exist.")
        print("Run example_architecture_comparison() first!")
        return
    
    # Create comprehensive visualization report
    create_summary_report(
        base_dir=comparison_dir,
        output_dir=os.path.join(comparison_dir, "visualizations")
    )
    
    print("\n✓ Visualization report complete!")
    print(f"  Plots saved to: {comparison_dir}/visualizations")


# ============================================================================
# EXAMPLE 5: Quick Hyperparameter Search
# ============================================================================

def example_hyperparameter_search():
    """Compare different learning rates"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Learning Rate Search")
    print("="*80 + "\n")
    
    # Load data
    data_dir = "./WaveNet_beat/data"
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))
    PATIENT_IDS = np.load(os.path.join(data_dir, "PATIENT_IDS.npy"))
    
    # Test different learning rates
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    
    model_configs = []
    for lr in learning_rates:
        model_configs.append({
            'name': f'LR_{lr:.0e}',
            'builder': lambda input_shape: modules.WaveNet_two_channel_combined_attention(
                input_shape=input_shape
            ),
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
        base_save_dir="./example_results/lr_search",
        apply_smoothing=True,
        smoothing_params={'fs': 284, 'cutoff_hz': 10.0, 'order': 4},
        sampling_rate=284,
        early_stopping_patience=10
    )
    
    print("\n✓ Learning rate search complete!")
    print(f"  Results saved to: ./example_results/lr_search")
    
    # Generate visualizations
    create_summary_report(
        base_dir="./example_results/lr_search",
        output_dir="./example_results/lr_search/visualizations"
    )
    
    return comparison_ml, comparison_clinical


# ============================================================================
# Main Function - Run All Examples
# ============================================================================

def main():
    """Run all examples"""
    
    print("\n" + "#"*80)
    print("# WaveNet Evaluation Framework - Examples")
    print("#"*80)
    
    # Check if data exists
    data_dir = "./WaveNet_beat/data"
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please update the data_dir path in the examples.")
        return
    
    # Uncomment the examples you want to run:
    
    # Example 1: Single model evaluation
    # ml_summary, clinical_summary = example_single_model_evaluation()
    
    # Example 2: Compare architectures
    # comparison_ml, comparison_clinical = example_architecture_comparison()
    
    # Example 3: Compare loss functions
    # comparison_ml, comparison_clinical = example_loss_comparison()
    
    # Example 4: Generate visualizations (run after example 2 or 3)
    # example_visualizations()
    
    # Example 5: Hyperparameter search
    # comparison_ml, comparison_clinical = example_hyperparameter_search()
    
    print("\n" + "#"*80)
    print("# Examples Complete!")
    print("#"*80)
    print("\nTo run specific examples, uncomment them in the main() function.")
    print("\nQuick start:")
    print("  1. Update data_dir path to your data location")
    print("  2. Uncomment the example you want to run")
    print("  3. Run: python example_usage.py")
    print("\nFor command-line usage:")
    print("  python evaluate_wavenet_kfold.py --help")
    print("  python compare_models.py --help")


if __name__ == "__main__":
    # Quick usage instructions
    print("\n" + "="*80)
    print("WaveNet Evaluation Framework - Quick Start")
    print("="*80)
    print("\nThis script contains 5 example use cases:")
    print("\n1. example_single_model_evaluation()")
    print("   - Evaluate a single model with 5-fold CV")
    print("   - Get ML/signal metrics and clinical metrics")
    print("\n2. example_architecture_comparison()")
    print("   - Compare 3 different WaveNet architectures")
    print("   - Side-by-side performance metrics")
    print("\n3. example_loss_comparison()")
    print("   - Compare MAE, MSE, and composite loss functions")
    print("\n4. example_visualizations()")
    print("   - Generate publication-quality plots")
    print("   - Bar charts, heatmaps, radar charts")
    print("\n5. example_hyperparameter_search()")
    print("   - Compare different learning rates")
    print("   - Includes automatic visualization")
    print("\nUncomment the examples in main() to run them.")
    print("="*80 + "\n")
    
    # Uncomment to run:
    # main()
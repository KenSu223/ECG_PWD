"""Post-processing visualization toolkit for evaluation outputs (bar charts, heatmaps, radar charts, and fold-variance views)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_ml_metrics_comparison(csv_path, save_path=None, figsize=(14, 10)):
    """
    Create bar plots comparing ML metrics across models.
    
    Parameters:
    -----------
    csv_path : str
        Path to all_models_ml_metrics.csv
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    df = pd.read_csv(csv_path)
    
    # Select key metrics for visualization
    key_metrics = ['MAE', 'RMSE', 'Pearson_Correlation', 'DTW', 
                   'Cross_Correlation_Zero', 'PSD_Correlation']
    
    df_plot = df[df['Metric'].isin(key_metrics)]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    models = df_plot['Model'].unique()
    colors = sns.color_palette("husl", len(models))
    
    for idx, metric in enumerate(key_metrics):
        ax = axes[idx]
        metric_data = df_plot[df_plot['Metric'] == metric]
        
        x = np.arange(len(models))
        means = metric_data['Mean'].values
        stds = metric_data['Std'].values
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=colors, edgecolor='black', linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' '))
        ax.set_title(f'{metric.replace("_", " ")}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.4f}\n±{std:.4f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('ML/Signal Metrics Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_clinical_metrics_comparison(csv_path, save_path=None, figsize=(16, 10)):
    """
    Create grouped bar plots for clinical metrics.
    
    Parameters:
    -----------
    csv_path : str
        Path to all_models_clinical_metrics.csv
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    df = pd.read_csv(csv_path)
    
    metrics = df['Metric'].unique()
    models = df['Model'].unique()
    
    # Create figure with subplots for each metric
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    x = np.arange(len(models))
    width = 0.25
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_data = df[df['Metric'] == metric]
        
        # Real values
        real_means = metric_data['Real_Mean'].values
        real_stds = metric_data['Real_Std'].values
        
        # Generated values
        gen_means = metric_data['Generated_Mean'].values
        gen_stds = metric_data['Generated_Std'].values
        
        # MAE values
        mae_means = metric_data['MAE'].values
        mae_stds = metric_data['MAE_Std'].values
        
        # Plot grouped bars
        ax.bar(x - width, real_means, width, yerr=real_stds, 
               label='Real', capsize=3, alpha=0.8, color='steelblue')
        ax.bar(x, gen_means, width, yerr=gen_stds,
               label='Generated', capsize=3, alpha=0.8, color='coral')
        ax.bar(x + width, mae_means, width, yerr=mae_stds,
               label='MAE', capsize=3, alpha=0.8, color='lightgreen')
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title(metric, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide extra subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Clinical Metrics Comparison', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_correlation_heatmap(csv_path, save_path=None, figsize=(10, 8)):
    """
    Create heatmap of correlations for clinical metrics.
    
    Parameters:
    -----------
    csv_path : str
        Path to all_models_clinical_metrics.csv
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    df = pd.read_csv(csv_path)
    
    # Pivot to get correlation values
    pivot = df.pivot(index='Metric', columns='Model', values='Correlation')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.8,
                vmin=0, vmax=1, cbar_kws={'label': 'Correlation'},
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_title('Clinical Metrics Correlation (Real vs Generated)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Clinical Metric', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_mae_heatmap(csv_path, save_path=None, figsize=(10, 8)):
    """
    Create heatmap of MAE values for clinical metrics.
    
    Parameters:
    -----------
    csv_path : str
        Path to all_models_clinical_metrics.csv
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    df = pd.read_csv(csv_path)
    
    # Pivot to get MAE values
    pivot = df.pivot(index='Metric', columns='Model', values='MAE')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap (lower is better, so reverse colormap)
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r',
                cbar_kws={'label': 'MAE (lower is better)'},
                linewidths=1, linecolor='black', ax=ax)
    
    ax.set_title('Clinical Metrics MAE (Real vs Generated)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Clinical Metric', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_radar_chart(csv_path, normalize=True, save_path=None, figsize=(12, 8)):
    """
    Create radar chart comparing models across key metrics.
    
    Parameters:
    -----------
    csv_path : str
        Path to all_models_ml_metrics.csv
    normalize : bool
        Whether to normalize metrics to [0, 1]
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    df = pd.read_csv(csv_path)
    
    # Select metrics for radar chart
    metrics = ['MAE', 'RMSE', 'Pearson_Correlation', 'DTW', 'PSD_Correlation']
    df_plot = df[df['Metric'].isin(metrics)]
    
    models = df_plot['Model'].unique()
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Prepare data
    values = np.zeros((n_models, n_metrics))
    for i, model in enumerate(models):
        model_data = df_plot[df_plot['Model'] == model]
        for j, metric in enumerate(metrics):
            metric_val = model_data[model_data['Metric'] == metric]['Mean'].values[0]
            values[i, j] = metric_val
    
    # Normalize if requested
    if normalize:
        # For metrics where lower is better, invert
        lower_is_better = ['MAE', 'RMSE', 'DTW']
        for j, metric in enumerate(metrics):
            col = values[:, j]
            if metric in lower_is_better:
                # Invert and normalize to [0, 1]
                col = 1 - (col - col.min()) / (col.max() - col.min() + 1e-8)
            else:
                # Normalize to [0, 1]
                col = (col - col.min()) / (col.max() - col.min() + 1e-8)
            values[:, j] = col
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    values_plot = np.concatenate([values, values[:, [0]]], axis=1)
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = sns.color_palette("husl", n_models)
    
    for i, model in enumerate(models):
        ax.plot(angles, values_plot[i], 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values_plot[i], alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1 if normalize else None)
    ax.set_title('Model Performance Radar Chart' + (' (Normalized)' if normalize else ''),
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def create_summary_report(base_dir, output_dir=None):
    """
    Create comprehensive visualization report from comparison results.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing all_models_*.csv files
    output_dir : str, optional
        Directory to save plots (default: base_dir/visualizations)
    """
    if output_dir is None:
        output_dir = Path(base_dir) / "visualizations"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    ml_csv = Path(base_dir) / "all_models_ml_metrics.csv"
    clinical_csv = Path(base_dir) / "all_models_clinical_metrics.csv"
    
    if not ml_csv.exists() or not clinical_csv.exists():
        print(f"Error: Could not find CSV files in {base_dir}")
        return
    
    print("Creating visualization report...")
    print("="*80)
    
    # ML metrics comparison
    print("1. ML metrics bar plot...")
    plot_ml_metrics_comparison(
        ml_csv,
        save_path=output_dir / "ml_metrics_comparison.png"
    )
    plt.close()
    
    # Clinical metrics comparison
    print("2. Clinical metrics grouped bar plot...")
    plot_clinical_metrics_comparison(
        clinical_csv,
        save_path=output_dir / "clinical_metrics_comparison.png"
    )
    plt.close()
    
    # Correlation heatmap
    print("3. Clinical correlation heatmap...")
    plot_correlation_heatmap(
        clinical_csv,
        save_path=output_dir / "clinical_correlation_heatmap.png"
    )
    plt.close()
    
    # MAE heatmap
    print("4. Clinical MAE heatmap...")
    plot_mae_heatmap(
        clinical_csv,
        save_path=output_dir / "clinical_mae_heatmap.png"
    )
    plt.close()
    
    # Radar chart
    print("5. Performance radar chart...")
    plot_radar_chart(
        ml_csv,
        normalize=True,
        save_path=output_dir / "performance_radar_normalized.png"
    )
    plt.close()
    
    print("="*80)
    print(f"Report complete! Visualizations saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  - ml_metrics_comparison.png")
    print("  - clinical_metrics_comparison.png")
    print("  - clinical_correlation_heatmap.png")
    print("  - clinical_mae_heatmap.png")
    print("  - performance_radar_normalized.png")
    print("="*80)


def plot_fold_variance(fold_details_json, metric_name='MAE', 
                       metric_type='ml', save_path=None, figsize=(12, 6)):
    """
    Plot metric values across folds to visualize variance.
    
    Parameters:
    -----------
    fold_details_json : str
        Path to fold_details.json file
    metric_name : str
        Name of metric to plot
    metric_type : str
        'ml' or 'clinical'
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    with open(fold_details_json, 'r') as f:
        data = json.load(f)
    
    if metric_type == 'ml':
        metrics_data = data['ml_metrics']
        values = [fold[metric_name]['mean'] for fold in metrics_data]
        stds = [fold[metric_name]['std'] for fold in metrics_data]
    else:  # clinical
        metrics_data = data['clinical_metrics']
        values = [fold[metric_name]['mae'] for fold in metrics_data]
        stds = [fold[metric_name].get('correlation', np.nan) for fold in metrics_data]
    
    folds = [fold['fold'] for fold in metrics_data]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot metric values
    axes[0].plot(folds, values, 'o-', linewidth=2, markersize=8)
    axes[0].axhline(np.mean(values), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(values):.4f}')
    axes[0].fill_between(folds, 
                         np.mean(values) - np.std(values),
                         np.mean(values) + np.std(values),
                         alpha=0.2, color='red')
    axes[0].set_xlabel('Fold', fontweight='bold')
    axes[0].set_ylabel(metric_name, fontweight='bold')
    axes[0].set_title(f'{metric_name} Across Folds', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(folds)
    
    # Plot distribution
    axes[1].boxplot([values], labels=[metric_name])
    axes[1].scatter([1]*len(values), values, alpha=0.6, s=50, color='steelblue')
    axes[1].set_ylabel('Value', fontweight='bold')
    axes[1].set_title(f'{metric_name} Distribution', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Fold-wise Variance Analysis: {metric_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, axes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize model evaluation results")
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing CSV files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    create_summary_report(args.base_dir, args.output_dir)
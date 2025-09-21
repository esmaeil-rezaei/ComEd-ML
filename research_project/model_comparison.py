
"""
Model Evaluation and Visualization Script

Trains and evaluates RMA-RBD and RMA-PCA models for short-term load forecasting,
generating visualizations for comparison. RMA-RBD achieves similar accuracy to RMA-PCA
while being ~30x faster. A HYBRID algorithm (~78x faster than PCA) is under development.
This is a pre-production version; the final HYBRID model and advanced dimensionality reduction
techniques will be added in future releases.
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

from src.exception import CustomException
from src.logger import logging
from src.pipelines.train_pipeline import TrainStrategiesModel
from src.utils import (
    load_object,
    estimate_coefficient_of_error,
    estimate_mean_absolute_percentage,
    relative_error
)
import cProfile


def setup_matplotlib():
    """Configure matplotlib settings for high-quality plots."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.size': 15
    })


def create_output_directory(dim_reduction_size, time_interval, r):
    """Create and return the output directory path."""
    figs_path = Path("research_project/FIGs") / f"dim_{dim_reduction_size}_time_{time_interval}_r_{r}"
    figs_path.mkdir(parents=True, exist_ok=True)
    return figs_path


def save_plot(figs_path, filename, time_interval, r):
    """Save the current matplotlib figure with standardized naming."""
    filepath = figs_path / f"{filename}_{time_interval}_{r}r.pdf"
    plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {filepath}")


def train_models(wants_data_ingestion, dim_reduction_size, time_interval, strategy_names, ml_models_name, r):
    """Train the models with given parameters."""
    print("Training models...")
    trainer = TrainStrategiesModel(
        wants_data_ingestion=wants_data_ingestion,   
        dim_reduction_size=dim_reduction_size,
        time_interval=time_interval,
        strategy_names=strategy_names,
        ml_models_name=ml_models_name,
        r=r,
    )
    trainer.train()
    print("Training completed.")


def load_and_evaluate_models():
    """Load trained models and calculate predictions."""
    print("Loading trained models...")
    
    # Load models
    trained_data_rma_rbd = load_object(file_path="models/model_rma_rbd.pkl")
    trained_data_rma_pca = load_object(file_path="models/model_rma_pca.pkl")
    
    # Extract RMA-RBD data
    rma_rbd_regressor = trained_data_rma_rbd["best_model"]
    X_test_rma_rbd = trained_data_rma_rbd["X_test_rma_rbd"]
    y_test_rma_rbd = trained_data_rma_rbd["y_test_rma_rbd"]
    y_pred_rma_rbd = rma_rbd_regressor.predict(X_test_rma_rbd)
    print(f"Best regressor for RMA-RBD is:\n{rma_rbd_regressor}")
    
    # Extract RMA-PCA data
    rma_pca_regressor = trained_data_rma_pca["best_model"]
    X_test_rma_pca = trained_data_rma_pca["X_test_rma_pca"]
    y_test_rma_pca = trained_data_rma_pca["y_test_rma_pca"]
    y_pred_rma_pca = rma_pca_regressor.predict(X_test_rma_pca)
    print(f"Best regressor for RMA-PCA is:\n{rma_pca_regressor}")
    
    return {
        'rma_rbd': {
            'regressor': rma_rbd_regressor,
            'X_test': X_test_rma_rbd,
            'y_test': y_test_rma_rbd,
            'y_pred': y_pred_rma_rbd
        },
        'rma_pca': {
            'regressor': rma_pca_regressor,
            'X_test': X_test_rma_pca,
            'y_test': y_test_rma_pca,
            'y_pred': y_pred_rma_pca
        }
    }


def calculate_metrics(model_results):
    """Calculate performance metrics for both models."""
    print("Calculating performance metrics...")
    
    metrics = {}
    for strategy, data in model_results.items():
        y_test = data['y_test']
        y_pred = data['y_pred']
        
        # Calculate metrics
        mape = estimate_mean_absolute_percentage(y_test, y_pred)
        coef_error = estimate_coefficient_of_error(y_test, y_pred)
        rel_error = relative_error(y_test, y_pred)
        
        metrics[strategy] = {
            'mape': mape,
            'coefficient_error': coef_error,
            'relative_error': rel_error,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        strategy_name = 'RMA-RBD' if strategy == 'rma_rbd' else 'RMA-PCA'
        print(f"{strategy_name} - MAPE: {mape:.4f}, Coefficient of Error: {coef_error:.4f}")
    
    return metrics


def plot_performance_comparison(metrics, figs_path, time_interval, r):
    """Create bar chart comparing model performance."""
    print("Creating performance comparison plot...")
    
    metric_names = ['MAPE', 'Coefficient of Error']
    rbd_values = [metrics['rma_rbd']['mape'], metrics['rma_rbd']['coefficient_error']]
    pca_values = [metrics['rma_pca']['mape'], metrics['rma_pca']['coefficient_error']]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rbd_values, width, label='RMA-RBD')
    rects2 = ax.bar(x + width/2, pca_values, width, label='RMA-PCA')
    
    ax.set_ylabel('Error Values')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_plot(figs_path, "model_performance_comparison", time_interval, r)
    plt.close()


def plot_actual_vs_predicted(metrics, figs_path, time_interval, r):
    """Create scatter plots of actual vs predicted values."""
    print("Creating actual vs predicted plots...")
    
    plt.figure(figsize=(12, 6))
    
    # RMA-RBD subplot
    plt.subplot(1, 2, 1)
    y_test_rbd = metrics['rma_rbd']['y_test']
    y_pred_rbd = metrics['rma_rbd']['y_pred']
    
    plt.scatter(y_test_rbd, y_pred_rbd, alpha=0.5)
    plt.plot([min(y_test_rbd), max(y_test_rbd)], 
            [min(y_test_rbd), max(y_test_rbd)], 'r--')
    plt.title('RMA-RBD: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    
    # RMA-PCA subplot
    plt.subplot(1, 2, 2)
    y_test_pca = metrics['rma_pca']['y_test']
    y_pred_pca = metrics['rma_pca']['y_pred']
    
    plt.scatter(y_test_pca, y_pred_pca, alpha=0.5)
    plt.plot([min(y_test_pca), max(y_test_pca)], 
            [min(y_test_pca), max(y_test_pca)], 'r--')
    plt.title('RMA-PCA: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(figs_path, "actual_vs_predicted", time_interval, r)
    plt.close()


def plot_cumulative_error_distribution(metrics, figs_path, time_interval, r):
    """Create cumulative error distribution plot."""
    print("Creating cumulative error distribution plot...")
    
    plt.figure(figsize=(10, 6))
    
    sorted_rbd = np.sort(np.abs(metrics['rma_rbd']['relative_error']))
    sorted_pca = np.sort(np.abs(metrics['rma_pca']['relative_error']))
    
    plt.plot(sorted_rbd, np.linspace(0, 1, len(sorted_rbd)), label='RMA-RBD', linewidth=2)
    plt.plot(sorted_pca, np.linspace(0, 1, len(sorted_pca)), label='RMA-PCA', linewidth=2)
    plt.title('Cumulative Error Distribution')
    plt.xlabel('Absolute Relative Error')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot(figs_path, "cumulative_error_distribution", time_interval, r)
    plt.close()


def plot_time_series_comparison(metrics, figs_path, time_interval, r):
    """Create time series comparison plots."""
    print("Creating time series comparison plot...")
    
    # Create time indices for plotting
    y_test_rbd = metrics['rma_rbd']['y_test']
    time_intervals = np.arange(len(y_test_rbd))
    time_labels = [f"Oct{t}" for t in range(26, 32)]  # Oct24 to Oct31
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # RMA-RBD Plot
    y_pred_rbd = metrics['rma_rbd']['y_pred']
    ax1.plot(time_intervals, y_test_rbd, 'b-', label='Actual Values', linewidth=1.5, alpha=0.8)
    ax1.plot(time_intervals, y_pred_rbd, 'r--', label='Predicted Values', linewidth=1.2)
    ax1.set_title('RMA-RBD: Actual vs Predicted Values Over Time', fontsize=12)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Add shaded error region
    error_rbd = y_pred_rbd - y_test_rbd
    ax1.fill_between(time_intervals, 
                    y_test_rbd - error_rbd, 
                    y_test_rbd + error_rbd,
                    color='gray', alpha=0.1, label='Error Region')
    
    # RMA-PCA Plot
    y_test_pca = metrics['rma_pca']['y_test']
    y_pred_pca = metrics['rma_pca']['y_pred']
    ax2.plot(time_intervals, y_test_pca, 'b-', label='Actual Values', linewidth=1.5, alpha=0.8)
    ax2.plot(time_intervals, y_pred_pca, 'g--', label='Predicted Values', linewidth=1.2)
    ax2.set_title('RMA-PCA: Actual vs Predicted Values Over Time', fontsize=12)
    ax2.set_xlabel('Time Index', fontsize=10)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Add shaded error region
    error_pca = y_pred_pca - y_test_pca
    ax2.fill_between(time_intervals, 
                    y_test_pca - error_pca, 
                    y_test_pca + error_pca,
                    color='gray', alpha=0.1, label='Error Region')
    
    # Set x-axis ticks
    n_ticks = min(6, len(time_labels))
    if len(time_intervals) > 0:
        ax2.set_xticks(np.linspace(0, len(time_intervals)-1, n_ticks))
        ax2.set_xticklabels(time_labels[:n_ticks])
    
    plt.tight_layout()
    save_plot(figs_path, "time_series_comparison", time_interval, r)
    plt.close()


def plot_error_distribution_analysis(metrics, figs_path, time_interval, r):
    """Create detailed error distribution analysis."""
    logging.info("Creating error distribution analysis...")
    
    # Set appealing color palette
    colors = ["#4C72B0", "#DD8452"]
    sns.set_palette(colors)
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    relative_error_rma_rbd = metrics['rma_rbd']['relative_error']
    relative_error_rma_pca = metrics['rma_pca']['relative_error']
    
    bins = max(10, int(15 / time_interval))
    sns.histplot(relative_error_rma_rbd, bins=bins, color=colors[0], 
                label="RMA-RBD", kde=True, alpha=0.6, edgecolor='w', linewidth=0.5)
    sns.histplot(relative_error_rma_pca, bins=bins, color=colors[1], 
                label="RMA-PCA", kde=True, alpha=0.6, edgecolor='w', linewidth=0.5)
    plt.title("Relative Error Distribution", fontsize=14, pad=20)
    plt.xlabel("Relative Error", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Boxplot
    plt.subplot(1, 2, 2)
    box_data = [relative_error_rma_rbd, relative_error_rma_pca]
    sns.boxplot(data=box_data, palette=colors, width=0.4, linewidth=2)
    plt.xticks([0, 1], ["RMA-RBD", "RMA-PCA"], fontsize=12)
    plt.title("Relative Error Comparison", fontsize=14, pad=20)
    plt.ylabel("Relative Error", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=2.0)
    sns.despine(left=True)
    
    save_plot(figs_path, "relative_error_distribution", time_interval, r)
    plt.close()


def print_summary_statistics(metrics):
    """Print comprehensive performance summary."""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    for strategy in ['rma_rbd', 'rma_pca']:
        strategy_name = 'RMA-RBD' if strategy == 'rma_rbd' else 'RMA-PCA'
        rel_errors = metrics[strategy]['relative_error']
        
        print(f"\n{strategy_name}:")
        print(f"  MAPE: {metrics[strategy]['mape']:.4f}")
        print(f"  Coefficient of Error: {metrics[strategy]['coefficient_error']:.4f}")
        print(f"  Mean Absolute Relative Error: {np.mean(np.abs(rel_errors)):.4f}")
        print(f"  Std Relative Error: {np.std(rel_errors):.4f}")
        print(f"  Max Absolute Error: {np.max(np.abs(rel_errors)):.4f}")
        print(f"  Min Absolute Error: {np.min(np.abs(rel_errors)):.4f}")


def main():
    """Main execution function."""
    # Configuration parameters
    wants_data_ingestion = True
    r = 11
    dim_reduction_size = 13
    time_interval = 0.5
    strategy_names = {"rma_rbd", "rma_pca"}
    ml_models_name = {
        "Random Forest",
        "Decision Tree",
        "Gradient Boosting",
        "Linear Regression",
        "XGBRegressor",
        "CatBoosting Regressor",
        "AdaBoost Regressor",
    }
    
    logging.info("Starting model evaluation and visualization pipeline...")
    logging.info(f"Configuration: r={r}, dim_reduction_size={dim_reduction_size}, time_interval={time_interval}")
    
    # Setup
    setup_matplotlib()
    figs_path = create_output_directory(dim_reduction_size, time_interval, r)
    
    try:
        # Train models
        train_models(wants_data_ingestion, dim_reduction_size, time_interval, strategy_names, ml_models_name, r)
        
        # Load and evaluate models
        model_results = load_and_evaluate_models()
        
        # Calculate metrics
        metrics = calculate_metrics(model_results)
        
        # Generate all plots
        print("\nGenerating visualizations...")
        plot_performance_comparison(metrics, figs_path, time_interval, r)
        plot_actual_vs_predicted(metrics, figs_path, time_interval, r)
        plot_cumulative_error_distribution(metrics, figs_path, time_interval, r)
        plot_time_series_comparison(metrics, figs_path, time_interval, r)
        plot_error_distribution_analysis(metrics, figs_path, time_interval, r)
        
        # Print summary
        print_summary_statistics(metrics)
        
        print(f"\nAll visualizations saved to: {figs_path}")
        print("Model evaluation and visualization completed successfully!")
        
    except Exception as e:
        custom_error = CustomException(e, sys)
        logging.error(custom_error)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    profiler.print_stats(sort='time')
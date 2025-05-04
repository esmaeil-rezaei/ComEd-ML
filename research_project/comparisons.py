
from src.exception import CustomException
from src.logger import logging
from src.pipelines.train_pipeline import TrainStrategiesModel
from src.utils import (
    load_object,
    estimate_coefficient_of_error,
    estimate_mean_absolute_percentage,
    relative_error)
import os


trainer = TrainStrategiesModel(   
    dim_reduction_size=10,
    time_interval=0.5,
    strategy_names={"rma_rbd", "rma_pca"},
    ml_models_name={"Linear Regression"},
    r=3,
)
trainer.train()



trained_data_rma_rbd = load_object(
    file_path="artifacts/model_rma_rbd.pkl",
)

trained_data_rma_pca = load_object(
    file_path="artifacts/model_rma_pca.pkl",
)


rma_rbd_regressor = trained_data_rma_rbd["best_model"]
X_test_rma_rbd = trained_data_rma_rbd["X_test_rma_rbd"]
y_test_rma_rbd = trained_data_rma_rbd["y_test_rma_rbd"]
y_pred_rma_rbd = rma_rbd_regressor.predict(X_test_rma_rbd)


rma_pca_regressor = trained_data_rma_pca["best_model"]
X_test_rma_pca = trained_data_rma_pca["X_test_rma_pca"]
y_test_rma_pca = trained_data_rma_pca["y_test_rma_pca"]
y_pred_rma_pca = rma_pca_regressor.predict(X_test_rma_pca)


e_map_rbd = estimate_mean_absolute_percentage(
    y_true=y_test_rma_rbd,
    y_pred=y_pred_rma_rbd
) 
e_ce_rbd = estimate_coefficient_of_error(
    y_true=y_test_rma_rbd,
    y_pred=y_pred_rma_rbd
)
e_map_pca = estimate_mean_absolute_percentage(
    y_true=y_test_rma_pca,
    y_pred=y_pred_rma_pca
)
c_ce_pca = estimate_coefficient_of_error(
    y_true=y_test_rma_pca,
    y_pred=y_pred_rma_pca
)

relative_error_rma_rbd = relative_error(
    y_true=y_test_rma_rbd,
    y_pred=y_pred_rma_rbd
)
relative_error_rma_pca = relative_error(
    y_true=y_test_rma_pca,
    y_pred=y_pred_rma_pca
)




import matplotlib.pyplot as plt
import numpy as np


figs_path = "research_project/FIGs/"
os.makedirs(figs_path, exist_ok=True)



metrics = ['MAPE', 'Coefficient of Error']
rbd_values = [e_map_rbd, e_ce_rbd]
pca_values = [e_map_pca, c_ce_pca]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width/2, rbd_values, width, label='RMA-HYBRID')
rects2 = ax.bar(x + width/2, pca_values, width, label='RMA-PCA')

ax.set_ylabel('Error Values')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.savefig(f"{figs_path}model_performance_comparison.pdf", dpi=300, bbox_inches='tight', pad_inches=0.05)



plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(y_test_rma_rbd, y_pred_rma_rbd, alpha=0.5)
plt.plot([min(y_test_rma_rbd), max(y_test_rma_rbd)], 
         [min(y_test_rma_rbd), max(y_test_rma_rbd)], 'r--')
plt.title('RMA-HYBRID: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1,2,2)
plt.scatter(y_test_rma_pca, y_pred_rma_pca, alpha=0.5)
plt.plot([min(y_test_rma_pca), max(y_test_rma_pca)], 
         [min(y_test_rma_pca), max(y_test_rma_pca)], 'r--')
plt.title('RMA-PCA: Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.tight_layout()
plt.savefig(f"{figs_path}actual_vs_predicted.pdf", dpi=300, bbox_inches='tight')




plt.figure(figsize=(10,6))
sorted_rbd = np.sort(np.abs(relative_error_rma_rbd))
sorted_pca = np.sort(np.abs(relative_error_rma_pca))

plt.plot(sorted_rbd, np.linspace(0,1,len(sorted_rbd)), label='RMA-HYBRID')
plt.plot(sorted_pca, np.linspace(0,1,len(sorted_pca)), label='RMA-PCA')
plt.title('Cumulative Error Distribution')
plt.xlabel('Absolute Relative Error')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.savefig(f"{figs_path}cumulative_error_distribution.pdf", dpi=300, bbox_inches='tight', pad_inches=0.05)




import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Create time indices for plotting (assuming sequential test data)
time_intervals = np.arange(len(y_test_rma_rbd))
time_labels = [f"Oct{t}" for t in range(24, 32)]  # Only Oct24 to Oct31

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# RMA-RBD Plot
ax1.plot(time_intervals, y_test_rma_rbd, 'b-', label='Actual Values', linewidth=1.5, alpha=0.8)
ax1.plot(time_intervals, y_pred_rma_rbd, 'r--', label='Predicted Values', linewidth=1.2)
ax1.set_title('RMA-HYBRID: Actual vs Predicted Values Over Time', fontsize=12)
ax1.set_ylabel('Value', fontsize=10)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.6)

# Add shaded error region
error_rbd = y_pred_rma_rbd - y_test_rma_rbd
ax1.fill_between(time_intervals, 
                y_test_rma_rbd - error_rbd, 
                y_test_rma_rbd + error_rbd,
                color='gray', alpha=0.1, label='Error Region')

# RMA-PCA Plot
ax2.plot(time_intervals, y_test_rma_pca, 'b-', label='Actual Values', linewidth=1.5, alpha=0.8)
ax2.plot(time_intervals, y_pred_rma_pca, 'g--', label='Predicted Values', linewidth=1.2)
ax2.set_title('RMA-PCA: Actual vs Predicted Values Over Time', fontsize=12)
ax2.set_xlabel('Time Index', fontsize=10)
ax2.set_ylabel('Value', fontsize=10)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.6)

# Add shaded error region
error_pca = y_pred_rma_pca - y_test_rma_pca
ax2.fill_between(time_intervals, 
                y_test_rma_pca - error_pca, 
                y_test_rma_pca + error_pca,
                color='gray', alpha=0.1, label='Error Region')

# Set x-axis ticks to show only Oct24-Oct31
# We assume the data starts at Oct24 and we want to show 8 labels
ax2.set_xticks(np.linspace(0, len(time_intervals)-1, 8))  # 8 evenly spaced ticks
ax2.set_xticklabels(time_labels)  # Use our predefined labels
plt.tight_layout()
plt.savefig(f"{figs_path}time_series_comparison.pdf", dpi=300, bbox_inches='tight')



import matplotlib.pyplot as plt
import seaborn as sns

# Set a more appealing color palette
colors = ["#4C72B0", "#DD8452"]  # Seaborn's muted blue and orange alternatives
sns.set_palette(colors)
sns.set_style("whitegrid")  # Clean background with grid lines

plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(relative_error_rma_rbd, bins=30, color=colors[0], 
             label="RMA-HYBRID", kde=True, alpha=0.6, edgecolor='w', linewidth=0.5)
sns.histplot(relative_error_rma_pca, bins=30, color=colors[1], 
             label="RMA-PCA", kde=True, alpha=0.6, edgecolor='w', linewidth=0.5)
plt.title("Relative Error Distribution", fontsize=14, pad=20)
plt.xlabel("Relative Error", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)

# Boxplot
plt.subplot(1, 2, 2)
box = sns.boxplot(data=[relative_error_rma_rbd, relative_error_rma_pca], 
                 palette=colors, width=0.4, linewidth=2)
plt.xticks([0, 1], ["RMA-HYBRID", "RMA-PCA"], fontsize=12)
plt.title("Relative Error Comparison", fontsize=14, pad=20)
plt.ylabel("Relative Error", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and appearance
plt.tight_layout(pad=2.0)
sns.despine(left=True)  # Remove left spine for cleaner look
plt.tight_layout()
plt.savefig(f"{figs_path}relative_error_boxplot.pdf", dpi=300, bbox_inches='tight')



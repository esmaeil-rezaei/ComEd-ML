# Short-Term Load Forecasting for Comet Industry

This project focuses on **short-term load forecasting** for ComEd industry. The main goal is to compare different forecasting strategies and evaluate their performance using machine learning models, with dimensionality reduction for efficiency in real-time applications.

## Overview

The key steps of this workflow are:

1. **Forecasting Strategies**  
   We preprocess the data using multiple strategies:
   - **CBA** (Cluster-Based Algorithms)
   - **AMA** (Aggregated Model Approach)
   - **RMA** (Reduced Model Approach)

2. **Dimensionality Reduction**  
   To speed up processing and enable real-time forecasting, we reduce the dimensionality of the data using two hierarchical algorithms:
   - **PCA** (Principal Component Analysis)
   - **RBD** (Reduced Basis Decomposition)  
   - **HYBRID** (HYper-reduced Basis Reduction via Interactive Decomposition)

   ⚠️ **Note:** This is the first version of our code. We have developed HYBRID dimensionality reduction algorithm that is **approximately 78 times faster than PCA**. Once our paper is published, this algorithm and its **time series** friendly version will be integrated into this code.

3. **Model Training and Prediction**  
   After preprocessing and dimensionality reduction, different machine learning models are trained to predict load. This includes generating predictions and evaluating the results.

4. **Evaluation and Visualization**  
   The script calculates multiple **evaluation metrics**, visualizes predictions, and saves figures in the project directory for analysis.

---

## Important Notes

- **Data Structure:**  
  Before running this code, ensure that your data columns match the expected structure in the transformation model code. If your dataset contains different columns or categorical features, you will need to adjust the preprocessing pipeline accordingly.

- **Visualization:**  
  All generated plots are saved in the designated project directory. These visualizations help compare model performance and error distributions for different strategies and dimensionality reduction techniques.

- **Pre-production Version:**  
  This is a **pre-production version** of the code. It is fully functional for testing and experimentation, but the final **HYBRID model** and our advanced dimensionality reduction algorithms will be integrated in future versions after publication.

---

## Installation

1. Make sure you are in the **comet directory** (root of this project).  
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Install the project as a package to avoid Python path issues:

```bash
python -m pip install -e .
```

## Running the Model
To run the forecasting script:
```bash
python -m research_project.comparisons
```

This will train the models, predict load values, compute evaluation metrics, and save visualizations in the project directory.
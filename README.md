# Smart Meter Load Forecasting: High-Dimensional Big Data Processing via Basis Reduction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Active-orange.svg)](https://ishmaelrezaei.github.io/ComEd/)
[![DOI](https://img.shields.io/badge/DOI-Pending-red.svg)]()

> **Handling High-dimensional Data through Basis Reduction via Interactive Decomposition: Application to Smart Meter Big Data**

This repository implements advanced dimensionality reduction techniques for **short-term load forecasting** using smart meter data from **Commonwealth Edison (ComEd)**. Our research demonstrates significant computational speedups while maintaining forecasting accuracy through novel basis reduction methods.

📊 **[View Live Results & Analysis](https://esmaeil-rezaei.github.io/)**

---

## 🎯 Research Overview

Smart meter datasets present unique challenges due to their high-dimensional nature and computational requirements. Traditional methods like **Principal Component Analysis (PCA)** become computationally prohibitive for large-scale applications.

### Key Research Questions
- How can we maintain forecasting accuracy while dramatically reducing computation time?
- What are the trade-offs between dimensionality reduction speed and model performance?
- Can hybrid approaches outperform traditional methods in real-world energy applications?

---

## 🚀 Key Contributions

### Performance Breakthroughs
| Method | Speed Improvement | Accuracy | Best Use Case |
|--------|-------------------|----------|---------------|
| **RMA-RBD** | **~30× faster** than PCA | Comparable | Real-time applications |
| **RMA-HYBRID** | **~78× faster** than PCA | Maintained | Real-time and large-scale pipelines |
| **RMA-PCA** | Baseline comparison | High | Low-dimensional data |

### Novel Methodologies
- 🔬 **RBD (Reduced Basis Decomposition)**: Novel fast dimensionality reduction
- ⚡ **HYBRID (HYbrid Basis Reduction via Interactive Decomposition)**: Ultra-fast hybrid approach
- 📊 **Comparative Analysis**: Comprehensive evaluation against established methods

---

## 📈 Impact & Applications

### Immediate Benefits
- **Real-time Energy Forecasting**: Sub-second prediction capabilities
- **Scalable Big Data Processing**: Handle millions of smart meter readings
- **Cost-Effective Operations**: Reduced computational infrastructure requirements

### Target Applications
- 🏭 **Utility Companies**: Large-scale load forecasting
- 🌐 **Smart Grid Systems**: Real-time demand response
- 📱 **Energy Management Platforms**: Consumer-facing applications

---

## 🏗️ Technical Architecture

### Implemented Algorithms

#### 1. RMA (Reduced Model Approach)
```
Hierarchical PCA-based approach introduced by Alemazkoor et al (2022).
✅ Established baseline
```

#### 2. RBD (Reduced Basis Decomposition)
```
Presented by Chen (2015)
⚡ 30× speed improvement over PCA
🎯 Minimal accuracy loss
```

#### 3. HYBRID Algorithm (HYBRID)
```
Our Advanced development of RBD methodology introduced by Rezaei et al.
⚡ 78× speed improvement over PCA
🔒 Publication pending
```

### Data Pipeline
```mermaid
graph LR
    A[Smart Meter Data] --> B[Preprocessing]
    B --> C[Data Transformation]
    C --> D[Dimensionality Reduction]
    D --> E[Modela and Parameter Selection]
    E --> F[Model Training]
    F --> G[Load Forecasting]
    G --> H[Performance Evaluation]
```

---

## 📊 Experimental Results

### Computational Performance
- **PCA Baseline**: 100% computation time
- **RBD**: ~3.3% of PCA time (**30× faster**)
- **HYBRID**: ~1.3% of PCA time (**78× faster**)

### Accuracy Metrics
All methods maintain **comparable forecasting accuracy** with negligible performance degradation, making the speed improvements practically valuable.

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+**
- **pip** or **conda** package manager
- **150GB+ RAM** recommended to process one month of data for this research (October 2024)

### Installation Options

#### Option 1: Standard Installation
```bash
# Clone the repository
git clone https://github.com/ishmaelrezaei/ComEd.git
cd ComEd

# Create virtual environment (recommended)
python -m venv comed-env
source comed-env/bin/activate  # Linux/macOS
# comed-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/ishmaelrezaei/ComEd.git
cd ComEd

# Install in editable mode
pip install -e .
```
This tells Python to “link” your local project into your environment. Any changes you make to the code in the folder are immediately reflected when you import the package. As a result, you can import your project like a regular Python module from **anywhere in the active environment**, across different scripts or projects, without needing to copy the code. However, the original folder must remain in place, and the installation must be performed in each environment you use. This makes editable mode ideal for development and testing, as updates to the code are instantly available everywhere the package is used.


### Dependencies
- `numpy >= 1.19.0` - Numerical computing
- `pandas >= 1.3.0` - Data manipulation
- `scikit-learn >= 1.0.0` - Machine learning utilities
- `matplotlib >= 3.3.0` - Visualization
- `seaborn >= 0.11.0` - Statistical plotting
- `xgboost >= 1.5.0` - Gradient boosting

---

## 🚀 Usage

### Basic Execution
```bash
# Run complete comparative analysis
python -m research_project.model_comparison
```

### Advanced Configuration
```python
from src.pipelines.train_pipeline import TrainStrategiesModel
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer

# Configure experiment parameters
trainer = TrainStrategiesModel(
    dim_reduction_size=13,
    time_interval=0.5,
    strategy_names={"rma_rbd", "rma_pca"},
    ml_models_name={"Linear Regression", "XGBRegressor"},
    r=11
)

# Execute training pipeline
trainer.train()

# Alternative: Use individual components to generate model objects
data_ingestion = DataIngestion()
model_trainer = ModelTrainer()
```

### Output Structure
```
research_project/
├── FIGs/                          # Generated visualizations
│   └── dim_13_time_0.5_r_11/     # Experiment-specific results
│       ├── model_performance_comparison.pdf
│       ├── actual_vs_predicted.pdf
│       └── cumulative_error_distribution.pdf
├── model_comparison.py            # Main analysis script
data/
├── processed/                     # Processed datasets
├── raw/                          # Original data files
└── raw_unzipped/                 # Extracted data
models/                           # Trained model artifacts
logs/                             # Detailed execution logs
```

---

## 📋 Project Structure

```
ComEd/
├── 📁 data/                       # Data storage and management
│   ├── 📁 processed/             # Processed datasets
│   ├── 📁 raw/                   # Original raw data files
│   └── 📁 raw_unzipped/          # Extracted raw data
├── 📁 logs/                      # Execution and error logs
│   ├── 📄 09_20_2025_15_14_49.log
│   ├── 📄 09_20_2025_15_23_48.log
│   └── 📄 09_20_2025_15_43_01.log
├── 📁 models/                    # Trained model artifacts
├── 📁 notebook/                  # Jupyter notebooks for analysis
├── 📁 research_project/          # Research experiments
│   ├── 📁 FIGs/                 # Generated visualizations
│   │   └── 📁 dim_13_time_0.5_r_11/  # Experiment-specific plots
│   └── 📄 model_comparison.py    # Main comparison script
├── 📁 src/                       # Core source code
│   ├── 📄 __init__.py
│   ├── 📁 components/            # Core ML components
│   │   ├── 📄 __init__.py
│   │   ├── 📄 data_ingestion.py  # Data loading and preprocessing
│   │   ├── 📄 data_transformation.py  # Feature engineering
│   │   ├── 📄 encoder_pipeline.py     # Encoding strategies
│   │   └── 📄 model_trainer.py   # Model training logic
│   ├── 📄 exception.py           # Custom exception handling
│   ├── 📄 logger.py              # Logging configuration
│   ├── 📁 pipelines/             # ML pipelines
│   │   ├── 📄 __init__.py
│   │   ├── 📄 predict_pipeline.py     # Prediction pipeline
│   │   └── 📄 train_pipeline.py       # Training pipeline
│   └── 📄 utils.py               # Utility functions
├── 📄 setup.py                   # Package installation script
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # This file
└── 📄 LICENSE                    # License information
```

---

## 📊 Reproducible Research

### Verification Steps
1. **Environment Setup**: Follow installation instructions
2. **Data Validation**: Ensure ComEd dataset accessibility
3. **Execution**: Run comparison script
4. **Results Analysis**: Review generated figures and metrics

### Expected Outputs
- Comparative performance metrics (MAPE, Coefficient of Error)
- Visualization plots (actual vs. predicted, error distributions)
- Model artifacts for further analysis
- Detailed execution logs

---

## 🔬 Related Work & References

### Primary Research
Our work builds upon and extends:

> **N. Alemazkoor, M. Tootkaboni, R. Nateghi, A. Louhghalam.**  
> *"Smart-Meter Big Data for Load Forecasting: An Alternative Approach to Clustering."*  
> **IEEE Access**, vol. 10, pp. 8377-8387, 2022.  
> DOI: [10.1109/ACCESS.2022.3144227](https://doi.org/10.1109/ACCESS.2022.3144227)

> **Y. Chen.**  
> *"Reduced Basis Decomposition: A Certified and Fast Lossy Data Compression Algorithm."*  
> **Computers & Mathematics with Applications**, vol. 70, no. 10, pp. 2566-2574, 2015.  
> Publisher: Elsevier.

> **Y. Chen, S. Gottlieb, L. Ji, Y. Maday.**  
> *"An EIM-Degradation Free Reduced Basis Method via Over Collocation and Residual Hyper Reduction-Based Error Estimation."*  
> **Journal of Computational Physics**, vol. 444, pp. 110545, 2021.  
> Publisher: Elsevier.


### Research Context
- **CBA (Cluster-Based Algorithm)**: Traditional clustering approaches
- **AMA (Aggregated Model Approach)**: Aggregation-based methods  
- **RMA (Reduced Model Approach)**: Hierarchical PCA foundation
- **RBD (Reduced Basis Decomposition)**: Certified data compression
- **ROC (Reduced Over Collocation)**: Reduced basis method
- **Our Contributions**: HYBRID (tabular and time series) methodologies

---


### Pending Publication
Our hybrid algorithm research is about to submit. Citation details will be updated upon publication.

---

## 🤝 Contributing

We welcome contributions from the research community!

### Areas for Contribution
- 🧪 **Algorithm Improvements**: Enhanced basis reduction methods  
- 🤖 **AI Integration**: Leveraging AI models for improved load forecasting  
- 🧩 **Multimodal Data**: Incorporating multiple data sources, such as smart meters, weather, and operational metadata  
- 📊 **Dataset Expansion**: Additional utility company data  
- 🔧 **Performance Optimization**: Computational efficiency gains through hierarchical and hybrid dimensionality reduction techniques  
- 📚 **Documentation**: Examples, tutorials, and guidance for replicating results and extending the code to new datasets and scenarios  



---

## 📞 Contact & Support

- 🧑‍🔬 **Author**: [Esmaeil (Ishmael) Rezaei, Ph.D.](https://esmaeil-rezaei.github.io/)  
- 🎓 **Institution**: University of Massachusetts Dartmouth  
- 🔗 **LinkedIn**: [linkedin.com/in/esmaeil-rezaei](https://www.linkedin.com/in/esmaeil-rezaei/)


### Getting Help
- 🐛 **Bug Reports**: Contact me on [LinkedIn](https://www.linkedin.com/in/esmaeil-rezaei/)  
- 💡 **Feature Requests**: Contact me on [LinkedIn](https://www.linkedin.com/in/esmaeil-rezaei/)  
- 📧 **Research Collaboration**: Contact me on [LinkedIn](https://www.linkedin.com/in/esmaeil-rezaei/)


---

## 📜 License

This project is licensed under the MIT License.

```
MIT License - Copyright (c) 2024 Esmaeil Rezaei
```

---

## 🚧 Roadmap

### Upcoming Features 🔄
- [ ] **HYBRID Algorithm for Tabular Data**
- [ ] **HYBRID Algorithm for Time Series Data**
- [ ] **Dimensionality Reduction Package for Big Data**: Consolidate and organize various dimensionality reduction algorithms into a unified package



### Future Research 🔬
- [ ] **Deep Learning Integration** with basis reduction  
- [ ] **Transformers for Time Series Forecasting** integration
- [ ] **Transformers for Multimodal Data**: modeling and optimization


---

<div align="center">

**⭐ If this research helps your work, please consider starring the repository!**

[![GitHub stars](https://img.shields.io/github/stars/esmaeil-rezaei/ComEd-ML.svg?style=social&label=Star)](https://github.com/esmaeil-rezaei/ComEd-ML)
[![GitHub forks](https://img.shields.io/github/forks/esmaeil-rezaei/ComEd-ML.svg?style=social&label=Fork)](https://github.com/esmaeil-rezaei/ComEd-ML/fork)

</div>

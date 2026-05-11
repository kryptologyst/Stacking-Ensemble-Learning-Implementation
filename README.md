# Stacking Ensemble Learning Implementation

A comprehensive implementation of stacking ensemble learning methods with advanced variants, evaluation frameworks, and interactive demonstrations.

## Overview

This project implements various stacking ensemble learning techniques, from basic stacking to advanced multi-level ensembles. It provides a complete framework for comparing different ensemble methods, evaluating their performance, and visualizing results.

### Key Features

- **Multiple Ensemble Methods**: Simple stacking, advanced multi-level stacking, blending, and voting ensembles
- **Comprehensive Evaluation**: Cross-validation, multiple metrics, and statistical analysis
- **Advanced Models**: Support for XGBoost, LightGBM, and other advanced base learners
- **Interactive Demo**: Streamlit-based web application for experimentation
- **Reproducible Research**: Deterministic seeding and comprehensive logging
- **Modern Architecture**: Clean, typed code with proper documentation

## Safety and Ethics Disclaimer

**⚠️ IMPORTANT: This is a research and educational demonstration only.**

- **Not for Production Use**: This implementation is designed for research, education, and algorithm comparison only
- **No Real-World Decisions**: Results should not be used for medical, financial, or critical decision-making
- **Research Focus**: Intended for understanding ensemble learning concepts and comparing algorithms
- **Expert Consultation**: Always consult domain experts for real-world applications

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For advanced features, install additional packages:

```bash
# For XGBoost support
pip install xgboost

# For LightGBM support  
pip install lightgbm

# For advanced visualization
pip install plotly

# For hyperparameter optimization
pip install optuna
```

## Quick Start

### Command Line Interface

Run a quick experiment:

```bash
python run_experiment.py --dataset breast_cancer --quick
```

Run a comprehensive experiment:

```bash
python run_experiment.py --dataset breast_cancer --models simple_stacking advanced_stacking blending
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

Then open your browser to `http://localhost:8501`

## Usage

### Basic Usage

```python
from src.train.trainer import run_quick_experiment

# Run a quick experiment
results = run_quick_experiment(
    dataset_name="breast_cancer",
    models=["simple_stacking", "advanced_stacking"],
    random_state=42
)

print(f"Best accuracy: {results['summary']['best_accuracy']:.4f}")
```

### Advanced Usage

```python
from src.train.trainer import EnsembleTrainer
from src.data.loader import DataLoader

# Initialize components
trainer = EnsembleTrainer(random_state=42)
data_loader = DataLoader(random_state=42)

# Load data
X_train, X_test, y_train, y_test, metadata = data_loader.load_dataset(
    "breast_cancer", return_metadata=True
)

# Run experiment
results = trainer.run_experiment(
    dataset_name="breast_cancer",
    models_to_test=["simple_stacking", "advanced_stacking", "blending"],
    save_results=True
)

# Run ablation study
ablation_results = trainer.run_ablation_study(
    dataset_name="breast_cancer",
    base_model="simple_stacking"
)
```

## Available Datasets

- **breast_cancer**: Breast cancer classification (30 features, 2 classes)
- **wine**: Wine quality classification (13 features, 3 classes)
- **iris**: Iris species classification (4 features, 3 classes)
- **synthetic**: Synthetic binary classification (20 features, 2 classes)

## Available Models

### Ensemble Methods

1. **Simple Stacking**: Basic stacking with cross-validation generated meta-features
2. **Advanced Stacking**: Multi-level stacking with advanced base models
3. **Blending**: Holdout-based blending ensemble
4. **Sklearn Stacking**: Built-in StackingClassifier for comparison
5. **Voting**: Soft voting ensemble for baseline comparison

### Base Models

- Random Forest
- Gradient Boosting
- Support Vector Machine
- k-Nearest Neighbors
- Decision Tree
- XGBoost (if available)
- LightGBM (if available)

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1 Score**: Weighted average F1 score
- **ROC AUC**: Area under ROC curve
- **Log Loss**: Logarithmic loss
- **Cross-Validation**: Mean and standard deviation of CV scores

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Ensemble model implementations
│   ├── metrics/           # Evaluation metrics and tools
│   ├── train/             # Training utilities
│   ├── viz/               # Visualization tools
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── demo/                  # Interactive demo application
├── assets/                # Generated results and plots
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── run_experiment.py      # Main experiment script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Configuration

Configuration is managed through YAML files in the `configs/` directory. Key settings include:

- Dataset selection and preprocessing
- Model hyperparameters
- Evaluation metrics
- Visualization settings
- Hyperparameter search parameters

## Results and Visualization

The framework generates comprehensive results including:

- **Performance Metrics**: Detailed evaluation of all models
- **Leaderboard**: Ranked comparison of model performance
- **Visualizations**: Performance plots, CV distributions, training times
- **Reports**: Text-based evaluation reports
- **Interactive Dashboard**: Plotly-based interactive visualizations

## Advanced Features

### Hyperparameter Search

```python
# Run hyperparameter optimization
search_results = trainer.run_hyperparameter_search(
    dataset_name="breast_cancer",
    model_name="simple_stacking",
    param_grid={
        'cv_folds': [3, 5, 7],
        'meta_model': ['logistic', 'ridge', 'svm']
    }
)
```

### Ablation Studies

```python
# Run ablation study
ablation_results = trainer.run_ablation_study(
    dataset_name="breast_cancer",
    base_model="simple_stacking"
)
```

### Custom Datasets

```python
# Load custom dataset
X_train, X_test, y_train, y_test = data_loader.load_custom_data(
    file_path="path/to/data.csv",
    target_column="target"
)
```

## Reproducibility

The implementation ensures reproducibility through:

- **Deterministic Seeding**: All random operations use fixed seeds
- **Logging**: Comprehensive logging of all operations
- **Version Control**: Clear versioning of all components
- **Documentation**: Detailed documentation of all methods

## Performance Considerations

- **Cross-Validation**: Proper CV implementation prevents data leakage
- **Memory Efficiency**: Optimized for reasonable memory usage
- **Parallel Processing**: Support for parallel model training where possible
- **Caching**: Results caching to avoid recomputation

## Contributing

This is a research and educational project. Contributions are welcome for:

- Additional ensemble methods
- New evaluation metrics
- Performance optimizations
- Documentation improvements
- Bug fixes

## License

This project is provided for educational and research purposes. Please respect the intended use case and safety guidelines.

## Author

**kryptologyst**  
GitHub: https://github.com/kryptologyst

## Citation

If you use this implementation in your research, please cite:

```
@software{stacking_ensemble_2026,
  title={Stacking Ensemble Learning Implementation},
  author={kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Stacking-Ensemble-Learning-Implementation}
}
```

## Acknowledgments

- Scikit-learn team for the excellent ML library
- Streamlit team for the interactive web framework
- Plotly team for visualization capabilities
- The open-source ML community for inspiration and tools

---

**Remember**: This is a research demonstration. Always consult domain experts for real-world applications and never use these results for critical decision-making without proper validation.
# Stacking-Ensemble-Learning-Implementation

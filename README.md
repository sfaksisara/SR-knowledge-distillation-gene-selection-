# Symbolic Regression-Guided Knowledge Distillation for Interpretable Gene Selection in Cancer Classification

This repository contains the implementation of our novel approach combining symbolic regression with knowledge distillation for interpretable gene selection in cancer classification tasks.

## Abstract

Our method leverages genetic programming to discover meaningful gene interactions while using knowledge distillation to transfer complex patterns from teacher models to interpretable student models, achieving high classification performance with enhanced biological interpretability.

## Repository Structure

```
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── code.py                   # Main implementation
└── data/                     # Dataset folder
    ├── prostate.csv          # Prostate cancer dataset
    └── README.md             # Data documentation
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+
- gplearn 0.4+
- 4GB+ RAM recommended

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Step 3: Run the Experiment
```bash
python code.py
```

### Expected Output
The script will display progress through several stages:

```
-- Teacher Model Evaluation --
Accuracy : 0.8750
Precision: 0.8571
Recall   : 0.8571
F1 Score : 0.8571

===== Attempt 1 =====
Selected GP genes: [2, 7, 15, 23, 45, 67, 89]
Best symbolic model: add(mul(X2, X7), sub(X15, X23))

-- Student Model Evaluation --
Accuracy : 0.8750
Precision: 0.8571
Recall   : 0.8571
F1 Score : 0.8571

-- Classical Classifier Evaluation on GP-selected genes --
Logistic Regression:
  Accuracy : 0.8750
  Precision: 0.8571
  Recall   : 0.8571
  F1 Score : 0.8571

Random Forest:
  Accuracy : 0.9000
  Precision: 0.9000
  Recall   : 0.9000
  F1 Score : 0.9000

[... additional classifier results ...]
```

### Runtime
- Expected runtime: 5-10 minutes on standard hardware
- GPU acceleration available for PyTorch components

## Methodology Overview

The pipeline consists of several key stages:

1. **Data Preprocessing**: Load prostate cancer dataset, apply standard scaling, create train/test split (80/20)

2. **Teacher Training**: Train MLP teacher model on full gene expression feature set

3. **Soft Label Generation**: Extract probabilistic predictions from teacher model to create soft targets

4. **GP Feature Selection**: Use symbolic regression with custom fitness function to identify important gene combinations:
   - Custom confidence-weighted fitness function
   - Protected mathematical operations (exp, sin, cos, etc.)
   - Multi-objective optimization balancing accuracy and complexity

5. **Student Training**: Train compact student model using knowledge distillation:
   - Temperature-scaled soft targets
   - KL divergence loss for knowledge transfer

6. **Evaluation**: Compare performance with classical ML classifiers on GP-selected features

## Key Features

- **Custom Fitness Function**: Confidence-weighted evaluation prioritizes high-confidence predictions
- **Protected Operations**: Safe mathematical functions prevent numerical instabilities
- **Knowledge Distillation**: Temperature-scaled soft target training preserves teacher knowledge
- **Ensemble Comparison**: Comprehensive evaluation against multiple classical and ensemble methods
- **Interpretable Output**: Symbolic expressions reveal meaningful gene interactions

## Dataset

The experiment uses a prostate cancer gene expression dataset for binary classification:
- **Features**: Gene expression values (numerical)
- **Target**: Binary labels (0: normal, 1: cancer)
- **Preprocessing**: StandardScaler normalization applied
- **Split**: 80% training, 20% testing with stratification

## Reproducibility

All experiments ensure reproducible results through:
- Fixed random seeds (`random_state=42` for data splitting)
- PyTorch manual seed setting for model initialization
- Controlled GP evolution parameters
- Deterministic evaluation procedures

To reproduce exact results from the paper:
```bash
python code.py  # Uses fixed seeds internally
```

## Model Architecture

### Teacher Model (MLP)
```python
Sequential(
    Linear(n_features, 128),
    ReLU(),
    Linear(128, 2)
)
```

### Student Model (MLP)
```python
Sequential(
    Linear(n_selected_features, 128), 
    ReLU(),
    Linear(128, 2)
)
```

### Symbolic Regression Configuration
- **Population size**: 300
- **Generations**: 200
- **Function set**: Linear (add, sub, mul, div) + nonlinear (exp, sin, cos, sqrt, log)
- **Selection pressure**: Tournament selection
- **Parsimony coefficient**: 0.001 (complexity penalty)

## Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall classification correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/true positive rate  
- **F1-Score**: Harmonic mean of precision and recall

## Troubleshooting

### Common Issues

**ImportError for gplearn:**
```bash
pip install gplearn
```

**CUDA out of memory:**
- Reduce batch size or use CPU-only mode
- Models are small enough to run efficiently on CPU

**Dataset not found:**
- Ensure `data/prostate.csv` exists in the repository
- Check file path in `code.py` points to `data/prostate.csv`

### System Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: ~50MB for code and data
- **CPU**: Any modern processor (GPU optional)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Submit pull requests
- Ask questions in the Issues section

## Contact

For questions about the implementation or methodology, please:
- Open an issue in this repository
- Contact: [sara.sfaksi@univ-biskra.dz]

## Acknowledgments

- Built using PyTorch, scikit-learn, and gplearn
- Inspired by recent advances in knowledge distillation and symbolic regression
- Dataset preprocessing follows standard bioinformatics practices

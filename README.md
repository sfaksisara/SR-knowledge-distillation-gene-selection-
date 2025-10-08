# Symbolic Regression-Guided Knowledge Distillation for Interpretable Gene Selection in Cancer Classification

This repository presents a novel hybrid machine learning pipeline that combines genetic programming (GP), knowledge distillation, and nested cross-validation for robust feature selection and classification in high-dimensional biomedical datasets.

## Abstract

We introduce a three-stage methodology that addresses the critical challenge of feature selection in genomic data while maintaining strict adherence to best practices for avoiding data leakage:
Teacher Network Training: A deep neural network (MLP) is trained on the full feature space to learn complex non-linear patterns
GP-Guided Feature Selection: Genetic programming evolves symbolic expressions to approximate the teacher's soft predictions, automatically identifying the most informative features through evolutionary search
Student Network Distillation: A compact student network is trained on selected features using knowledge distillation from the teacher's soft labels

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

Methodology NotesCore Pipeline (Nested 5-Fold Cross-Validation)Validation Protocol

Outer CV: 5 stratified folds maintaining class proportions
Training/Test Split: 80%/20% per fold
Key Principle: All feature selection occurs exclusively within training folds to prevent data leakage
Step-by-Step Process (Per Fold k)Step 1: Fold-Specific Normalization

Compute mean and standard deviation exclusively from training data
Normalize training data using these parameters
Apply same parameters to test data (computed only from training)
Critical: Test set statistics never influence normalization
Step 2: Teacher Model Training
Architecture:

Input: M genes (full feature set)
Hidden: 128 neurons (ReLU activation)
Output: 2 neurons (Softmax activation)
Training Configuration:

Loss: CrossEntropyLoss on hard labels (0/1)
Optimizer: Adam (learning rate=10⁻⁴, β₁=0.9, β₂=0.999)
Early stopping: 15 epochs patience, 20% validation split
Random seed: 42 + k (fold-specific)
Trained exclusively on training fold
Step 3: Soft Label Generation

Use temperature-scaled softmax (T=3.0) to generate probabilistic predictions
Extract positive class probability: ŷᵢ = probability of class 1
Purpose: Encode prediction confidence and uncertainty (not available in hard labels)
Higher temperature creates "softer" distributions preserving inter-class relationships
Step 4: Genetic Programming Feature SelectionConfiguration:

Runs per fold: 5 independent runs (handles GP stochasticity)
Generations: 300
Population: 500 individuals
Selection: Best run by lowest fitness value
Function Sets:

Primary (algebraic): add, sub, mul, div, sqrt, abs - for interpretability
Secondary (transcendental): exp, log, sin, cos, neg, inv, min, max

Constrained to outer 2 tree levels, max nesting depth = 2


Confidence-Weighted Fitness:

Sample weights: wᵢ = 2 × |ŷᵢ - 0.5|

Higher weight for confident predictions (near 0 or 1)
Lower weight for uncertain predictions (near 0.5)


Weighted R² fitness measures how well expression predicts soft labels
Multi-Objective Optimization:

Balance accuracy and interpretability
Fitness = (1 - R²_weighted) + λ × Complexity
Complexity = tree depth + 0.01 × number of nodes
Parsimony coefficient λ = 0.001
Genetic Operators:

Tournament selection (k=3) for diversity
Crossover probability: 0.7 (exchange subtrees between parents)
Subtree mutation: 0.1 (random alterations)
Hoist mutation: 0.05 (promote subtree to root)
Point mutation: 0.1 (change single node)
Max tree depth: 6
Gene Extraction:

Parse best expression string
Extract all feature indices appearing in expression
Result: subset S* of selected genes
Step 5: Model Training on Selected FeaturesStudent MLP (Knowledge Distillation):

Input: |S*| selected genes only
Hidden: 64 neurons (ReLU activation)
Output: 2 neurons (Softmax activation)
Loss: KL divergence between student and teacher soft predictions
Temperature T=3.0 for distillation, reset to 1.0 for inference
Classical Classifiers (all trained on selected features):

Logistic Regression (max_iter=1000)
Random Forest (100 trees)
SVM (linear kernel, C=1.0)
Linear Discriminant Analysis (default parameters)
K-Nearest Neighbors (k=5, euclidean distance)
Gradient Boosting (100 estimators, learning rate=0.1)
AdaBoost (50 estimators, learning rate=1.0)
Voting Ensemble (LR, RF, SVM with soft voting)
Stacking Ensemble (LR, RF, SVM with LR meta-learner)
All classifiers use random_state=42 for determinismStep 6: Held-Out Evaluation
Only after training completes, evaluate on isolated test fold:

Classification: Accuracy, Precision, Recall, F1-Score
Discrimination: ROC-AUC, PR-AUC
Calibration: Expected Calibration Error (ECE), Brier Score
Confusion Matrix: True/False Positives/Negatives
Statistical Aggregation (Across 5 Folds)After all K=5 folds complete:

Compute mean performance metrics
Compute standard deviation across folds
Calculate 95% confidence intervals using t-distribution (t₄,₀.₉₇₅ = 2.776)
Statistical Validation:

Permutation p-values: 1000 random label shuffles per fold
Significance threshold: α = 0.05
All models achieved p < 0.001 (highly significant)

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

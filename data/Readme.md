# Data folder
This folder contains Some of datasets used in the experiments.
# Dataset Information

- **Format**: CSV file with gene expression profiles
- **Size**: [X samples] × [Y features]  
- **Target**: Binary classification (0: normalor type1, 1: cancer or type2)
- **Usage**: Main dataset for the experiments in main.py

## Data Format
- Features: Numerical gene expression values
- Target column: 'label' with values 0 (normal) and 1 (cancer)
- Preprocessing: StandardScaler normalization applied in the code

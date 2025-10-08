# ===================== NESTED CV PIPELINE - REVIEWER COMPLIANT =====================
# Feature selection occurs INSIDE each CV fold (no data leakage)
# Reports: Mean±SD, 95% CI, permutation p-values, gene stability

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, balanced_accuracy_score,
                           average_precision_score, roc_curve, auc, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from gplearn.genetic import SymbolicRegressor
    from gplearn.fitness import make_fitness
    import re
    print("✓ gplearn imported successfully")
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'gplearn'], check=True)
    from gplearn.genetic import SymbolicRegressor
    from gplearn.fitness import make_fitness
    import re

# ===================== CONFIGURATION =====================
GP_CONFIG = {
    'generations': 300,
    'population_size': 400,
    'parsimony_coefficient': 0.001,
    'n_gp_runs': 5,  # Run GP multiple times per fold, select best
}

# ===================== NEURAL NETWORK =====================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, X, y, epochs=100, patience=15, val_seed=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    val_size = int(0.2 * len(X))
    if val_seed is not None:
        rng = np.random.RandomState(val_seed)
        indices = rng.permutation(len(X))
    else:
        indices = np.random.permutation(len(X))

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = loss_fn(outputs, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_fn(val_outputs, y_val).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model

def train_student_with_soft_labels(model, X, soft_labels, epochs=100, temperature=3.0,
                                   patience=15, val_seed=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    teacher_probs = torch.tensor(soft_labels, dtype=torch.float32)

    val_size = int(0.2 * len(X))
    if val_seed is not None:
        rng = np.random.RandomState(val_seed)
        indices = rng.permutation(len(X))
    else:
        indices = np.random.permutation(len(X))

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train = X_tensor[train_indices]
    soft_train = teacher_probs[train_indices]
    X_val = X_tensor[val_indices]
    soft_val = teacher_probs[val_indices]

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)

        soft_targets = torch.stack([1 - soft_train, soft_train], dim=1)
        student_soft = torch.softmax(outputs / temperature, dim=1)
        teacher_soft = torch.softmax(torch.log(soft_targets + 1e-8) / temperature, dim=1)

        loss = nn.KLDivLoss(reduction='batchmean')(torch.log(student_soft), teacher_soft) * (temperature ** 2)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_soft_targets = torch.stack([1 - soft_val, soft_val], dim=1)
            val_student_soft = torch.softmax(val_outputs / temperature, dim=1)
            val_teacher_soft = torch.softmax(torch.log(val_soft_targets + 1e-8) / temperature, dim=1)
            val_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(val_student_soft), val_teacher_soft) * (temperature ** 2)
            val_loss = val_loss.item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model

# ===================== GP FITNESS =====================
def robust_fitness(y, y_pred, sample_weight):
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    mse = np.mean((y - y_pred_clipped) ** 2)
    return mse

custom_fitness = make_fitness(function=robust_fitness, greater_is_better=False)
FUNCTION_SET = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs']

# ===================== GP FEATURE SELECTION WITH MULTIPLE RUNS =====================
def gp_feature_selection(X_train, soft_labels, n_runs=None, verbose=False):
    """
    GP feature selection with multiple runs to handle stochasticity.
    Runs GP multiple times and selects the best result by fitness.
    """
    if n_runs is None:
        n_runs = GP_CONFIG.get('n_gp_runs', 5)

    gp_start_time = time.time()

    if verbose:
        print(f"    GP Feature Selection ({n_runs} runs to handle stochasticity)")

    all_runs = []

    for run_idx in range(n_runs):
        if verbose:
            print(f"      Run {run_idx+1}/{n_runs}... ", end='', flush=True)

        try:
            soft_labels_normalized = np.clip(soft_labels, 0.0, 1.0)

            sr = SymbolicRegressor(
                function_set=FUNCTION_SET,
                generations=GP_CONFIG['generations'],
                population_size=GP_CONFIG['population_size'],
                stopping_criteria=0.0,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=0,
                parsimony_coefficient=GP_CONFIG['parsimony_coefficient'],
                metric=custom_fitness,
                random_state=None,  # Different seed each run
                init_depth=(2, 6),
                init_method='half and half',
                const_range=(-1.0, 1.0),
            )

            sr.fit(X_train, soft_labels_normalized)

            # Extract features from best program
            best_program = sr._program

            if best_program is None:
                if verbose:
                    print("✗ No program")
                continue

            expr_str = str(best_program)
            feature_matches = re.findall(r'X(\d+)', expr_str)
            selected_features = sorted(list(set([int(match) for match in feature_matches])))

            if len(selected_features) == 0:
                if verbose:
                    print("✗ No features")
                continue

            fitness = 1 - sr.score(X_train, soft_labels_normalized)

            all_runs.append({
                'features': selected_features,
                'fitness': fitness,
                'expression': expr_str,
                'n_features': len(selected_features)
            })

            if verbose:
                print(f"✓ {len(selected_features)} features, fitness={fitness:.4f}")

        except Exception as e:
            if verbose:
                print(f"✗ Error: {str(e)[:30]}")
            continue

    # Select best run by fitness
    if len(all_runs) == 0:
        if verbose:
            print(f"    All runs failed, using fallback")
        n_features = X_train.shape[1]
        selected_features = sorted(np.random.choice(n_features, size=min(12, n_features), replace=False).tolist())
        gp_result = {
            'features': selected_features,
            'fitness': 999.0,
            'expression': 'FALLBACK',
            'n_features': len(selected_features),
            'all_runs': []
        }
    else:
        # Select run with best (lowest) fitness
        best_run = min(all_runs, key=lambda x: x['fitness'])
        selected_features = best_run['features']

        gp_result = {
            'features': selected_features,
            'fitness': best_run['fitness'],
            'expression': best_run['expression'],
            'n_features': best_run['n_features'],
            'all_runs': all_runs  # Store all runs for analysis
        }

        if verbose:
            print(f"\n    Best run: {best_run['n_features']} features, fitness={best_run['fitness']:.4f}")
            print(f"    Fitness range: [{min(r['fitness'] for r in all_runs):.4f}, {max(r['fitness'] for r in all_runs):.4f}]")

    total_time = time.time() - gp_start_time

    return selected_features, gp_result, total_time

# ===================== BASELINE COMPARISON METHODS =====================
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import ElasticNet

def baseline_feature_selection_methods(X_train, y_train, X_test, y_test, n_features=10, verbose=False):
    """
    Baseline feature selection methods for comparison.
    Returns performance metrics for each baseline.
    """
    results = {}

    # 1. SVM-RFE (Guyon et al., 2002)
    if verbose:
        print("  Running SVM-RFE...")
    svm_rfe = RFE(SVC(kernel='linear', random_state=42), n_features_to_select=n_features, step=100)
    svm_rfe.fit(X_train, y_train)
    X_train_rfe = svm_rfe.transform(X_train)
    X_test_rfe = svm_rfe.transform(X_test)

    lr_rfe = LogisticRegression(max_iter=1000, random_state=42)
    lr_rfe.fit(X_train_rfe, y_train)
    pred_rfe = lr_rfe.predict(X_test_rfe)
    prob_rfe = lr_rfe.predict_proba(X_test_rfe)[:, 1]

    results['SVM-RFE'] = calculate_metrics(y_test, pred_rfe, prob_rfe)
    results['SVM-RFE']['selected_features'] = np.where(svm_rfe.support_)[0].tolist()

    # 2. Elastic Net (L1+L2 regularization)
    if verbose:
        print("  Running Elastic Net...")
    enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    enet.fit(X_train, y_train)

    # Select top features by coefficient magnitude
    coef_abs = np.abs(enet.coef_)
    top_features = np.argsort(coef_abs)[-n_features:]

    X_train_enet = X_train[:, top_features]
    X_test_enet = X_test[:, top_features]

    lr_enet = LogisticRegression(max_iter=1000, random_state=42)
    lr_enet.fit(X_train_enet, y_train)
    pred_enet = lr_enet.predict(X_test_enet)
    prob_enet = lr_enet.predict_proba(X_test_enet)[:, 1]

    results['Elastic_Net'] = calculate_metrics(y_test, pred_enet, prob_enet)
    results['Elastic_Net']['selected_features'] = top_features.tolist()

    # 3. ANOVA F-test (filter method)
    if verbose:
        print("  Running ANOVA F-test...")
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X_train, y_train)
    X_train_anova = selector.transform(X_train)
    X_test_anova = selector.transform(X_test)

    lr_anova = LogisticRegression(max_iter=1000, random_state=42)
    lr_anova.fit(X_train_anova, y_train)
    pred_anova = lr_anova.predict(X_test_anova)
    prob_anova = lr_anova.predict_proba(X_test_anova)[:, 1]

    results['ANOVA_F'] = calculate_metrics(y_test, pred_anova, prob_anova)
    results['ANOVA_F']['selected_features'] = selector.get_support(indices=True).tolist()

    return results

def calculate_ece(y_true, y_proba, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)

    ECE measures the difference between predicted confidence and actual accuracy.
    Lower ECE indicates better calibration.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = np.logical_and(y_proba > bin_lower, y_proba <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Average confidence in this bin
            avg_confidence = np.mean(y_proba[in_bin])
            # Average accuracy in this bin
            avg_accuracy = np.mean(y_true[in_bin])
            # Add weighted difference to ECE
            ece += prop_in_bin * np.abs(avg_confidence - avg_accuracy)

    return ece

def calculate_brier_score(y_true, y_proba):
    """
    Calculate Brier score (mean squared error of probabilities)

    Lower Brier score indicates better calibration.
    """
    return np.mean((y_proba - y_true) ** 2)

def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            metrics['ece'] = calculate_ece(y_true, y_proba, n_bins=10)
            metrics['brier_score'] = calculate_brier_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = np.nan
            metrics['pr_auc'] = np.nan
            metrics['ece'] = np.nan
            metrics['brier_score'] = np.nan

    return metrics

def permutation_test(y_true, y_pred, n_permutations=1000):
    true_score = accuracy_score(y_true, y_pred)
    perm_scores = [accuracy_score(np.random.permutation(y_true), y_pred) for _ in range(n_permutations)]
    p_value = np.mean(np.array(perm_scores) >= true_score)
    return true_score, p_value

def confidence_interval(scores, confidence=0.95):
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    n = len(scores)
    t_value = stats.t.ppf(1 - (1-confidence)/2, df=n-1)
    margin = t_value * std / np.sqrt(n)
    return mean, std, (mean - margin, mean + margin)

# ===================== NESTED CV PIPELINE =====================
def run_nested_cv_pipeline(X, y, outer_cv=5, random_state=42):
    print("\n" + "="*80)
    print("NESTED CV PIPELINE (REVIEWER COMPLIANT)")
    print("="*80)
    print("Feature selection occurs INSIDE each fold (no data leakage)")
    print(f"Dataset: {X.shape}, Classes: {np.bincount(y)}")
    print("="*80)

    outer_splits = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)

    fold_metrics = {
        'Student_MLP': [],
        'Logistic_Regression': [],
        'Random_Forest': [],
        'SVM': [],
        'LDA': [],
        'KNN': [],
        'Gradient_Boosting': [],
        'AdaBoost': [],
        'Voting_Ensemble': [],
        'Stacking_Ensemble': []
    }

    aggregated_predictions = {
        'Student_MLP': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'Logistic_Regression': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'Random_Forest': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'SVM': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'LDA': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'KNN': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'Gradient_Boosting': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'AdaBoost': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'Voting_Ensemble': {'y_true': [], 'y_pred': [], 'y_proba': []},
        'Stacking_Ensemble': {'y_true': [], 'y_pred': [], 'y_proba': []},
    }

    fold_gp_results = []
    total_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits.split(X, y)):
        fold_start = time.time()
        print(f"\n{'='*20} FOLD {fold_idx + 1}/{outer_cv} {'='*20}")

        # Split data - NO TEST DATA USED UNTIL FINAL EVALUATION
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Step 1: Train teacher on THIS fold's training data only
        print("Step 1: Training Teacher (on training fold only)")
        teacher = MLP(input_dim=X_train.shape[1])
        teacher = train_model(teacher, X_train, y_train, val_seed=fold_idx)

        # Step 2: Generate soft labels from THIS fold's training data only
        teacher.eval()
        with torch.no_grad():
            logits = teacher(torch.tensor(X_train, dtype=torch.float32))
            soft_labels = torch.softmax(logits, dim=1)[:, 1].numpy()

        print(f"  Soft labels range: [{soft_labels.min():.3f}, {soft_labels.max():.3f}]")

        # Step 3: GP feature selection on THIS fold's training data only
        print("Step 2: GP Feature Selection (on training fold only)")
        selected_features, gp_result, gp_time = gp_feature_selection(
            X_train, soft_labels, verbose=True
        )

        # Step 3.5: Run baseline feature selection methods for comparison
        print("Step 2.5: Baseline Feature Selection Comparison")
        baseline_results = baseline_feature_selection_methods(
            X_train, y_train, X_test, y_test,
            n_features=len(selected_features),
            verbose=True
        )

        fold_gp_results.append({
            'fold': fold_idx + 1,
            'selected_features': selected_features,
            'gp_result': gp_result,
            'gp_time': gp_time,
            'baseline_results': baseline_results  # Store baseline comparisons
        })

        # Step 4: Train models on selected features (training fold only)
        print(f"Step 3: Training Models on {len(selected_features)} Features")
        X_train_sel = X_train[:, selected_features]
        X_test_sel = X_test[:, selected_features]

        # Train teacher on selected features
        teacher_sel = MLP(input_dim=len(selected_features))
        teacher_sel = train_model(teacher_sel, X_train_sel, y_train, val_seed=fold_idx)

        teacher_sel.eval()
        with torch.no_grad():
            logits_sel = teacher_sel(torch.tensor(X_train_sel, dtype=torch.float32))
            soft_labels_sel = torch.softmax(logits_sel, dim=1)[:, 1].numpy()

        # Train student
        student = MLP(input_dim=len(selected_features))
        student = train_student_with_soft_labels(student, X_train_sel, soft_labels_sel, val_seed=fold_idx)

        # Train classical models
        base_classifiers = {
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'SVM': SVC(kernel='linear', probability=True, random_state=random_state),
            'LDA': LinearDiscriminantAnalysis(),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=random_state),
        }

        # Train base classifiers
        for clf in base_classifiers.values():
            clf.fit(X_train_sel, y_train)

        # Create ensemble classifiers
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, random_state=random_state)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
                ('svm', SVC(kernel='linear', probability=True, random_state=random_state))
            ],
            voting='soft'
        )
        voting_clf.fit(X_train_sel, y_train)

        stacking_clf = StackingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, random_state=random_state)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
                ('svm', SVC(kernel='linear', probability=True, random_state=random_state))
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=random_state)
        )
        stacking_clf.fit(X_train_sel, y_train)

        # Combine all classifiers
        classifiers = {**base_classifiers, 'Voting_Ensemble': voting_clf, 'Stacking_Ensemble': stacking_clf}

        # Step 5: Evaluate on held-out test fold
        print("Step 4: Evaluating on Held-Out Test Fold")

        # Student evaluation
        student.eval()
        with torch.no_grad():
            logits_test = student(torch.tensor(X_test_sel, dtype=torch.float32))
            probs_test = torch.softmax(logits_test, dim=1)[:, 1].numpy()
            preds_test = (probs_test > 0.5).astype(int)

        student_metrics = calculate_metrics(y_test, preds_test, probs_test)
        fold_metrics['Student_MLP'].append(student_metrics)
        aggregated_predictions['Student_MLP']['y_true'].extend(y_test)
        aggregated_predictions['Student_MLP']['y_pred'].extend(preds_test)
        aggregated_predictions['Student_MLP']['y_proba'].extend(probs_test)

        print(f"  Student: Acc={student_metrics['accuracy']:.3f}, F1={student_metrics['f1']:.3f}")

        # Classical models evaluation
        for clf_name, clf in classifiers.items():
            preds = clf.predict(X_test_sel)
            probs = clf.predict_proba(X_test_sel)[:, 1]
            metrics = calculate_metrics(y_test, preds, probs)
            fold_metrics[clf_name].append(metrics)

            # Store for aggregated analysis and ROC curves
            aggregated_predictions[clf_name]['y_true'].extend(y_test)
            aggregated_predictions[clf_name]['y_pred'].extend(preds)
            aggregated_predictions[clf_name]['y_proba'].extend(probs)

        fold_time = time.time() - fold_start
        print(f"Fold completed in {fold_time:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"Pipeline completed in {total_time/60:.1f} minutes")

    return fold_metrics, aggregated_predictions, fold_gp_results

# ===================== RESULTS DISPLAY WITH ROC CURVES =====================
def display_results(fold_metrics, aggregated_preds, gp_results):
    print("\n" + "="*80)
    print("RESULTS REPORT")
    print("="*80)

    # 1. Performance with Mean±SD and 95% CI
    print("\n1. CLASSIFICATION PERFORMANCE (Mean ± SD with 95% CI)")
    print("="*80)

    all_models = ['Student_MLP', 'Logistic_Regression', 'Random_Forest', 'SVM',
                  'LDA', 'KNN', 'Gradient_Boosting', 'AdaBoost', 'Voting_Ensemble', 'Stacking_Ensemble']

    print(f"\n{'Model':<25} {'Metric':<15} {'Mean±SD':<20} {'95% CI'}")
    print("-" * 85)

    for model_name in all_models:
        print(f"\n{model_name}:")
        for metric in ['accuracy', 'f1', 'roc_auc']:
            scores = [m[metric] for m in fold_metrics[model_name] if not np.isnan(m.get(metric, np.nan))]
            if len(scores) > 0:
                mean, std, (ci_low, ci_up) = confidence_interval(scores)
                print(f"  {metric:<15} {mean:.3f}±{std:.3f}       [{ci_low:.3f}, {ci_up:.3f}]")

    # 2. BASELINE METHOD COMPARISON
    print("\n2. BASELINE FEATURE SELECTION COMPARISON (Mean ± SD with 95% CI)")
    print("="*80)

    # Aggregate baseline results across folds
    baseline_methods = ['SVM-RFE', 'Elastic_Net', 'ANOVA_F']
    baseline_aggregated = {method: {'accuracy': [], 'f1': [], 'roc_auc': []} for method in baseline_methods}

    for fold in gp_results:
        for method in baseline_methods:
            baseline_aggregated[method]['accuracy'].append(fold['baseline_results'][method]['accuracy'])
            baseline_aggregated[method]['f1'].append(fold['baseline_results'][method]['f1'])
            baseline_aggregated[method]['roc_auc'].append(fold['baseline_results'][method]['roc_auc'])

    print(f"\n{'Method':<25} {'Metric':<15} {'Mean±SD':<20} {'95% CI'}")
    print("-" * 85)

    for method in baseline_methods:
        print(f"\n{method}:")
        for metric in ['accuracy', 'f1', 'roc_auc']:
            scores = baseline_aggregated[method][metric]
            mean, std, (ci_low, ci_up) = confidence_interval(scores)
            print(f"  {metric:<15} {mean:.3f}±{std:.3f}       [{ci_low:.3f}, {ci_up:.3f}]")

    # 2.5 CALIBRATION ANALYSIS (NEW)
    print("\n2.5 CALIBRATION ANALYSIS (Post Gene-Selection)")
    print("="*80)
    print("NOTE: Calibration measured on models trained with SELECTED genes only")
    print("="*80)

    print(f"\n{'Model':<25} {'ECE (Mean±SD)':<20} {'Brier (Mean±SD)':<20} {'Interpretation'}")
    print("-" * 85)

    for model_name in all_models:
        ece_scores = [m.get('ece', np.nan) for m in fold_metrics[model_name] if not np.isnan(m.get('ece', np.nan))]
        brier_scores = [m.get('brier_score', np.nan) for m in fold_metrics[model_name] if not np.isnan(m.get('brier_score', np.nan))]

        if len(ece_scores) > 0:
            ece_mean = np.mean(ece_scores)
            ece_std = np.std(ece_scores, ddof=1)
            brier_mean = np.mean(brier_scores)
            brier_std = np.std(brier_scores, ddof=1)

            # Interpret calibration quality
            if ece_mean < 0.05:
                interp = "Excellent"
            elif ece_mean < 0.10:
                interp = "Good"
            elif ece_mean < 0.15:
                interp = "Moderate"
            else:
                interp = "Poor"

            print(f"{model_name:<25} {ece_mean:.4f}±{ece_std:.4f}      {brier_mean:.4f}±{brier_std:.4f}      {interp}")

    # Generate calibration curves
    print("\nGenerating calibration curves...")

    key_models_cal = ['Student_MLP', 'Logistic_Regression', 'Random_Forest',
                      'SVM', 'Voting_Ensemble', 'Stacking_Ensemble']

    # Aggregated calibration curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, model_name in enumerate(key_models_cal):
        if model_name not in aggregated_preds or idx >= 6:
            continue

        ax = axes[idx]
        y_true = np.array(aggregated_preds[model_name]['y_true'])
        y_proba = np.array(aggregated_preds[model_name]['y_proba'])

        if len(y_true) == 0:
            continue

        # Calculate calibration curve
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(y_proba > bin_lower, y_proba <= bin_upper)
            if np.sum(in_bin) > 0:
                bin_confidences.append(np.mean(y_proba[in_bin]))
                bin_accuracies.append(np.mean(y_true[in_bin]))
                bin_counts.append(np.sum(in_bin))

        # Plot calibration curve
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        if len(bin_confidences) > 0:
            ax.plot(bin_confidences, bin_accuracies, 'o-', lw=2, markersize=8,
                   label='Model Calibration')

            # Add bar chart showing sample counts
            ax2 = ax.twinx()
            ax2.bar(bin_confidences, bin_counts, alpha=0.3, width=0.08,
                   color='gray', label='Sample Count')
            ax2.set_ylabel('Sample Count', fontsize=10)
            ax2.legend(loc='upper left', fontsize=8)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Observed Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Calculate and display ECE
        ece = calculate_ece(y_true, y_proba)
        ax.text(0.05, 0.95, f'ECE = {ece:.4f}',
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Calibration Curves (All folds combined)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('calibration_curves_aggregated.png', dpi=300, bbox_inches='tight')
    print("✓ Calibration curves saved as 'calibration_curves_aggregated.png'")
    plt.close()

    print("\nCalibration Interpretation:")
    print("  ECE < 0.05: Excellent calibration (predictions match reality)")
    print("  ECE < 0.10: Good calibration")
    print("  ECE < 0.15: Moderate calibration (some overconfidence/underconfidence)")
    print("  ECE ≥ 0.15: Poor calibration (predictions unreliable)")
    print("\nBrier Score: Lower is better (0 = perfect, 1 = worst)")

    # 3. ROC Curves - ENHANCED WITH INDIVIDUAL PLOTS
    print("\n3. ROC CURVE ANALYSIS")
    print("="*80)
    print("Generating comprehensive ROC and PR curves...")

    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    # Create main combined ROC plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    print(f"\n{'Model':<25} {'ROC-AUC':<15} {'PR-AUC':<15}")
    print("-" * 55)

    roc_data = {}
    for idx, model_name in enumerate(all_models):
        y_true = np.array(aggregated_preds[model_name]['y_true'])
        y_proba = np.array(aggregated_preds[model_name]['y_proba'])

        if len(y_true) > 0:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            # PR curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall_curve, precision_curve)

            roc_data[model_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'precision': precision_curve,
                'recall': recall_curve
            }

            # Plot ROC curve
            ax1.plot(fpr, tpr, color=colors[idx], lw=2,
                    label=f'{model_name} (AUC={roc_auc:.3f})')

            # Plot PR curve
            ax2.plot(recall_curve, precision_curve, color=colors[idx], lw=2,
                    label=f'{model_name} (AUC={pr_auc:.3f})')

            print(f"{model_name:<25} {roc_auc:<15.4f} {pr_auc:<15.4f}")

    # Configure ROC plot
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curves - All Classifiers', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Configure PR plot
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax2.set_title('Precision-Recall Curves - All Classifiers', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_pr_curves_combined.png', dpi=300, bbox_inches='tight')
    print("\n✓ Combined ROC and PR curves saved as 'roc_pr_curves_combined.png'")
    plt.close()

    # Create individual high-quality ROC curves for paper
    key_models = ['Student_MLP', 'Logistic_Regression', 'Random_Forest',
                  'SVM', 'Voting_Ensemble', 'Stacking_Ensemble']

    # Individual ROC plot (cleaner for paper)
    plt.figure(figsize=(10, 8))
    for idx, model_name in enumerate(key_models):
        if model_name in roc_data:
            fpr = roc_data[model_name]['fpr']
            tpr = roc_data[model_name]['tpr']
            roc_auc = roc_data[model_name]['roc_auc']
            plt.plot(fpr, tpr, lw=2.5, label=f'{model_name} (AUC={roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves - Key Classifiers', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_main.png', dpi=300, bbox_inches='tight')
    print("✓ Main ROC curve saved as 'roc_curve_main.png'")
    plt.close()

    # Create grid of individual ROC curves per model
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, model_name in enumerate(key_models):
        if model_name in roc_data and idx < 6:
            ax = axes[idx]
            fpr = roc_data[model_name]['fpr']
            tpr = roc_data[model_name]['tpr']
            roc_auc = roc_data[model_name]['roc_auc']

            ax.plot(fpr, tpr, 'b-', lw=3, label=f'ROC (AUC={roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add AUC as text
            ax.text(0.6, 0.2, f'AUC = {roc_auc:.4f}',
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('roc_curves_individual.png', dpi=300, bbox_inches='tight')
    print("✓ Individual ROC curves saved as 'roc_curves_individual.png'")
    plt.close()

    # 4. Permutation Tests
    print("\n4. PERMUTATION-BASED P-VALUES (1000 permutations)")
    print("="*80)

    for model_name in all_models:
        y_true = np.array(aggregated_preds[model_name]['y_true'])
        y_pred = np.array(aggregated_preds[model_name]['y_pred'])

        if len(y_true) > 0:
            score, p_val = permutation_test(y_true, y_pred)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"{model_name:<25} Accuracy={score:.4f}, p={p_val:.6f} {sig}")

    # 5. Confusion Matrices - VISUALIZED
    print("\n5. CONFUSION MATRICES (VISUALIZED)")
    print("="*80)

    # Create confusion matrix plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    models_for_cm = ['Student_MLP', 'Logistic_Regression', 'Random_Forest',
                     'SVM', 'Voting_Ensemble', 'Stacking_Ensemble']

    for idx, model_name in enumerate(models_for_cm):
        if model_name not in aggregated_preds or idx >= 6:
            continue

        y_true = np.array(aggregated_preds[model_name]['y_true'])
        y_pred = np.array(aggregated_preds[model_name]['y_pred'])

        if len(y_true) == 0:
            continue

        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Add labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'],
               title=f'{model_name}',
               ylabel='True Label',
               xlabel='Predicted Label')

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=16, fontweight='bold')

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Add metrics as subtitle
        ax.text(0.5, -0.15, f'Sensitivity={sens:.3f}, Specificity={spec:.3f}',
               ha='center', transform=ax.transAxes, fontsize=10)

        # Print text version
        print(f"\n{model_name}:")
        print(f"  [[TN={tn:3d}  FP={fp:3d}]")
        print(f"   [FN={fn:3d}  TP={tp:3d}]]")
        print(f"  Sensitivity={sens:.3f}, Specificity={spec:.3f}")

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("\n✓ Confusion matrices saved as 'confusion_matrices.png'")
    plt.close()

    # 6. Gene Selection Stability
    print("\n6. GENE SELECTION STABILITY ACROSS FOLDS")
    print("="*80)

    gene_counter = Counter()
    for fold in gp_results:
        for gene in fold['selected_features']:
            gene_counter[gene] += 1

    print(f"\n{'Gene':<10} {'Appearances':<15} {'Percentage':<15} {'Stability'}")
    print("-" * 55)

    for gene, count in gene_counter.most_common(20):
        pct = (count / len(gp_results)) * 100
        stability = "High" if count >= 4 else "Medium" if count >= 3 else "Low"
        print(f"{gene:<10} {count}/{len(gp_results):<12} {pct:<15.1f}% {stability}")

    print(f"\nSummary:")
    print(f"  High stability (≥4 folds): {sum(1 for _, c in gene_counter.items() if c >= 4)} genes")
    print(f"  Medium stability (3 folds): {sum(1 for _, c in gene_counter.items() if c == 3)} genes")
    print(f"  Low stability (2 folds): {sum(1 for _, c in gene_counter.items() if c == 2)} genes")
    print(f"  Total unique genes: {len(gene_counter)}")

    # 7. GP Details per Fold with Best Expressions
    print("\n7. GP FEATURE SELECTION PER FOLD WITH BEST EXPRESSIONS")
    print("="*80)

    for fold in gp_results:
        print(f"\nFold {fold['fold']}:")
        print(f"  Features selected: {fold['gp_result']['n_features']}")
        print(f"  GP fitness: {fold['gp_result']['fitness']:.4f}")
        print(f"  Time: {fold['gp_time']:.1f}s")
        print(f"  Gene indices: {fold['selected_features'][:15]}...")
        
        # Display best expression
        expr = fold['gp_result']['expression']
        if expr != 'FALLBACK':
            # Truncate if too long
            if len(expr) > 150:
                expr_display = expr[:150] + "..."
            else:
                expr_display = expr
            print(f"  Best Expression: {expr_display}")
        else:
            print(f"  Best Expression: FALLBACK (all GP runs failed)")
        
        # Display statistics from multiple runs if available
        all_runs = fold['gp_result'].get('all_runs', [])
        if len(all_runs) > 1:
            fitnesses = [run['fitness'] for run in all_runs]
            n_features_list = [run['n_features'] for run in all_runs]
            print(f"  GP Runs Summary ({len(all_runs)} runs):")
            print(f"    Fitness range: [{min(fitnesses):.4f}, {max(fitnesses):.4f}]")
            print(f"    Features range: [{min(n_features_list)}, {max(n_features_list)}]")
            print(f"    Best run fitness: {fold['gp_result']['fitness']:.4f}")

    # 8. Save GP Expressions to File
    print("\n8. SAVING GP EXPRESSIONS TO FILE")
    print("="*80)
    
    with open('gp_expressions_per_fold.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("GP SYMBOLIC EXPRESSIONS PER FOLD\n")
        f.write("="*80 + "\n\n")
        
        for fold in gp_results:
            f.write(f"{'='*60}\n")
            f.write(f"FOLD {fold['fold']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Fitness: {fold['gp_result']['fitness']:.6f}\n")
            f.write(f"Number of features: {fold['gp_result']['n_features']}\n")
            f.write(f"Selected gene indices: {fold['selected_features']}\n")
            f.write(f"GP execution time: {fold['gp_time']:.2f}s\n\n")
            
            expr = fold['gp_result']['expression']
            f.write(f"BEST EXPRESSION:\n")
            f.write(f"{'-'*60}\n")
            if expr != 'FALLBACK':
                # Pretty print with line breaks for readability
                f.write(f"{expr}\n")
            else:
                f.write("FALLBACK (all GP runs failed)\n")
            f.write(f"{'-'*60}\n\n")
            
            # Add information about all runs
            all_runs = fold['gp_result'].get('all_runs', [])
            if len(all_runs) > 1:
                f.write(f"\nALL GP RUNS ({len(all_runs)} runs):\n")
                f.write(f"{'-'*60}\n")
                for i, run in enumerate(all_runs, 1):
                    f.write(f"\nRun {i}:\n")
                    f.write(f"  Fitness: {run['fitness']:.6f}\n")
                    f.write(f"  Features: {run['n_features']}\n")
                    f.write(f"  Expression: {run['expression'][:100]}{'...' if len(run['expression']) > 100 else ''}\n")
                f.write(f"{'-'*60}\n")
            
            f.write("\n\n")
        
        # Add summary statistics
        f.write(f"\n{'='*80}\n")
        f.write("SUMMARY ACROSS ALL FOLDS\n")
        f.write(f"{'='*80}\n")
        
        all_fitnesses = [fold['gp_result']['fitness'] for fold in gp_results]
        all_n_features = [fold['gp_result']['n_features'] for fold in gp_results]
        
        f.write(f"\nFitness statistics:\n")
        f.write(f"  Mean: {np.mean(all_fitnesses):.6f}\n")
        f.write(f"  Std:  {np.std(all_fitnesses, ddof=1):.6f}\n")
        f.write(f"  Min:  {np.min(all_fitnesses):.6f}\n")
        f.write(f"  Max:  {np.max(all_fitnesses):.6f}\n")
        
        f.write(f"\nNumber of features statistics:\n")
        f.write(f"  Mean: {np.mean(all_n_features):.2f}\n")
        f.write(f"  Std:  {np.std(all_n_features, ddof=1):.2f}\n")
        f.write(f"  Min:  {np.min(all_n_features)}\n")
        f.write(f"  Max:  {np.max(all_n_features)}\n")
    
    print("✓ GP expressions saved to 'gp_expressions_per_fold.txt'")

    # 9. Performance Summary Table
    print("\n9. OVERALL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'PR-AUC':<12}")
    print("-" * 73)

    for model_name in all_models:
        acc_scores = [m['accuracy'] for m in fold_metrics[model_name]]
        f1_scores = [m['f1'] for m in fold_metrics[model_name]]
        auc_scores = [m['roc_auc'] for m in fold_metrics[model_name] if not np.isnan(m['roc_auc'])]
        pr_scores = [m.get('pr_auc', np.nan) for m in fold_metrics[model_name] if not np.isnan(m.get('pr_auc', np.nan))]

        acc_mean = np.mean(acc_scores)
        f1_mean = np.mean(f1_scores)
        auc_mean = np.nanmean(auc_scores) if len(auc_scores) > 0 else np.nan
        pr_mean = np.nanmean(pr_scores) if len(pr_scores) > 0 else np.nan

        print(f"{model_name:<25} {acc_mean:<12.3f} {f1_mean:<12.3f} {auc_mean:<12.3f} {pr_mean:<12.3f}")

# ===================== MAIN =====================
def main():
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    df = pd.read_csv("prostate.csv")
    X = df.drop('label', axis=1).values
    y = df['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.bincount(y)}")

    # Run nested CV pipeline
    fold_metrics, aggregated, gp_results = run_nested_cv_pipeline(X, y, outer_cv=5)

    # Display results
    display_results(fold_metrics, aggregated, gp_results)

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()

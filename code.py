# ===================== 1. Setup & Imports =====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import torch
import torch.nn as nn
import torch.optim as optim

!pip install gplearn
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
import re


# ===================== 2. Load & Preprocess Dataset =====================
df = pd.read_csv("yourdatasetname.csv")

X = df.drop('label', axis=1).values
y = df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ===================== 3. Define MLP Model =====================
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

def train_model(model, X, y, epochs=30):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.long)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    return model

def train_student_with_soft_labels(model, X, soft_labels, epochs=30, temperature=3.0):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = model(inputs)
        
        # Convert soft labels to proper format for distillation
        teacher_probs = torch.tensor(soft_labels, dtype=torch.float32)
        # Create soft targets as [1-prob, prob] for binary classification
        soft_targets = torch.stack([1 - teacher_probs, teacher_probs], dim=1)
        
        # Apply temperature scaling
        student_soft = torch.softmax(outputs / temperature, dim=1)
        teacher_soft = torch.softmax(torch.log(soft_targets + 1e-8) / temperature, dim=1)
        
        # KL divergence loss for knowledge distillation
        loss = nn.KLDivLoss(reduction='batchmean')(torch.log(student_soft), teacher_soft) * (temperature ** 2)
        
        loss.backward()
        optimizer.step()

    return model

# ===================== 4. Train Teacher and Generate Soft Labels =====================
teacher = MLP(input_dim=X_train.shape[1])
teacher = train_model(teacher, X_train, y_train)

# Evaluate teacher model
teacher.eval()
with torch.no_grad():
    teacher_test_preds = teacher(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
    teacher_train_logits = teacher(torch.tensor(X_train, dtype=torch.float32))
    teacher_probs = torch.softmax(teacher_train_logits, dim=1)[:, 1].numpy()

# Display teacher evaluation metrics
teacher_acc = accuracy_score(y_test, teacher_test_preds)
teacher_prec = precision_score(y_test, teacher_test_preds)
teacher_rec = recall_score(y_test, teacher_test_preds)
teacher_f1 = f1_score(y_test, teacher_test_preds)

print("\n-- Teacher Model Evaluation --")
print(f"Accuracy : {teacher_acc:.4f}")
print(f"Precision: {teacher_prec:.4f}")
print(f"Recall   : {teacher_rec:.4f}")
print(f"F1 Score : {teacher_f1:.4f}")

# ===================== 5. GP Feature Selection with Linear & Nonlinear Operators =====================
# --- Safe nonlinear functions ---
def protected_exp(x):
    with np.errstate(over='ignore'):
        return np.where(x < 100, np.exp(x), 1e6)

exp = make_function(function=protected_exp, name='exp', arity=1)
sin = make_function(function=np.sin, name='sin', arity=1)
cos = make_function(function=np.cos, name='cos', arity=1)

# --- Function set including linear and nonlinear operations ---
function_set = [
    'add', 'sub', 'mul', 'div',       # linear
    'sqrt', 'log', 'abs', 'neg',      # common math
    'max', 'min', 'inv',              # more math
    exp, sin, cos                     # nonlinear
]

def confidence_weighted_fitness(y, y_pred, sample_weight):
    confidence = np.abs(y - 0.5) * 2
    return 1 - r2_score(y, y_pred, sample_weight=confidence)

custom_fitness = make_fitness(function=confidence_weighted_fitness, greater_is_better=False)

def gp_feature_selection(X, soft_labels, generations=200):
    best_expr = None
    best_score = float("inf")
    used_features = set()

    for _ in range(5):
        sr = SymbolicRegressor(
            function_set=function_set,
            generations=generations,
            population_size=300,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=0,
            parsimony_coefficient=0.001,
            metric=custom_fitness,
            random_state=None
        )
        sr.fit(X, soft_labels)
        expr = sr._program
        complexity = expr.length_
        fitness = 1 - sr.score(X, soft_labels)
        total_score = fitness + 0.001 * complexity

        if total_score < best_score:
            best_score = total_score
            best_expr = str(expr)
            used = [int(re.findall(r'X(\d+)', tok)[0]) for tok in str(expr).split() if 'X' in tok]
            used_features = set(used)

    return sorted(list(used_features)), best_expr

# ===================== 6. Repeated GP + Student Training Loop =====================
desired_accuracy = 1.0
max_attempts = 1
attempt = 0
best_accuracy = 0
best_results = None

while best_accuracy < desired_accuracy and attempt < max_attempts:
    attempt += 1
    print(f"\n===== Attempt {attempt} =====")

    selected_gp_indices, best_expr = gp_feature_selection(X_train, teacher_probs)
    X_train_gp = X_train[:, selected_gp_indices]
    X_test_gp = X_test[:, selected_gp_indices]

    # Get teacher soft labels for selected features
    teacher_gp = MLP(input_dim=X_train_gp.shape[1])
    teacher_gp = train_model(teacher_gp, X_train_gp, y_train)
    
    teacher_gp.eval()
    with torch.no_grad():
        teacher_gp_logits = teacher_gp(torch.tensor(X_train_gp, dtype=torch.float32))
        teacher_gp_probs = torch.softmax(teacher_gp_logits, dim=1)[:, 1].numpy()

    student_final = MLP(input_dim=X_train_gp.shape[1])
    student_final = train_student_with_soft_labels(student_final, X_train_gp, teacher_gp_probs)

    student_final.eval()
    with torch.no_grad():
        final_preds = student_final(torch.tensor(X_test_gp, dtype=torch.float32)).argmax(dim=1).numpy()

    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)

    print("\n-- Student Model Evaluation --")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_results = {
            'indices': selected_gp_indices,
            'expr': best_expr,
            'metrics': (acc, prec, rec, f1),
            'X_train_gp': X_train_gp,
            'X_test_gp': X_test_gp
        }

# ===================== 7. Final Evaluation with Classical Classifiers =====================
if best_results:
    print("\n=== Best Attempt Summary ===")
    print(f"Selected GP genes: {best_results['indices']}")
    print(f"Best symbolic model: {best_results['expr']}")
    print(f"Accuracy : {best_results['metrics'][0]:.4f}")
    print(f"Precision: {best_results['metrics'][1]:.4f}")
    print(f"Recall   : {best_results['metrics'][2]:.4f}")
    print(f"F1 Score : {best_results['metrics'][3]:.4f}")

    X_train_gp = best_results['X_train_gp']
    X_test_gp = best_results['X_test_gp']

    base_classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='linear', probability=True),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LDA': LinearDiscriminantAnalysis()
    }

    for clf in base_classifiers.values():
        clf.fit(X_train_gp, y_train)

    ensemble_classifiers = {
        'Voting Ensemble': VotingClassifier(
            estimators=[(name, clf) for name, clf in base_classifiers.items()],
            voting='soft'
        ),
        'Stacking Ensemble': StackingClassifier(
            estimators=[(name, clf) for name, clf in base_classifiers.items()],
            final_estimator=LogisticRegression(max_iter=1000)
        ),
        'AdaBoost': AdaBoostClassifier(n_estimators=100)
    }

    all_classifiers = {**base_classifiers, **ensemble_classifiers}

    print("\n-- Classical Classifier Evaluation on GP-selected genes --")
    for name, clf in all_classifiers.items():
        clf.fit(X_train_gp, y_train)
        pred = clf.predict(X_test_gp)
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        print(f"{name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}\n")
else:

    print("Did not achieve desired accuracy in any attempt.")

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    roc_auc_score, confusion_matrix
)

def prepare_splits(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target into stratified train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_logistic_regression(X_train, X_test, y_train, y_test, balanced: bool = False):
    """
    Trains a Logistic Regression model. Scales features as required.
    Allows toggling balanced class weights to address target imbalance.
    """
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model configuration
    class_weight = 'balanced' if balanced else None
    log_reg = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    
    # Predict
    preds = log_reg.predict(X_test_scaled)
    probs = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    return log_reg, scaler, preds, probs

def tune_decision_tree(X_train, y_train, cv: int = 5):
    """
    Tunes a Decision Tree classifier using GridSearchCV, optimizing for F1.
    """
    tree_base = DecisionTreeClassifier(random_state=42)
    
    param_grid = {
        'max_depth': [4, 6, 8, 10],
        'min_samples_leaf': [2, 5, 10, 20]
    }
    
    # Run GridSearch optimizing for F1 score, tracking multiple metrics
    grid = GridSearchCV(
        estimator=tree_base,
        param_grid=param_grid,
        scoring=['f1', 'recall', 'precision', 'accuracy', 'roc_auc'],
        refit='f1',
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    return grid

def tune_random_forest(X_train, y_train, cv: int = 4):
    """
    Tunes a Random Forest classifier using GridSearchCV, optimizing for ROC-AUC.
    Corrects max_features parameter to allow proper feature bagging ('sqrt', 0.5, 0.7)
    instead of locking it to 1.0 (Bagged Trees).
    """
    rf_base = RandomForestClassifier(random_state=42)
    
    cv_params = {
        'max_depth': [3, 5, None],
        'max_features': ['sqrt', 0.5, 0.7],  # Fixed: no longer restricted to 1.0
        'max_samples': [0.7, 1.0],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2],
        'n_estimators': [300]
    }
    
    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=cv_params,
        scoring=['f1', 'recall', 'precision', 'accuracy', 'roc_auc'],
        refit='roc_auc',
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    return grid

def evaluate_model(model_name: str, y_true, y_pred, y_prob) -> pd.DataFrame:
    """
    Computes performance metrics and returns a dataframe with results.
    """
    metrics = {
        'model': [model_name],
        'precision': [precision_score(y_true, y_pred)],
        'recall': [recall_score(y_true, y_pred)],
        'f1': [f1_score(y_true, y_pred)],
        'accuracy': [accuracy_score(y_true, y_pred)],
        'auc': [roc_auc_score(y_true, y_prob)]
    }
    return pd.DataFrame(metrics)

def save_model(model, file_path: str):
    """
    Saves a trained model object to a pickle file.
    """
    import os
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[Modeling] Model saved to {file_path}")

def load_model(file_path: str):
    """
    Loads a model object from a pickle file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_roc_curves(models_dict, X_test, y_test, save_path: str):
    """
    Plots ROC curves for multiple models on a single plot and saves the figure.
    models_dict is expected to be a dictionary:
    {
        'Model Name': {
            'probs': array_of_probabilities_for_class_1,
            'color': 'color_code'
        }
    }
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import os

    plt.figure(figsize=(8, 6))
    
    for name, info in models_dict.items():
        probs = info['probs']
        color = info.get('color', None)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Modeling] ROC curves saved to {save_path}")


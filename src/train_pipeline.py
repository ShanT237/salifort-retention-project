import pandas as pd
import numpy as np
import time
from preprocessing import preprocess_pipeline
from modeling import (
    prepare_splits, train_logistic_regression, tune_decision_tree,
    tune_random_forest, evaluate_model, save_model
)

def run_pipeline():
    print("=== STARTING DATA SCIENCE PIPELINE ===")
    
    # 1. Preprocess data
    raw_data_path = 'data/raw/HR_capstone_dataset.csv'
    X, y = preprocess_pipeline(raw_data_path)
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = prepare_splits(X, y)
    print(f"[Pipeline] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 3. Model 1: Logistic Regression Baseline
    print("\n--- Training Logistic Regression (Baseline) ---")
    log_model, scaler, log_preds, log_probs = train_logistic_regression(X_train, X_test, y_train, y_test, balanced=False)
    lr_baseline_metrics = evaluate_model("Logistic Regression (Baseline)", y_test, log_preds, log_probs)
    
    # Model 1b: Logistic Regression Balanced
    print("--- Training Logistic Regression (Balanced Class Weights) ---")
    log_model_bal, scaler_bal, log_preds_bal, log_probs_bal = train_logistic_regression(X_train, X_test, y_train, y_test, balanced=True)
    lr_balanced_metrics = evaluate_model("Logistic Regression (Balanced)", y_test, log_preds_bal, log_probs_bal)
    
    # 4. Model 2: Tuned Decision Tree
    print("\n--- Tuning Decision Tree ---")
    start_time = time.time()
    dt_grid = tune_decision_tree(X_train, y_train)
    print(f"[Pipeline] Decision Tree tuning took {time.time() - start_time:.2f} seconds")
    best_dt = dt_grid.best_estimator_
    print(f"  Best params: {dt_grid.best_params_}")
    
    dt_preds = best_dt.predict(X_test)
    # Get probabilities for class 1
    dt_probs = best_dt.predict_proba(X_test)[:, 1]
    dt_metrics = evaluate_model("Decision Tree (Tuned)", y_test, dt_preds, dt_probs)
    
    # 5. Model 3: Tuned Random Forest
    print("\n--- Tuning Random Forest ---")
    start_time = time.time()
    rf_grid = tune_random_forest(X_train, y_train)
    print(f"[Pipeline] Random Forest tuning took {time.time() - start_time:.2f} seconds")
    best_rf = rf_grid.best_estimator_
    print(f"  Best params: {rf_grid.best_params_}")
    
    rf_preds = best_rf.predict(X_test)
    rf_probs = best_rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_model("Random Forest (Tuned)", y_test, rf_preds, rf_probs)
    
    # 6. Compare Models
    print("\n=== MODEL PERFORMANCE COMPARISON (TEST SET) ===")
    comparison_df = pd.concat([
        lr_baseline_metrics,
        lr_balanced_metrics,
        dt_metrics,
        rf_metrics
    ], ignore_index=True)
    
    # Format floats for cleaner output
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(comparison_df.to_string(index=False))
    
    # 7. Save the best model (Random Forest GridSearchCV object)
    # We save to notebooks/models/hr_rf1.pickle to maintain compatibility with the notebook's pickling
    model_save_path = 'notebooks/models/hr_rf1.pickle'
    save_model(rf_grid, model_save_path)
    
    # Also save the scaler and other models in case they are needed for deployment
    save_model(log_model, 'notebooks/models/hr_lr_baseline.pickle')
    save_model(scaler, 'notebooks/models/scaler.pickle')
    
    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    run_pipeline()

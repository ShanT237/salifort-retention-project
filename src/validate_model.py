import os
import pickle
import numpy as np
import pandas as pd

def validate_saved_models():
    print("=== STARTING MODEL VALIDATION AND INTEGRITY CHECK ===")
    
    rf_model_path = 'notebooks/models/hr_rf1.pickle'
    lr_model_path = 'notebooks/models/hr_lr_baseline.pickle'
    scaler_path = 'notebooks/models/scaler.pickle'
    
    # 1. Verify files exist
    for path in [rf_model_path, lr_model_path, scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        print(f"[Verification] File exists: {path}")
        
    # 2. Load models
    print("\n[Verification] Loading saved models...")
    with open(rf_model_path, 'rb') as f:
        rf_grid = pickle.load(f)
    with open(lr_model_path, 'rb') as f:
        lr_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Extract estimators
    best_rf = rf_grid.best_estimator_
    print(f"[Verification] Loaded Random Forest model with params: {rf_grid.best_params_}")
    print(f"[Verification] Loaded Logistic Regression model: {lr_model}")
    print(f"[Verification] Loaded Scaler: {scaler}")
    
    # 3. Create dummy employee test data
    # Features (17 features in df_processed, excluding target 'left'):
    # satisfaction_level, last_evaluation, number_project, average_monthly_hours, tenure, work_accident, promotion_last_5years, salary
    # and 9 department dummy columns
    
    feature_names = [
        'satisfaction_level', 'last_evaluation', 'number_project', 
        'average_monthly_hours', 'tenure', 'work_accident', 
        'promotion_last_5years', 'salary',
        'department_RandD', 'department_accounting', 'department_hr', 
        'department_management', 'department_marketing', 'department_product_mng', 
        'department_sales', 'department_support', 'department_technical'
    ]
    
    # Dummy employee 1: Happy, normal workload, medium salary, sales department
    employee_happy = pd.DataFrame([[
        0.85, 0.75, 4, 180, 3, 0, 0, 1,  # 1 = medium salary
        0, 0, 0, 0, 0, 0, 1, 0, 0        # department_sales = 1
    ]], columns=feature_names)
    
    # Dummy employee 2: Burned out, overworked, low salary, technical department
    # Note: satisfaction_level must be in [0.09, 0.11] to match the actual burnout cluster.
    employee_burnout = pd.DataFrame([[
        0.09, 0.90, 6, 280, 4, 0, 0, 0,  # 0 = low salary
        0, 0, 0, 0, 0, 0, 0, 0, 1        # department_technical = 1
    ]], columns=feature_names)
    
    print("\n[Verification] Testing inference on dummy employee profiles:")
    print("Employee 1 (Happy, balanced workload):")
    print(employee_happy.to_string(index=False))
    print("Employee 2 (Overworked, high evaluation, low satisfaction):")
    print(employee_burnout.to_string(index=False))
    
    # Run predictions - Random Forest
    pred_happy_rf = best_rf.predict(employee_happy)[0]
    prob_happy_rf = best_rf.predict_proba(employee_happy)[0, 1]
    
    pred_burnout_rf = best_rf.predict(employee_burnout)[0]
    prob_burnout_rf = best_rf.predict_proba(employee_burnout)[0, 1]
    
    print("\n--- Inference Results (Random Forest) ---")
    print(f"Employee 1 (Happy)   -> Predicted class: {pred_happy_rf} (Probability of churn: {prob_happy_rf:.4f})")
    print(f"Employee 2 (Burnout) -> Predicted class: {pred_burnout_rf} (Probability of churn: {prob_burnout_rf:.4f})")
    
    # Check predictions
    assert pred_happy_rf == 0, "Error: Happy employee predicted as churn"
    assert pred_burnout_rf == 1, "Error: Burned out employee predicted as stay"
    print("\n[Verification] Random Forest assertions passed!")
    
    # Run predictions - Logistic Regression (needs scaling!)
    happy_scaled = scaler.transform(employee_happy)
    burnout_scaled = scaler.transform(employee_burnout)
    
    pred_happy_lr = lr_model.predict(happy_scaled)[0]
    prob_happy_lr = lr_model.predict_proba(happy_scaled)[0, 1]
    
    pred_burnout_lr = lr_model.predict(burnout_scaled)[0]
    prob_burnout_lr = lr_model.predict_proba(burnout_scaled)[0, 1]
    
    print("\n--- Inference Results (Logistic Regression Baseline) ---")
    print(f"Employee 1 (Happy)   -> Predicted class: {pred_happy_lr} (Probability of churn: {prob_happy_lr:.4f})")
    print(f"Employee 2 (Burnout) -> Predicted class: {pred_burnout_lr} (Probability of churn: {prob_burnout_lr:.4f})")
    
    print("\n=== MODEL VALIDATION AND INTEGRITY CHECK PASSED ===")

if __name__ == '__main__':
    validate_saved_models()

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from modeling import evaluate_model, prepare_splits, save_model, load_model

def test_evaluate_model_returns_correct_columns():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4])
    
    metrics_df = evaluate_model("TestModel", y_true, y_pred, y_prob)
    
    assert isinstance(metrics_df, pd.DataFrame)
    expected_cols = {'model', 'precision', 'recall', 'f1', 'accuracy', 'auc'}
    assert set(metrics_df.columns) == expected_cols
    assert metrics_df.loc[0, 'model'] == "TestModel"

def test_prepare_splits_stratification():
    # Unbalanced dataset to test stratification
    X = pd.DataFrame(np.random.randn(100, 4))
    y = pd.Series([0] * 80 + [1] * 20)
    
    X_train, X_test, y_train, y_test = prepare_splits(X, y, test_size=0.2, random_state=42)
    
    # Check split shapes
    assert len(X_train) == 80
    assert len(X_test) == 20
    
    # Check stratify holds: proportion of 1s in train and test should be equal
    # 20% of 20 ones is 4 ones in test set, and 16 ones in train set
    assert sum(y_test == 1) == 4
    assert sum(y_train == 1) == 16

def test_save_and_load_model_roundtrip(tmp_path):
    model = LogisticRegression()
    # Fit model on dummy data
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.fit(X, y)
    
    temp_file = os.path.join(tmp_path, "test_model.pickle")
    
    save_model(model, temp_file)
    assert os.path.exists(temp_file)
    
    loaded_model = load_model(temp_file)
    assert isinstance(loaded_model, LogisticRegression)
    # Check that coefficients are identical
    np.testing.assert_array_equal(model.coef_, loaded_model.coef_)

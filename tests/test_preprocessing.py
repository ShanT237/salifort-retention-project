import pandas as pd
import numpy as np
import pytest
from preprocessing import clean_data, engineer_features

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        'satisfaction_level': [0.5, 0.5, 0.8],
        'last_evaluation': [0.7, 0.7, 0.9],
        'number_project': [3, 3, 4],
        'average_montly_hours': [180, 180, 200],  # average_montly_hours misspelled
        'time_spend_company': [3, 3, 5],            # time_spend_company
        'Work_accident': [0, 0, 1],                # Work_accident capitalized
        'left': [0, 0, 1],
        'promotion_last_5years': [0, 0, 0],
        'Department': ['sales', 'sales', 'technical'], # Department capitalized
        'salary': ['low', 'low', 'medium']
    })

def test_clean_data_removes_duplicates(sample_raw_data):
    df_clean = clean_data(sample_raw_data)
    # The fixture has 3 rows, but the first two are identical duplicates.
    # After deduplication, there should be exactly 2 rows left.
    assert len(df_clean) == 2

def test_clean_data_renames_columns(sample_raw_data):
    df_clean = clean_data(sample_raw_data)
    # Columns renamed to snake_case
    assert 'work_accident' in df_clean.columns
    assert 'average_monthly_hours' in df_clean.columns
    assert 'tenure' in df_clean.columns
    assert 'department' in df_clean.columns
    
    # Capitalized / misspelled versions shouldn't exist anymore
    assert 'Work_accident' not in df_clean.columns
    assert 'average_montly_hours' not in df_clean.columns
    assert 'time_spend_company' not in df_clean.columns
    assert 'Department' not in df_clean.columns

def test_engineer_features_salary_mapping(sample_raw_data):
    df_clean = clean_data(sample_raw_data)
    df_feat = engineer_features(df_clean)
    
    # low -> 0, medium -> 1
    # Check that the mapped values are correct
    # The first row (after clean) has salary low (mapped to 0)
    # The second row has salary medium (mapped to 1)
    assert list(df_feat['salary']) == [0, 1]

def test_engineer_features_department_dummies(sample_raw_data):
    df_clean = clean_data(sample_raw_data)
    df_feat = engineer_features(df_clean)
    
    # 'department' was one-hot encoded and 'department_sales' dropped or kept?
    # department has values 'sales' and 'technical'.
    # pd.get_dummies with drop_first=True on department will drop 'sales' (first alphabetically is sales).
    # Therefore, department_technical should exist, but department_sales should not.
    assert 'department_technical' in df_feat.columns
    assert 'department_sales' not in df_feat.columns
    assert 'department' not in df_feat.columns

def test_engineer_features_no_bool_columns(sample_raw_data):
    df_clean = clean_data(sample_raw_data)
    df_feat = engineer_features(df_clean)
    
    # Any columns resulting from pd.get_dummies shouldn't be boolean
    bool_cols = df_feat.select_dtypes(include=['bool']).columns
    assert len(bool_cols) == 0
    # Also verify that the dummy columns are of type int or similar
    assert df_feat['department_technical'].dtype in [np.int64, np.int32, np.int8, int]

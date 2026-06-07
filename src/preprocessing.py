import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input dataframe:
    - Renames columns to snake_case and correct spelling mistakes.
    - Removes duplicate rows.
    """
    # Create copy to avoid mutating the original dataframe
    df_clean = df.copy()
    
    # Rename columns for consistency and spelling
    df_clean = df_clean.rename(columns={
        'Work_accident': 'work_accident',
        'average_montly_hours': 'average_monthly_hours',
        'time_spend_company': 'tenure',
        'Department': 'department'
    })
    
    # Drop duplicates
    duplicates_count = df_clean.duplicated().sum()
    print(f"[Preprocessing] Number of duplicate rows detected: {duplicates_count}")
    df_clean = df_clean.drop_duplicates(keep='first')
    print(f"[Preprocessing] Dataset shape after deduplication: {df_clean.shape}")
    
    return df_clean

def get_outliers_info(df: pd.DataFrame, column: str = 'tenure'):
    """
    Computes and prints IQR-based outliers info for a specified column.
    Outliers are kept in the data as they represent a valid part of the population
    (highly relevant for predicting employee churn).
    """
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    iqr = q75 - q25
    lower_limit = q25 - 1.5 * iqr
    upper_limit = q75 + 1.5 * iqr
    
    outliers = df[(df[column] > upper_limit) | (df[column] < lower_limit)]
    print(f"[Preprocessing] IQR Outliers in '{column}':")
    print(f"  - 25th percentile: {q25}, 75th percentile: {q75}, IQR: {iqr}")
    print(f"  - Non-outlier range: [{lower_limit}, {upper_limit}]")
    print(f"  - Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}% of data)")
    
    return lower_limit, upper_limit, len(outliers)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers and encodes features for model training:
    - Maps 'salary' (ordinal category) to integers (low: 0, medium: 1, high: 2).
    - One-hot encodes 'department' using dummy variables with drop_first=True.
    - Converts boolean columns resulting from encoding into integer 0 or 1.
    """
    df_feat = df.copy()
    
    # 1. Map salary ordinally
    salary_map = {'low': 0, 'medium': 1, 'high': 2}
    df_feat['salary'] = df_feat['salary'].map(salary_map)
    
    # 2. One-hot encode department
    df_feat = pd.get_dummies(df_feat, columns=['department'], drop_first=True)
    
    # 3. Convert any boolean columns to integers (0 or 1)
    bool_cols = df_feat.select_dtypes(include=['bool']).columns
    df_feat[bool_cols] = df_feat[bool_cols].astype(int)
    
    return df_feat

def preprocess_pipeline(raw_data_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw data, cleans it, engineers features, and splits into features (X) and target (y).
    """
    df_raw = pd.read_csv(raw_data_path)
    df_clean = clean_data(df_raw)
    get_outliers_info(df_clean, 'tenure')
    df_processed = engineer_features(df_clean)
    
    X = df_processed.drop(columns=['left'])
    y = df_processed['left']
    
    return X, y

import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Introduction and Business Context
    cells.append(nbf.v4.new_markdown_cell(
"""# Executive Report on Employee Retention: Salifort Project

This report presents the findings, analysis, and predictive modeling for employee retention at Salifort Motors. The main goal is to identify the root causes of employee turnover and build a highly reliable predictive model to identify employees at risk of leaving.

---

## 1. Business Context and Objectives

Losing key talent (employee churn) is a significant cost to organizations, including recruitment expenses, loss of institutional knowledge, and decreased team morale.

In this project, data from **11,991 employees** (after removing duplicates) was analyzed to:
1. **Predict the probability** of an employee leaving the company.
2. **Identify the root causes** driving employee turnover.
3. **Propose actionable recommendations** for the Human Resources (HR) department."""
    ))
    
    # 2. Setup cell
    cells.append(nbf.v4.new_code_cell(
"""import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Style settings for professional plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Ensure we import from src/
sys.path.append(os.path.abspath('../src'))
from preprocessing import clean_data, engineer_features"""
    ))
    
    # 3. Data loading
    cells.append(nbf.v4.new_code_cell(
"""# Data loading and cleaning
raw_data_path = '../data/raw/HR_capstone_dataset.csv'
df = pd.read_csv(raw_data_path)
df_clean = clean_data(df)

print(f"Cleaned dataset dimensions: {df_clean.shape}")
df_clean.head()"""
    ))
    
    # 4. EDA Section Markdown
    cells.append(nbf.v4.new_markdown_cell(
"""## 2. Key Insights from Exploratory Data Analysis (EDA)

Through exploratory analysis, we identified two main profiles or clusters of employees who leave the company voluntarily."""
    ))
    
    # 5. EDA Visual 1
    cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(10, 6))
# Filter stayed vs left for visualization
sns.scatterplot(
    data=df_clean,
    x='satisfaction_level',
    y='average_monthly_hours',
    hue='left',
    alpha=0.6,
    palette={0: '#2ca02c', 1: '#d62728'}
)
plt.title('Satisfaction Level vs. Average Monthly Hours (by Retention Status)', fontsize=14, pad=15)
plt.xlabel('Satisfaction Level', fontsize=12)
plt.ylabel('Average Monthly Hours', fontsize=12)
plt.legend(title='Left?', labels=['Stayed (0)', 'Left (1)'])
plt.tight_layout()
plt.show()"""
    ))
    
    # 6. EDA Visual 1 Commentary
    cells.append(nbf.v4.new_markdown_cell(
"""*Note on the plot above*: We can clearly observe two distinct churn clusters:
1. **Burnout Cluster**: High-performing employees (evaluation >= 0.8) with very high monthly hours (>= 240 hours/month, equivalent to 50+ hours a week) but extremely low satisfaction (<= 0.11).
2. **Disengaged Cluster**: Employees with low/moderate evaluations (~0.5), low hours (~150 hours/month), and low satisfaction (~0.4), who likely feel stagnant or underutilized."""
    ))
    
    # 7. EDA Visual 2
    cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_clean,
    x='tenure',
    y='satisfaction_level',
    hue='left',
    palette={0: '#2ca02c', 1: '#d62728'}
)
plt.title('Satisfaction Level Distribution by Tenure (Years)', fontsize=14, pad=15)
plt.xlabel('Tenure (Years)', fontsize=12)
plt.ylabel('Satisfaction Level', fontsize=12)
plt.legend(title='Left?', labels=['Stayed (0)', 'Left (1)'])
plt.tight_layout()
plt.show()"""
    ))
    
    # 8. EDA Visual 2 Commentary
    cells.append(nbf.v4.new_markdown_cell(
"""*Note on tenure*: Employees who leave tend to do so between **year 3 and year 6**. Specifically, in year 4 there is a dramatic drop in satisfaction for employees who end up leaving. Employees with 7 or more years of tenure show virtually zero turnover, indicating stability once they cross the 6-year threshold."""
    ))
    
    # 9. Model Comparison Markdown
    cells.append(nbf.v4.new_markdown_cell(
"""## 3. Machine Learning Model Comparison

We compared four model configurations to predict employee turnover: Logistic Regression (Baseline and Balanced), Decision Tree, and Random Forest.

### Performance Metrics Summary on the Test Set (N=2,399)

| Model | Precision | Recall | F1-Score | Accuracy | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.5000 | 0.1910 | 0.2764 | 0.8341 | 0.8440 |
| **Logistic Regression (Balanced)** | 0.4209 | 0.8417 | 0.5611 | 0.7816 | 0.8504 |
| **Decision Tree (Tuned)** | 0.9738 | 0.9322 | 0.9525 | 0.9846 | 0.9690 |
| **Random Forest (Tuned)** | **0.9893** | **0.9271** | **0.9572** | **0.9862** | **0.9814** |

### Comparative ROC Curves

Below are the ROC curves generated during model training:

![Comparative ROC Curves](../reports/figures/roc_comparison.png)"""
    ))
    
    # 10. Selected Model Intro
    cells.append(nbf.v4.new_markdown_cell(
"""## 4. Detailed Evaluation of the Selected Model (Random Forest)

The **Tuned Random Forest** was selected as the best model due to its excellent balance between precision and recall (F1-score of 0.9572) and its outstanding discriminative power (AUC-ROC of 0.9814).

Let's load the saved model and analyze its detailed behavior."""
    ))
    
    # 11. Selected Model Eval
    cells.append(nbf.v4.new_code_cell(
"""# Load trained model and process validation data
model_path = '../notebooks/models/hr_rf1.pickle'
with open(model_path, 'rb') as f:
    rf_grid = pickle.load(f)

best_rf = rf_grid.best_estimator_

# Preprocess full dataset to extract X and y for testing
df_processed = engineer_features(df_clean)
X = df_processed.drop(columns=['left'])
y = df_processed['left']

# Split exactly the same way as in the pipeline (using stratify and random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Predictions
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

# Classification Report
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))"""
    ))
    
    # 12. Confusion Matrix Plot
    cells.append(nbf.v4.new_code_cell(
"""cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Left'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Confusion Matrix - Random Forest', fontsize=14, pad=15)
plt.grid(False)
plt.tight_layout()
plt.show()"""
    ))
    
    # 13. CM commentary
    cells.append(nbf.v4.new_markdown_cell(
"""The model committed only **4 false positives** (predicted that an employee would leave when they stayed, a precision of 98.9%) and **29 false negatives** (failed to detect an employee who left, a recall of 92.7%). This level of precision is extremely valuable in business settings, as it prevents wasting retention resources on stable employees."""
    ))
    
    # 14. Feature Importance Intro
    cells.append(nbf.v4.new_markdown_cell(
"""## 5. Feature Importance

Random Forest allows us to extract the global importance of each variable in its predictions."""
    ))
    
    # 15. Feature Importance Plot
    cells.append(nbf.v4.new_code_cell(
"""# Extract importances
importances = best_rf.feature_importances_
feature_names = X.columns

# Create DataFrame and sort
df_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

# Plot top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_importances.tail(10),
    x='Importance',
    y='Feature',
    palette='viridis'
)
plt.title('Top 10 Feature Importances for Predicting Turnover', fontsize=14, pad=15)
plt.xlabel('Relative Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()"""
    ))
    
    # 16. Feature Importance Commentary
    cells.append(nbf.v4.new_markdown_cell(
"""The most critical variables for predicting turnover are:
1. **Satisfaction Level (satisfaction_level)**: The most dominant variable by far (representing nearly 50% of the predictive weight).
2. **Performance Evaluation (last_evaluation)**: Employees with extremely high or low evaluations are highly prone to leave.
3. **Number of Projects (number_project)**: Having too many projects leads to burnout, while too few leads to disengagement.
4. **Tenure (tenure)**: The years spent at the company, particularly years 3 through 6.
5. **Average Monthly Hours (average_monthly_hours)**: A key indicator of workload and burnout.

---

## 6. Strategic Recommendations for Human Resources

Based on the model findings, the following concrete actions are proposed to mitigate the risk of turnover at Salifort Motors:

### 1. Cap the Number of Projects and Overtime (Burnout Prevention)
* **Finding**: There is a critical group of high-performing employees (evaluation >= 0.8) with 6 or more projects and over 240 monthly hours who leave unsatisfied.
* **Action**: Set a strict limit of 5 active projects per employee and monitor monthly hours to alert managers if any team member exceeds 200 hours. Implement workload distribution policies.

### 2. Career Progression Plans and Mentoring in Critical Years (Years 3 to 6)
* **Finding**: Turnover is heavily concentrated between the 3rd and 6th year of tenure.
* **Action**: Design career progression reviews at the 36-month mark (year 3). Offer promotions, lateral rotations, or skill development opportunities to re-engage employees.

### 3. Salary Corrections for High Performers with Low/Medium Salary
* **Finding**: Cross-analysis shows high-performing employees receiving low or medium salaries have a high propensity to leave.
* **Action**: Perform salary audits to ensure outstanding employees (evaluation >= 0.8) are paid competitively and offered performance incentives.

### 4. Continuous Feedback Channels and Pulse Surveys
* **Finding**: Satisfaction is the number one predictor of turnover.
* **Action**: Implement anonymous bi-monthly pulse surveys and structured 1-on-1 meetings, focusing on employees in their 3rd and 4th years, to detect dissatisfaction before it leads to resignation."""
    ))
    
    nb['cells'] = cells
    
    output_path = 'notebooks/02_executive_report.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Created notebook outline at {output_path}")

if __name__ == '__main__':
    create_notebook()

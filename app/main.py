import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Employee Churn Prediction | Salifort Motors",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src/ directory to system path to import local preprocessing
sys.path.append(os.path.abspath('src'))
from preprocessing import clean_data

# Premium custom styling
st.markdown("""
<style>
    /* Main container background */
    .reportview-container {
        background-color: #f8f9fa;
    }
    /* Metric and result card container */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    /* Button and selector styling custom tweaks */
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
    }
    /* Typography settings */
    .app-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .app-subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Load resources (model and comparison stats)
@st.cache_resource
def load_resources():
    model_path = 'notebooks/models/hr_rf1.pickle'
    with open(model_path, 'rb') as f:
        rf_grid = pickle.load(f)
    model = rf_grid.best_estimator_
    
    raw_data_path = 'data/raw/HR_capstone_dataset.csv'
    df_raw = pd.read_csv(raw_data_path)
    df_clean = clean_data(df_raw)
    
    return model, df_clean

try:
    model, df_clean = load_resources()
except Exception as e:
    st.error(f"Error loading resources. Ensure you have run the training pipeline first. Detail: {e}")
    st.stop()

# Main page headers
st.markdown('<h1 class="app-title">🚗 Salifort Motors — Churn Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Interactive platform for real-time employee retention and risk prediction</p>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("### 📋 Employee Profile")
st.sidebar.markdown("Input the employee's characteristics below to evaluate their retention risk.")

satisfaction_level = st.sidebar.slider(
    "Satisfaction Level",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Employee's self-reported satisfaction level (0.0 to 1.0)"
)

last_evaluation = st.sidebar.slider(
    "Last Evaluation",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Employee's score in their last performance evaluation (0.0 to 1.0)"
)

number_project = st.sidebar.number_input(
    "Number of Projects",
    min_value=2,
    max_value=7,
    value=3,
    step=1,
    help="Number of active projects currently assigned"
)

average_monthly_hours = st.sidebar.slider(
    "Average Monthly Hours",
    min_value=96,
    max_value=310,
    value=180,
    step=1,
    help="Average hours worked per month"
)

tenure = st.sidebar.slider(
    "Tenure (Years)",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
    help="Years spent at the company"
)

work_accident = st.sidebar.selectbox(
    "Work Accident History?",
    options=[("No", 0), ("Yes", 1)],
    format_func=lambda x: x[0]
)[1]

promotion_last_5years = st.sidebar.selectbox(
    "Promoted in the Last 5 Years?",
    options=[("No", 0), ("Yes", 1)],
    format_func=lambda x: x[0]
)[1]

salary = st.sidebar.selectbox(
    "Salary Level",
    options=["low", "medium", "high"],
    format_func=lambda x: "Low (low)" if x == "low" else "Medium (medium)" if x == "medium" else "High (high)"
)

department = st.sidebar.selectbox(
    "Department",
    options=["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales", "support", "technical"],
    format_func=lambda x: x.upper() if x == "IT" else x.capitalize()
)

# Align input data with model expected feature format
feature_names = [
    'satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours',
    'tenure', 'work_accident', 'promotion_last_5years', 'salary',
    'department_RandD', 'department_accounting', 'department_hr', 'department_management',
    'department_marketing', 'department_product_mng', 'department_sales',
    'department_support', 'department_technical'
]

salary_map = {'low': 0, 'medium': 1, 'high': 2}
salary_val = salary_map[salary]

input_dict = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_monthly_hours': average_monthly_hours,
    'tenure': tenure,
    'work_accident': work_accident,
    'promotion_last_5years': promotion_last_5years,
    'salary': salary_val
}

for dept in ['RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng', 'sales', 'support', 'technical']:
    input_dict[f'department_{dept}'] = 1 if department == dept else 0

input_df = pd.DataFrame([input_dict], columns=feature_names)

# Model inference
prob_leaving = model.predict_proba(input_df)[0, 1]

# Risk logic mapping
if prob_leaving >= 0.7:
    risk_level = "High"
    risk_color = "#dc3545"  # Red
    risk_icon = "🔴"
    risk_text = "This employee presents a high risk of voluntarily leaving the company soon. Immediate retention actions are strongly recommended."
elif prob_leaving >= 0.3:
    risk_level = "Medium"
    risk_color = "#ffc107"  # Yellow/Amber
    risk_icon = "🟡"
    risk_text = "Moderate risk of churn. Consider reviewing their workload and satisfaction level in upcoming 1-on-1s."
else:
    risk_level = "Low"
    risk_color = "#198754"  # Green
    risk_icon = "🟢"
    risk_text = "Low risk of churn. The profile matches historically stable retention patterns."

# Display outputs
col1, col2 = st.columns([1.1, 1.0])

with col1:
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📊 Churn Risk Assessment")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### Churn Probability")
        st.markdown(f"<h1 style='color:{risk_color}; font-size: 4rem; margin-top: -10px; margin-bottom: 0px;'>{prob_leaving * 100:.1f}%</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"#### Risk Category")
        st.markdown(f"<h1 style='color:{risk_color}; font-size: 2.2rem; margin-top: 5px; margin-bottom: 0px;'>{risk_icon} {risk_level}</h1>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='margin-top: 15px; font-size: 1.05rem;'>{risk_text}</p>", unsafe_allow_html=True)
    
    # Progress gauge
    st.markdown("<p style='margin-bottom: 5px; font-weight: bold;'>Risk Thermometer</p>", unsafe_allow_html=True)
    st.progress(prob_leaving)
    st.markdown(f"</div>", unsafe_allow_html=True)
    
    # Dynamic text recommendations
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("💡 Key Risk Factors & Recommendations")
    
    factors = []
    
    # Burnout cluster
    if satisfaction_level <= 0.11 and average_monthly_hours >= 240 and number_project >= 6:
        factors.append((
            "⚠️ **Extreme Burnout Syndrome Detected**",
            "The employee has critically low satisfaction, works over 240 hours monthly, and handles 6 or more projects. This aligns exactly with the main historical churn cluster (burnout).",
            "**Recommended Action**: Limit active projects to a maximum of 5 and cap monthly hours below 200 immediately."
        ))
    
    # High evaluation vs low salary
    if last_evaluation >= 0.8 and salary == "low":
        factors.append((
            "💸 **High Performance vs. Low Salary**",
            "The employee has an outstanding evaluation score but is on a low salary. Historically, this mismatch creates dissatisfaction and high susceptibility to competitors' offers.",
            "**Recommended Action**: Review salary benchmark and consider a salary increase to 'medium' or offer performance bonuses."
        ))
        
    # Tenure critical range
    if tenure in [3, 4, 5]:
        factors.append((
            "⏳ **Critical Tenure Window (Years 3-5)**",
            f"The employee is in year {tenure} at the company. Historical turnover at Salifort Motors is heavily clustered in years 3 through 5, corresponding with drops in satisfaction.",
            "**Recommended Action**: Schedule a career development talk to set new professional milestones."
        ))
        
    # Underutilization
    if number_project <= 2 and satisfaction_level <= 0.4:
        factors.append((
            "📉 **Risk of Disengagement / Underutilization**",
            f"The employee has only {number_project} projects and satisfaction below 0.4. This indicates potential underutilization, which can cause lack of motivation.",
            "**Recommended Action**: Assign at least one additional challenging project to reactivate their professional interest."
        ))
        
    # Work accident correlation
    if work_accident == 1:
        factors.append((
            "🏥 **Work Accident History**",
            "The employee has suffered a work accident in the past. Although accidents show a slight negative correlation with churn globally (possibly due to legal protections or stability incentives), it warrants occupational health monitoring.",
            "**Recommended Action**: Ensure proper follow-up and verify if workplace safety affects their well-being."
        ))
        
    if not factors:
        st.write("✅ No extreme risk factors detected. Maintain standard check-ins.")
    else:
        for title, desc, action in factors:
            st.markdown(f"**{title}**")
            st.write(desc)
            st.markdown(f"<span style='color:#0d6efd;'>{action}</span>", unsafe_allow_html=True)
            st.markdown("---")
            
    st.markdown(f"</div>", unsafe_allow_html=True)

with col2:
    # Averages table
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("🔄 Comparison vs. Historical Averages")
    
    avg_stayed = df_clean[df_clean['left'] == 0].mean(numeric_only=True)
    avg_left = df_clean[df_clean['left'] == 1].mean(numeric_only=True)
    
    comparison_data = {
        "Metric": [
            "Satisfaction Level",
            "Last Evaluation",
            "Number of Projects",
            "Average Monthly Hours",
            "Tenure (Years)",
            "Work Accident (%)",
            "Promoted Last 5 Years (%)",
            "Salary Level Mapped (0-2)"
        ],
        "Current Employee": [
            satisfaction_level,
            last_evaluation,
            number_project,
            average_monthly_hours,
            tenure,
            work_accident * 100.0,
            promotion_last_5years * 100.0,
            salary_val
        ],
        "Average (Stayed)": [
            avg_stayed['satisfaction_level'],
            avg_stayed['last_evaluation'],
            avg_stayed['number_project'],
            avg_stayed['average_monthly_hours'],
            avg_stayed['tenure'],
            avg_stayed['work_accident'] * 100.0,
            avg_stayed['promotion_last_5years'] * 100.0,
            avg_stayed['salary']
        ],
        "Average (Left)": [
            avg_left['satisfaction_level'],
            avg_left['last_evaluation'],
            avg_left['number_project'],
            avg_left['average_monthly_hours'],
            avg_left['tenure'],
            avg_left['work_accident'] * 100.0,
            avg_left['promotion_last_5years'] * 100.0,
            avg_left['salary']
        ]
    }
    
    df_comp = pd.DataFrame(comparison_data)
    
    # Formatting
    df_comp_formatted = df_comp.copy()
    for col in ["Current Employee", "Average (Stayed)", "Average (Left)"]:
        df_comp_formatted[col] = df_comp_formatted[col].apply(lambda x: f"{x:.2f}" if abs(x) > 5 else f"{x:.4f}" if x < 1 else f"{x:.2f}")
        
    st.table(df_comp_formatted.set_index("Metric"))
    st.markdown(f"</div>", unsafe_allow_html=True)
    
    # Feature importances
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("🎯 Global Feature Importances (RF Model)")
    
    importances = model.feature_importances_
    
    feat_imp = pd.DataFrame({
        'Feature': [f.replace('department_', 'Dept: ').replace('satisfaction_level', 'Satisfaction').replace('last_evaluation', 'Evaluation').replace('number_project', 'No. Projects').replace('average_monthly_hours', 'Monthly Hours').replace('tenure', 'Tenure').replace('work_accident', 'Work Accident').replace('promotion_last_5years', 'Promoted 5y').replace('salary', 'Salary') for f in feature_names],
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.barplot(
        data=feat_imp.tail(8),
        x='Importance',
        y='Feature',
        palette='viridis',
        ax=ax
    )
    ax.set_xlabel('Relative Importance')
    ax.set_ylabel('Feature')
    ax.set_title('Top 8 Most Influential Predictors')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown(f"</div>", unsafe_allow_html=True)

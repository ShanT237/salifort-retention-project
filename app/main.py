import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Churn Prediction | Salifort Motors",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Path resolution (works from any working directory) ─────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from preprocessing import clean_data

# ── Load model + data ──────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model_path = os.path.join(PROJECT_ROOT, 'notebooks', 'models', 'hr_rf1.pickle')
    with open(model_path, 'rb') as f:
        rf_grid = pickle.load(f)
    model = rf_grid.best_estimator_

    raw_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'HR_capstone_dataset.csv')
    df_raw = pd.read_csv(raw_data_path)
    df_clean = clean_data(df_raw)

    # Add numeric salary for averaging (df_clean salary is still a string here)
    salary_map = {'low': 0, 'medium': 1, 'high': 2}
    df_clean['salary_num'] = df_clean['salary'].map(salary_map)

    return model, df_clean

try:
    model, df_clean = load_resources()
except Exception as e:
    st.error(f"Error loading resources. Make sure you ran the training pipeline first.\n\n`{e}`")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🚗 Salifort Motors — Employee Churn Analytics")
st.caption("Real-time employee retention risk prediction powered by Random Forest (AUC = 0.9814)")
st.divider()

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Employee Profile")
    st.caption("Adjust the sliders to evaluate a specific employee's churn risk.")
    st.divider()

    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01,
        help="Employee's self-reported satisfaction (0 = lowest, 1 = highest)")
    last_evaluation = st.slider("Last Performance Evaluation", 0.0, 1.0, 0.70, 0.01,
        help="Score from last performance review (0 = lowest, 1 = highest)")
    number_project = st.slider("Number of Active Projects", 2, 7, 3, 1,
        help="Current number of projects assigned")
    average_monthly_hours = st.slider("Average Monthly Hours", 96, 310, 180, 1,
        help="Average hours worked per month")
    tenure = st.slider("Tenure (Years at Company)", 2, 10, 3, 1)
    salary = st.selectbox("Salary Level", ["low", "medium", "high"],
        format_func=lambda x: {"low": "Low", "medium": "Medium", "high": "High"}[x])
    department = st.selectbox("Department",
        ["IT", "RandD", "accounting", "hr", "management",
         "marketing", "product_mng", "sales", "support", "technical"])
    work_accident = st.selectbox("Work Accident History?", [("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0])[1]
    promotion_last_5years = st.selectbox("Promoted in Last 5 Years?", [("No", 0), ("Yes", 1)],
        format_func=lambda x: x[0])[1]

# ── Build input vector ─────────────────────────────────────────────────────────
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
for dept in ['RandD', 'accounting', 'hr', 'management', 'marketing',
             'product_mng', 'sales', 'support', 'technical']:
    input_dict[f'department_{dept}'] = 1 if department == dept else 0

input_df = pd.DataFrame([input_dict], columns=feature_names)

# ── Inference ──────────────────────────────────────────────────────────────────
prob_leaving = float(model.predict_proba(input_df)[0, 1])

if prob_leaving >= 0.70:
    risk_level, risk_emoji = "High Risk", "🔴"
    risk_color = "#ef4444"
    risk_msg = "Immediate retention action recommended. This profile closely matches the historical high-churn cluster."
elif prob_leaving >= 0.30:
    risk_level, risk_emoji = "Medium Risk", "🟡"
    risk_color = "#f59e0b"
    risk_msg = "Moderate churn probability. Review workload and satisfaction in upcoming 1-on-1s."
else:
    risk_level, risk_emoji = "Low Risk", "🟢"
    risk_color = "#10b981"
    risk_msg = "Profile matches stable retention patterns. Continue standard engagement practices."

# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.05, 1.0], gap="large")

# ── LEFT COLUMN ────────────────────────────────────────────────────────────────
with col_left:

    # Risk score card
    st.subheader("📊 Churn Risk Assessment")
    m1, m2 = st.columns(2)
    m1.metric("Churn Probability", f"{prob_leaving * 100:.1f}%")
    m2.metric("Risk Level", f"{risk_emoji} {risk_level}")

    st.progress(prob_leaving)
    st.info(risk_msg)
    st.divider()

    # Dynamic risk factors
    st.subheader("💡 Key Risk Factors & Recommendations")

    factors = []

    if satisfaction_level <= 0.11 and average_monthly_hours >= 240 and number_project >= 6:
        factors.append((
            "⚠️ Extreme Burnout Detected",
            "Critically low satisfaction + 240+ monthly hours + 6+ projects. This is the primary historical churn cluster.",
            "Reduce active projects to ≤5 and cap monthly hours below 200 immediately."
        ))

    if last_evaluation >= 0.8 and salary == "low":
        factors.append((
            "💸 High Performer, Low Pay",
            "Top evaluation score but on the lowest salary tier — a strong predictor of departure.",
            "Review salary band and consider a raise to 'medium' or a performance bonus."
        ))

    if tenure in [3, 4, 5]:
        factors.append((
            f"⏳ Critical Tenure Window (Year {tenure})",
            "Years 3–5 are the highest-risk period for voluntary churn at Salifort Motors.",
            "Schedule a career development conversation and set new milestones."
        ))

    if number_project <= 2 and satisfaction_level <= 0.4:
        factors.append((
            "📉 Underutilized / Disengaged",
            f"Only {number_project} project(s) with low satisfaction — typical disengagement profile.",
            "Assign an additional stretch project to re-engage the employee."
        ))

    if work_accident == 1:
        factors.append((
            "🏥 Work Accident History",
            "Past work accident on record. Accidents correlate weakly with reduced churn (stability), but still warrants occupational health follow-up.",
            "Confirm ongoing support and check that workplace safety concerns are resolved."
        ))

    if not factors:
        st.success("✅ No critical risk factors detected. Maintain regular check-ins.")
    else:
        for title, desc, action in factors:
            with st.expander(title, expanded=True):
                st.write(desc)
                st.markdown(f"**→ Action:** {action}")

    # Feature importances chart
    st.divider()
    st.subheader("🎯 Global Feature Importances (Random Forest)")

    label_map = {
        'satisfaction_level': 'Satisfaction Level',
        'last_evaluation': 'Last Evaluation',
        'number_project': 'No. of Projects',
        'average_monthly_hours': 'Monthly Hours',
        'tenure': 'Tenure',
        'work_accident': 'Work Accident',
        'promotion_last_5years': 'Promoted (5y)',
        'salary': 'Salary Level'
    }
    feat_labels = [
        label_map.get(f, f.replace('department_', 'Dept: ').capitalize())
        for f in feature_names
    ]

    feat_imp = pd.DataFrame({
        'Feature': feat_labels,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(8)

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'],
                   color=plt.cm.viridis(np.linspace(0.2, 0.85, len(feat_imp))))
    ax.set_xlabel('Relative Importance', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_title('Top 8 Predictors', color='white', pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ── RIGHT COLUMN ───────────────────────────────────────────────────────────────
with col_right:

    # Historical comparison table
    st.subheader("🔄 Comparison vs. Historical Averages")

    avg_stayed = df_clean[df_clean['left'] == 0].agg({
        'satisfaction_level': 'mean',
        'last_evaluation': 'mean',
        'number_project': 'mean',
        'average_monthly_hours': 'mean',
        'tenure': 'mean',
        'work_accident': 'mean',
        'promotion_last_5years': 'mean',
        'salary_num': 'mean'
    })
    avg_left = df_clean[df_clean['left'] == 1].agg({
        'satisfaction_level': 'mean',
        'last_evaluation': 'mean',
        'number_project': 'mean',
        'average_monthly_hours': 'mean',
        'tenure': 'mean',
        'work_accident': 'mean',
        'promotion_last_5years': 'mean',
        'salary_num': 'mean'
    })

    comp_metrics = [
        ("Satisfaction Level",    satisfaction_level,               avg_stayed['satisfaction_level'],  avg_left['satisfaction_level']),
        ("Last Evaluation",       last_evaluation,                  avg_stayed['last_evaluation'],     avg_left['last_evaluation']),
        ("No. of Projects",       number_project,                   avg_stayed['number_project'],      avg_left['number_project']),
        ("Monthly Hours",         average_monthly_hours,            avg_stayed['average_monthly_hours'], avg_left['average_monthly_hours']),
        ("Tenure (Years)",        tenure,                           avg_stayed['tenure'],              avg_left['tenure']),
        ("Work Accident (%)",     work_accident * 100,              avg_stayed['work_accident'] * 100, avg_left['work_accident'] * 100),
        ("Promoted 5y (%)",       promotion_last_5years * 100,      avg_stayed['promotion_last_5years'] * 100, avg_left['promotion_last_5years'] * 100),
        ("Salary (0-2 mapped)",   salary_val,                       avg_stayed['salary_num'],          avg_left['salary_num']),
    ]

    df_comp = pd.DataFrame(comp_metrics, columns=["Metric", "This Employee", "Avg (Stayed)", "Avg (Left)"])
    for col in ["This Employee", "Avg (Stayed)", "Avg (Left)"]:
        df_comp[col] = df_comp[col].apply(lambda x: f"{x:.2f}")

    st.dataframe(df_comp.set_index("Metric"), use_container_width=True)
    st.divider()

    # ROC curves (static image)
    roc_path = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'roc_comparison.png')
    if os.path.exists(roc_path):
        st.subheader("📈 ROC Curves — All Models")
        st.image(roc_path, use_container_width=True)
    else:
        st.info("ROC curves not found. Run `src/train_pipeline.py` to generate them.")

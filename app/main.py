import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de página
st.set_page_config(
    page_title="Predicción de Rotación de Empleados | Salifort Motors",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar la carpeta src/ al path para cargar módulos locales
sys.path.append(os.path.abspath('src'))
from preprocessing import clean_data

# CSS para estilo Premium
st.markdown("""
<style>
    /* Estilo del contenedor principal */
    .reportview-container {
        background-color: #f8f9fa;
    }
    /* Tarjetas de métricas y resultados */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    /* Estilos de botones y selectores */
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
    }
    /* Títulos e iconos */
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

# Cargar recursos (modelo y datos para comparación)
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
    st.error(f"Error al cargar el modelo o los datos. Asegúrate de haber ejecutado el pipeline de entrenamiento primero. Detalle: {e}")
    st.stop()

# Estructura de la aplicación
st.markdown('<h1 class="app-title">🚗 Salifort Motors — Churn Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Plataforma interactiva para la predicción de riesgo de fuga de talento en tiempo real</p>', unsafe_allow_html=True)

# Sidebar para ingresar los datos del empleado
st.sidebar.markdown("### 📋 Datos del Empleado")
st.sidebar.markdown("Ingrese las características del empleado para evaluar su probabilidad de permanencia.")

# Inputs del usuario en el sidebar
satisfaction_level = st.sidebar.slider(
    "Nivel de Satisfacción",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Nivel de satisfacción reportado por el empleado (0 a 1)"
)

last_evaluation = st.sidebar.slider(
    "Última Evaluación",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Calificación obtenida por el empleado en su última evaluación (0 a 1)"
)

number_project = st.sidebar.number_input(
    "Número de Proyectos",
    min_value=2,
    max_value=7,
    value=3,
    step=1,
    help="Número de proyectos asignados actualmente"
)

average_monthly_hours = st.sidebar.slider(
    "Horas Mensuales Promedio",
    min_value=96,
    max_value=310,
    value=180,
    step=1,
    help="Horas de trabajo promedio al mes"
)

tenure = st.sidebar.slider(
    "Antigüedad en la Empresa (Años)",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
    help="Años transcurridos desde el ingreso a la empresa"
)

work_accident = st.sidebar.selectbox(
    "¿Ha tenido algún accidente laboral?",
    options=[("No", 0), ("Sí", 1)],
    format_func=lambda x: x[0]
)[1]

promotion_last_5years = st.sidebar.selectbox(
    "¿Ha sido ascendido en los últimos 5 años?",
    options=[("No", 0), ("Sí", 1)],
    format_func=lambda x: x[0]
)[1]

salary = st.sidebar.selectbox(
    "Nivel de Salario",
    options=["low", "medium", "high"],
    format_func=lambda x: "Bajo (low)" if x == "low" else "Medio (medium)" if x == "medium" else "Alto (high)"
)

department = st.sidebar.selectbox(
    "Departamento",
    options=["IT", "RandD", "accounting", "hr", "management", "marketing", "product_mng", "sales", "support", "technical"],
    format_func=lambda x: x.upper() if x == "IT" else x.capitalize()
)

# Procesar inputs para el modelo
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

# Realizar Predicción
prob_leaving = model.predict_proba(input_df)[0, 1]
prob_retaining = 1 - prob_leaving

# Definir nivel de riesgo
if prob_leaving >= 0.7:
    risk_level = "Alto"
    risk_color = "#dc3545"  # Rojo
    risk_icon = "🔴"
    risk_text = "Este empleado presenta un alto riesgo de abandonar la empresa voluntariamente en el corto plazo. Se sugiere tomar medidas de retención inmediatas."
elif prob_leaving >= 0.3:
    risk_level = "Medio"
    risk_color = "#ffc107"  # Amarillo/Naranja
    risk_icon = "🟡"
    risk_text = "Riesgo moderado de abandono. Monitorear su nivel de carga de trabajo y satisfacción en las próximas semanas."
else:
    risk_level = "Bajo"
    risk_color = "#198754"  # Verde
    risk_icon = "🟢"
    risk_text = "Bajo riesgo de abandono. El perfil del empleado coincide con patrones de retención estables."

# Diseño del layout principal
col1, col2 = st.columns([1.1, 1.0])

with col1:
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📊 Análisis de Riesgo de Churn")
    
    # Mostrar resultados
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### Probabilidad de Fuga")
        st.markdown(f"<h1 style='color:{risk_color}; font-size: 4rem; margin-top: -10px; margin-bottom: 0px;'>{prob_leaving * 100:.1f}%</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"#### Nivel de Riesgo")
        st.markdown(f"<h1 style='color:{risk_color}; font-size: 2.2rem; margin-top: 5px; margin-bottom: 0px;'>{risk_icon} {risk_level}</h1>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='margin-top: 15px; font-size: 1.05rem;'>{risk_text}</p>", unsafe_allow_html=True)
    
    # Barra de progreso customizada
    st.markdown("<p style='margin-bottom: 5px; font-weight: bold;'>Termómetro de Riesgo</p>", unsafe_allow_html=True)
    st.progress(prob_leaving)
    st.markdown(f"</div>", unsafe_allow_html=True)
    
    # Factores de Riesgo / Recomendaciones Dinámicas
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("💡 Factores y Recomendaciones Dinámicas")
    
    factors = []
    
    # Burnout cluster check
    if satisfaction_level <= 0.11 and average_monthly_hours >= 240 and number_project >= 6:
        factors.append((
            "⚠️ **Síndrome de Burnout Extremo Detectado**",
            "El empleado tiene un nivel de satisfacción críticamente bajo y trabaja más de 240 horas mensuales con 6 o más proyectos. Este perfil coincide perfectamente con el principal clúster histórico de abandono voluntario.",
            "**Acción recomendada**: Reducir su asignación de proyectos a un máximo de 5 y limitar sus horas mensuales por debajo de las 200 horas de forma obligatoria."
        ))
    
    # High evaluation / Low salary check
    if last_evaluation >= 0.8 and salary == "low":
        factors.append((
            "💸 **Alto Rendimiento vs. Compensación Baja**",
            "El empleado tiene una evaluación sobresaliente pero percibe un salario bajo. Históricamente, este desbalance produce un fuerte sentimiento de descontento o aumenta la probabilidad de ser contratado por competidores.",
            "**Acción recomendada**: Evaluar un ajuste salarial a nivel 'medio' u ofrecer bonos de desempeño en el corto plazo."
        ))
        
    # Tenure risk check
    if tenure in [3, 4, 5]:
        factors.append((
            "⏳ **Antigüedad en Periodo Crítico (Año 3 a 5)**",
            f"El empleado se encuentra en su año {tenure} en la empresa. La tasa de abandono en Salifort Motors se concentra fuertemente entre los años 3 y 5 de antigüedad, coincidiendo con la caída de satisfacción laboral.",
            "**Acción recomendada**: Programar una sesión de coaching o revisión de plan de carrera para ofrecer nuevas metas dentro del departamento."
        ))
        
    # Underutilization check
    if number_project <= 2 and satisfaction_level <= 0.4:
        factors.append((
            "📉 **Riesgo de Desconexión / Aburrimiento**",
            f"El empleado tiene solo {number_project} proyectos y satisfacción menor a 0.4. Esto suele indicar subutilización del talento, lo que produce aburrimiento o desinterés.",
            "**Acción recomendada**: Asignarle al menos un proyecto adicional que represente un reto profesional para reactivar su motivación."
        ))
        
    # Accident and Churn correlation
    if work_accident == 1:
        factors.append((
            "🏥 **Historial de Accidente Laboral**",
            "El empleado ha sufrido un accidente laboral en el pasado. Aunque los accidentes tienen una correlación negativa global con la rotación (los que tienen accidentes tienden a irse menos, posiblemente por mayor protección legal o estabilidad), requiere monitoreo de su salud ocupacional.",
            "**Acción recomendada**: Asegurar seguimiento médico adecuado y evaluar si la seguridad laboral influye en su estado emocional."
        ))
        
    if not factors:
        st.write("✅ No se han detectado combinaciones de riesgo extremas. Mantenga el monitoreo regular.")
    else:
        for title, desc, action in factors:
            st.markdown(f"**{title}**")
            st.write(desc)
            st.markdown(f"<span style='color:#0d6efd;'>{action}</span>", unsafe_allow_html=True)
            st.markdown("---")
            
    st.markdown(f"</div>", unsafe_allow_html=True)

with col2:
    # Comparativa vs Promedios Históricos
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("🔄 Comparación vs. Promedios Históricos")
    
    # Calcular promedios del dataset limpio
    avg_stayed = df_clean[df_clean['left'] == 0].mean(numeric_only=True)
    avg_left = df_clean[df_clean['left'] == 1].mean(numeric_only=True)
    
    # Preparar DataFrame de comparación
    comparison_data = {
        "Métrica": [
            "Nivel de Satisfacción",
            "Última Evaluación",
            "Número de Proyectos",
            "Horas Mensuales Promedio",
            "Antigüedad (Años)",
            "Accidente Laboral (%)",
            "Ascendido últ. 5 años (%)",
            "Salario Mapeado (0-2)"
        ],
        "Empleado Actual": [
            satisfaction_level,
            last_evaluation,
            number_project,
            average_monthly_hours,
            tenure,
            work_accident * 100.0,
            promotion_last_5years * 100.0,
            salary_val
        ],
        "Promedio (Se Quedaron)": [
            avg_stayed['satisfaction_level'],
            avg_stayed['last_evaluation'],
            avg_stayed['number_project'],
            avg_stayed['average_monthly_hours'],
            avg_stayed['tenure'],
            avg_stayed['work_accident'] * 100.0,
            avg_stayed['promotion_last_5years'] * 100.0,
            avg_stayed['salary']
        ],
        "Promedio (Se Fueron)": [
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
    
    # Formatear valores
    df_comp_formatted = df_comp.copy()
    for col in ["Empleado Actual", "Promedio (Se Quedaron)", "Promedio (Se Fueron)"]:
        df_comp_formatted[col] = df_comp_formatted[col].apply(lambda x: f"{x:.2f}" if abs(x) > 5 else f"{x:.4f}" if x < 1 else f"{x:.2f}")
        
    st.table(df_comp_formatted.set_index("Métrica"))
    st.markdown(f"</div>", unsafe_allow_html=True)
    
    # Importancias de las características en el modelo
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("🎯 Importancia Global de Variables (Modelo RF)")
    
    # Obtener feature importances del modelo
    importances = model.feature_importances_
    # Para simplificar la visualización, agrupamos las dummies de department
    # y mostramos el top de variables
    # El orden de las features es el que tenemos en feature_names
    feat_imp = pd.DataFrame({
        'Variable': [f.replace('department_', 'Depto: ').replace('satisfaction_level', 'Satisfacción').replace('last_evaluation', 'Evaluación').replace('number_project', 'Nº Proyectos').replace('average_monthly_hours', 'Horas Mensuales').replace('tenure', 'Antigüedad').replace('work_accident', 'Accidente Laboral').replace('promotion_last_5years', 'Ascendido 5a').replace('salary', 'Salario') for f in feature_names],
        'Importancia': importances
    }).sort_values('Importancia', ascending=True)
    
    # Gráfico seaborn/matplotlib
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Seleccionar top 8 para claridad
    sns.barplot(
        data=feat_imp.tail(8),
        x='Importancia',
        y='Variable',
        palette='viridis',
        ax=ax
    )
    ax.set_xlabel('Importancia Relativa')
    ax.set_ylabel('Variable')
    ax.set_title('Top 8 Predictores Más Influyentes')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown(f"</div>", unsafe_allow_html=True)

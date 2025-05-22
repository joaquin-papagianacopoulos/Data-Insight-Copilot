import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Data Insight Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: ##552d7a;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #b3d9ff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Genera datos de ejemplo para testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = {
        'fecha': dates,
        'ventas': np.random.normal(1000, 200, 100) + np.random.exponential(50, 100),
        'clientes': np.random.poisson(25, 100),
        'categoria': np.random.choice(['A', 'B', 'C'], 100, p=[0.5, 0.3, 0.2]),
        'region': np.random.choice(['Norte', 'Sur', 'Centro'], 100, p=[0.4, 0.35, 0.25])
    }
    return pd.DataFrame(data)

def analyze_basic_stats(df):
    """Análisis estadístico básico"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'missing': df[col].isnull().sum()
        }
    
    return stats

def detect_outliers(df, column):
    """Detecta outliers usando IQR"""
    if column not in df.select_dtypes(include=[np.number]).columns:
        return []
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def create_visualizations(df):
    """Genera visualizaciones automáticas"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    visualizations = []
    
    # Histogramas para columnas numéricas
    for col in numeric_cols[:3]:  # Limitar a 3 para no saturar
        fig = px.histogram(df, x=col, title=f'Distribución de {col}')
        visualizations.append(('histogram', col, fig))
    
    # Gráficos de barras para categóricas
    for col in categorical_cols[:2]:
        if df[col].nunique() <= 10:  # Solo si no hay demasiadas categorías
            value_counts = df[col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f'Frecuencia de {col}')
            visualizations.append(('bar', col, fig))
    
    # Correlación si hay múltiples columnas numéricas
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Matriz de Correlación")
        visualizations.append(('correlation', 'correlation', fig))
    
    return visualizations

def main():
    # Header principal
    st.markdown('<h1 class="main-header">🧠 Data Insight Copilot</h1>', unsafe_allow_html=True)
    st.markdown("### Analiza tus datos CSV con inteligencia artificial")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    
    # Opción para datos de ejemplo
    use_sample = st.sidebar.checkbox("Usar datos de ejemplo", value=False)
    
    if use_sample:
        df = load_sample_data()
        st.sidebar.success("✅ Datos de ejemplo cargados")
    else:
        # Upload de archivo
        uploaded_file = st.sidebar.file_uploader(
            "Sube tu archivo CSV",
            type=['csv'],
            help="Formatos soportados: CSV"
        )
        
        if uploaded_file is not None:
            try:
                # Leer el archivo
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Archivo cargado: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Error al cargar archivo: {str(e)}")
                return
        else:
            st.info("👈 Sube un archivo CSV o usa los datos de ejemplo desde la barra lateral")
            return
    
    # Información básica del dataset
    st.header("📊 Información General del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📝 Filas", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📋 Columnas", len(df.columns))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        missing_count = df.isnull().sum().sum()
        st.metric("❓ Valores Faltantes", f"{missing_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("💾 Tamaño (MB)", f"{memory_usage:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Preview de los datos
    st.header("👀 Vista Previa de los Datos")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Información de columnas
    st.header("🔍 Información de Columnas")
    
    col_info = []
    for col in df.columns:
        col_info.append({
            'Columna': col,
            'Tipo': str(df[col].dtype),
            'No Nulos': df[col].count(),
            'Nulos': df[col].isnull().sum(),
            '% Nulos': f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
            'Únicos': df[col].nunique()
        })
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, use_container_width=True)
    
    # Análisis estadístico
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.header("📈 Estadísticas Descriptivas")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Visualizaciones automáticas
    st.header("📊 Visualizaciones Automáticas")
    
    visualizations = create_visualizations(df)
    
    for viz_type, col_name, fig in visualizations:
        st.subheader(f"📊 {fig.layout.title.text}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detección de outliers
    if len(numeric_cols) > 0:
        st.header("🎯 Detección de Anomalías")
        
        selected_col = st.selectbox("Selecciona columna para análisis de outliers:", numeric_cols)
        
        if selected_col:
            outliers = detect_outliers(df, selected_col)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🚨 Outliers Detectados", len(outliers))
            
            with col2:
                if len(outliers) > 0:
                    outlier_percentage = (len(outliers) / len(df)) * 100
                    st.metric("📊 % del Total", f"{outlier_percentage:.2f}%")
            
            if len(outliers) > 0:
                st.subheader("Valores Atípicos Encontrados:")
                st.dataframe(outliers, use_container_width=True)
    
    # Insights automáticos básicos
    st.header("💡 Insights Iniciales")
    
    insights = []
    
    # Dataset size insight
    if len(df) > 10000:
        insights.append("📊 Dataset grande: Más de 10,000 registros disponibles para análisis profundo.")
    elif len(df) < 100:
        insights.append("📊 Dataset pequeño: Considera recopilar más datos para obtener insights más robustos.")
    
    # Missing data insight
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_percentage > 20:
        insights.append("⚠️ Alta cantidad de datos faltantes: Considera estrategias de limpieza de datos.")
    elif missing_percentage < 5:
        insights.append("✅ Excelente calidad de datos: Muy pocos valores faltantes detectados.")
    
    # Numeric columns insight
    if len(numeric_cols) > len(df.columns) * 0.7:
        insights.append("🔢 Dataset predominantemente numérico: Ideal para análisis estadísticos y machine learning.")
    
    # Display insights
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Footer con información del proyecto
    st.markdown("---")
    st.markdown("**Data Insight Copilot** v1.0 - Powered by **Joaquin Papagianacopoulos** 🚀")
    st.markdown(f"*Análisis generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()
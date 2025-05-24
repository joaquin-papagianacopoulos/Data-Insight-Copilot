import streamlit as st
import pandas as pd
import plotly.express as px
from smart_analyzer import analyze_with_smart_analyzer
from visualization_engine import VisualizationEngine
from report_generator import PDFReportGenerator
from data_validator import DataValidator
import time
import os

# Configuración de la página
st.set_page_config(
    page_title="Data Insight Copilot",
    page_icon="📊",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1e88e5;
        padding: 1rem 0;
    }
    .insight-box {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1e88e5;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🔥 Data Insight Copilot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Análisis inteligente de datos con IA</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Cargar Datos")
    uploaded_file = st.file_uploader(
        "Seleccioná tu archivo CSV",
        type=['csv'],
        help="Máximo 200MB"
    )
    
    st.divider()
    
    st.header("⚙️ Configuración")
    analysis_depth = st.select_slider(
        "Profundidad del análisis",
        options=["Rápido", "Normal", "Profundo"],
        value="Normal"
    )
    
    generate_visualizations = st.checkbox("Generar visualizaciones", value=True)
    
    st.divider()
    
    st.info("""
    **¿Cómo funciona?**
    1. Subí tu CSV
    2. Los agentes IA analizan los datos
    3. Obtenés insights y recomendaciones
    4. Descargá el reporte final
    """)

# Main content
if uploaded_file is not None:
    # Cargar datos
    try:
        df = pd.read_csv(uploaded_file)
        
        # Mostrar preview
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("📋 Vista previa de los datos")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.metric("Filas", f"{len(df):,}")
            st.metric("Columnas", len(df.columns))
        
        with col3:
            st.metric("Tamaño", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.metric("Datos faltantes", f"{missing_percent:.1f}%")
        
        # Botón de análisis
        if st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True):
            with st.spinner("🤖 Los agentes están analizando tus datos..."):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simular progreso
                status_text.text("Inicializando agentes...")
                progress_bar.progress(10)
                time.sleep(1)
                
                # Ejecutar análisis optimizado
                results = analyze_with_smart_analyzer(df)
                
                # Generar visualizaciones
                status_text.text("📊 Generando visualizaciones...")
                progress_bar.progress(75)
                viz_engine = VisualizationEngine()
                visualizations = viz_engine.generate_all_visualizations(df)
                
                progress_bar.progress(100)
                status_text.text("¡Análisis completado!")
                
                # Guardar en session state
                st.session_state['analysis_results'] = results
                st.session_state['visualizations'] = visualizations
                st.session_state['dataframe'] = df
                
                status_text.text("Detectando patrones...")
                progress_bar.progress(60)
                time.sleep(1)
                
                status_text.text("Generando reporte...")
                progress_bar.progress(90)
                time.sleep(1)
                
                progress_bar.progress(100)
                status_text.text("¡Análisis completado!")
                
                # Mostrar resultados
                st.success("✅ Análisis completado exitosamente")
                
                # Tabs para organizar resultados
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Resumen", "🔍 Insights", "📈 Visualizaciones", "🎯 Anomalías", "📄 Reporte"])
                
                with tab1:
                    st.markdown("### 📊 Resumen del Análisis")
                    
                    # Mostrar métricas clave
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if df.select_dtypes(include=['number']).shape[1] > 0:
                            st.metric("Media Global", f"{df.select_dtypes(include=['number']).mean().mean():.2f}")
                    
                    with col2:
                        unique_ratio = df.nunique().mean() / len(df) * 100
                        st.metric("Ratio de Unicidad", f"{unique_ratio:.1f}%")
                    
                    with col3:
                        correlation_strength = 0
                        numeric_cols = df.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 1:
                            corr_matrix = df[numeric_cols].corr()
                            correlation_strength = (corr_matrix.abs().sum().sum() - len(numeric_cols)) / (len(numeric_cols) * (len(numeric_cols) - 1))
                        st.metric("Correlación Promedio", f"{correlation_strength:.2f}")
                    
                    # Estadísticas descriptivas
                    if df.select_dtypes(include=['number']).shape[1] > 0:
                        st.markdown("#### Estadísticas Descriptivas")
                        st.dataframe(df.describe(), use_container_width=True)
                    
                    # Tipos de datos
                    st.markdown("#### Información de Variables")
                    dtype_df = pd.DataFrame({
                        'Variable': df.columns,
                        'Tipo': df.dtypes.astype(str),
                        'Valores únicos': [df[col].nunique() for col in df.columns],
                        'Valores faltantes': df.isnull().sum().values,
                        '% Faltantes': (df.isnull().sum() / len(df) * 100).round(2).values
                    })
                    st.dataframe(dtype_df, use_container_width=True)
                
                with tab2:
                    st.markdown("### 🔍 Insights Descubiertos")
                    
                    # Mostrar secciones del análisis
                    if 'sections' in results:
                        sections = results['sections']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Calidad de Datos")
                            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
                            st.write(sections.get('calidad', 'No disponible'))
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("#### 🎯 Patrones Clave")
                            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
                            st.write(sections.get('patrones', 'No disponible'))
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### 💡 Recomendaciones")
                            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
                            st.write(sections.get('recomendaciones', 'No disponible'))
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown("#### 📋 Resumen Ejecutivo")
                            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
                            st.write(sections.get('resumen', 'No disponible'))
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.write(results.get('full_analysis', 'No se pudo generar el análisis'))
                
                with tab3:
                    st.markdown("### 📈 Visualizaciones Automáticas")
                    
                    if 'visualizations' in st.session_state:
                        viz_options = list(st.session_state['visualizations'].keys())
                        
                        # Selector de visualización
                        selected_viz = st.selectbox(
                            "Seleccionar visualización:",
                            viz_options,
                            format_func=lambda x: {
                                'overview': '📊 Vista General',
                                'distributions': '📈 Distribuciones',
                                'correlation': '🔗 Correlaciones',
                                'categorical': '📊 Variables Categóricas',
                                'bivariate': '🔍 Análisis Bivariado',
                                'outliers': '🎯 Detección de Outliers',
                                'timeseries': '📅 Serie Temporal'
                            }.get(x, x)
                        )
                        
                        # Mostrar visualización seleccionada
                        if selected_viz in st.session_state['visualizations']:
                            fig = st.session_state['visualizations'][selected_viz]
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"No hay datos suficientes para generar: {selected_viz}")
                        
                        # Botón para descargar todas las visualizaciones
                        with st.expander("💾 Opciones de exportación"):
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📥 Descargar visualización actual"):
                                    fig = st.session_state['visualizations'][selected_viz]
                                    if fig:
                                        img_bytes = fig.to_image(format="png", width=1200, height=800)
                                        st.download_button(
                                            label="Descargar PNG",
                                            data=img_bytes,
                                            file_name=f"{selected_viz}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.png",
                                            mime="image/png"
                                        )
                            
                            with col2:
                                st.info("Las visualizaciones también se incluyen en el reporte PDF")
                
                with tab4:
                    st.markdown("### 🎯 Detección de Anomalías")
                    
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    
                    if numeric_cols:
                        selected_col = st.selectbox("Seleccionar variable para análisis de anomalías:", numeric_cols)
                        
                        if selected_col:
                            # Detectar anomalías
                            viz_engine = VisualizationEngine()
                            anomalies, summary = viz_engine.detect_anomalies(df, selected_col)
                            
                            # Mostrar resumen
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total de datos", summary['total_data_points'])
                                st.metric("Media", f"{summary['mean']:.2f}")
                            
                            with col2:
                                st.metric("Anomalías IQR", summary['iqr_anomalies'])
                                st.metric("Desviación estándar", f"{summary['std']:.2f}")
                            
                            with col3:
                                st.metric("Anomalías Z-Score", summary['zscore_anomalies'])
                                st.metric("Asimetría", f"{summary['skewness']:.2f}")
                            
                            # Mostrar detalles
                            with st.expander("📊 Ver detalles de anomalías"):
                                st.write(f"**Límites IQR:** {summary['iqr_bounds'][0]:.2f} - {summary['iqr_bounds'][1]:.2f}")
                                st.write(f"**Curtosis:** {summary['kurtosis']:.2f}")
                                
                                if summary['iqr_anomalies'] > 0:
                                    st.write(f"**Índices de anomalías IQR:** {anomalies['iqr_outliers'][:10]}...")
                                
                                if summary['isolation_forest_anomalies'] > 0:
                                    st.write(f"**Anomalías por Isolation Forest:** {summary['isolation_forest_anomalies']} detectadas")
                            
                            # Visualización de outliers
                            if 'outliers' in st.session_state.get('visualizations', {}):
                                st.plotly_chart(st.session_state['visualizations']['outliers'], use_container_width=True)
                    else:
                        st.info("No hay variables numéricas para análisis de anomalías")
                
                with tab5:
                    st.markdown("### 📄 Reporte Final")
                    
                    # Vista previa del contenido
                    #df = st.session_state.get("df", "dataset.csv")

                    with st.expander("👁️ Vista previa del reporte"):
                        report_content = f"""
                            # Data Insight Report
                            ## Dataset: {df}
                            ## Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

                            ### Resumen Ejecutivo
                            {results.get('sections', {}).get('resumen', results.get('full_analysis', 'No disponible'))}

                            ### Calidad de Datos
                            {results.get('sections', {}).get('calidad', 'No disponible')}

                            ### Patrones Identificados
                            {results.get('sections', {}).get('patrones', 'No disponible')}

                            ### Recomendaciones
                            {results.get('sections', {}).get('recomendaciones', 'No disponible')}

                            ### Información del Dataset
                            - Total de filas: {len(df)}
                            - Total de columnas: {len(df.columns)}
                            - Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
                        """
                        st.text_area("", report_content, height=400)
                    
                    # Opciones de descarga
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Descargar como texto
                        st.download_button(
                            label="📥 Descargar Reporte (TXT)",
                            data=report_content,
                            file_name=f"reporte_{df}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        # Generar PDF
                        if st.button("📑 Generar Reporte PDF", type="primary"):
                            with st.spinner("Generando PDF..."):
                                try:
                                    # Crear generador de PDF
                                    pdf_gen = PDFReportGenerator()
                                    
                                    # Preparar datos para el PDF
                                    analysis_results = st.session_state.get('analysis_results', results)
                                    analysis_results['dataset_name'] = df
                                    
                                    # Generar PDF
                                    pdf_bytes = pdf_gen.generate_report(
                                        df=df,
                                        analysis_results=analysis_results,
                                        visualizations=st.session_state.get('visualizations', {}),
                                        filename="report.pdf"
                                    )
                                    
                                    # Botón de descarga para PDF
                                    st.download_button(
                                        label="📥 Descargar PDF",
                                        data=pdf_bytes,
                                        file_name=f"data_insight_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                                        mime="application/pdf"
                                    )
                                    
                                    st.success("✅ PDF generado exitosamente!")
                                except Exception as e:
                                    st.error(f"Error al generar PDF: {str(e)}")
                                    st.info("Instala las dependencias necesarias: pip install reportlab pillow kaleido")
        
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        st.info("Asegurate de que el archivo sea un CSV válido")

else:
    # Landing page
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2>👋 ¡Bienvenido a Data Insight Copilot!</h2>
        <p style='font-size: 1.2rem; color: #666; margin: 2rem 0;'>
            Transformá tus datos en insights accionables con el poder de la IA
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🧹 Limpieza Automática</h3>
            <p>Detecta y sugiere correcciones para problemas en los datos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>🔍 Detección de Patrones</h3>
            <p>Encuentra correlaciones y anomalías ocultas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Reportes Inteligentes</h3>
            <p>Genera resúmenes ejecutivos listos para presentar</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem;'>
        <p style='font-size: 1.1rem;'>
            ⬅️ Subí tu archivo CSV desde la barra lateral para comenzar
        </p>
    </div>
    """, unsafe_allow_html=True)
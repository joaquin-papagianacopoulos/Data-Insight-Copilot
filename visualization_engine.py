import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

class VisualizationEngine:
    """Motor inteligente para generar visualizaciones automáticas"""
    
    def __init__(self):
        self.color_palette = px.colors.sequential.Viridis
        self.template = "plotly_white"
        
    def generate_all_visualizations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Genera todas las visualizaciones relevantes para el dataset"""
        visualizations = {}
        
        # Identificar tipos de columnas
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # 1. Overview del dataset
        visualizations['overview'] = self._create_overview_plot(df)
        
        # 2. Distribuciones numéricas
        if numeric_cols:
            visualizations['distributions'] = self._create_distribution_plots(df, numeric_cols[:6])
            visualizations['correlation'] = self._create_correlation_heatmap(df, numeric_cols)
            
        # 3. Análisis categórico
        if categorical_cols:
            visualizations['categorical'] = self._create_categorical_analysis(df, categorical_cols[:4])
            
        # 4. Análisis bivariado
        if numeric_cols and categorical_cols:
            visualizations['bivariate'] = self._create_bivariate_analysis(
                df, categorical_cols[0], numeric_cols[0]
            )
            
        # 5. Detección de outliers
        if numeric_cols:
            visualizations['outliers'] = self._create_outlier_analysis(df, numeric_cols[:4])
            
        # 6. Serie temporal si hay fechas
        if datetime_cols and numeric_cols:
            visualizations['timeseries'] = self._create_timeseries_plot(
                df, datetime_cols[0], numeric_cols[0]
            )
            
        return visualizations
    
    def _create_overview_plot(self, df: pd.DataFrame) -> go.Figure:
        """Crea un gráfico de overview del dataset"""
        # Calcular estadísticas básicas
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        # Crear subplot con información general
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Tipos de Datos', 'Datos Faltantes (%)', 
                          'Distribución de Tipos', 'Top 10 Columnas con Más Faltantes'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'pie'}, {'type': 'bar'}]]
        )
        
        # 1. Tipos de datos
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Bar(x=dtype_counts.index.astype(str), y=dtype_counts.values,
                   marker_color='lightblue', name='Tipos'),
            row=1, col=1
        )
        
        # 2. Porcentaje de datos faltantes general
        completeness = [100 - missing_percent.mean(), missing_percent.mean()]
        fig.add_trace(
            go.Bar(x=['Completos', 'Faltantes'], y=completeness,
                   marker_color=['green', 'red'], name='Completitud'),
            row=1, col=2
        )
        
        # 3. Pie chart de tipos
        fig.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values,
                   hole=0.3),
            row=2, col=1
        )
        
        # 4. Top columnas con datos faltantes
        top_missing = missing_percent.nlargest(10)
        if len(top_missing) > 0:
            fig.add_trace(
                go.Bar(x=top_missing.values, y=top_missing.index,
                       orientation='h', marker_color='orange', name='Faltantes %'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="📊 Vista General del Dataset",
            showlegend=False,
            height=800,
            template=self.template
        )
        
        return fig
    
    def _create_distribution_plots(self, df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Crea histogramas y box plots para columnas numéricas"""
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 1) // 2
        
        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=[f'Distribución de {col}' for col in numeric_cols],
            vertical_spacing=0.1
        )
        
        for idx, col in enumerate(numeric_cols):
            row = (idx // 2) + 1
            col_pos = (idx % 2) + 1
            
            # Eliminar NaN para el histograma
            data = df[col].dropna()
            
            # Agregar histograma con KDE
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=col,
                    nbinsx=30,
                    opacity=0.7,
                    marker_color=self.color_palette[idx % len(self.color_palette)]
                ),
                row=row, col=col_pos
            )
            
            # Agregar línea de media
            mean_val = data.mean()
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"μ={mean_val:.2f}",
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title_text="📈 Distribuciones de Variables Numéricas",
            showlegend=False,
            height=300 * n_rows,
            template=self.template
        )
        
        return fig
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Crea un heatmap de correlación interactivo"""
        try:
            # Usar el validador para calcular correlación de forma segura
            from data_validator import DataValidator
            validator = DataValidator()
            corr_matrix = validator.safe_correlation_matrix(df[numeric_cols])
            
            if corr_matrix.empty:
                # Crear figura vacía con mensaje
                fig = go.Figure()
                fig.add_annotation(
                    text="No se pudo calcular la matriz de correlación",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                return fig
            
            # Crear máscara para el triángulo superior
            mask = np.triu(np.ones_like(corr_matrix), k=1)
            corr_masked = corr_matrix.where(~mask)
            
            # Crear heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_masked.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(corr_masked.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlación")
            ))
            
            fig.update_layout(
                title="🔗 Matriz de Correlación",
                width=800,
                height=800,
                template=self.template,
                xaxis={'side': 'bottom'}
            )
            
            return fig
            
        except Exception as e:
            # Crear figura con mensaje de error
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generando correlación: {str(e)[:50]}...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="red")
            )
            return fig
    
    def _create_categorical_analysis(self, df: pd.DataFrame, categorical_cols: List[str]) -> go.Figure:
        """Analiza variables categóricas"""
        n_cols = len(categorical_cols)
        fig = make_subplots(
            rows=n_cols, cols=2,
            subplot_titles=[f'{col} - Distribución' for col in categorical_cols] + 
                         [f'{col} - Top 10' for col in categorical_cols],
            specs=[[{'type': 'bar'}, {'type': 'bar'}] for _ in range(n_cols)]
        )
        
        for idx, col in enumerate(categorical_cols):
            # Contar valores
            value_counts = df[col].value_counts()
            
            # Gráfico de barras completo
            fig.add_trace(
                go.Bar(
                    x=value_counts.index[:20],  # Limitar a 20 categorías
                    y=value_counts.values[:20],
                    name=col,
                    marker_color=self.color_palette[idx % len(self.color_palette)]
                ),
                row=idx+1, col=1
            )
            
            # Top 10 con porcentajes
            top10 = value_counts.head(10)
            percentages = (top10 / len(df)) * 100
            
            fig.add_trace(
                go.Bar(
                    x=top10.index,
                    y=percentages,
                    name=f'{col} %',
                    text=[f'{p:.1f}%' for p in percentages],
                    textposition='auto',
                    marker_color=self.color_palette[(idx+2) % len(self.color_palette)]
                ),
                row=idx+1, col=2
            )
        
        fig.update_layout(
            title_text="📊 Análisis de Variables Categóricas",
            showlegend=False,
            height=400 * n_cols,
            template=self.template
        )
        
        return fig
    
    def _create_bivariate_analysis(self, df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
        """Análisis bivariado entre categórica y numérica"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{num_col} por {cat_col} - Box Plot',
                f'{num_col} por {cat_col} - Violin Plot',
                f'Media de {num_col} por {cat_col}',
                f'Distribución de {cat_col}'
            ),
            specs=[[{'type': 'box'}, {'type': 'violin'}],
                   [{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Preparar datos
        categories = df[cat_col].unique()[:10]  # Limitar a 10 categorías
        
        # 1. Box plot
        for cat in categories:
            data = df[df[cat_col] == cat][num_col].dropna()
            fig.add_trace(
                go.Box(y=data, name=str(cat), showlegend=False),
                row=1, col=1
            )
        
        # 2. Violin plot
        for cat in categories:
            data = df[df[cat_col] == cat][num_col].dropna()
            fig.add_trace(
                go.Violin(y=data, name=str(cat), showlegend=False),
                row=1, col=2
            )
        
        # 3. Medias por categoría
        means = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=means.index, y=means.values, marker_color='lightcoral'),
            row=2, col=1
        )
        
        # 4. Distribución de categorías
        cat_dist = df[cat_col].value_counts().head(10)
        fig.add_trace(
            go.Pie(labels=cat_dist.index, values=cat_dist.values, hole=0.3),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"🔍 Análisis Bivariado: {num_col} vs {cat_col}",
            showlegend=False,
            height=800,
            template=self.template
        )
        
        return fig
    
    def _create_outlier_analysis(self, df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Detecta y visualiza outliers usando IQR y Z-score"""
        n_cols = len(numeric_cols)
        
        fig = make_subplots(
            rows=n_cols, cols=2,
            subplot_titles=[f'{col} - Box Plot con Outliers' for col in numeric_cols] + 
                         [f'{col} - Distribución con Outliers' for col in numeric_cols]
        )
        
        outlier_summary = {}
        
        for idx, col in enumerate(numeric_cols):
            data = df[col].dropna()
            
            # Calcular IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identificar outliers
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'bounds': (lower_bound, upper_bound)
            }
            
            # 1. Box plot con outliers marcados
            fig.add_trace(
                go.Box(
                    y=data,
                    name=col,
                    boxpoints='outliers',
                    marker_color='lightblue',
                    marker=dict(
                        outliercolor='red',
                        symbol='diamond',
                        size=8
                    )
                ),
                row=idx+1, col=1
            )
            
            # 2. Histograma con outliers marcados
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=col,
                    opacity=0.7,
                    marker_color='lightgreen'
                ),
                row=idx+1, col=2
            )
            
            # Marcar outliers en el histograma
            if len(outliers) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=outliers,
                        y=[0] * len(outliers),
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='diamond'),
                        name='Outliers',
                        showlegend=False
                    ),
                    row=idx+1, col=2
                )
            
            # Agregar líneas de límites
            fig.add_hline(y=lower_bound, line_dash="dash", line_color="orange", 
                         annotation_text=f"Lower: {lower_bound:.2f}", row=idx+1, col=1)
            fig.add_hline(y=upper_bound, line_dash="dash", line_color="orange", 
                         annotation_text=f"Upper: {upper_bound:.2f}", row=idx+1, col=1)
        
        fig.update_layout(
            title_text="🎯 Detección de Outliers (Método IQR)",
            showlegend=False,
            height=300 * n_cols,
            template=self.template
        )
        
        return fig
    
    def _create_timeseries_plot(self, df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        """Crea un gráfico de serie temporal si hay fechas"""
        # Asegurar que la columna de fecha sea datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Agregar por fecha
        ts_data = df.groupby(date_col)[value_col].agg(['mean', 'sum', 'count']).reset_index()
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f'Promedio de {value_col} en el tiempo',
                          f'Suma de {value_col} en el tiempo',
                          f'Cantidad de registros en el tiempo'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # 1. Promedio
        fig.add_trace(
            go.Scatter(
                x=ts_data[date_col],
                y=ts_data['mean'],
                mode='lines+markers',
                name='Promedio',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Suma
        fig.add_trace(
            go.Scatter(
                x=ts_data[date_col],
                y=ts_data['sum'],
                mode='lines+markers',
                name='Suma',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # 3. Conteo
        fig.add_trace(
            go.Bar(
                x=ts_data[date_col],
                y=ts_data['count'],
                name='Cantidad',
                marker_color='lightcoral'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title_text=f"📅 Análisis Temporal: {value_col}",
            showlegend=False,
            height=900,
            template=self.template
        )
        
        return fig
    
    def detect_anomalies(self, df: pd.DataFrame, column: str) -> Tuple[pd.Series, Dict[str, Any]]:
        """Detecta anomalías usando múltiples métodos"""
        data = df[column].dropna()
        
        anomalies = {
            'iqr_outliers': [],
            'zscore_outliers': [],
            'isolation_forest': []
        }
        
        # 1. Método IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_mask = (data < lower_bound) | (data > upper_bound)
        anomalies['iqr_outliers'] = data[iqr_mask].index.tolist()
        
        # 2. Z-Score (para distribuciones normales)
        z_scores = np.abs(stats.zscore(data))
        zscore_mask = z_scores > 3
        anomalies['zscore_outliers'] = data.index[zscore_mask].tolist()
        
        # 3. Isolation Forest (si hay suficientes datos)
        if len(data) > 50:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(data.values.reshape(-1, 1))
            iso_mask = predictions == -1
            anomalies['isolation_forest'] = data.index[iso_mask].tolist()
        
        # Resumen
        summary = {
            'total_data_points': len(data),
            'iqr_anomalies': len(anomalies['iqr_outliers']),
            'zscore_anomalies': len(anomalies['zscore_outliers']),
            'isolation_forest_anomalies': len(anomalies['isolation_forest']),
            'iqr_bounds': (lower_bound, upper_bound),
            'mean': data.mean(),
            'std': data.std(),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
        
        return anomalies, summary
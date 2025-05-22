import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Clase principal para análisis de datos CSV"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
    def clean_data(self) -> Dict[str, Any]:
        """Limpieza básica de datos"""
        cleaning_report = {
            'original_shape': self.df.shape,
            'duplicates_removed': 0,
            'columns_cleaned': []
        }
        
        # Eliminar duplicados
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        cleaning_report['duplicates_removed'] = initial_rows - len(self.df)
        
        # Limpiar columnas numéricas
        for col in self.numeric_cols:
            # Reemplazar infinitos con NaN
            self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
            cleaning_report['columns_cleaned'].append(f"{col}: infinitos reemplazados")
        
        # Limpiar columnas categóricas
        for col in self.categorical_cols:
            # Eliminar espacios extra
            self.df[col] = self.df[col].astype(str).str.strip()
            cleaning_report['columns_cleaned'].append(f"{col}: espacios eliminados")
        
        cleaning_report['final_shape'] = self.df.shape
        return cleaning_report
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Estadísticas básicas del dataset"""
        return {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'unique_counts': self.df.nunique().to_dict()
        }
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> pd.DataFrame:
        """Detecta outliers usando método IQR"""
        if column not in self.numeric_cols:
            return pd.DataFrame()
        
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = self.df[
            (self.df[column] < lower_bound) | 
            (self.df[column] > upper_bound)
        ]
        
        return outliers
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3) -> pd.DataFrame:
        """Detecta outliers usando Z-Score"""
        if column not in self.numeric_cols:
            return pd.DataFrame()
        
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        outliers = self.df[z_scores > threshold]
        
        return outliers
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """Análisis de correlaciones entre variables numéricas"""
        if len(self.numeric_cols) < 2:
            return {}
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Encontrar correlaciones fuertes
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Correlación fuerte
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_corr
        }
    
    def get_distribution_insights(self, column: str) -> Dict[str, Any]:
        """Analiza la distribución de una columna"""
        if column not in self.df.columns:
            return {}
        
        insights = {'column': column}
        
        if column in self.numeric_cols:
            data = self.df[column].dropna()
            insights.update({
                'type': 'numeric',
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min()
            })
            
            # Determinar tipo de distribución
            if abs(insights['skewness']) < 0.5:
                insights['distribution_type'] = 'aproximadamente normal'
            elif insights['skewness'] > 0.5:
                insights['distribution_type'] = 'sesgada hacia la derecha'
            else:
                insights['distribution_type'] = 'sesgada hacia la izquierda'
                
        elif column in self.categorical_cols:
            value_counts = self.df[column].value_counts()
            insights.update({
                'type': 'categorical',
                'unique_values': len(value_counts),
                'most_frequent': value_counts.index[0],
                'most_frequent_count': value_counts.iloc[0],
                'least_frequent': value_counts.index[-1],
                'least_frequent_count': value_counts.iloc[-1]
            })
        
        return insights
    
    def generate_summary_insights(self) -> List[str]:
        """Genera insights automáticos del dataset"""
        insights = []
        
        # Insights sobre el tamaño del dataset
        rows, cols = self.df.shape
        if rows > 100000:
            insights.append(f"📊 Dataset muy grande con {rows:,} registros - Excelente para análisis estadísticos robustos")
        elif rows > 10000:
            insights.append(f"📊 Dataset de tamaño considerable con {rows:,} registros - Apropiado para machine learning")
        elif rows < 1000:
            insights.append(f"📊 Dataset pequeño con {rows:,} registros - Considera recopilar más datos")
        
        # Insights sobre calidad de datos
        total_missing = self.df.isnull().sum().sum()
        missing_percentage = (total_missing / (rows * cols)) * 100
        
        if missing_percentage > 30:
            insights.append("⚠️ Alta cantidad de datos faltantes (>30%) - Requiere limpieza significativa")
        elif missing_percentage > 10:
            insights.append("⚠️ Cantidad moderada de datos faltantes (>10%) - Considera estrategias de imputación")
        elif missing_percentage < 5:
            insights.append("✅ Excelente calidad de datos - Menos del 5% de valores faltantes")
        
        # Insights sobre tipos de variables
        if len(self.numeric_cols) > len(self.categorical_cols):
            insights.append("🔢 Dataset predominantemente numérico - Ideal para análisis estadísticos y correlaciones")
        elif len(self.categorical_cols) > len(self.numeric_cols):
            insights.append("📝 Dataset predominantemente categórico - Útil para análisis de frecuencias y segmentación")
        
        # Insights sobre outliers en columnas numéricas
        outlier_cols = []
        for col in self.numeric_cols[:3]:  # Revisar solo las primeras 3 para no saturar
            outliers = self.detect_outliers_iqr(col)
            if len(outliers) > len(self.df) * 0.05:  # Más del 5% son outliers
                outlier_cols.append(col)
        
        if outlier_cols:
            insights.append(f"🎯 Outliers detectados en: {', '.join(outlier_cols)} - Considera revisar estos valores")
        
        # Insights sobre correlaciones
        if len(self.numeric_cols) >= 2:
            corr_analysis = self.analyze_correlations()
            if corr_analysis.get('strong_correlations'):
                strong_pairs = len(corr_analysis['strong_correlations'])
                insights.append(f"🔗 {strong_pairs} correlaciones fuertes detectadas - Potencial multicolinealidad")
        
        return insights

def create_advanced_visualizations(df: pd.DataFrame) -> List[Tuple[str, str, Any]]:
    """Crea visualizaciones avanzadas basadas en el tipo de datos"""
    analyzer = DataAnalyzer(df)
    visualizations = []
    
    # 1. Distribuciones para variables numéricas
    for col in analyzer.numeric_cols[:4]:  # Limitar para performance
        # Histograma con curva de densidad
        fig = px.histogram(df, x=col, marginal="box", title=f'Distribución de {col}')
        visualizations.append(('distribution', col, fig))
    
    # 2. Gráficos de barras para categóricas
    for col in analyzer.categorical_cols[:3]:
        if df[col].nunique() <= 15:  # Solo si no hay demasiadas categorías
            value_counts = df[col].value_counts().head(10)
            fig = px.bar(x=value_counts.values, y=value_counts.index, 
                        orientation='h', title=f'Top categorías en {col}')
            visualizations.append(('categorical', col, fig))
    
    # 3. Matriz de correlación
    if len(analyzer.numeric_cols) > 1:
        corr_matrix = df[analyzer.numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale="RdBu",
                       title="Matriz de Correlación")
        visualizations.append(('correlation', 'correlation_matrix', fig))
    
    # 4. Boxplots para detectar outliers
    for col in analyzer.numeric_cols[:3]:
        fig = px.box(df, y=col, title=f'Detección de Outliers - {col}')
        visualizations.append(('outliers', col, fig))
    
    # 5. Scatter plots para correlaciones fuertes
    if len(analyzer.numeric_cols) >= 2:
        corr_analysis = analyzer.analyze_correlations()
        for corr_pair in corr_analysis.get('strong_correlations', [])[:2]:  # Solo 2 para no saturar
            fig = px.scatter(df, x=corr_pair['var1'], y=corr_pair['var2'],
                           title=f"Correlación: {corr_pair['var1']} vs {corr_pair['var2']} (r={corr_pair['correlation']:.3f})")
            visualizations.append(('scatter', f"{corr_pair['var1']}_vs_{corr_pair['var2']}", fig))
    
    return visualizations

def generate_sample_datasets() -> Dict[str, pd.DataFrame]:
    """Genera datasets de ejemplo para testing"""
    np.random.seed(42)
    
    datasets = {}
    
    # Dataset 1: Ventas e-commerce
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    ecommerce_data = {
        'fecha': dates,
        'ventas_diarias': np.random.normal(5000, 1500, 365) + 
                         np.random.exponential(500, 365) + 
                         np.sin(np.arange(365) * 2 * np.pi / 7) * 1000,  # Patrón semanal
        'pedidos': np.random.poisson(150, 365),
        'categoria': np.random.choice(['Electrónicos', 'Ropa', 'Hogar', 'Deportes'], 365),
        'canal': np.random.choice(['Web', 'App', 'Tienda Física'], 365, p=[0.5, 0.3, 0.2]),
        'descuento_aplicado': np.random.uniform(0, 0.3, 365),
        'satisfaccion_cliente': np.random.normal(4.2, 0.8, 365)
    }
    datasets['ecommerce'] = pd.DataFrame(ecommerce_data)
    
    # Dataset 2: Datos de empleados
    employee_data = {
        'empleado_id': range(1, 501),
        'departamento': np.random.choice(['IT', 'Ventas', 'Marketing', 'RRHH', 'Finanzas'], 500),
        'salario': np.random.normal(60000, 20000, 500),
        'años_experiencia': np.random.exponential(5, 500),
        'edad': np.random.normal(35, 10, 500),
        'genero': np.random.choice(['M', 'F'], 500),
        'educacion': np.random.choice(['Bachillerato', 'Universidad', 'Postgrado'], 500, p=[0.2, 0.6, 0.2]),
        'satisfaccion_laboral': np.random.normal(7, 2, 500),
        'horas_semanales': np.random.normal(40, 5, 500)
    }
    datasets['employees'] = pd.DataFrame(employee_data)
    
    # Dataset 3: Datos financieros
    financial_data = {
        'mes': pd.date_range('2020-01-01', periods=48, freq='M'),
        'ingresos': np.random.normal(100000, 25000, 48) * (1 + np.arange(48) * 0.02),  # Crecimiento
        'gastos': np.random.normal(75000, 15000, 48),
        'marketing_spend': np.random.normal(15000, 5000, 48),
        'nuevos_clientes': np.random.poisson(200, 48),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 48),
        'tipo_campania': np.random.choice(['Digital', 'Tradicional', 'Mixta'], 48)
    }
    datasets['financial'] = pd.DataFrame(financial_data)
    
    return datasets
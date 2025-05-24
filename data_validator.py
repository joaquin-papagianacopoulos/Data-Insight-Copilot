import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

class DataValidator:
    """Valida y limpia datos antes del análisis"""
    
    @staticmethod
    def validate_and_clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Valida y limpia el DataFrame, retorna el df limpio y un reporte"""
        report = {
            'original_shape': df.shape,
            'issues_found': [],
            'fixes_applied': [],
            'columns_modified': []
        }
        
        # Copia para no modificar el original
        df_clean = df.copy()
        
        # 1. Verificar y corregir tipos de datos problemáticos
        for col in df_clean.columns:
            try:
                # Detectar columnas que deberían ser numéricas
                if df_clean[col].dtype == 'object':
                    # Intentar convertir a numérico
                    df_temp = pd.to_numeric(df_clean[col], errors='coerce')
                    if df_temp.notna().sum() > len(df_clean) * 0.5:  # Si más del 50% son números
                        df_clean[col] = df_temp
                        report['fixes_applied'].append(f"Convertida columna '{col}' a numérico")
                        report['columns_modified'].append(col)
                
                # Verificar columnas booleanas mal codificadas
                if df_clean[col].dtype == 'object':
                    unique_vals = df_clean[col].dropna().unique()
                    if len(unique_vals) <= 3:  # Posible booleano
                        unique_vals_lower = [str(v).lower() for v in unique_vals]
                        if all(v in ['true', 'false', '1', '0', 'yes', 'no', 'si', 'no'] for v in unique_vals_lower):
                            # Convertir a booleano
                            df_clean[col] = df_clean[col].map({
                                'true': True, 'false': False,
                                '1': True, '0': False,
                                'yes': True, 'no': False,
                                'si': True, 'no': False,
                                'True': True, 'False': False,
                                'YES': True, 'NO': False,
                                'SI': True, 'NO': False
                            })
                            report['fixes_applied'].append(f"Convertida columna '{col}' a booleano")
                            report['columns_modified'].append(col)
                
            except Exception as e:
                report['issues_found'].append(f"Error procesando columna '{col}': {str(e)}")
        
        # 2. Manejar valores infinitos en columnas numéricas
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df_clean[col]).any():
                n_inf = np.isinf(df_clean[col]).sum()
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                report['fixes_applied'].append(f"Reemplazados {n_inf} valores infinitos en '{col}' con NaN")
                report['issues_found'].append(f"Columna '{col}' tenía valores infinitos")
        
        # 3. Verificar y limpiar nombres de columnas
        new_columns = []
        for col in df_clean.columns:
            # Limpiar caracteres especiales
            new_col = str(col).strip()
            new_col = new_col.replace(' ', '_')
            new_col = ''.join(c if c.isalnum() or c == '_' else '_' for c in new_col)
            new_columns.append(new_col)
        
        if list(df_clean.columns) != new_columns:
            df_clean.columns = new_columns
            report['fixes_applied'].append("Nombres de columnas limpiados (espacios y caracteres especiales)")
        
        # 4. Eliminar columnas completamente vacías
        empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
        if empty_cols:
            df_clean = df_clean.drop(columns=empty_cols)
            report['fixes_applied'].append(f"Eliminadas {len(empty_cols)} columnas vacías: {empty_cols}")
            report['issues_found'].append(f"Se encontraron columnas completamente vacías")
        
        # 5. Verificar duplicados
        n_duplicates = df_clean.duplicated().sum()
        if n_duplicates > 0:
            report['issues_found'].append(f"Se encontraron {n_duplicates} filas duplicadas")
        
        # 6. Reporte final
        report['final_shape'] = df_clean.shape
        report['data_types'] = df_clean.dtypes.to_dict()
        report['validation_passed'] = len(report['issues_found']) == 0
        
        return df_clean, report
    
    @staticmethod
    def safe_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Calcula matriz de correlación de forma segura"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return pd.DataFrame()
        
        # Eliminar columnas con varianza cero
        std = numeric_df.std()
        cols_to_keep = std[std > 0].index
        numeric_df = numeric_df[cols_to_keep]
        
        if numeric_df.empty:
            return pd.DataFrame()
        
        # Calcular correlación manejando NaN
        try:
            corr_matrix = numeric_df.corr(method='pearson', numeric_only=True)
            # Reemplazar NaN con 0 en la diagonal
            np.fill_diagonal(corr_matrix.values, 1.0)
            return corr_matrix
        except Exception:
            return pd.DataFrame()
    
    @staticmethod
    def prepare_for_visualization(series: pd.Series) -> pd.Series:
        """Prepara una serie para visualización"""
        # Eliminar NaN e infinitos
        clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Si es numérica, verificar que sea válida
        if pd.api.types.is_numeric_dtype(clean_series):
            # Eliminar outliers extremos si es necesario
            if len(clean_series) > 0:
                q1 = clean_series.quantile(0.01)
                q99 = clean_series.quantile(0.99)
                # Solo si el rango es muy extremo
                if (q99 - q1) > 0 and (clean_series.max() - clean_series.min()) / (q99 - q1) > 100:
                    clean_series = clean_series[(clean_series >= q1) & (clean_series <= q99)]
        
        return clean_series
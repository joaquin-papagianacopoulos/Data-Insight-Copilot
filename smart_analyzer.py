import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import time
from typing import Dict, Any, List
import json

load_dotenv()

class SmartDataAnalyzer:
    """Analizador optimizado para trabajar con límites de Groq"""
    
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "gemma2-9b-it"  # Modelo eficiente
        self.tokens_used = 0
        self.last_call_time = 0
        
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análisis completo con una sola llamada optimizada"""
        
        # Preparar datos de manera eficiente
        dataset_summary = self._prepare_efficient_summary(df)
        
        # Crear un prompt único y optimizado
        prompt = self._create_unified_prompt(dataset_summary)
        
        # Hacer UNA SOLA llamada a la API
        analysis = self._get_analysis_with_retry(prompt)
        
        # Extraer secciones del análisis
        sections = self._parse_analysis_sections(analysis)
        
        return {
            'full_analysis': analysis,
            'sections': sections,
            'dataset_info': dataset_summary,
            'visualizations': self._suggest_visualizations(df)
        }
    
    def _prepare_efficient_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepara un resumen eficiente del dataset"""
        # Limitar información para no exceder tokens
        summary = {
            'shape': df.shape,
            'columns': list(df.columns)[:20],  # Max 20 columnas
            'dtypes': {col: str(dtype) for col, dtype in list(df.dtypes.items())[:10]},
            'missing': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            'sample': df.head(3).to_dict('records')  # Solo 3 filas
        }
        
        # Agregar estadísticas básicas solo para numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns[:5]  # Max 5
        if len(numeric_cols) > 0:
            summary['stats'] = {
                col: {
                    'mean': round(df[col].mean(), 2),
                    'std': round(df[col].std(), 2),
                    'min': round(df[col].min(), 2),
                    'max': round(df[col].max(), 2)
                }
                for col in numeric_cols
            }
        
        return summary
    
    def _create_unified_prompt(self, summary: Dict[str, Any]) -> str:
        """Crea un prompt único que cubra todo el análisis"""
        return f"""Analiza este dataset y proporciona un análisis completo estructurado.

DATOS:
- Forma: {summary['shape']}
- Columnas: {summary['columns']}
- Tipos: {list(summary['dtypes'].keys())}
- Faltantes: {summary['missing']}

MUESTRA (3 filas):
{summary['sample']}

{f"ESTADÍSTICAS: {summary.get('stats', {})}" if 'stats' in summary else ''}

Proporciona un análisis estructurado con EXACTAMENTE estas secciones:

### CALIDAD DE DATOS
(2-3 puntos sobre la calidad, limpieza necesaria, problemas encontrados)

### PATRONES CLAVE
(3 patrones o insights más importantes que encuentres)

### RECOMENDACIONES
(3 acciones concretas basadas en el análisis)

### RESUMEN EJECUTIVO
(1 párrafo de 2-3 líneas con lo más importante)

Sé conciso pero específico. Usa bullets donde sea apropiado."""
    
    def _get_analysis_with_retry(self, prompt: str, max_retries: int = 2) -> str:
        """Obtiene análisis con reintentos inteligentes"""
        for attempt in range(max_retries):
            try:
                # Esperar si es necesario
                self._wait_if_needed()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un analista de datos experto. Sé conciso y específico."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=800  # Limitado para no exceder
                )
                
                self.last_call_time = time.time()
                return response.choices[0].message.content
                
            except Exception as e:
                if "rate_limit" in str(e) and attempt < max_retries - 1:
                    wait_time = self._extract_wait_time(str(e))
                    print(f"⏳ Rate limit alcanzado. Esperando {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return f"Error en el análisis: {str(e)[:100]}"
        
        return "No se pudo completar el análisis debido a límites de API."
    
    def _wait_if_needed(self):
        """Espera si han pasado menos de 2 segundos desde la última llamada"""
        elapsed = time.time() - self.last_call_time
        if elapsed < 2:
            time.sleep(2 - elapsed)
    
    def _extract_wait_time(self, error_msg: str) -> float:
        """Extrae tiempo de espera del mensaje de error"""
        import re
        match = re.search(r'try again in (\d+\.?\d*)s', error_msg)
        return float(match.group(1)) + 1 if match else 15
    
    def _parse_analysis_sections(self, analysis: str) -> Dict[str, str]:
        """Parsea el análisis en secciones"""
        sections = {
            'calidad': '',
            'patrones': '',
            'recomendaciones': '',
            'resumen': ''
        }
        
        # Buscar cada sección
        import re
        
        # Patterns para cada sección
        patterns = {
            'calidad': r'### CALIDAD DE DATOS\s*(.*?)(?=###|$)',
            'patrones': r'### PATRONES CLAVE\s*(.*?)(?=###|$)',
            'recomendaciones': r'### RECOMENDACIONES\s*(.*?)(?=###|$)',
            'resumen': r'### RESUMEN EJECUTIVO\s*(.*?)(?=###|$)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, analysis, re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()
        
        return sections
    
    def _suggest_visualizations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Sugiere visualizaciones basadas en los tipos de datos"""
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Sugerencias basadas en tipos de columnas
        if numeric_cols:
            suggestions.append({
                'type': 'histogram',
                'columns': numeric_cols[:3],
                'description': 'Distribución de variables numéricas'
            })
            
            if len(numeric_cols) > 1:
                suggestions.append({
                    'type': 'correlation_matrix',
                    'columns': numeric_cols,
                    'description': 'Correlaciones entre variables numéricas'
                })
        
        if categorical_cols and numeric_cols:
            suggestions.append({
                'type': 'box_plot',
                'x': categorical_cols[0],
                'y': numeric_cols[0],
                'description': f'{numeric_cols[0]} por {categorical_cols[0]}'
            })
        
        return suggestions


# Función para usar en Streamlit
def analyze_with_smart_analyzer(df: pd.DataFrame) -> Dict[str, Any]:
    """Función wrapper para usar en Streamlit"""
    analyzer = SmartDataAnalyzer()
    return analyzer.analyze_dataset(df)
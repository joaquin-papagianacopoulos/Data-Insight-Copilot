"""
Configuración global del proyecto Data Insight Copilot
"""
import os
from pathlib import Path

# Directorios del proyecto
PROJECT_ROOT = Path(__file__).parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

# Crear directorios si no existen
PROMPTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR.mkdir(exist_ok=True)

# Configuración de Streamlit
STREAMLIT_CONFIG = {
    "page_title": "Data Insight Copilot",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Configuración de análisis de datos
DATA_CONFIG = {
    "max_file_size_mb": 200,
    "supported_formats": [".csv", ".xlsx", ".xls"],
    "outlier_methods": ["iqr", "zscore"],
    "correlation_threshold": 0.7,
    "max_categories_display": 15,
    "max_visualizations": 10
}

# Configuración de visualizaciones
VIZ_CONFIG = {
    "color_palette": "viridis",
    "figure_size": (12, 8),
    "dpi": 100,
    "style": "whitegrid"
}

# Configuración de IA (para días posteriores)
AI_CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "max_tokens": 1500,
    "temperature": 0.3,
    "timeout": 30
}

# Variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Mensajes del sistema
SYSTEM_MESSAGES = {
    "file_upload_success": "✅ Archivo cargado exitosamente",
    "file_upload_error": "❌ Error al cargar el archivo",
    "analysis_complete": "📊 Análisis completado",
    "no_data": "📁 No hay datos para analizar",
    "processing": "⏳ Procesando datos...",
}

# Límites de procesamiento
PROCESSING_LIMITS = {
    "max_rows_preview": 1000,
    "max_columns_analysis": 50,
    "max_unique_values_categorical": 100,
    "timeout_seconds": 120
}
"""
Script de demostración para Data Insight Copilot
Genera una demo automatizada mostrando todas las funcionalidades
"""

import time
import subprocess
import os
from datetime import datetime

def crear_estructura_proyecto():
    """Crea la estructura de carpetas del proyecto"""
    carpetas = ['examples', 'outputs', 'docs', 'screenshots']
    
    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok=True)
    
    print("✅ Estructura del proyecto creada")

def generar_datos_ejemplo():
    """Ejecuta el generador de datos de ejemplo"""
    print("\n📊 Generando datasets de ejemplo...")
    subprocess.run(['python', 'generate_sample_data.py'])

def iniciar_streamlit():
    """Inicia la aplicación de Streamlit"""
    print("\n🚀 Iniciando Data Insight Copilot...")
    print("La aplicación se abrirá en tu navegador en http://localhost:8501")
    print("\nPara la demo:")
    print("1. Carga uno de los archivos de la carpeta 'examples/'")
    print("2. Haz click en '🚀 Analizar con IA'")
    print("3. Explora las diferentes pestañas")
    print("4. Genera y descarga el reporte PDF")
    
    subprocess.run(['streamlit', 'run', 'app.py'])

def crear_documentacion_visual():
    """Guía para crear documentación visual"""
    print("\n📸 Guía para capturas de pantalla:")
    print("\n1. PÁGINA PRINCIPAL:")
    print("   - Captura la landing page vacía")
    print("   - Nombre: screenshots/01_landing.png")
    
    print("\n2. CARGA DE DATOS:")
    print("   - Captura el momento de arrastrar el CSV")
    print("   - Nombre: screenshots/02_upload.png")
    
    print("\n3. VISTA PREVIA:")
    print("   - Captura las métricas y preview de datos")
    print("   - Nombre: screenshots/03_preview.png")
    
    print("\n4. ANÁLISIS EN PROGRESO:")
    print("   - Captura la barra de progreso")
    print("   - Nombre: screenshots/04_analyzing.png")
    
    print("\n5. RESULTADOS:")
    print("   - Captura cada pestaña (Resumen, Insights, etc.)")
    print("   - Nombres: screenshots/05_summary.png, 06_insights.png, etc.")
    
    print("\n6. PDF GENERADO:")
    print("   - Captura el botón de descarga del PDF")
    print("   - Nombre: screenshots/09_pdf_download.png")

def crear_gif_demo():
    """Instrucciones para crear GIF de demo"""
    print("\n🎬 Para crear un GIF de demostración:")
    print("\n1. Usa una herramienta como:")
    print("   - ScreenToGif (Windows)")
    print("   - Kap (macOS)")
    print("   - Peek (Linux)")
    
    print("\n2. Graba estos pasos:")
    print("   a) Abrir la aplicación")
    print("   b) Cargar un CSV")
    print("   c) Click en analizar")
    print("   d) Mostrar resultados (rápido)")
    print("   e) Generar PDF")
    
    print("\n3. Configuración recomendada:")
    print("   - FPS: 15")
    print("   - Duración: 30-45 segundos")
    print("   - Resolución: 1280x720")
    print("   - Optimizar para < 10MB")

def main():
    """Ejecuta el script de demo completo"""
    print("🔥 DATA INSIGHT COPILOT - Script de Demo")
    print("=" * 50)
    
    # Crear estructura
    crear_estructura_proyecto()
    
    # Generar datos
    generar_datos_ejemplo()
    
    # Mostrar guías
    crear_documentacion_visual()
    crear_gif_demo()
    
    # Preguntar si iniciar la app
    print("\n" + "=" * 50)
    respuesta = input("\n¿Deseas iniciar la aplicación ahora? (s/n): ")
    
    if respuesta.lower() == 's':
        iniciar_streamlit()
    else:
        print("\nPara iniciar la aplicación más tarde, ejecuta:")
        print("streamlit run app.py")

if __name__ == "__main__":
    main()
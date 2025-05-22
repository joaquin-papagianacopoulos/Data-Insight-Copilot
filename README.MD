# 🧠 Data Insight Copilot

Un asistente inteligente que analiza archivos CSV, detecta patrones automáticamente y genera reportes con insights accionables usando IA.

## 🚀 Características

- **Análisis Automático**: Carga un CSV y obtén insights inmediatos
- **Detección de Anomalías**: Identifica outliers automáticamente  
- **Visualizaciones Inteligentes**: Gráficos generados según el tipo de datos
- **Limpieza de Datos**: Preprocessing automático
- **Insights con IA**: Resúmenes en lenguaje natural (próximamente)
- **Interfaz Intuitiva**: Streamlit responsive y fácil de usar

## 📋 Instalación

### Pasos de instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/data-insight-copilot.git
cd data-insight-copilot

# 2. Crear entorno virtual
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

## 🎯 Uso Rápido

1. **Ejecutar la app**: `streamlit run app.py`
2. **Subir CSV**: Usa el panel lateral para cargar tu archivo
3. **Explorar**: Revisa automáticamente las visualizaciones y estadísticas
4. **Analizar**: Detecta outliers y patrones automáticamente

### Datos de Ejemplo
Si no tienes un CSV, puedes usar los datos de ejemplo incluidos:
- ✅ Marca "Usar datos de ejemplo" en la barra lateral
- Explora las funcionalidades con datos sintéticos realistas

## 📊 Funcionalidades Actuales (Día 1)

### ✅ Implementado
- [x] Carga de archivos CSV
- [x] Vista previa de datos
- [x] Estadísticas descriptivas automáticas
- [x] Detección de valores faltantes
- [x] Visualizaciones automáticas (histogramas, correlaciones, boxplots)
- [x] Detección de outliers (IQR)
- [x] Información de columnas y tipos de datos
- [x] Datasets de ejemplo para testing
- [x] Interfaz responsive con Streamlit

### 🔄 En Desarrollo (Próximos días)
- [ ] Agente IA con LangChain/LangGraph (Día 2)
- [ ] Generación de insights con GPT (Día 2-3)
- [ ] Reportes automatizados en PDF (Día 4)
- [ ] Análisis predictivo básico (Día 4)
- [ ] Recomendaciones de acciones (Día 5)

## 🛠️ Arquitectura Técnica

```
data-insight-copilot/
├── app.py                  # Streamlit app principal
├── data_utils.py          # Utilidades de análisis de datos
├── agent_graph.py         # Flujo de agentes IA (próximamente)
├── prompts/               # Templates de prompts para IA
│   └── analyst_prompt.txt
├── outputs/               # Reportes generados
├── examples/              # Datasets de ejemplo
├── requirements.txt       # Dependencias Python
└── README.md             # Esta documentación
```

## 🔧 Tecnologías

- **Frontend**: Streamlit
- **Análisis**: Pandas, NumPy
- **Visualización**: Plotly, Matplotlib, Seaborn  
- **IA**: LangChain + LangGraph (próximamente)
- **LLM**: OpenAI GPT (próximamente)

## 📝 Ejemplos de Uso

### Cargar y Analizar Ventas
```python
# Los datos se cargan automáticamente desde la interfaz
# El análisis incluye:
# - Estadísticas descriptivas
# - Detección de outliers
# - Correlaciones entre variables
# - Visualizaciones automáticas
```

### Insights Automáticos Generados
- Calidad de datos (% valores faltantes)
- Distribución de variables numéricas
- Correlaciones fuertes entre variables  
- Detección de anomalías
- Recomendaciones de limpieza

## 🚀 Roadmap de Desarrollo

### Día 1 ✅ (Completado)
- Setup del proyecto
- Interfaz básica de Streamlit
- Carga y análisis de CSV
- Visualizaciones automáticas

### Día 2 🔄 (En progreso)
- Integración de LangChain
- Agente IA para análisis
- Generación de insights con GPT

### Día 3
- Detección avanzada de patrones
- Visualizaciones contextualizadas
- Insights más sofisticados

### Día 4
- Generación de reportes PDF
- Análisis predictivo básico
- Refinamiento de prompts

### Día 5
- Deploy y documentación final
- Video demo
- Optimizaciones finales

## 🤝 Contribuciones

Este es un proyecto en desarrollo activo. Ideas y sugerencias son bienvenidas.

## 📄 Licencia

MIT License - Ver archivo LICENSE para detalles.

---

**Desarrollado con ❤️ para demostrar capacidades de IA aplicada a análisis de datos** Prerrequisitos
- Python 3.8+
- pip

###
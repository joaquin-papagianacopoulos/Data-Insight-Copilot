from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
import pandas as pd
from datetime import datetime
import io
import base64
from typing import Dict, Any, List
import plotly.graph_objects as go

class PDFReportGenerator:
    """Generador de reportes PDF profesionales"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Crea estilos personalizados para el reporte"""
        # Título principal
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        # Subtítulos
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#3949ab'),
            spaceAfter=12
        ))
        
        # Texto normal
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
        
        # Insights destacados
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['BodyText'],
            fontSize=11,
            leftIndent=20,
            rightIndent=20,
            textColor=colors.HexColor('#1565c0'),
            backColor=colors.HexColor('#e3f2fd'),
            borderColor=colors.HexColor('#1976d2'),
            borderWidth=1,
            borderPadding=10,
            spaceAfter=12
        ))
        
    def generate_report(self, 
                       df: pd.DataFrame, 
                       analysis_results: Dict[str, Any],
                       visualizations: Dict[str, go.Figure],
                       filename: str = "data_insight_report.pdf") -> bytes:
        """Genera un reporte PDF completo"""
        
        # Crear buffer en memoria
        buffer = io.BytesIO()
        
        # Crear documento
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Contenido del reporte
        story = []
        
        # 1. Portada
        story.extend(self._create_cover_page(df, analysis_results))
        story.append(PageBreak())
        
        # 2. Resumen Ejecutivo
        story.extend(self._create_executive_summary(analysis_results))
        story.append(PageBreak())
        
        # 3. Análisis de Calidad de Datos
        story.extend(self._create_data_quality_section(df, analysis_results))
        story.append(PageBreak())
        
        # 4. Insights y Patrones
        story.extend(self._create_insights_section(analysis_results))
        
        # 5. Visualizaciones (si están disponibles)
        if visualizations:
            story.append(PageBreak())
            story.extend(self._create_visualizations_section(visualizations))
        
        # 6. Recomendaciones
        story.append(PageBreak())
        story.extend(self._create_recommendations_section(analysis_results))
        
        # 7. Apéndice técnico
        story.append(PageBreak())
        story.extend(self._create_technical_appendix(df))
        
        # Generar PDF
        doc.build(story)
        
        # Obtener bytes del PDF
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _create_cover_page(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List:
        """Crea la página de portada"""
        elements = []
        
        # Título
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("DATA INSIGHT REPORT", self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Línea decorativa
        elements.append(Table([['']], colWidths=[6*inch], rowHeights=[2],
                            style=TableStyle([
                                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#3949ab')),
                            ])))
        
        elements.append(Spacer(1, 0.5*inch))
        
        # Información del dataset
        info_data = [
            ['Dataset:', analysis_results.get('dataset_name', 'Análisis de Datos')],
            ['Fecha:', datetime.now().strftime('%d de %B de %Y')],
            ['Registros:', f"{len(df):,}"],
            ['Variables:', f"{len(df.columns)}"],
            ['Generado por:', 'Data Insight Copilot']
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(info_table)
        
        return elements
    
    def _create_executive_summary(self, analysis_results: Dict[str, Any]) -> List:
        """Crea el resumen ejecutivo"""
        elements = []
        
        elements.append(Paragraph("Resumen Ejecutivo", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Extraer resumen del análisis
        sections = analysis_results.get('sections', {})
        summary = sections.get('resumen', 'No se generó resumen ejecutivo.')
        
        elements.append(Paragraph(summary, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Métricas clave
        elements.append(Paragraph("Métricas Clave", self.styles['Heading2']))
        
        dataset_info = analysis_results.get('dataset_info', {})
        metrics_data = []
        
        if 'shape' in dataset_info:
            metrics_data.append(['Dimensiones del Dataset:', f"{dataset_info['shape'][0]} filas × {dataset_info['shape'][1]} columnas"])
        
        if 'missing' in dataset_info:
            total_missing = sum(dataset_info['missing'].values())
            total_cells = dataset_info['shape'][0] * dataset_info['shape'][1]
            missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
            metrics_data.append(['Datos Faltantes:', f"{missing_pct:.1f}% del total"])
        
        if 'stats' in dataset_info:
            metrics_data.append(['Variables Numéricas:', f"{len(dataset_info['stats'])} columnas analizadas"])
        
        if metrics_data:
            metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(metrics_table)
        
        return elements
    
    def _create_data_quality_section(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List:
        """Crea la sección de calidad de datos"""
        elements = []
        
        elements.append(Paragraph("Análisis de Calidad de Datos", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Calidad según el análisis
        sections = analysis_results.get('sections', {})
        quality_text = sections.get('calidad', 'No se encontraron problemas de calidad significativos.')
        
        elements.append(Paragraph(quality_text, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Tabla de calidad por columna
        elements.append(Paragraph("Detalle por Variable", self.styles['Heading2']))
        
        quality_data = [['Variable', 'Tipo', 'Valores Únicos', 'Datos Faltantes', 'Estado']]
        
        for col in df.columns[:15]:  # Limitar a 15 columnas para el reporte
            dtype = str(df[col].dtype)
            unique_vals = df[col].nunique()
            missing = df[col].isnull().sum()
            missing_pct = (missing / len(df)) * 100
            
            # Determinar estado
            if missing_pct > 50:
                status = '⚠️ Crítico'
                status_color = colors.red
            elif missing_pct > 20:
                status = '⚠️ Revisar'
                status_color = colors.orange
            else:
                status = '✓ OK'
                status_color = colors.green
            
            quality_data.append([
                col[:30],  # Truncar nombres largos
                dtype[:10],
                str(unique_vals),
                f"{missing} ({missing_pct:.1f}%)",
                status
            ])
        
        quality_table = Table(quality_data, colWidths=[2*inch, 1*inch, 1*inch, 1.5*inch, 0.8*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        
        elements.append(quality_table)
        
        return elements
    
    def _create_insights_section(self, analysis_results: Dict[str, Any]) -> List:
        """Crea la sección de insights y patrones"""
        elements = []
        
        elements.append(Paragraph("Insights y Patrones Descubiertos", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        sections = analysis_results.get('sections', {})
        patterns = sections.get('patrones', 'No se identificaron patrones específicos.')
        
        # Dividir los patrones en puntos si es posible
        if '•' in patterns or '-' in patterns or '1.' in patterns:
            elements.append(Paragraph(patterns, self.styles['CustomBody']))
        else:
            # Si no hay formato, crear un highlight box
            elements.append(Paragraph(patterns, self.styles['Insight']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Agregar información adicional si está disponible
        if 'visualizations' in analysis_results:
            elements.append(Paragraph("Análisis Visual", self.styles['Heading2']))
            elements.append(Paragraph(
                "Se han generado visualizaciones detalladas que muestran distribuciones, "
                "correlaciones y tendencias en los datos. Consulte las gráficas adjuntas "
                "para un análisis visual completo.",
                self.styles['CustomBody']
            ))
        
        return elements
    
    def _create_visualizations_section(self, visualizations: Dict[str, go.Figure]) -> List:
        """Crea la sección de visualizaciones"""
        elements = []
        
        elements.append(Paragraph("Visualizaciones y Gráficos", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Nota sobre las visualizaciones
        elements.append(Paragraph(
            "Las siguientes visualizaciones fueron generadas automáticamente para "
            "proporcionar una comprensión visual de los patrones y distribuciones en los datos.",
            self.styles['CustomBody']
        ))
        elements.append(Spacer(1, 0.3*inch))
        
        # Convertir cada figura a imagen y agregar
        for viz_name, fig in list(visualizations.items())[:5]:  # Limitar a 5 visualizaciones
            if fig is not None:
                try:
                    # Convertir plotly figure a imagen
                    img_bytes = fig.to_image(format="png", width=600, height=400)
                    img_buffer = io.BytesIO(img_bytes)
                    
                    # Agregar imagen al PDF
                    img = Image(img_buffer, width=5*inch, height=3.33*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
                    
                    # Agregar descripción
                    desc = self._get_visualization_description(viz_name)
                    elements.append(Paragraph(desc, self.styles['Caption']))
                    elements.append(Spacer(1, 0.5*inch))
                    
                except:
                    # Si falla la conversión, agregar nota
                    elements.append(Paragraph(
                        f"[Visualización '{viz_name}' disponible en la versión interactiva]",
                        self.styles['CustomBody']
                    ))
        
        return elements
    
    def _create_recommendations_section(self, analysis_results: Dict[str, Any]) -> List:
        """Crea la sección de recomendaciones"""
        elements = []
        
        elements.append(Paragraph("Recomendaciones y Próximos Pasos", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        sections = analysis_results.get('sections', {})
        recommendations = sections.get('recomendaciones', 'No se generaron recomendaciones específicas.')
        
        elements.append(Paragraph(recommendations, self.styles['CustomBody']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Agregar recomendaciones genéricas basadas en el análisis
        elements.append(Paragraph("Acciones Sugeridas", self.styles['Heading2']))
        
        action_items = [
            "1. Revisar y tratar los valores faltantes identificados en las variables críticas.",
            "2. Investigar más a fondo los patrones y anomalías detectadas.",
            "3. Considerar la creación de nuevas variables derivadas basadas en los insights.",
            "4. Validar los hallazgos con expertos del dominio del negocio.",
            "5. Implementar un proceso de monitoreo continuo de la calidad de datos."
        ]
        
        for item in action_items:
            elements.append(Paragraph(item, self.styles['CustomBody']))
        
        return elements
    
    def _create_technical_appendix(self, df: pd.DataFrame) -> List:
        """Crea el apéndice técnico"""
        elements = []
        
        elements.append(Paragraph("Apéndice Técnico", self.styles['CustomHeading']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Información técnica del dataset
        tech_info = [
            ['Propiedad', 'Valor'],
            ['Formato de archivo', 'CSV'],
            ['Codificación estimada', 'UTF-8'],
            ['Tamaño en memoria', f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"],
            ['Índice', f"{df.index.name or 'RangeIndex'}"],
            ['Columnas numéricas', str(len(df.select_dtypes(include=['number']).columns))],
            ['Columnas categóricas', str(len(df.select_dtypes(include=['object']).columns))],
            ['Fecha de análisis', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        tech_table = Table(tech_info, colWidths=[3*inch, 3*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(tech_table)
        
        return elements
    
    def _get_visualization_description(self, viz_name: str) -> str:
        """Obtiene una descripción para cada tipo de visualización"""
        descriptions = {
            'overview': "Vista general del dataset mostrando tipos de datos y completitud.",
            'distributions': "Distribuciones de las variables numéricas principales.",
            'correlation': "Matriz de correlación entre variables numéricas.",
            'categorical': "Análisis de frecuencias para variables categóricas.",
            'bivariate': "Análisis bivariado mostrando relaciones entre variables.",
            'outliers': "Detección de valores atípicos usando el método IQR.",
            'timeseries': "Análisis temporal de las variables principales."
        }
        return descriptions.get(viz_name, f"Visualización: {viz_name}")
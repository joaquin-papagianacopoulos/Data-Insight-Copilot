import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sales_data():
    """Genera un dataset de ventas de ejemplo"""
    np.random.seed(42)
    
    # Configuración
    n_records = 1000
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Generar fechas
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = np.random.choice(date_range, n_records)
    
    # Categorías
    regions = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    categories = ['Electrónica', 'Ropa', 'Alimentos', 'Hogar', 'Deportes']
    customer_types = ['Premium', 'Standard', 'Básico']
    payment_methods = ['Tarjeta', 'Efectivo', 'Transferencia', 'Débito']
    
    # Generar datos
    data = {
        'fecha': dates,
        'region': np.random.choice(regions, n_records, p=[0.25, 0.20, 0.20, 0.20, 0.15]),
        'categoria': np.random.choice(categories, n_records),
        'tipo_cliente': np.random.choice(customer_types, n_records, p=[0.2, 0.5, 0.3]),
        'metodo_pago': np.random.choice(payment_methods, n_records),
        'cantidad': np.random.poisson(3, n_records) + 1,
        'precio_unitario': np.random.gamma(2, 50, n_records),
        'descuento_pct': np.random.choice([0, 5, 10, 15, 20], n_records, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # Calcular campos derivados
    df['precio_total'] = df['cantidad'] * df['precio_unitario']
    df['descuento'] = df['precio_total'] * df['descuento_pct'] / 100
    df['venta_neta'] = df['precio_total'] - df['descuento']
    
    # Agregar algunos patrones
    # Ventas más altas en diciembre
    december_mask = df['fecha'].dt.month == 12
    df.loc[december_mask, 'venta_neta'] *= 1.5
    
    # Premium compra más
    premium_mask = df['tipo_cliente'] == 'Premium'
    df.loc[premium_mask, 'venta_neta'] *= 1.3
    
    # Agregar algunos valores faltantes realistas
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'descuento_pct'] = np.nan
    
    # Agregar algunos outliers
    outlier_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    df.loc[outlier_indices, 'venta_neta'] *= np.random.uniform(3, 5, size=len(outlier_indices))
    
    # Ordenar por fecha
    df = df.sort_values('fecha').reset_index(drop=True)
    
    return df

def generate_hr_data():
    """Genera un dataset de recursos humanos de ejemplo"""
    np.random.seed(42)
    
    n_employees = 500
    
    # Departamentos y cargos
    departments = ['Ventas', 'IT', 'RRHH', 'Marketing', 'Operaciones', 'Finanzas']
    positions = ['Junior', 'Semi-Senior', 'Senior', 'Lead', 'Manager']
    
    # Generar datos básicos
    data = {
        'empleado_id': [f'EMP{i:04d}' for i in range(1, n_employees + 1)],
        'edad': np.random.normal(35, 8, n_employees).astype(int).clip(22, 65),
        'departamento': np.random.choice(departments, n_employees),
        'cargo': np.random.choice(positions, n_employees, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
        'años_empresa': np.random.exponential(5, n_employees).clip(0, 25).astype(int),
        'educacion': np.random.choice(['Secundaria', 'Universitario', 'Posgrado'], n_employees, p=[0.2, 0.6, 0.2]),
        'genero': np.random.choice(['M', 'F'], n_employees, p=[0.52, 0.48]),
    }
    
    df = pd.DataFrame(data)
    
    # Salario basado en cargo y experiencia
    base_salary = {
        'Junior': 30000,
        'Semi-Senior': 45000,
        'Senior': 65000,
        'Lead': 85000,
        'Manager': 100000
    }
    
    df['salario_base'] = df['cargo'].map(base_salary)
    df['salario'] = df['salario_base'] * (1 + df['años_empresa'] * 0.03) * np.random.uniform(0.9, 1.1, n_employees)
    
    # Performance score
    df['performance_score'] = np.random.beta(8, 2, n_employees) * 100
    
    # Satisfacción laboral (correlacionada con salario y performance)
    satisfaction_base = (df['salario'] / df['salario'].max() * 50 + 
                        df['performance_score'] / 2 + 
                        np.random.normal(0, 10, n_employees))
    df['satisfaccion'] = satisfaction_base.clip(0, 100)
    
    # Días de ausencia (inversamente correlacionado con satisfacción)
    df['dias_ausencia'] = np.random.poisson(20 - df['satisfaccion'] / 5).clip(0, 30)
    
    # Algunos valores faltantes
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'satisfaccion'] = np.nan
    
    df = df.drop('salario_base', axis=1)
    
    return df

def generate_iot_sensor_data():
    """Genera un dataset de sensores IoT de ejemplo"""
    np.random.seed(42)
    
    # Configuración
    n_sensors = 10
    hours = 24 * 7  # Una semana de datos
    
    # Generar timestamps
    start_time = datetime.now() - timedelta(hours=hours)
    timestamps = [start_time + timedelta(hours=h) for h in range(hours)]
    
    # Crear datos para cada sensor
    all_data = []
    
    for sensor_id in range(1, n_sensors + 1):
        sensor_type = np.random.choice(['temperatura', 'humedad', 'presion'])
        location = np.random.choice(['Planta A', 'Planta B', 'Almacén', 'Oficina'])
        
        for timestamp in timestamps:
            # Generar valores con patrones diarios
            hour = timestamp.hour
            
            if sensor_type == 'temperatura':
                # Temperatura con ciclo diario
                base_temp = 20 + 5 * np.sin((hour - 6) * np.pi / 12)
                value = base_temp + np.random.normal(0, 1)
            elif sensor_type == 'humedad':
                # Humedad inversa a temperatura
                base_humidity = 60 - 10 * np.sin((hour - 6) * np.pi / 12)
                value = base_humidity + np.random.normal(0, 3)
            else:  # presion
                value = 1013 + np.random.normal(0, 5)
            
            # Agregar anomalías ocasionales
            if np.random.random() < 0.01:
                value *= np.random.choice([0.5, 1.5])
            
            all_data.append({
                'timestamp': timestamp,
                'sensor_id': f'SENSOR_{sensor_id:03d}',
                'tipo_sensor': sensor_type,
                'ubicacion': location,
                'valor': value,
                'unidad': {'temperatura': '°C', 'humedad': '%', 'presion': 'hPa'}[sensor_type]
            })
    
    df = pd.DataFrame(all_data)
    
    # Agregar algunos valores faltantes (sensores desconectados)
    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'valor'] = np.nan
    
    return df

# Generar todos los datasets
if __name__ == "__main__":
    print("Generando datasets de ejemplo...")
    
    # Dataset 1: Ventas
    sales_df = generate_sales_data()
    sales_df.to_csv('examples/ventas_ejemplo.csv', index=False)
    print(f"✅ ventas_ejemplo.csv - {len(sales_df)} registros")
    
    # Dataset 2: RRHH
    hr_df = generate_hr_data()
    hr_df.to_csv('examples/rrhh_ejemplo.csv', index=False)
    print(f"✅ rrhh_ejemplo.csv - {len(hr_df)} registros")
    
    # Dataset 3: IoT
    iot_df = generate_iot_sensor_data()
    iot_df.to_csv('examples/iot_sensores_ejemplo.csv', index=False)
    print(f"✅ iot_sensores_ejemplo.csv - {len(iot_df)} registros")
    
    print("\n📊 Datasets de ejemplo generados en la carpeta 'examples/'")
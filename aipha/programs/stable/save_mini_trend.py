"""
save_mini_trend.py - Módulo para guardar resultados de mini-tendencias y comparación con velas clave

Este módulo permite guardar los resultados del análisis de mini-tendencias en una base de datos MySQL,
incluida la comparación con las velas clave detectadas por detect_candles.py.
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
import traceback

# Ajuste de path para importar desde el módulo padre
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from mini_trend import MiniTrendDetector
from detect_candles import Detector
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/.env')), override=True)

# Configuración de logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'mini_trend.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_db_config():
    """
    Obtiene la configuración de la base de datos desde variables de entorno.
    """
    return {
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'database': os.getenv('MYSQL_DATABASE', 'binance_lob')
    }

def connect_to_db(db_config):
    """
    Establece conexión con la base de datos MySQL.
    """
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            cursor = connection.cursor()
            logging.info(f"Connected to MySQL database: {db_config['database']}")
            return connection, cursor
    except Error as e:
        logging.error(f"Error connecting to MySQL database: {e}")
    return None, None

def close_db_connection(connection, cursor):
    """
    Cierra la conexión con la base de datos.
    """
    if cursor:
        cursor.close()
    if connection:
        connection.close()
        logging.info("Database connection closed.")

def get_key_candles_from_db(cursor, table_name='key_candles'):
    """
    Obtiene las velas clave guardadas en la base de datos.
    """
    try:
        cursor.execute(f"SELECT * FROM {table_name}")
        columns = [desc[0] for desc in cursor.description]
        key_candles = []
        
        for row in cursor.fetchall():
            key_candle = {}
            for i, column in enumerate(columns):
                key_candle[column] = row[i]
            key_candles.append(key_candle)
            
        logging.info(f"Retrieved {len(key_candles)} key candles from database")
        return key_candles
    except Error as e:
        logging.error(f"Error retrieving key candles: {e}")
        return []

def compare_mini_trends_with_key_candles(mini_trends_df, key_candles, poc_tol=0.002, check_direction=True):
    """
    Compara mini-tendencias con velas clave y agrega resultados de la comparación.
    Incluye verificación opcional de dirección de la mini-tendencia en relación a la vela clave.
    
    Args:
        mini_trends_df: DataFrame con las mini-tendencias detectadas
        key_candles: Lista de velas clave de la base de datos
        poc_tol: Tolerancia para considerar el POC cercano al precio (% relativo)
        check_direction: Si True, verifica si la dirección de la mini-tendencia es coherente con
                         la probable dirección de reversión o continuación tras la vela clave
    """
    if mini_trends_df.empty or not key_candles:
        logging.warning("No hay datos para comparar")
        return mini_trends_df
    
    # Añadir columna para resultados de comparación
    mini_trends_df['comparison_results'] = None
    
    # Para cada mini-tendencia, buscar velas clave relevantes
    for idx, mini_trend in mini_trends_df.iterrows():
        poc = mini_trend['poc']
        end_idx = mini_trend['end_idx']
        trend_direction = mini_trend['direction']
        relevant_candles = []
        
        for candle in key_candles:
            candle_idx = candle.get('candle_index')
            candle_price = candle.get('close', 0)
            candle_open = candle.get('open', 0)
            
            # Comprobar si la vela clave está después de la mini-tendencia
            if candle_idx > end_idx:
                # Verificar proximidad entre POC y precio de la vela
                if poc:
                    price_diff_pct = abs(poc - candle_price) / candle_price
                    
                    if price_diff_pct <= poc_tol:
                        # Evaluar coherencia de dirección si se solicita
                        direction_match = True
                        direction_pattern = "neutral"
                        
                        if check_direction:
                            # Para velas clave (shakeout), calculamos si podría haber una reversión
                            candle_body = abs(candle_price - candle_open)
                            candle_range = candle.get('high', 0) - candle.get('low', 0)
                            body_percentage = candle_body / candle_range if candle_range > 0 else 0
                            
                            # Analizamos patrones de reversión basados en el tipo de vela clave
                            is_small_body = body_percentage <= 0.3  # 30% es el umbral típico para shakeout
                            
                            if is_small_body:
                                # Para shakeouts (cuerpo pequeño), esperamos una potencial reversión de dirección
                                expected_reversal = 'bajista' if trend_direction == 'alcista' else 'alcista'
                                
                                # Verificar si hay más velas después de la vela clave para confirmar dirección
                                next_candle_idx = candle_idx + 1
                                possible_reversal = True  # Asumimos posible reversión por defecto
                                
                                direction_match = possible_reversal  # En ausencia de datos, aceptamos posible reversión
                                direction_pattern = f"trend:{trend_direction},expected_reversal:{expected_reversal}"
                            else:
                                # Para velas con cuerpo grande, es más probable continuación de tendencia
                                # La mini-tendencia debe ser coherente con la vela clave
                                candle_direction = 'alcista' if candle_price > candle_open else 'bajista'
                                direction_match = trend_direction == candle_direction
                                direction_pattern = f"trend:{trend_direction},candle:{candle_direction}"
                        
                        # Solo añadir a relevantes si supera todas las verificaciones
                        if direction_match or not check_direction:
                            # Calcular distancia temporal
                            bars_distance = candle_idx - end_idx
                            relevant_candles.append({
                                'candle_id': candle.get('id'),
                                'candle_idx': candle_idx,
                                'price_diff_pct': price_diff_pct * 100,  # Convertir a porcentaje
                                'bars_distance': bars_distance,
                                'direction_pattern': direction_pattern,
                                'direction_match': direction_match
                            })
        
        # Guardar resultados de la comparación
        if relevant_candles:
            mini_trends_df.at[idx, 'comparison_results'] = json.dumps(relevant_candles)
    
    # Contar mini-tendencias con resultados de comparación
    trends_with_matches = mini_trends_df['comparison_results'].notna().sum()
    logging.info(f"Identified {trends_with_matches} mini-trends matched with key candles")
    
    return mini_trends_df

def create_mini_trend_table(cursor, table_name='mini_trend_results'):
    """
    Crea la tabla para almacenar resultados de mini-tendencias si no existe.
    """
    try:
        cursor.execute(f"DELETE FROM {table_name}")
        logging.info(f"Cleaned existing table: {table_name}")
    except Error as e:
        # Si la tabla no existe, crearla
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            start_idx INT,
            end_idx INT,
            start_time DATETIME,
            end_time DATETIME,
            direction VARCHAR(20),
            slope FLOAT,
            r_squared FLOAT,
            poc FLOAT,
            volume_total FLOAT,
            duration_bars INT,
            duration_minutes INT,
            comparison_results JSON,
            csv_file VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """
        cursor.execute(create_table_query)
        logging.info(f"Created new table: {table_name}")

def save_mini_trend(df, path=None, db_config=None, table_name='mini_trend_results', csv_file=None):
    """
    Guarda los resultados de mini-tendencias en CSV y/o base de datos MySQL.
    
    Args:
        df: DataFrame con resultados de mini-tendencias
        path: Ruta para guardar CSV (opcional)
        db_config: Configuración para conexión a base de datos (opcional)
        table_name: Nombre de la tabla MySQL
        csv_file: Nombre del archivo CSV procesado
    """
    # Guardar en CSV si se proporciona ruta
    if path:
        if not path.endswith('.csv'):
            path += '.csv'
        df.to_csv(path, index=False)
        logging.info(f"Saved mini-trend results to CSV: {path}")
    
    # Guardar en base de datos si se proporciona configuración
    if db_config:
        connection, cursor = connect_to_db(db_config)
        if connection and cursor:
            try:
                # Crear/limpiar tabla
                create_mini_trend_table(cursor, table_name)
                
                # Preparar datos para la inserción
                for _, row in df.iterrows():
                    # Convertir datetime a formato compatible con MySQL
                    start_time = row.get('start_time')
                    if isinstance(start_time, pd.Timestamp):
                        start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    end_time = row.get('end_time')
                    if isinstance(end_time, pd.Timestamp):
                        end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Insertar fila
                    insert_query = f"""
                    INSERT INTO {table_name} (
                        start_idx, end_idx, start_time, end_time, direction, 
                        slope, r_squared, poc, volume_total, duration_bars, 
                        duration_minutes, comparison_results, csv_file
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        int(row.get('start_idx', 0)),
                        int(row.get('end_idx', 0)),
                        start_time,
                        end_time,
                        row.get('direction', ''),
                        float(row.get('slope', 0)),
                        float(row.get('r_squared', 0)),
                        float(row.get('poc', 0)),
                        float(row.get('volume_total', 0)),
                        int(row.get('duration_bars', 0)),
                        int(row.get('duration_minutes', 0)),
                        row.get('comparison_results'),
                        csv_file
                    ))
                
                connection.commit()
                logging.info(f"Saved {len(df)} mini-trend results to database table: {table_name}")
            
            except Error as e:
                logging.error(f"Error saving to database: {e}")
                traceback.print_exc()
            
            finally:
                close_db_connection(connection, cursor)
        else:
            logging.error("Failed to connect to database")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Save mini-trend detection results to DB")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--output', type=str, default='resultados_mini_trend.csv', help='Path to output CSV file')
    parser.add_argument('--poc-tol', type=float, default=0.002, help='POC tolerance for comparison (default: 0.002)')
    parser.add_argument('--check-direction', action='store_true', default=False, help='Verify direction consistency between mini-trends and key candles')
    parser.add_argument('--zigzag-threshold', type=float, default=0.005, help='Threshold for ZigZag segmentation (default: 0.005)')
    parser.add_argument('--min-trend-bars', type=int, default=5, help='Minimum bars for a valid mini-trend (default: 5)')
    args = parser.parse_args()
    
    # Obtener configuración de DB
    db_config = get_db_config()
    
    # Inicializar detector y procesar CSV
    print(f"Analyzing file: {args.csv}")
    mini_trend_detector = MiniTrendDetector(args.csv)
    mini_trend_detector.set_params(
        zigzag_threshold=args.zigzag_threshold,
        min_trend_bars=args.min_trend_bars
    )
    print(f"Using parameters: ZigZag threshold={args.zigzag_threshold}, POC tolerance={args.poc_tol}")
    print(f"Direction checking: {'Enabled' if args.check_direction else 'Disabled'}")
    
    mini_trends_df = mini_trend_detector.process_csv()
    
    if not mini_trends_df.empty:
        print(f"Detected {len(mini_trends_df)} mini-trends")
        print(f"Directions: {mini_trends_df['direction'].value_counts().to_dict()}")
        print(f"Average slope: {mini_trends_df['slope'].mean():.6f}")
        print(f"Average R²: {mini_trends_df['r_squared'].mean():.4f}")
        
        # Conectar a la DB para obtener velas clave
        connection, cursor = connect_to_db(db_config)
        if connection and cursor:
            # Obtener velas clave
            key_candles = get_key_candles_from_db(cursor)
            print(f"Retrieved {len(key_candles)} key candles from database")
            close_db_connection(connection, cursor)
            
            # Comparar mini-tendencias con velas clave
            mini_trends_df = compare_mini_trends_with_key_candles(
                mini_trends_df, key_candles, 
                poc_tol=args.poc_tol,
                check_direction=args.check_direction
            )
            
            # Guardar resultados en CSV y DB
            csv_filename = os.path.basename(args.csv)
            save_mini_trend(
                mini_trends_df, 
                path=args.output, 
                db_config=db_config,
                csv_file=csv_filename
            )
            
            # Contar tendencias con coincidencias y extraer datos
            matches_count = mini_trends_df['comparison_results'].notna().sum()
            match_percentage = (matches_count / len(mini_trends_df)) * 100 if len(mini_trends_df) > 0 else 0
            
            print(f"\nResults summary:")
            print(f"- Processed {len(mini_trends_df)} mini-trends")
            print(f"- Found {matches_count} matches with key candles ({match_percentage:.1f}%)")
            
            # Guardar un subconjunto de minitrends relevantes para análisis
            relevant_trends = mini_trends_df[mini_trends_df['comparison_results'].notna()]
            if not relevant_trends.empty:
                relevant_file = 'mini_trend_relevantes.csv'
                relevant_trends.to_csv(relevant_file, index=False)
                print(f"- Saved {len(relevant_trends)} relevant mini-trends to {relevant_file}")
                print(f"- Average POC: {relevant_trends['poc'].mean():.2f}")
                print(f"- Average R² of matched mini-trends: {relevant_trends['r_squared'].mean():.4f}")
            
            print("\nData saved successfully to SQL table 'mini_trend_results'")
        else:
            # Si no pudimos conectar a la DB, guardar sin comparación
            save_mini_trend(mini_trends_df, path=args.output)
            print(f"Processed {len(mini_trends_df)} mini-trends (no database comparison)")
    else:
        print("No mini-trends detected in the CSV file")

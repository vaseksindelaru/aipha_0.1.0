"""
save_detect_accumulation_zone.py - Módulo para guardar resultados de la detección de zonas de acumulación en base de datos

Este módulo permite guardar los resultados de la detección de zonas de acumulación en una base de datos MySQL para su análisis posterior.
Ubicación: aipha/programs/stable/save_detect_accumulation_zone.py
"""

import os
import sys
import pandas as pd
import mysql.connector
import argparse
import json
import logging
import traceback
from dotenv import load_dotenv
from datetime import datetime
from detect_accumulation_zone import AccumulationZoneDetector
from mini_trend import MiniTrendDetector

# Configuración de logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'save_accumulation_zone.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/.env')), override=True)

# Ajuste de path para importar desde el módulo padre
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class AccumulationZoneResultSaver:
    """
    Clase para guardar resultados de la detección de zonas de acumulación en la base de datos binance_lob.
    Maneja la conexión y métodos para almacenar datos de zonas y parámetros de detección.
    """
    def __init__(self, host=None, user=None, password=None, database=None):
        self.db_config = {
            'host': host or os.getenv('MYSQL_HOST', 'localhost'),
            'user': user or os.getenv('MYSQL_USER', 'root'),
            'password': password or os.getenv('MYSQL_PASSWORD', ''),
            'database': database or os.getenv('MYSQL_DATABASE', 'binance_lob')
        }
        print(f"Database configuration: host={self.db_config['host']}, user={self.db_config['user']}, database={self.db_config['database']}")
        self.connection = None
        self.cursor = None
        self.csv_file = None
        self.num_candles = None
        self.mini_trend_detector = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                print(f"Connected to MySQL database: {self.db_config['database']}")
                return True
        except mysql.connector.Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return False

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def update_key_candles_in_zones(self, zones):
        """
        Actualiza la tabla key_candles para marcar las velas que están dentro de las zonas de acumulación.
        """
        if not self.connection or not zones:
            return
        
        print("Updating key_candles with in_accumulation_zone information...")
        
        # Asegurarse que la columna existe en la tabla
        try:
            # Primero verificamos si la columna existe
            check_column_query = """SHOW COLUMNS FROM key_candles LIKE 'in_accumulation_zone';"""
            self.cursor.execute(check_column_query)
            column_exists = self.cursor.fetchone()
            
            if not column_exists:
                # Si no existe, la añadimos
                add_column_query = """ALTER TABLE key_candles ADD COLUMN in_accumulation_zone BOOLEAN DEFAULT FALSE;"""
                self.cursor.execute(add_column_query)
                self.connection.commit()
                print("Added in_accumulation_zone column to key_candles table")
                
            # Verificamos si la columna mini_trend_id existe
            check_column_query = """SHOW COLUMNS FROM key_candles LIKE 'mini_trend_id';"""
            self.cursor.execute(check_column_query)
            column_exists = self.cursor.fetchone()
            
            if not column_exists:
                # Si no existe, la añadimos
                add_column_query = """ALTER TABLE key_candles ADD COLUMN mini_trend_id INT NULL;"""
                self.cursor.execute(add_column_query)
                self.connection.commit()
                print("Added mini_trend_id column to key_candles table")
        except Exception as e:
            print(f"Error checking/adding column: {e}")
            return
        
        # Actualizar la columna in_accumulation_zone para cada zona de acumulación
        for zone in zones:
            start_idx = zone.get('start_idx')
            end_idx = zone.get('end_idx')
            
            if start_idx is not None and end_idx is not None:
                update_query = """
                UPDATE key_candles
                SET in_accumulation_zone = TRUE
                WHERE index_in_csv >= %s AND index_in_csv <= %s
                AND csv_file = %s;
                """
                self.cursor.execute(update_query, (start_idx, end_idx, self.csv_file))
                self.connection.commit()
            print(f"Actualizadas velas clave en zonas de acumulación")
            return True

    def analyze_mini_trends(self, data_df, zones):
        """
        Analiza mini-tendencias en relación con las zonas de acumulación detectadas.
        Añade información de mini-tendencias a las zonas de acumulación y guarda las mini-tendencias relevantes.
        """
        if self.mini_trend_detector is None:
            self.mini_trend_detector = MiniTrendDetector()
            if self.csv_file:
                self.mini_trend_detector.load_csv(self.csv_file)
            else:
                # Si no tenemos archivo CSV, usamos el DataFrame proporcionado
                self.mini_trend_detector.data = data_df
            
            # Configurar parámetros de mini-tendencias
            self.mini_trend_detector.set_params(
                zigzag_threshold=0.003,  # Menor umbral para captar más mini-tendencias
                lookback_window=200,
                min_trend_bars=4,  # Reducir el mínimo de barras para tendencias más cortas
                volume_profile_bins=100  # Mayor resolución en el volume profile
            )
        
        # Detectar mini-tendencias
        self.mini_trend_detector.segment_mini_trends()
        mini_trends = self.mini_trend_detector.mini_trends
        
        if not mini_trends:
            print("No se detectaron mini-tendencias")
            return zones
        
        print(f"Se detectaron {len(mini_trends)} mini-tendencias")
        
        # Crear tabla para mini-tendencias si no existe
        self.connect()
        if not self.connection:
            return zones
        
        try:
            create_mini_trends_table = """
            CREATE TABLE IF NOT EXISTS mini_trends (
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
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                csv_file VARCHAR(255),
                related_accumulation_zone_id INT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.cursor.execute(create_mini_trends_table)
            self.connection.commit()
            
            # Extraer símbolo y timeframe del archivo CSV
            symbol = None
            timeframe = None
            if self.csv_file:
                parts = os.path.basename(self.csv_file).split('-')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1]
                    
            # Eliminar mini-tendencias anteriores para este símbolo y timeframe
            if symbol and timeframe:
                delete_query = "DELETE FROM mini_trends WHERE symbol = %s AND timeframe = %s"
                self.cursor.execute(delete_query, (symbol, timeframe))
                self.connection.commit()
            
            # Guardar mini-tendencias y relacionarlas con zonas de acumulación
            insert_mini_trend_query = """
            INSERT INTO mini_trends (
                start_idx, end_idx, start_time, end_time, direction, slope, r_squared,
                poc, volume_total, duration_bars, symbol, timeframe, csv_file
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Relaciones entre mini-tendencias y zonas de acumulación
            mini_trend_zone_relations = []
            
            for trend in mini_trends:
                # Verificar si la mini-tendencia tiene relación con alguna zona de acumulación
                trend_start = trend['start_idx']
                trend_end = trend['end_idx']
                
                related_zones = []
                for i, zone in enumerate(zones):
                    zone_start = zone.get('start_idx')
                    zone_end = zone.get('end_idx')
                    
                    # Verificar si hay solapamiento entre la mini-tendencia y la zona
                    if not (trend_end < zone_start or trend_start > zone_end):
                        related_zones.append(i)
                
                # Guardar la mini-tendencia
                row = (
                    trend['start_idx'],
                    trend['end_idx'],
                    trend['start_time'],
                    trend['end_time'],
                    trend['direction'],
                    trend['slope'],
                    trend['r_squared'],
                    trend['poc'],
                    trend['volume_total'],
                    trend['duration_bars'],
                    symbol,
                    timeframe,
                    self.csv_file
                )
                self.cursor.execute(insert_mini_trend_query, row)
                mini_trend_id = self.cursor.lastrowid
                
                # Guardar relaciones para actualización posterior
                for zone_idx in related_zones:
                    mini_trend_zone_relations.append((mini_trend_id, zone_idx))
            
            self.connection.commit()
            
            # Ahora actualizar las velas clave con su mini_trend_id si corresponde
            for mini_trend_id, zone_idx in mini_trend_zone_relations:
                zone = zones[zone_idx]
                zone_start = zone.get('start_idx')
                zone_end = zone.get('end_idx')
                
                # Actualizar mini_trend_id para key_candles dentro de la zona
                update_candles_query = """
                UPDATE key_candles
                SET mini_trend_id = %s
                WHERE index_in_csv >= %s AND index_in_csv <= %s
                AND csv_file = %s AND in_accumulation_zone = TRUE;
                """
                self.cursor.execute(update_candles_query, (mini_trend_id, zone_start, zone_end, self.csv_file))
            
            self.connection.commit()
            print(f"Saved {len(mini_trends)} mini-trends and updated key candles with mini-trend relationships")
            
        except Exception as e:
            print(f"Error analyzing mini-trends: {e}")
        finally:
            self.close()
        
        return zones
        
    def save_results(self, zones, detection_params, data_df=None):
        """
        Guarda los resultados de las zonas de acumulación en la tabla detect_accumulation_zone_results.
        Si la tabla no existe, la crea con la estructura adecuada.
        Borra solo los datos anteriores del mismo símbolo y timeframe antes de insertar los nuevos.
        También actualiza la tabla key_candles para marcar qué velas están dentro de zonas de acumulación.
        """
        if not zones:
            print("No accumulation zones to save.")
            return False
        
        # Analizar mini-tendencias en relación con las zonas detectadas
        zones = self.analyze_mini_trends(data_df, zones)
        
        self.connect()
        if not self.connection:
            return False
        
        try:
            # Crea la tabla si no existe
            create_table_query = """
            CREATE TABLE IF NOT EXISTS detect_accumulation_zone_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                start_idx INT,
                end_idx INT,
                high FLOAT,
                low FLOAT,
                volume_avg FLOAT,
                vol_total FLOAT,
                vwap FLOAT,
                poc FLOAT,
                mfi FLOAT,
                quality_score FLOAT,
                datetime_start DATETIME,
                datetime_end DATETIME,
                detection_params TEXT,
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                csv_file VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            
            # Extraer símbolo y timeframe del nombre del archivo CSV
            symbol = None
            timeframe = None
            if self.csv_file:
                parts = os.path.basename(self.csv_file).split('-')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1]
            
            # Borra solo los datos anteriores del mismo símbolo y timeframe
            if symbol and timeframe:
                print(f"Borrando datos anteriores para {symbol}-{timeframe}...")
                delete_query = "DELETE FROM detect_accumulation_zone_results WHERE symbol = %s AND timeframe = %s"
                self.cursor.execute(delete_query, (symbol, timeframe))
                self.connection.commit()
            else:
                print("No se pudo determinar símbolo y timeframe para borrar datos antiguos")
            
            # Prepara la consulta de inserción
            insert_query = """
            INSERT INTO detect_accumulation_zone_results (
                start_idx, end_idx, high, low, volume_avg, vol_total, 
                vwap, poc, mfi, quality_score, datetime_start, datetime_end,
                detection_params, symbol, timeframe, csv_file
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            # Usamos el symbol y timeframe que ya extrajimos anteriormente
            
            # Inserta los datos de cada zona detectada
            for zone in zones:
                row = (
                    zone.get('start_idx'),
                    zone.get('end_idx'),
                    zone.get('high'),
                    zone.get('low'),
                    zone.get('volume_avg'),
                    zone.get('vol_total'),
                    zone.get('vwap'),
                    zone.get('poc'),
                    zone.get('mfi'),
                    zone.get('quality_score'),
                    zone.get('datetime_start'),
                    zone.get('datetime_end'),
                    json.dumps(detection_params),
                    symbol,
                    timeframe,
                    os.path.basename(self.csv_file) if self.csv_file else None
                )
                self.cursor.execute(insert_query, row)
            
            self.connection.commit()
            print(f"Saved {len(zones)} accumulation zones")
            
            # Actualiza la tabla key_candles para marcar las velas dentro de zonas de acumulación
            self.update_key_candles_in_zones(zones)
            
            return True
        
        except Exception as e:
            print(f"Error saving results: {e}")
            return False
        
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description="Save detected accumulation zones to DB")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with OHLCV data')
    parser.add_argument('--key-candles', type=str, help='Path to CSV file with key candles (optional)')
    parser.add_argument('--atr-period', type=int, default=14, help='Periodo ATR para la detección (default: 14)')
    parser.add_argument('--atr-multiplier', type=float, default=1.0, help='Multiplicador ATR para la tolerancia de rango (default: 1.0)')
    parser.add_argument('--volume-threshold', type=float, default=1.1, help='Umbral de volumen relativo (default: 1.1)')
    parser.add_argument('--quality-threshold', type=float, default=3.0, help='Umbral de puntuación de calidad mínima (default: 3.0)')
    parser.add_argument('--recency-bonus', type=float, default=0.1, help='Bonificación por proximidad temporal')
    parser.add_argument('--use-mini-trends', type=bool, default=True, help='Usar detector de mini-tendencias para enriquecer resultados')
    parser.add_argument('--verbose', action='store_true', help='Mostrar información detallada durante la ejecución')
    
    args = parser.parse_args()
    
    # Inicializa el detector y carga los datos
    try:
        detector = AccumulationZoneDetector()
        detector.set_params(
            atr_period=args.atr_period,
            atr_multiplier=args.atr_multiplier,
            volume_threshold=args.volume_threshold,
            quality_threshold=args.quality_threshold,
            recency_bonus=args.recency_bonus  # Pasar recency_bonus al detector
        )
        
        # Cargar los datos para tenerlos disponibles
        data_df = None
        try:
            # Intentamos cargar los datos usando las mismas columnas que usa MiniTrendDetector
            binance_columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            data_df = pd.read_csv(args.csv, names=binance_columns, header=None)
            
            # Verificar si ya tiene encabezados
            if data_df.columns[0] in ['timestamp', 'open_time']:
                data_df = data_df.iloc[1:].reset_index(drop=True)
                
            # Convertir a tipos numéricos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data_df[col] = pd.to_numeric(data_df[col])
                
            # Convertir timestamp a datetime para referencia
            data_df['datetime'] = pd.to_datetime(data_df['timestamp'], unit='us')
            
            print(f"Loaded CSV with {len(data_df)} rows for mini-trend analysis")
        except Exception as e:
            print(f"Error loading data for mini-trend analysis: {e}")
            data_df = None
        
        # Ejecutar la detección
        results = detector.process_csv(args.csv)
        
        if results:
            # Guardar resultados en la base de datos
            saver = AccumulationZoneResultSaver()
            saver.csv_file = args.csv
            # Pasar también el DataFrame para el análisis de mini-tendencias
            saver.save_results(results, detector.params, data_df)
        else:
            print("No accumulation zones detected")
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

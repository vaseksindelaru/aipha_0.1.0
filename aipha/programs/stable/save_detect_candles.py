"""
save_detect_candles.py - Módulo para guardar resultados de la detección Shakeout en base de datos

Este módulo permite guardar los resultados de la detección de velas clave en una base de datos MySQL para su análisis posterior.
Ubicación: aipha/programs/stable/save_detect_candles.py
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/.env')), override=True)

import sys
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import json
from datetime import datetime
import traceback

# Ajuste de path para importar desde el módulo padre
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from detect_candles import Detector

class DetectionResultSaver:
    """
    Clase para guardar resultados de la estrategia Shakeout en la base de datos binance_lob.
    Maneja la conexión y métodos para almacenar datos de velas clave y parámetros de detección.
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

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                print(f"Connected to MySQL database: {self.db_config['database']}")
                return True
        except Error as e:
            print(f"Error connecting to MySQL database: {e}")
            return False

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def save_results(self, results, detection_params, tables):
        """
        Guarda los resultados en cada tabla de la lista 'tables'.
        - Si la tabla es 'detection_sessions', guarda solo una fila con los parámetros de la sesión y la fecha/hora.
        - Las otras tablas reciben los resultados completos de velas.
        Si la tabla existe, detecta sus columnas y solo inserta los campos que existan.
        Si no existe, la crea con la estructura estándar.
        
        Extrae el símbolo y timeframe del nombre del archivo CSV para guardarlos en las tablas.
        """
        # Extraer símbolo y timeframe del nombre del archivo CSV
        self.symbol = None
        self.timeframe = None
        if hasattr(self, 'csv_file') and self.csv_file:
            parts = os.path.basename(self.csv_file).split('-')
            if len(parts) >= 2:
                self.symbol = parts[0]
                self.timeframe = parts[1]
        from datetime import datetime
        if not self.connection or not self.connection.is_connected():
            print("Not connected to database.")
            return False
        try:
            # Ordena las tablas para borrar primero las hijas y luego la padre
            tablas_borrado = []
            if 'key_candles' in tables:
                tablas_borrado.append('key_candles')
            if 'detection_params' in tables:
                tablas_borrado.append('detection_params')
            if 'detection_sessions' in tables:
                tablas_borrado.append('detection_sessions')
            # Borrado en orden seguro
            for table_name in tablas_borrado:
                # Verifica si la tabla existe
                self.cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                exists = self.cursor.fetchone() is not None
                if not exists:
                    # Crea la tabla con la estructura estándar o de sesión
                    if table_name in ['detection_sessions', 'detection_params']:
                        create_table_query = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            detection_params JSON,
                            csv_file VARCHAR(255),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ) ENGINE=InnoDB;
                        """
                    else:
                        create_table_query = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            candle_index INT,
                            open FLOAT,
                            high FLOAT,
                            low FLOAT,
                            close FLOAT,
                            volume FLOAT,
                            volume_percentile FLOAT,
                            body_percentage FLOAT,
                            is_key_candle BOOLEAN,
                            symbol VARCHAR(20),
                            timeframe VARCHAR(10),
                            in_accumulation_zone BOOLEAN DEFAULT FALSE,
                            datetime DATETIME,
                            detection_params JSON,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ) ENGINE=InnoDB;
                        """
                    self.cursor.execute(create_table_query)
                    self.connection.commit()
                # Limpia la tabla antes de insertar nuevos datos
                self.cursor.execute(f"DELETE FROM {table_name}")
                # Obtiene las columnas existentes en la tabla
                self.cursor.execute(f"SHOW COLUMNS FROM {table_name}")
                columns_info = self.cursor.fetchall()
                table_columns = [col[0] for col in columns_info if col[0] != 'id' and col[0] != 'created_at']
                insert_cols = []
                for col in table_columns:
                    if col == 'detection_params':
                        insert_cols.append('detection_params')
                    else:
                        insert_cols.append(col)
                insert_query = f"INSERT INTO {table_name} ({', '.join(insert_cols)}) VALUES ({', '.join(['%s']*len(insert_cols))})"
                if table_name in ['detection_sessions', 'detection_params']:
                    # Solo una fila por sesión
                    row = []
                    # Extrae los valores de los parámetros
                    vpt = detection_params.get('volume_percentile_threshold', None)
                    bpt = detection_params.get('body_percentage_threshold', None)
                    lookback = detection_params.get('lookback_candles', None)
                    num_candles = getattr(self, 'num_candles', None)
                    for col in insert_cols:
                        if col == 'detection_params':
                            row.append(json.dumps(detection_params))
                        elif col == 'csv_file':
                            row.append(self.csv_file if hasattr(self, 'csv_file') else None)
                        elif col == 'created_at':
                            row.append(datetime.now())
                        elif col == 'volume_threshold':
                            row.append(vpt)
                        elif col == 'body_threshold':
                            row.append(bpt)
                        elif col == 'lookback':
                            row.append(lookback)
                        elif col == 'num_candles':
                            row.append(num_candles)
                        else:
                            row.append(None)
                    self.cursor.execute(insert_query, tuple(row))
                else:
                    for res in results:
                        row = []
                        for col in insert_cols:
                            if col == 'detection_params':
                                row.append(json.dumps(detection_params))
                            elif col == 'candle_index' and 'index' in res:
                                row.append(res['index'])
                            elif col == 'symbol':
                                row.append(self.symbol)
                            elif col == 'timeframe':
                                row.append(self.timeframe)
                            elif col == 'datetime' and 'timestamp' in res:
                                row.append(res['timestamp'])
                            else:
                                row.append(res.get(col, None))
                        self.cursor.execute(insert_query, tuple(row))
                self.connection.commit()
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Save Shakeout detection results to DB (AIPHA version)")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--volume-percentile', type=int, default=70, help='Percentil para considerar volumen alto (70 = top 30%)')
    parser.add_argument('--body-threshold', type=int, default=40, help='Porcentaje máximo del cuerpo de la vela respecto al rango')
    parser.add_argument('--lookback', type=int, default=30, help='Número de velas para calcular percentiles')
    parser.add_argument('--verbose', action='store_true', help='Mostrar información detallada')
    args = parser.parse_args()

    if args.verbose:
        print(f"Parámetros de detección: VPT={args.volume_percentile}, BPT={args.body_threshold}, lookback={args.lookback}")
    
    detector = Detector(args.csv)
    detector.set_detection_params(args.volume_percentile, args.body_threshold, args.lookback)
    results = detector.process_csv()

    if args.verbose:
        print(f"Detectadas {len(results)} velas clave en {os.path.basename(args.csv)}")

    saver = DetectionResultSaver()
    if saver.connect():
        tablas_destino = ["detection_params", "detection_sessions", "key_candles"]
        # Guarda el nombre del archivo CSV en el saver para usarlo en detection_sessions
        saver.csv_file = os.path.basename(args.csv)
        # Guarda el número de velas analizadas
        saver.num_candles = len(detector.data) if detector.data is not None else None
        success = saver.save_results(results, detector.detection_params, tablas_destino)
        saver.close()
        
        if success and args.verbose:
            print(f"Velas clave guardadas exitosamente en la base de datos")
            # Extraer símbolo y timeframe del nombre del archivo
            parts = os.path.basename(args.csv).split('-')
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1]
                print(f"Para consultar resultados: SELECT * FROM key_candles WHERE symbol = '{symbol}' AND timeframe = '{timeframe}';")
        elif not success:
            print("Error al guardar las velas clave en la base de datos")

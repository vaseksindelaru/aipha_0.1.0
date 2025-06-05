"""
detect_accumulation_zone.py - Módulo para detectar y guardar zonas de acumulación en base de datos

Este módulo permite detectar zonas de acumulación previas a velas clave y guardar los resultados
en una base de datos MySQL para su análisis posterior.
Ubicación: aipha/programs/stable/detect_accumulation_zone.py
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import json
from datetime import datetime
import traceback
import argparse
import logging
# Reemplazamos talib por pandas_ta
import pandas_ta as ta

# Configuración de logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'accumulation_zone.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/.env')), override=True)

# Ajuste de path para importar desde el módulo padre
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

class AccumulationZoneDetector:
    """
    Detector autónomo de zonas de acumulación previas a velas clave, sin dependencia de TradingView.
    """
    def __init__(self, csv_path=None):
        """
        Inicializa el detector con datos OHLCV.
        :param csv_path: Ruta al archivo CSV en formato Binance
        """
        self.data = None
        self.params = {
            'atr_period': 14,              # Período para ATR
            'atr_multiplier': 1.5,         # Multiplicador para rango estrecho
            'volume_threshold': 1.2,       # Umbral para volumen elevado
            'min_zone_bars': 5,            # Mínimo de barras para una zona
            'volume_profile_bins': 50,     # Bins para Volume Profile
            'mfi_period': 14,              # Período para MFI
            'sma_period': 200,             # Período para SMA (contexto)
            'quality_threshold': 0.7        # Umbral para índice de calidad
        }
        if csv_path:
            self.load_csv(csv_path)

    def load_csv(self, csv_path):
        """
        Carga datos de un archivo CSV en formato Binance.
        :param csv_path: Ruta al archivo CSV
        """
        if not os.path.exists(csv_path):
            logging.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        try:
            binance_columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            self.data = pd.read_csv(csv_path, names=binance_columns, header=None)
            if self.data.columns[0] in ['timestamp', 'open_time']:
                self.data = self.data.iloc[1:].reset_index(drop=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.data[col] = pd.to_numeric(self.data[col])
            self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='us')
            logging.info(f"Loaded CSV with {len(self.data)} rows from {csv_path}")
        except Exception as e:
            logging.error(f"Error loading CSV: {str(e)}")
            raise

    def set_params(self, atr_period=14, atr_multiplier=1.5, volume_threshold=1.2, 
                   min_zone_bars=5, volume_profile_bins=50, mfi_period=14, 
                   sma_period=200, quality_threshold=0.7):
        """
        Establece los parámetros para la detección de zonas de acumulación.
        """
        self.params.update({
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'volume_threshold': volume_threshold,
            'min_zone_bars': min_zone_bars,
            'volume_profile_bins': volume_profile_bins,
            'mfi_period': mfi_period,
            'sma_period': sma_period,
            'quality_threshold': quality_threshold
        })
        logging.info(f"Parameters set: {self.params}")

    def calculate_dynamic_lookback(self, index):
        """
        Calcula un lookback dinámico basado en ATR.
        :param index: Índice de la vela actual
        :return: Número de velas para el lookback
        """
        # Calculamos ATR usando pandas_ta
        atr_series = self.data.ta.atr(length=self.params['atr_period'])
        atr = atr_series.iloc[index] if index < len(atr_series) else atr_series.iloc[-1]
        lookback = max(self.params['min_zone_bars'], int(atr / self.data['close'].iloc[index] * 1000))
        return min(lookback, 50)

    def calculate_volume_profile(self, start_idx, end_idx):
        """
        Calcula el Volume Profile y POC para un rango de velas.
        :param start_idx: Índice inicial
        :param end_idx: Índice final
        :return: POC (precio), volumen total
        """
        segment = self.data.iloc[start_idx:end_idx]
        if len(segment) == 0:
            return None, 0

        price_min = segment['low'].min()
        price_max = segment['high'].max()
        price_range = price_max - price_min
        if price_range < 0.001:
            price_range = 0.001

        num_bins = self.params['volume_profile_bins']
        bin_size = price_range / num_bins
        bins = [price_min + i * bin_size for i in range(num_bins + 1)]
        volume_profile = np.zeros(num_bins)

        for _, candle in segment.iterrows():
            low_price = candle['low']
            high_price = candle['high']
            candle_volume = candle['volume']
            low_bin_idx = max(0, min(int((low_price - price_min) / bin_size), num_bins - 1))
            high_bin_idx = max(0, min(int((high_price - price_min) / bin_size), num_bins - 1))

            if low_bin_idx == high_bin_idx:
                volume_profile[low_bin_idx] += candle_volume
            else:
                candle_range = high_price - low_price or 0.001
                for bin_idx in range(low_bin_idx, high_bin_idx + 1):
                    bin_low = price_min + bin_idx * bin_size
                    bin_high = price_min + (bin_idx + 1) * bin_size
                    overlap_low = max(bin_low, low_price)
                    overlap_high = min(bin_high, high_price)
                    overlap_range = max(0, overlap_high - overlap_low)
                    bin_volume = candle_volume * (overlap_range / candle_range)
                    volume_profile[bin_idx] += bin_volume

        poc_bin_idx = np.argmax(volume_profile)
        poc_price = price_min + (poc_bin_idx + 0.5) * bin_size
        vol_total = segment['volume'].sum()
        return poc_price, vol_total

    def calculate_vwap(self, start_idx, end_idx):
        """
        Calcula VWAP para un rango de velas.
        :param start_idx: Índice inicial
        :param end_idx: Índice final
        :return: VWAP
        """
        prices = (self.data['high'].iloc[start_idx:end_idx] + 
                  self.data['low'].iloc[start_idx:end_idx] + 
                  self.data['close'].iloc[start_idx:end_idx]) / 3
        volumes = self.data['volume'].iloc[start_idx:end_idx]
        vwap = (prices * volumes).cumsum() / volumes.cumsum()
        return vwap.iloc[-1]

    def calculate_quality_score(self, start_idx, end_idx, range_width, avg_volume_zone, vwap, poc, mfi):
        """
        Calcula un índice de calidad para la zona de acumulación con criterios más permisivos.
        :param start_idx, end_idx: Índices de la zona
        :param range_width: Ancho del rango
        :param avg_volume_zone: Volumen promedio
        :param vwap: VWAP de la zona
        :param poc: POC de la zona
        :param mfi: MFI al final de la zona
        :return: Puntuación de calidad (0-1)
        """
        try:
            # Calculamos ATR y SMA usando pandas_ta con manejo de errores
            try:
                atr_series = self.data.ta.atr(length=self.params['atr_period'])
                atr = atr_series.iloc[start_idx:end_idx].mean()
                if pd.isna(atr) or atr == 0:
                    atr = self.data['close'].iloc[end_idx] * 0.01  # 1% como valor por defecto
            except Exception:
                atr = self.data['close'].iloc[end_idx] * 0.01  # 1% como valor por defecto
            
            # Calculamos percentil de volumen con un rango más amplio
            try:
                volume_percentile = np.percentile(self.data['volume'].iloc[max(0, start_idx - 150):end_idx], 65)  # Menos exigente (65 en vez de 75)
            except Exception:
                volume_percentile = avg_volume_zone * 1.2  # 20% más que el promedio como valor por defecto
            
            # Calculamos SMA con manejo de errores
            try:
                sma_series = self.data.ta.sma(length=self.params['sma_period'])
                sma = sma_series.iloc[end_idx]
                if pd.isna(sma):
                    sma = self.data['close'].iloc[end_idx]
            except Exception:
                sma = self.data['close'].iloc[end_idx]  # Precio actual como fallback

            # Criterios de calidad más permisivos
            # CRITERIO 1: Rango estrecho (más permisivo)
            range_threshold = self.params['atr_multiplier'] * atr * 1.5
            range_score = 1 - min(range_width / range_threshold, 1)  
            
            # CRITERIO 2: Volumen (más permisivo)
            min_volume_score = 0.3  # Valor mínimo como base
            volume_score = min(avg_volume_zone / (volume_percentile * 0.8), 1) if volume_percentile > 0 else min_volume_score
            volume_score = max(volume_score, min_volume_score)  # Garantizamos un mínimo
            
            # CRITERIO 3: Precio cerca de VWAP (más permisivo)
            vwap_threshold = 0.03  # 3% en vez del 1% original
            vwap_score = 1 if abs(self.data['close'].iloc[end_idx] - vwap) / vwap <= vwap_threshold else 0.6
            
            # CRITERIO 4: MFI en rango más amplio
            mfi_score = 1 if 30 <= mfi <= 70 else 0.6  # Rango 30-70 en vez de 40-60
            
            # CRITERIO 5: Contexto de tendencia más flexible
            context_score = 0.8  # Base mínima razonable
            if not pd.isna(sma):
                context_score = 1 if abs(self.data['close'].iloc[end_idx] - sma) / sma <= 0.02 else 0.8  # Cerca de SMA es válido

            # Ponderación ajustada para favorecer rango y volumen
            quality = (0.35 * range_score + 0.35 * volume_score + 0.15 * vwap_score + 
                    0.1 * mfi_score + 0.05 * context_score)
            
            # Añadimos bonificación por número de velas - zonas más largas pueden ser mejores
            num_bars = end_idx - start_idx
            if num_bars >= 3:
                bar_bonus = min(0.15, 0.05 * (num_bars - 2))  # Hasta 15% extra
                quality += bar_bonus
            
            return min(quality, 1.0)  # Aseguramos que no exceda 1.0
            
        except Exception as e:
            logging.error(f"Error calculating quality score: {str(e)}")
            return 0.5  # Valor neutro por defecto en caso de error

    def detect_accumulation_zone(self, candle_index):
        """
        Detecta una zona de acumulación previa a una vela clave.
        :param candle_index: Índice de la vela clave
        :return: Diccionario con detalles de la zona o None
        """
        if self.data is None or candle_index < self.params['min_zone_bars'] + self.params['atr_period']:
            logging.error("Insufficient data or invalid candle index")
            return None

        try:
            # Calcular lookback dinámico - MÁS AMPLIO para encontrar patrones
            lookback = min(self.calculate_dynamic_lookback(candle_index) * 2, 50)  # Ampliamos el lookback 
            start_idx = max(0, candle_index - lookback)
            print(f"DEBUG: Evaluando vela {candle_index}, lookback={lookback}, rango={start_idx}-{candle_index}")

            # Verificar mínimo de barras - MENOS RESTRICTIVO
            if candle_index - start_idx < self.params['min_zone_bars']:
                print(f"DEBUG: Insuficientes barras: {candle_index - start_idx} < {self.params['min_zone_bars']}")
                return None

            # Calcular ATR y rango de precios 
            atr_series = self.data.ta.atr(length=self.params['atr_period'])
            atr = atr_series.iloc[start_idx:candle_index].mean()
            if pd.isna(atr) or atr == 0:  # Manejo de casos con ATR nulo o cero
                atr = self.data['close'].iloc[candle_index] * 0.01  # 1% del precio como ATR por defecto

            # NUEVO: Buscar subrangos más estrechos dentro del rango completo
            best_zone = None
            best_quality = 0
            
            # Probamos diferentes tamaños de ventana para encontrar la mejor zona
            min_window = max(self.params['min_zone_bars'], 2)  # Ventana mínima de 2 velas o min_zone_bars
            for window_size in range(min_window, min(lookback, 15) + 1):
                for window_start in range(start_idx, candle_index - window_size + 1):
                    window_end = window_start + window_size
                    
                    # Extraer rango de precios del subrango
                    high_max = self.data['high'].iloc[window_start:window_end].max()
                    low_min = self.data['low'].iloc[window_start:window_end].min()
                    range_width = high_max - low_min
                    
                    # CRITERIO 1: Rango estrecho - MÁS PERMISIVO
                    range_threshold = self.params['atr_multiplier'] * atr * 1.5  # 50% más permisivo
                    if range_width <= range_threshold:
                        # CRITERIO 2: Volumen - MENOS ESTRICTO
                        avg_volume_zone = self.data['volume'].iloc[window_start:window_end].mean()
                        global_avg_volume = self.data['volume'].iloc[max(0, start_idx - 50):candle_index].mean()
                        
                        # Comparamos con el promedio global, no solo con el periodo anterior
                        volume_threshold = max(0.5, self.params['volume_threshold']) * global_avg_volume
                        if avg_volume_zone >= volume_threshold * 0.7:  # 30% más permisivo
                            # CRITERIO 3: VWAP - ELIMINADO (demasiado restrictivo)
                            vwap = self.calculate_vwap(window_start, window_end)
                            
                            # CRITERIO 4: Relación con la vela clave - MÁS PERMISIVO
                            candle_high = self.data['high'].iloc[candle_index]
                            candle_low = self.data['low'].iloc[candle_index]
                            
                            # Calculamos proximidad en lugar de superposición
                            # Si el rango de la zona y la vela están cerca (dentro del 2% del precio), consideramos válido
                            price_2pct = self.data['close'].iloc[candle_index] * 0.02
                            
                            zone_touches_candle = (
                                (low_min <= candle_high + price_2pct and high_max >= candle_low - price_2pct) or
                                (abs(high_max - candle_low) <= price_2pct) or
                                (abs(low_min - candle_high) <= price_2pct)
                            )
                            
                            if zone_touches_candle:
                                # CRITERIO 5: Volume Profile y POC - MISMO
                                poc, vol_total = self.calculate_volume_profile(window_start, window_end)
                                
                                # CRITERIO 6: MFI - MÁS AMPLIO
                                try:
                                    mfi_series = self.data.ta.mfi(
                                        high=self.data['high'], 
                                        low=self.data['low'], 
                                        close=self.data['close'], 
                                        volume=self.data['volume'], 
                                        length=self.params['mfi_period']
                                    )
                                    mfi = mfi_series.iloc[candle_index] if not pd.isna(mfi_series.iloc[candle_index]) else 50
                                except Exception:
                                    mfi = 50  # Valor neutral por defecto
                                
                                # Calcular calidad con criterios más permisivos
                                quality = self.calculate_quality_score(window_start, window_end, range_width, 
                                                                     avg_volume_zone, vwap, poc, mfi)
                                
                                # Bonificación por proximidad temporal a la vela clave
                                recency_bonus = 0.2 * (1 - (candle_index - window_end) / lookback)
                                quality += recency_bonus
                                
                                # DEBUG
                                print(f"DEBUG: Ventana {window_start}-{window_end}, ATR={atr:.2f}, range={range_width:.2f}, Quality={quality:.2f}")
                                
                                # Guardar la mejor zona encontrada
                                if quality > best_quality and quality >= self.params['quality_threshold'] * 0.8:  # 20% más permisivo
                                    best_quality = quality
                                    best_zone = {
                                        'start_idx': window_start,
                                        'end_idx': window_end,
                                        'high': high_max,
                                        'low': low_min,
                                        'volume_avg': avg_volume_zone,
                                        'vol_total': vol_total,
                                        'vwap': vwap,
                                        'poc': poc,
                                        'mfi': mfi,
                                        'quality_score': quality,
                                        'datetime_start': self.data['datetime'].iloc[window_start],
                                        'datetime_end': self.data['datetime'].iloc[window_end]
                                    }
            
            # Retornamos la mejor zona encontrada
            if best_zone:
                logging.info(f"Accumulation zone detected at index {candle_index}: {best_zone}")
                return best_zone
        except Exception as e:
            logging.error(f"Error detecting accumulation zone: {str(e)}")
        return None

    def process_candles(self, key_candle_indices):
        """
        Procesa una lista de índices de velas clave para detectar zonas de acumulación.
        :param key_candle_indices: Lista de índices de velas clave
        :return: Lista de zonas de acumulación detectadas
        """
        if self.data is None:
            logging.error("No data loaded for processing")
            return []

        print(f"Procesando {len(key_candle_indices)} velas clave: {key_candle_indices[:10]}...")
        zones = []
        for idx in key_candle_indices:
            zone = self.detect_accumulation_zone(idx)
            if zone:
                zones.append(zone)
        
        logging.info(f"Detected {len(zones)} accumulation zones for {len(key_candle_indices)} key candles")
        print(f"Zonas de acumulación detectadas: {len(zones)} de {len(key_candle_indices)} velas clave")
        return zones


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

    def save_results(self, zones, detection_params):
        """
        Guarda los resultados de las zonas de acumulación en la tabla detect_accumulation_zone_results.
        Si la tabla no existe, la crea con la estructura adecuada.
        """
        if not self.connection or not self.connection.is_connected():
            print("Not connected to database.")
            return False
        
        try:
            # Verifica si la tabla existe
            self.cursor.execute("SHOW TABLES LIKE 'detect_accumulation_zone_results'")
            exists = self.cursor.fetchone() is not None
            
            if not exists:
                # Crea la tabla con la estructura adecuada
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
                    detection_params JSON,
                    symbol VARCHAR(20),
                    timeframe VARCHAR(10),
                    csv_file VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB;
                """
                self.cursor.execute(create_table_query)
                self.connection.commit()
            
            # Limpia la tabla antes de insertar nuevos datos
            # self.cursor.execute("DELETE FROM detect_accumulation_zone_results")
            
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
            
            # Extrae información de símbolo y timeframe del nombre del archivo CSV
            symbol = None
            timeframe = None
            if self.csv_file:
                parts = os.path.basename(self.csv_file).split('-')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1]
            
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
            print(f"Saved {len(zones)} accumulation zones to database.")
            return True
        
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            traceback.print_exc()
            return False


def load_key_candles(csv_file_path, key_candles_path=None, db_saver=None):
    """
    Carga los índices de velas clave desde la base de datos, archivo CSV o utiliza un rango si no se proporciona.
    
    :param csv_file_path: Ruta al archivo CSV con datos OHLCV
    :param key_candles_path: Ruta opcional al archivo CSV con velas clave
    :param db_saver: Objeto de conexión a la base de datos (opcional)
    :return: Lista de índices de velas clave
    """
    # Primero intentar obtener los datos de la base de datos si está disponible
    if db_saver and hasattr(db_saver, 'connection') and hasattr(db_saver, 'cursor'):
        try:
            # Extraer symbol y timeframe del csv_file_path
            parts = os.path.basename(csv_file_path).split('-')
            symbol = parts[0] if len(parts) > 0 else None
            timeframe = parts[1] if len(parts) > 1 else None
            
            if symbol and timeframe:
                query = "SELECT candle_index FROM key_candles WHERE symbol = %s AND timeframe = %s"
                db_saver.cursor.execute(query, (symbol, timeframe))
                indices = [row[0] for row in db_saver.cursor.fetchall()]
                if indices:
                    logging.info(f"Loaded {len(indices)} key candles from database for {symbol}-{timeframe}")
                    return indices
        except Exception as e:
            logging.error(f"Error querying key_candles table: {str(e)}")
    
    # Si no se pudo obtener de la base de datos, intentar con el archivo CSV
    if key_candles_path and os.path.exists(key_candles_path):
        try:
            key_candles_df = pd.read_csv(key_candles_path)
            if 'candle_index' in key_candles_df.columns:
                return key_candles_df['candle_index'].tolist()
            elif 'index' in key_candles_df.columns:
                return key_candles_df['index'].tolist()
        except Exception as e:
            logging.error(f"Error loading key candles from CSV: {str(e)}")
    
    # Si no hay archivo de velas clave, selecciona algunos índices del archivo OHLCV
    try:
        # Carga el archivo CSV para contar las filas
        binance_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        data = pd.read_csv(csv_file_path, names=binance_columns, header=None)
        
        # Genera un conjunto de índices espaciados a lo largo del archivo
        total_rows = len(data)
        if total_rows > 100:
            # Selecciona índices cada 50 filas, saltando las primeras 50
            indices = list(range(50, total_rows, 50))
            # Limita a 20 índices como máximo
            if len(indices) > 20:
                indices = indices[:20]
            return indices
        else:
            # Para archivos pequeños, usa algunos índices fijos
            return [10, 20, 30, 40, 50]
    except Exception as e:
        logging.error(f"Error generating key candle indices: {str(e)}")
        # Retorna algunos índices por defecto
        return [100, 150, 200, 250, 300]


def main():
    parser = argparse.ArgumentParser(description="Detect and save accumulation zones to DB")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with OHLCV data')
    parser.add_argument('--key-candles', type=str, help='Path to CSV file with key candles (optional)')
    parser.add_argument('--atr-period', type=int, default=14, help='ATR period')
    parser.add_argument('--atr-multiplier', type=float, default=1.5, help='ATR multiplier for range detection')
    parser.add_argument('--volume-threshold', type=float, default=1.2, help='Volume threshold multiplier')
    parser.add_argument('--min-zone-bars', type=int, default=5, help='Minimum bars for a valid zone')
    parser.add_argument('--quality-threshold', type=float, default=0.7, help='Quality threshold for zone validation')
    
    args = parser.parse_args()
    
    # Inicializa el detector y carga los datos
    try:
        detector = AccumulationZoneDetector(args.csv)
        detector.set_params(
            atr_period=args.atr_period,
            atr_multiplier=args.atr_multiplier,
            volume_threshold=args.volume_threshold,
            min_zone_bars=args.min_zone_bars,
            quality_threshold=args.quality_threshold
        )
        
        # Crea un objeto de base de datos para consultar las velas clave
        from save_detect_accumulation_zone import AccumulationZoneResultSaver
        db_saver = AccumulationZoneResultSaver()
        db_saver.connect()
        
        # Carga o genera índices de velas clave
        key_candle_indices = load_key_candles(args.csv, args.key_candles, db_saver=db_saver)
        logging.info(f"Processing {len(key_candle_indices)} key candles")
        
        # Detecta zonas de acumulación
        zones = detector.process_candles(key_candle_indices)
        
        # Guarda los resultados en la base de datos
        if zones:
            saver = AccumulationZoneResultSaver()
            if saver.connect():
                # Guarda el nombre del archivo CSV en el saver
                saver.csv_file = args.csv
                # Guarda el número de velas analizadas
                saver.num_candles = len(detector.data) if detector.data is not None else None
                
                # Guarda los resultados
                saver.save_results(zones, detector.params)
                saver.close()
                print(f"Successfully saved {len(zones)} accumulation zones to database")
            else:
                print("Failed to connect to database")
        else:
            print("No accumulation zones detected")
            
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

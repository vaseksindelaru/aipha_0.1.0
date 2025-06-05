"""
detect_candle.py - Módulo de Detección de Velas Clave (integración de detect.py)

Este archivo integra la lógica de detect.py para identificar velas clave según la estrategia Shakeout.
Se utiliza para análisis de datos de velas, detección de patrones de reversión y pruebas en entornos simulados.
"""

import pandas as pd
import numpy as np
import os

class Detector:
    """
    Detector para identificar velas clave en la estrategia Shakeout.
    """
    def __init__(self, csv_path=None):
        self.detection_params = {}
        self.detection_data = {}
        self.results = []
        self.data = None
        if csv_path:
            self.load_csv(csv_path)

    def load_csv(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        binance_columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        self.data = pd.read_csv(csv_path, names=binance_columns, header=None)
        if self.data.columns[0] in ['timestamp', 'open_time']:
            self.data = self.data.iloc[:, 1:]
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"CSV missing required columns: {required_cols - set(self.data.columns)}")

    def set_detection_params(self, volume_percentile_threshold=70, body_percentage_threshold=40, lookback_candles=30):
        """
        Establece los parámetros para la detección de velas clave.
        
        :param volume_percentile_threshold: Percentil para considerar volumen alto (70 = top 30% del volumen)
        :param body_percentage_threshold: Porcentaje máximo de tamaño del cuerpo respecto al rango
        :param lookback_candles: Número de velas anteriores para calcular el percentil de volumen
        """
        self.detection_params = {
            'volume_percentile_threshold': volume_percentile_threshold,
            'body_percentage_threshold': body_percentage_threshold,
            'lookback_candles': lookback_candles
        }

    def detect_key_candle(self, index):
        params = self.detection_params
        vpt = params.get('volume_percentile_threshold', 80)
        bpt = params.get('body_percentage_threshold', 30)
        lookback = params.get('lookback_candles', 50)
        if self.data is None or index < lookback:
            return False
        volume_percentile = np.percentile(self.data['volume'].iloc[index - lookback:index], vpt)
        current = self.data.iloc[index]
        current_volume = current['volume']
        current_body_size = abs(current['close'] - current['open'])
        current_range = current['high'] - current['low']
        if current_range == 0:
            return False
        body_percentage = 100 * current_body_size / current_range
        is_high_volume = current_volume >= volume_percentile
        is_small_body = body_percentage <= bpt
        return is_high_volume and is_small_body

    def process_csv(self):
        if self.data is None:
            return []
        key_candles = []
        params = self.detection_params
        vpt = params.get('volume_percentile_threshold', 80)
        bpt = params.get('body_percentage_threshold', 30)
        lookback = params.get('lookback_candles', 50)
        for idx in range(len(self.data)):
            if idx < lookback:
                continue
            volume_percentile = float(np.percentile(self.data['volume'].iloc[idx - lookback:idx], vpt))
            current = self.data.iloc[idx]
            current_volume = float(current['volume'])
            current_body_size = abs(float(current['close']) - float(current['open']))
            current_range = float(current['high']) - float(current['low'])
            if current_range == 0:
                continue
            body_percentage = 100 * current_body_size / current_range
            is_high_volume = current_volume >= volume_percentile
            is_small_body = body_percentage <= bpt
            is_key_candle = is_high_volume and is_small_body
            if is_key_candle:
                # Obtener timestamp si está disponible
                timestamp = None
                if 'timestamp' in self.data.columns:
                    timestamp = self.data.iloc[idx]['timestamp']
                elif 'open_time' in self.data.columns:
                    timestamp = self.data.iloc[idx]['open_time']
                
                key_candles.append({
                    'index': idx,
                    'open': float(current['open']),
                    'high': float(current['high']),
                    'low': float(current['low']),
                    'close': float(current['close']),
                    'volume': current_volume,
                    'volume_percentile': volume_percentile,
                    'body_percentage': body_percentage,
                    'is_key_candle': is_key_candle,
                    'timestamp': timestamp
                })
        return key_candles

# Ejemplo de uso:
# detector = Detector('archivo.csv')
# detector.set_detection_params(80, 30, 50)
# resultados = detector.process_csv()
# print(resultados)

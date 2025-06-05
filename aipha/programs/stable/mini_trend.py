"""
mini_trend.py - Módulo para la detección y análisis de mini-tendencias

Este módulo implementa algoritmos para la segmentación automática de series de precios
en mini-tendencias, cálculo de volume profile y POC (Point of Control).
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configuración de logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'mini_trend.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MiniTrendDetector:
    """
    Detector de mini-tendencias que implementa varios métodos de segmentación
    y análisis de volume profile.
    """
    def __init__(self, csv_path=None):
        self.data = None
        self.params = {
            'zigzag_threshold': 0.005,  # 0.5% por defecto
            'lookback_window': 100,
            'min_trend_bars': 5,
            'volume_profile_bins': 50
        }
        self.mini_trends = []
        if csv_path:
            self.load_csv(csv_path)
            
    def load_csv(self, csv_path):
        """
        Carga datos de un archivo CSV en formato Binance.
        """
        if not os.path.exists(csv_path):
            logging.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            # Nombres de columnas para formato Binance
            binance_columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            self.data = pd.read_csv(csv_path, names=binance_columns, header=None)
            
            # Verificar si ya tiene encabezados
            if self.data.columns[0] in ['timestamp', 'open_time']:
                self.data = self.data.iloc[1:].reset_index(drop=True)
                
            # Convertir a tipos numéricos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.data[col] = pd.to_numeric(self.data[col])
                
            # Convertir timestamp a datetime para referencia
            self.data['datetime'] = pd.to_datetime(self.data['timestamp'], unit='us')
            
            logging.info(f"Loaded CSV with {len(self.data)} rows from {csv_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading CSV: {str(e)}")
            raise
    
    def set_params(self, zigzag_threshold=0.005, lookback_window=100, min_trend_bars=5, volume_profile_bins=50):
        """
        Establece los parámetros para la detección de mini-tendencias.
        """
        self.params['zigzag_threshold'] = zigzag_threshold
        self.params['lookback_window'] = lookback_window
        self.params['min_trend_bars'] = min_trend_bars
        self.params['volume_profile_bins'] = volume_profile_bins
        logging.info(f"Parameters set: {self.params}")
    
    def detect_zigzag_pivots(self):
        """
        Detecta pivotes usando un algoritmo ZigZag simple basado en umbral de precio.
        """
        if self.data is None or len(self.data) < 2:
            logging.error("No data available for zigzag detection")
            return []
        
        # Inicializar variables
        pivots = []
        trend = None  # None = no establecido, True = subiendo, False = bajando
        last_pivot_idx = 0
        last_pivot_price = self.data.iloc[0]['close']
        threshold = self.params['zigzag_threshold']
        
        # Recorrer los datos para encontrar pivotes
        for i in range(1, len(self.data)):
            current_price = self.data.iloc[i]['close']
            price_change = (current_price - last_pivot_price) / last_pivot_price
            
            # Si no hay tendencia establecida, determinarla con el primer movimiento significativo
            if trend is None and abs(price_change) >= threshold:
                trend = price_change > 0
                continue
                
            # Si hay un cambio de dirección más grande que el umbral
            if (trend and price_change <= -threshold) or (not trend and price_change >= threshold):
                # Agregar el último pivote
                pivots.append(last_pivot_idx)
                # Actualizar al nuevo pivote
                last_pivot_idx = i
                last_pivot_price = current_price
                # Cambiar la dirección de la tendencia
                trend = not trend
            # Si continúa en la misma dirección pero con un nuevo extremo
            elif (trend and current_price > last_pivot_price) or (not trend and current_price < last_pivot_price):
                last_pivot_idx = i
                last_pivot_price = current_price
        
        # Agregar el último pivote si no está en la lista
        if pivots and pivots[-1] != last_pivot_idx:
            pivots.append(last_pivot_idx)
        
        logging.info(f"Detected {len(pivots)} zigzag pivots")
        return pivots
    
    def segment_mini_trends(self):
        """
        Segmenta los datos en mini-tendencias usando el método ZigZag.
        """
        pivots = self.detect_zigzag_pivots()
        if not pivots or len(pivots) < 2:
            logging.warning("Not enough pivots to create mini-trends")
            return []
        
        mini_trends = []
        for i in range(len(pivots) - 1):
            start_idx = pivots[i]
            end_idx = pivots[i+1]
            
            # Solo considerar segmentos con suficientes barras
            if end_idx - start_idx + 1 >= self.params['min_trend_bars']:
                segment = self.data.iloc[start_idx:end_idx+1]
                
                # Calcular propiedades de la mini-tendencia
                start_price = segment.iloc[0]['close']
                end_price = segment.iloc[-1]['close']
                direction = 'alcista' if end_price > start_price else 'bajista'
                slope = (end_price - start_price) / len(segment)
                
                # Calcular R^2 para medir la suavidad de la tendencia
                x = np.arange(len(segment))
                y = segment['close'].values
                coeffs = np.polyfit(x, y, 1)
                p = np.poly1d(coeffs)
                y_fit = p(x)
                ss_total = np.sum((y - np.mean(y))**2)
                ss_residual = np.sum((y - y_fit)**2)
                r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                
                # Calcular volume profile y POC
                poc, vol_total = self.calculate_volume_profile(segment)
                
                mini_trend = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_time': segment.iloc[0]['datetime'],
                    'end_time': segment.iloc[-1]['datetime'],
                    'direction': direction,
                    'slope': slope,
                    'r_squared': r_squared,
                    'poc': poc,
                    'volume_total': vol_total,
                    'duration_bars': len(segment),
                    'duration_minutes': len(segment) * 15  # Asumiendo barras de 15 minutos
                }
                
                mini_trends.append(mini_trend)
        
        self.mini_trends = mini_trends
        logging.info(f"Segmented {len(mini_trends)} mini-trends")
        return mini_trends
    
    def calculate_volume_profile(self, segment):
        """
        Calcula el volume profile y POC para un segmento de datos.
        Distribuye el volumen proporcionalmente en el rango [low, high] de cada vela.
        """
        if len(segment) == 0:
            return None, 0
        
        # Extraer rango de precios
        price_min = segment['low'].min()
        price_max = segment['high'].max()
        price_range = price_max - price_min
        
        # Si el rango es demasiado pequeño, usar un valor mínimo
        if price_range < 0.001:
            price_range = 0.001
        
        # Crear bins para el volume profile
        num_bins = self.params['volume_profile_bins']
        bin_size = price_range / num_bins
        bins = [price_min + i * bin_size for i in range(num_bins + 1)]
        
        # Inicializar arreglo para volume profile
        volume_profile = np.zeros(num_bins)
        
        # Para cada vela, distribuir el volumen proporcionalmente entre los bins cubiertos por [low, high]
        for _, candle in segment.iterrows():
            low_price = candle['low']
            high_price = candle['high']
            candle_volume = candle['volume']
            
            # Calcular bins que cubre la vela
            low_bin_idx = max(0, min(int((low_price - price_min) / bin_size), num_bins - 1))
            high_bin_idx = max(0, min(int((high_price - price_min) / bin_size), num_bins - 1))
            
            # Si la vela cubre un solo bin
            if low_bin_idx == high_bin_idx:
                volume_profile[low_bin_idx] += candle_volume
            else:
                # Calcular rango de precio de la vela
                candle_range = high_price - low_price
                if candle_range <= 0:
                    # Si no hay rango válido, asignar al bin del precio de cierre
                    close_bin_idx = max(0, min(int((candle['close'] - price_min) / bin_size), num_bins - 1))
                    volume_profile[close_bin_idx] += candle_volume
                else:
                    # Distribuir el volumen proporcionalmente entre los bins
                    for bin_idx in range(low_bin_idx, high_bin_idx + 1):
                        # Precio inferior y superior del bin
                        bin_low = price_min + bin_idx * bin_size
                        bin_high = price_min + (bin_idx + 1) * bin_size
                        
                        # Calcular la intersección entre el bin y el rango de la vela
                        overlap_low = max(bin_low, low_price)
                        overlap_high = min(bin_high, high_price)
                        overlap_range = overlap_high - overlap_low
                        
                        # Asignar volumen proporcional
                        if overlap_range > 0:
                            bin_volume = candle_volume * (overlap_range / candle_range)
                            volume_profile[bin_idx] += bin_volume
        
        # Encontrar el POC (Point of Control)
        poc_bin_idx = np.argmax(volume_profile)
        poc_price = price_min + (poc_bin_idx + 0.5) * bin_size  # Precio central del bin con más volumen
        
        # Volumen total del segmento
        vol_total = segment['volume'].sum()
        
        return poc_price, vol_total
    
    def find_minitrends_for_candle(self, candle, price_col='close', poc_tol=0.002):
        """
        Encuentra las mini-tendencias relevantes para una vela específica.
        Compara si el POC de la mini-tendencia está cerca del precio de la vela.
        
        Args:
            candle: Diccionario con los datos de la vela clave
            price_col: Columna de precio a comparar ('open', 'close', etc.)
            poc_tol: Tolerancia para considerar que el POC está cerca del precio (% relativo)
            
        Returns:
            DataFrame con las mini-tendencias relevantes
        """
        if not self.mini_trends or not candle:
            logging.warning("No hay mini-tendencias o vela para comparar")
            return pd.DataFrame([])  # Devolver DataFrame vacío si no hay datos
        
        # Extraer precio y timestamp de la vela clave
        price = float(candle.get(price_col, 0))
        candle_idx = int(candle.get('index', 0))
        
        # Inicializar lista para mini-tendencias relevantes
        relevant_trends = []
        
        for trend in self.mini_trends:
            # Verificar si la vela clave está después de la mini-tendencia
            if candle_idx > trend['end_idx']:
                # Calcular distancia relativa entre POC y precio de vela
                if trend['poc'] is not None:
                    price_diff_pct = abs(trend['poc'] - price) / price
                    
                    # Si el POC está suficientemente cerca del precio
                    if price_diff_pct <= poc_tol:
                        # Calcular tiempo transcurrido entre fin de mini-tendencia y vela clave
                        bars_since_trend = candle_idx - trend['end_idx']
                        
                        # Añadir información de comparación
                        trend_copy = trend.copy()
                        trend_copy['price_diff_pct'] = price_diff_pct * 100  # Convertir a porcentaje
                        trend_copy['bars_since_trend'] = bars_since_trend
                        trend_copy['candle_idx'] = candle_idx
                        trend_copy['candle_price'] = price
                        
                        relevant_trends.append(trend_copy)
        
        # Crear DataFrame con mini-tendencias relevantes
        if relevant_trends:
            return pd.DataFrame(relevant_trends)
        else:
            return pd.DataFrame([])  # Devolver DataFrame vacío si no hay relevantes
        
    def process_csv(self):
        """
        Procesa todo el archivo CSV y genera resultados de mini-tendencias.
        """
        if self.data is None:
            logging.error("No data loaded for processing")
            return pd.DataFrame([])
        
        try:
            # Detectar mini-tendencias
            self.segment_mini_trends()
            
            if not self.mini_trends:
                logging.warning("No mini-trends detected")
                return pd.DataFrame([])
            
            # Convertir a DataFrame para facilitar la manipulación
            results_df = pd.DataFrame(self.mini_trends)
            
            logging.info(f"Processed CSV successfully, found {len(results_df)} mini-trends")
            return results_df
        
        except Exception as e:
            logging.error(f"Error processing CSV: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame([])

# Ejemplo de uso:
# detector = MiniTrendDetector('data.csv')
# detector.set_params(zigzag_threshold=0.005)
# results = detector.process_csv()
# print(results)

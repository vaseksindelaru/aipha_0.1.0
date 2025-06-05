"""
Prueba la lógica del detector de velas clave con datos sintéticos.
Ubicación: aipha/programs/stable/tests/test_detect_candle.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Ajuste de path para importar desde el módulo padre
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detect_candles import Detector

def create_sample_data(num_candles=100):
    np.random.seed(42)
    base_price = 10000
    random_walk = np.cumsum(np.random.normal(0, 100, num_candles))
    data = []
    for i in range(num_candles):
        price = base_price + random_walk[i]
        range_size = abs(np.random.normal(0, 50))
        body_size = range_size * np.random.uniform(0.1, 0.9)
        volume = abs(np.random.normal(1000, 200))
        if np.random.random() < 0.1:
            volume *= np.random.uniform(2, 5)
        if np.random.random() < 0.05:
            volume *= np.random.uniform(3, 6)
            body_size = range_size * np.random.uniform(0.05, 0.2)
        high = price + range_size/2
        low = price - range_size/2
        if np.random.random() < 0.5:
            open_price = low + (range_size - body_size) * np.random.uniform(0, 1)
            close = open_price + body_size
        else:
            close = low + (range_size - body_size) * np.random.uniform(0, 1)
            open_price = close + body_size
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    return pd.DataFrame(data)

def test_detector():
    df = create_sample_data(100)
    df.to_csv('sample_data.csv', index=False)
    detector = Detector('sample_data.csv')
    detector.set_detection_params(80, 30, 20)
    results = detector.process_csv()
    print(f"Key candles detected: {len(results)}")
    for candle in results:
        print(f"Index: {candle['index']}, Open: {candle['open']:.2f}, Close: {candle['close']:.2f}, Volume: {candle['volume']:.2f}")

if __name__ == "__main__":
    test_detector()

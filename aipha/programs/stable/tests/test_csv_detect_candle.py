"""
Prueba el detector de velas clave con datos reales en CSV.
Ubicación: aipha/programs/stable/tests/test_csv_detect_candle.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detect_candles import Detector

def main():
    print("Testing Shakeout Strategy Detector with real CSV data...")
    detector = Detector('sample_data.csv')
    detector.set_detection_params(
        volume_percentile_threshold=80,
        body_percentage_threshold=30,
        lookback_candles=20
    )
    print("Detector initialized with parameters:")
    print(f"  - volume_percentile_threshold: {detector.detection_params['volume_percentile_threshold']}")
    print(f"  - body_percentage_threshold: {detector.detection_params['body_percentage_threshold']}")
    print(f"  - lookback_candles: {detector.detection_params['lookback_candles']}")
    results = detector.process_csv()
    if results:
        print(f"\nKey candles detected: {len(results)}")
        for i, candle in enumerate(results):
            print(f"Key candle {i+1}: Index: {candle['index']}, Volume: {candle['volume']:.2f}, Open: {candle['open']:.2f}, Close: {candle['close']:.2f}")
    else:
        print("\nNo key candles detected.")

if __name__ == "__main__":
    main()

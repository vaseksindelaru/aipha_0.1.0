#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_combined_detection.py - Script para ejecutar la detección de velas clave y zonas de acumulación en secuencia

Este script coordina la ejecución de save_detect_candles.py y save_detect_accumulation_zone.py
para asegurar que usen los mismos parámetros y datos, permitiendo una integración efectiva
de la detección de velas clave con zonas de acumulación.
"""

import os
import sys
import argparse
import logging
import subprocess
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_command(command, description):
    """
    Ejecuta un comando y registra su salida en el log
    """
    logging.info(f"Ejecutando: {description}")
    logging.info(f"Comando: {command}")
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logging.info(f"Resultado exitoso: {description}")
        logging.info(f"Salida: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error en {description}: {e}")
        logging.error(f"Salida de error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Ejecutar detección combinada de velas clave y zonas de acumulación")
    parser.add_argument('--csv', type=str, required=True, help='Path a archivo CSV con datos OHLCV')
    parser.add_argument('--volume-percentile', type=int, default=70, help='Percentil para considerar volumen alto (70 = top 30%)')
    parser.add_argument('--body-threshold', type=int, default=40, help='Porcentaje máximo del cuerpo de la vela respecto al rango')
    parser.add_argument('--lookback', type=int, default=30, help='Número de velas para calcular percentiles')
    parser.add_argument('--atr-period', type=int, default=14, help='Periodo ATR para zonas de acumulación')
    parser.add_argument('--atr-multiplier', type=float, default=1.0, help='Multiplicador ATR (menor = más zonas)')
    parser.add_argument('--volume-threshold', type=float, default=1.1, help='Umbral de volumen (menor = más zonas)')
    parser.add_argument('--quality-threshold', type=float, default=3.0, help='Umbral de calidad para validación')
    parser.add_argument('--recency-bonus', type=float, default=0.1, help='Bonificación por proximidad temporal')
    parser.add_argument('--use-mini-trends', action='store_true', help='Activar análisis de mini-tendencias')
    parser.add_argument('--verbose', action='store_true', help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    # Verifica si el archivo CSV existe
    if not os.path.exists(args.csv):
        logging.error(f"El archivo CSV no existe: {args.csv}")
        return False
    
    # 1. Primero ejecutar detección de velas clave
    candles_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "save_detect_candles.py"),
        "--csv", args.csv
    ]
    
    # Añade la lógica para modificar los parámetros de detección de velas clave por código
    cmd_str = " ".join(candles_cmd)
    logging.info(f"NOTA: Los parámetros de detección de velas clave se modificarán en el código (VPT={args.volume_percentile}, BPT={args.body_threshold}, lookback={args.lookback})")
    
    success = run_command(candles_cmd, "Detección de velas clave")
    if not success:
        logging.error("Falló la detección de velas clave, abortando.")
        return False
    
    # 2. Luego ejecutar detección de zonas de acumulación con los mismos parámetros
    accum_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "save_detect_accumulation_zone.py"),
        "--csv", args.csv,
        "--atr-period", str(args.atr_period),
        "--atr-multiplier", str(args.atr_multiplier),
        "--volume-threshold", str(args.volume_threshold),
        "--quality-threshold", str(args.quality_threshold),
        "--recency-bonus", str(args.recency_bonus)
    ]
    
    # Añadir el parámetro de uso de mini-tendencias si está activado
    if args.use_mini_trends:
        accum_cmd.extend(["--use-mini-trends", "True"])
    
    if args.verbose:
        accum_cmd.append("--verbose")
    
    success = run_command(accum_cmd, "Detección de zonas de acumulación")
    if not success:
        logging.error("Falló la detección de zonas de acumulación.")
        return False
    
    logging.info("Proceso combinado completado con éxito.")
    logging.info(f"Las velas clave que coinciden con zonas de acumulación pueden consultarse con:")
    logging.info(f"SELECT * FROM key_candles WHERE symbol = '{os.path.basename(args.csv).split('-')[0]}' AND timeframe = '{os.path.basename(args.csv).split('-')[1]}' AND in_accumulation_zone = TRUE;")
    
    # Mostrar consulta para velas clave que también están en mini-tendencias
    if args.use_mini_trends:
        logging.info(f"Las velas clave que también están en mini-tendencias pueden consultarse con:")
        logging.info(f"SELECT kc.*, mt.direction, mt.r_squared, mt.poc FROM key_candles kc ")
        logging.info(f"JOIN mini_trends mt ON kc.mini_trend_id = mt.id ")
        logging.info(f"WHERE kc.symbol = '{os.path.basename(args.csv).split('-')[0]}' ")
        logging.info(f"AND kc.timeframe = '{os.path.basename(args.csv).split('-')[1]}' ")
        logging.info(f"AND kc.in_accumulation_zone = TRUE;")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Error inesperado: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

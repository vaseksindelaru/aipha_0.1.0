#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
view_triple_signals.py - Muestra las señales de triple coincidencia con su sistema de puntuación
"""

import os
import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import json

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
db_config = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', '21blackjack'),
    'database': os.getenv('MYSQL_DATABASE', 'binance_lob')
}

def connect_db():
    """Conectar a la base de datos."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    print(f"Conectado a base de datos MySQL: {db_config['database']}")
    return conn, cursor

def view_triple_signals(symbol="BTCUSDT", timeframe="5m"):
    """Obtener y mostrar las señales de triple coincidencia"""
    conn, cursor = connect_db()
    
    try:
        # Obtener señales
        query = """
        SELECT 
            id, symbol, timeframe, candle_index, datetime, 
            open, high, low, close, volume, body_percentage,
            zone_id, zone_quality_score, zone_start_datetime, zone_end_datetime,
            mini_trend_id, trend_direction, trend_slope, trend_r_squared,
            trend_start_datetime, trend_end_datetime,
            signal_strength, combined_score,
            zone_score, trend_score, candle_score, 
            direction_factor, slope_factor,
            divergence_factor, reliability_bonus, profit_potential
        FROM 
            triple_signals
        WHERE 
            symbol = %s AND timeframe = %s
        ORDER BY 
            combined_score DESC, signal_strength DESC
        """
        
        cursor.execute(query, (symbol, timeframe))
        signals = cursor.fetchall()
        
        if not signals:
            print(f"No se encontraron señales de triple coincidencia para {symbol}-{timeframe}")
            return
        
        # Mostrar información resumida
        print(f"\n===== SEÑALES DE TRIPLE COINCIDENCIA ({symbol}-{timeframe}) =====")
        print(f"Total de señales: {len(signals)}\n")
        
        # Crear tabla resumida con formato de texto simple
        headers = ["ID", "Indice", "Fecha/Hora", "Dirección", "Precio", "Volumen", "Cuerpo %", 
                 "Calidad Zona", "R2 Tendencia", "Fuerza Señal", "Puntuación"]
        
        # Imprimir encabezados
        header_format = "{:<4} {:<7} {:<19} {:<10} {:<8} {:<8} {:<9} {:<12} {:<12} {:<12} {:<10}"
        print(header_format.format(*headers))
        print("-" * 105)
        
        # Imprimir datos
        row_format = "{:<4} {:<7} {:<19} {:<10} {:<8} {:<8} {:<9} {:<12} {:<12} {:<12} {:<10}"
        for signal in signals:
            print(row_format.format(
                signal['id'],
                signal['candle_index'],
                str(signal['datetime']),
                signal['trend_direction'],
                f"{signal['close']:.2f}",
                f"{signal['volume']:.2f}",
                f"{signal['body_percentage']:.2f}%",
                f"{signal['zone_quality_score']:.4f}",
                f"{signal['trend_r_squared']:.4f}",
                f"{signal['signal_strength']:.4f}",
                f"{signal['combined_score']:.4f}"
            ))
        
        # Mostrar detalles avanzados de la puntuación
        print("\n===== ANÁLISIS DETALLADO DE PUNTUACIÓN =====")
        for signal in signals:
            print(f"\nSeñal #{signal['id']} (Índice: {signal['candle_index']}, Datetime: {signal['datetime']})")
            print(f"Precio: {signal['close']:.2f}, Dirección: {signal['trend_direction']}")
            # Mostrar detalles avanzados de la puntuación en formato texto simple
            print("\nAnálisis detallado de componentes:")
            print("{:<25} {:<12} {:<6} {:<50}".format("Componente", "Puntuación", "Peso", "Factores"))
            print("-" * 93)
            
            # Imprimir filas de datos
            print("{:<25} {:<12} {:<6} {:<50}".format(
                "Zona Acumulación", f"{signal['zone_score']:.4f}", "35%", f"Calidad: {signal['zone_quality_score']:.4f}"))
            print("{:<25} {:<12} {:<6} {:<50}".format(
                "Mini-Tendencia", f"{signal['trend_score']:.4f}", "35%", 
                f"R2: {signal['trend_r_squared']:.4f}, Dir: {signal['direction_factor']:.2f}, Pend: {signal['slope_factor']:.2f}"))
            print("{:<25} {:<12} {:<6} {:<50}".format(
                "Vela Clave", f"{signal['candle_score']:.4f}", "30%", 
                f"Vol: {signal['volume']:.2f}, Cuerpo: {signal['body_percentage']:.2f}%"))
            print("{:<25} {:<12} {:<6} {:<50}".format(
                "Divergencia", f"{signal['divergence_factor']:.4f}", "20%", "Convergencia de factores"))
            print("{:<25} {:<12} {:<6} {:<50}".format(
                "Fiabilidad", f"{signal['reliability_bonus']:.4f}", "15%", f"Bonus por R2 alto"))
            print("{:<25} {:<12} {:<6} {:<50}".format(
                "Potencial Rentabilidad", f"{signal['profit_potential']:.4f}", "15%", "Basado en patrones históricos"))
            print("-" * 93)
            print("{:<25} {:<12} {:<6} {:<50}".format(
                "PUNTUACIÓN FINAL", f"{signal['combined_score']:.4f}", "100%", ""))
            
            # Intentar obtener y mostrar scoring_details si existe
            query_details = "SELECT scoring_details FROM triple_signals WHERE id = %s"
            cursor.execute(query_details, (signal['id'],))
            details_row = cursor.fetchone()
            
            if details_row and details_row['scoring_details']:
                try:
                    details = json.loads(details_row['scoring_details'])
                    print("\nDetalles adicionales de puntuación:")
                    for key, value in details.items():
                        if key not in ['zone_score', 'trend_score', 'candle_score', 'direction_factor', 
                                     'slope_factor', 'divergence_factor', 'reliability_bonus', 
                                     'profit_potential', 'base_strength', 'final_score']:
                            print(f"  - {key}: {value}")
                except:
                    pass
    
    except Exception as e:
        print(f"Error consultando señales de triple coincidencia: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cursor.close()
        conn.close()
        print("\nConexión cerrada.")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ver señales de triple coincidencia")
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Símbolo (por defecto: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe (por defecto: 5m)')
    
    args = parser.parse_args()
    
    view_triple_signals(args.symbol, args.timeframe)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error general: {e}")
        import traceback
        traceback.print_exc()

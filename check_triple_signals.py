import mysql.connector
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

try:
    # Conectar a la base de datos
    print("Conectando a la base de datos...")
    conn = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE')
    )
    print("Conexión exitosa!")

    cursor = conn.cursor(dictionary=True)  # Usar dictionary=True para obtener resultados más legibles

    # Consultar las señales guardadas
    print("\nConsultando tabla triple_signals...")
    cursor.execute("SELECT * FROM triple_signals")
    results = cursor.fetchall()

    if not results:
        print("No se encontraron registros en la tabla triple_signals")
    else:
        # Mostrar resultados
        print(f"\nSe encontraron {len(results)} registros:")
        print("-" * 100)
        
        # Mostrar las columnas
        columns = results[0].keys()
        for column in columns:
            print(f"{column:<20}", end="")
        print("\n" + "-" * 100)

        # Mostrar los datos
        for row in results:
            for value in row.values():
                print(f"{str(value):<20}", end="")
            print()

except mysql.connector.Error as err:
    print(f"Error: {err}")
finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("\nConexión cerrada.")

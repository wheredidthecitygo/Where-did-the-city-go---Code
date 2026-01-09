import argparse
import sys
import logging
from pathlib import Path
from src.data_loader import LaionDataLoader

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(
        description="Paso 1: Descargar metadatos de LAION-5B y filtrar por ciudad/palabra clave."
    )
    
    # Argumentos obligatorios
    parser.add_argument('--token', type=str, required=True, help='Tu token de Hugging Face (con permisos de lectura)')
    parser.add_argument('--keywords', nargs='+', required=True, help='Lista de palabras clave (ej: Madrid "City of Madrid")')
    
    # Argumentos opcionales
    parser.add_argument('--subset', type=str, default='multi', choices=['multi', 'en', 'nolang'], 
                        help='Subconjunto de LAION (default: multi)')
    parser.add_argument('--limit', type=int, default=0, 
                        help='Límite de archivos a procesar (0 = procesar todo). Útil para pruebas.')
    
    args = parser.parse_args()

    # Definir ruta de salida (fija para mantener consistencia)
    output_path = Path("data") / "metadata"
    
    logging.info(f"--- INICIO PASO 1: DESCARGA Y FILTRADO ---")
    logging.info(f"📁 Directorio de salida: {output_path}")
    logging.info(f"🔑 Palabras clave: {args.keywords}")
    
    # Inicializar el cargador
    try:
        loader = LaionDataLoader(
            output_dir=output_path,
            hf_token=args.token,
            dataset_subset=args.subset
        )
        
        # Ejecutar proceso
        limit = args.limit if args.limit > 0 else None
        loader.filter_and_process(
            keywords=args.keywords, 
            max_files=limit
        )
        
        logging.info("--- PASO 1 COMPLETADO EXITOSAMENTE ---")
        
    except Exception as e:
        logging.error(f"❌ Ocurrió un error crítico: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
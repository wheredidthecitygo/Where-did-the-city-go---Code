import argparse
import logging
from pathlib import Path
from src.data_cleaning import MetadataCleaner

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(
        description="Paso 3: Limpieza de metadatos (Filtrado negativo)."
    )
    
    parser.add_argument('--exclude', nargs='+', required=True, 
                        help='Lista de palabras prohibidas. Si aparecen en el caption, la fila se borra. (ej: football soccer "real madrid")')
    
    args = parser.parse_args()

    # Definimos rutas
    # IMPORTANTE: Ahora el siguiente paso (imágenes) deberá leer de 'metadata_clean'
    input_dir = Path("data") / "metadata"
    output_dir = Path("data") / "metadata_cleaned"
    
    if not input_dir.exists():
        logging.error(f"No existe el directorio de entrada: {input_dir}")
        return

    logging.info(f"--- INICIO PASO 3: LIMPIEZA ---")
    logging.info(f"Palabras a excluir: {args.exclude}")
    
    cleaner = MetadataCleaner(input_dir, output_dir)
    cleaner.clean_dataset(exclude_keywords=args.exclude)
    
    logging.info("--- PASO 3 COMPLETADO ---")
    logging.info(f"Ahora puedes usar 'data/metadata_cleaned' para el procesamiento de imágenes.")

if __name__ == "__main__":
    main()
import argparse
import logging
import sys
from pathlib import Path
from src.image_processing import ClipProcessor

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(
        description="Paso 4: Descarga de imágenes y generación de Embeddings CLIP."
    )
    
    # Directorios
    parser.add_argument('--input_dir', type=str, default='data/metadata_cleaned',
                        help='Carpeta con metadatos filtrados (default: data/metadata_cleaned)')
    parser.add_argument('--output_dir', type=str, default='data/embeddings',
                        help='Carpeta de salida (default: data/embeddings)')
    
    # Configuración Modelo
    parser.add_argument('--model', type=str, default='ViT-L/14@336px',
                        help='Modelo CLIP a usar (default: ViT-L/14@336px)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    if not input_path.exists():
        logging.error(f"No existe el directorio de entrada: {input_path}")
        logging.info("Asegúrate de haber ejecutado el Paso 3 (limpieza) o usa '--input_dir data/metadata'")
        sys.exit(1)

    logging.info("--- INICIO PASO 4: PROCESAMIENTO DE IMÁGENES (CLIP) ---")
    logging.info(f"Input: {args.input_dir}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"Modelo: {args.model}")

    try:
        processor = ClipProcessor(model_name=args.model)
        processor.process_directory(args.input_dir, args.output_dir)
        
        logging.info("--- PASO 4 COMPLETADO ---")
        
    except Exception as e:
        logging.error(f"Error crítico en el proceso: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
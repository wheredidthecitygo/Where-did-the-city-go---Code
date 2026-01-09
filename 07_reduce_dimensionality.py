import argparse
import logging
import sys
from pathlib import Path
from src.projection.py import ProjectionProcessor # Nota: El nombre del modulo es src.projection

# Corrección de importación si es necesario, pero asumimos src/projection.py
try:
    from src.projection import ProjectionProcessor
except ImportError:
    # Fallback por si se ejecuta desde otro lado
    sys.path.append(str(Path(__file__).parent))
    from src.projection import ProjectionProcessor

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(
        description="Paso 5: Reducción de dimensionalidad (UMAP 768D -> 2D)."
    )
    
    # Directorios
    parser.add_argument('--input_dir', type=str, default='data/embeddings',
                        help='Carpeta con embeddings (default: data/embeddings)')
    parser.add_argument('--output_dir', type=str, default='data/projection',
                        help='Carpeta de salida (default: data/projection)')
    
    # Parámetros UMAP
    parser.add_argument('--neighbors', type=int, default=25,
                        help='UMAP n_neighbors (default: 25)')
    parser.add_argument('--dist', type=float, default=0.1,
                        help='UMAP min_dist (default: 0.1)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logging.error(f"No existe el directorio de entrada: {input_path}")
        sys.exit(1)

    logging.info("--- INICIO PASO 5: UMAP ---")
    
    try:
        processor = ProjectionProcessor(args.input_dir, args.output_dir)
        
        # 1. Cargar
        metadata, embeddings = processor.load_data()
        if metadata is None:
            logging.error("No se pudieron cargar datos.")
            sys.exit(1)
            
        # 2. Procesar
        embedding_2d = processor.run_umap(
            embeddings, 
            n_neighbors=args.neighbors, 
            min_dist=args.dist
        )
        
        # 3. Guardar
        processor.save_results(metadata, embedding_2d)
        
        logging.info("--- PASO 5 COMPLETADO ---")
        
    except Exception as e:
        logging.error(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
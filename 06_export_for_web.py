import argparse
import logging
import asyncio
import sys
from pathlib import Path
from src.export import WebExporter

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

async def run_process(args):
    try:
        exporter = WebExporter(
            input_parquet=args.input_file,
            output_dir=args.output_dir
        )
        
        # Generar grids y descargar imágenes
        g256, g128, g64 = await exporter.generate_grids()
        
        # Guardar archivos JSON
        exporter.save_json_split(g64, "grid_64")
        exporter.save_json_split(g128, "grid_128")
        exporter.save_json_split(g256, "grid_256")
        
        logging.info("--- PASO 6 COMPLETADO EXITOSAMENTE ---")
        logging.info(f"Datos exportados en: {args.output_dir}")
        
    except Exception as e:
        logging.error(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Paso 6: Exportación para Web (Grids + Imágenes)."
    )
    
    parser.add_argument('--input_file', type=str, default='data/projection/umap_projection.parquet',
                        help='Archivo parquet con proyección UMAP (default: data/projection/umap_projection.parquet)')
    parser.add_argument('--output_dir', type=str, default='data/export',
                        help='Directorio de salida para la web (default: data/export)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        logging.error(f"No existe el archivo de entrada: {input_path}")
        sys.exit(1)

    logging.info("--- INICIO PASO 6: EXPORTACIÓN WEB ---")
    
    # Ejecutar loop asíncrono
    asyncio.run(run_process(args))

if __name__ == "__main__":
    main()
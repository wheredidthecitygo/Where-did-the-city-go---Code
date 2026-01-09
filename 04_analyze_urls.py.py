import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from src.url_analysis import UrlAnalyzer

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(
        description="Paso 4: Análisis de Dominios (URLs)."
    )
    
    # Opcional: permitir elegir la carpeta de entrada (por si te saltaste la limpieza)
    parser.add_argument('--input_dir', type=str, default='data/metadata_cleaned',
                        help='Directorio con los archivos .parquet (default: data/metadata_cleaned)')
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path("data") / "analysis"
    
    if not input_dir.exists():
        logging.error(f"No existe el directorio de entrada: {input_dir}")
        logging.info("Consejo: Si no has ejecutado el paso 3 (limpieza), usa '--input_dir data/metadata'")
        return

    # Buscar archivos
    files = list(input_dir.glob("*.parquet"))
    
    if not files:
        logging.error(f"No hay archivos .parquet en {input_dir}")
        return

    logging.info(f"--- INICIO PASO 4: ANÁLISIS DE DOMINIOS ---")
    logging.info(f"Analizando {len(files)} archivos desde: {input_dir}")

    analyzer = UrlAnalyzer()

    # Procesar
    for file_path in tqdm(files, desc="Extrayendo dominios"):
        analyzer.process_file(file_path)

    # Guardar
    analyzer.save_results(output_dir, top_n=2000)
    
    logging.info("--- PASO 4 COMPLETADO ---")
    logging.info(f"Revisa 'data/analysis/top_domains.csv' para ver las fuentes más comunes.")

if __name__ == "__main__":
    main()
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from src.text_analysis import TextAnalyzer

# Configuración de logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    parser = argparse.ArgumentParser(
        description="Paso 2: Análisis de Texto (Frecuencias, Bigramas, Trigramas)."
    )
    
    parser.add_argument('--city', type=str, required=True, 
                        help='Nombre de la ciudad (para excluirla del conteo de palabras simples).')
    
    args = parser.parse_args()

    # Rutas (asumimos estructura estándar)
    input_dir = Path("data") / "metadata"
    output_dir = Path("data") / "analysis"
    
    if not input_dir.exists():
        logging.error(f"No se encuentra el directorio de entrada: {input_dir}")
        logging.error("Por favor, ejecuta el paso 1 (01_download_and_filter.py) primero.")
        return

    # Buscar archivos parquet generados en el paso 1
    files = list(input_dir.glob("*.parquet"))
    
    if not files:
        logging.error("No se encontraron archivos .parquet en data/metadata.")
        return

    logging.info(f"--- INICIO PASO 2: ANÁLISIS DE TEXTO ---")
    logging.info(f"Ciudad objetivo: {args.city}")
    logging.info(f"Archivos a procesar: {len(files)}")

    # Inicializar analizador
    analyzer = TextAnalyzer(city_name=args.city)

    # Procesar archivos
    for file_path in tqdm(files, desc="Analizando captions"):
        analyzer.process_file(file_path)

    # Guardar resultados
    logging.info("Guardando resultados en CSV...")
    analyzer.save_results(output_dir, top_n=1000)
    
    logging.info(f"--- PASO 2 COMPLETADO ---")
    logging.info(f"Resultados disponibles en: {output_dir}")

if __name__ == "__main__":
    main()
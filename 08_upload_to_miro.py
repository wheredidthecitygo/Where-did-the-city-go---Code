import argparse
import json
import logging
import sys
import time
from pathlib import Path
from tqdm import tqdm
from src.miro_client import MiroClient

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Paso 7: Subir visualización a Miro.")
    
    # Credenciales (Obligatorias)
    parser.add_argument('--token', type=str, required=True, help='Miro Access Token')
    parser.add_argument('--board', type=str, required=True, help='Miro Board ID')
    
    # Configuración de entrada
    parser.add_argument('--grid_json', type=str, default='data/export/grid_256.json',
                        help='Archivo JSON del grid 256 (default: data/export/grid_256.json)')
    parser.add_argument('--images_dir', type=str, default='data/export/images',
                        help='Carpeta raíz de imágenes (default: data/export/images)')
    
    # Configuración Visual
    parser.add_argument('--base_width', type=int, default=400,
                        help='Ancho máximo de imagen en px Miro (para la celda más densa)')
    parser.add_argument('--min_width', type=int, default=100,
                        help='Ancho mínimo de imagen en px Miro')
    parser.add_argument('--spacing', type=int, default=450,
                        help='Espacio entre celdas del grid en Miro')
    parser.add_argument('--limit', type=int, default=0,
                        help='Límite de imágenes a subir (para pruebas). 0 = sin límite.')

    args = parser.parse_args()

    # 1. Validar archivos
    json_path = Path(args.grid_json)
    images_path = Path(args.images_dir)
    
    if not json_path.exists():
        logging.error(f"No se encuentra {json_path}")
        sys.exit(1)

    logging.info("--- INICIO PASO 7: SUBIDA A MIRO ---")
    
    # 2. Cargar datos del Grid
    logging.info(f"Cargando {json_path}...")
    with open(json_path, 'r') as f:
        grid_data = json.load(f)
    
    # Filtrar solo celdas que tengan imágenes físicas
    valid_cells = []
    max_count = 0
    
    for key, data in grid_data.items():
        # grid_256 usa claves "x,y" y rutas relativas "images/256/x_y.webp"
        # Necesitamos resolver la ruta absoluta
        # La ruta en el JSON suele ser relativa "images/256/...", así que ajustamos
        rel_path = data['img']
        if rel_path.startswith("images/"):
            rel_path = rel_path.replace("images/", "")
            
        full_img_path = images_path / rel_path
        
        if full_img_path.exists():
            cell_x, cell_y = map(int, key.split(','))
            count = data['count']
            max_count = max(max_count, count)
            
            valid_cells.append({
                'x': cell_x,
                'y': cell_y,
                'count': count,
                'path': full_img_path,
                'caption': data.get('caption', '')
            })
    
    logging.info(f"Celdas válidas encontradas: {len(valid_cells)}")
    logging.info(f"Densidad máxima (max_count): {max_count}")
    
    if not valid_cells:
        logging.error("No se encontraron imágenes válidas para subir.")
        sys.exit(1)

    # 3. Inicializar Cliente
    client = MiroClient(args.token, args.board)
    
    # 4. Loop de subida
    # Aplicar límite si es necesario
    cells_to_process = valid_cells[:args.limit] if args.limit > 0 else valid_cells
    
    successful = 0
    failed = 0
    
    logging.info(f"Comenzando subida de {len(cells_to_process)} elementos...")
    
    for item in tqdm(cells_to_process, desc="Subiendo a Miro"):
        # Calcular coordenadas en Miro
        # Centramos el mapa: (x - 128) * spacing
        miro_x = (item['x'] - 128) * args.spacing
        miro_y = (item['y'] - 128) * args.spacing
        
        # Calcular tamaño basado en densidad
        # Fórmula: min_width + (ratio * (base_width - min_width))
        ratio = item['count'] / max_count
        width = args.min_width + (ratio * (args.base_width - args.min_width))
        width = int(width)
        
        # Subir Imagen
        # Ponemos el caption como título (tooltip) de la imagen
        short_caption = (item['caption'][:97] + '...') if len(item['caption']) > 100 else item['caption']
        
        res_img = client.upload_image_from_file(
            file_path=item['path'],
            x=miro_x,
            y=miro_y,
            width=width,
            title=short_caption
        )
        
        if res_img:
            # Crear texto debajo
            # Posición Y: miro_y + (altura_imagen / 2) + margen
            # Estimamos altura asumiendo imagen cuadrada o ratio similar
            text_y = miro_y + (width // 2) + 20
            
            client.create_caption_text(
                text=short_caption,
                x=miro_x,
                y=text_y,
                width=width # El texto tiene el mismo ancho que la imagen
            )
            successful += 1
        else:
            failed += 1
            
        # Pequeña pausa para ser amable con la API
        time.sleep(0.2)

    logging.info("--- PASO 7 COMPLETADO ---")
    logging.info(f"✅ Subidos: {successful}")
    logging.info(f"❌ Fallidos: {failed}")
    logging.info(f"🔗 Tablero: https://miro.com/app/board/{args.board}/")

if __name__ == "__main__":
    main()
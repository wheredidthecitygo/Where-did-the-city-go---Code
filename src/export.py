import pandas as pd
import asyncio
import aiohttp
import json
import math
import logging
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebExporter:
    """
    Prepares the UMAP data for web visualization.
    1. Grids the data into zoom levels (256x256, 128x128, 64x64).
    2. Downloads thumbnails for representative points in each grid cell.
    3. Exports JSON files optimized for web loading.
    """

    def __init__(self, input_parquet, output_dir):
        self.input_path = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        (self.images_dir / "256").mkdir(exist_ok=True) # Only need leaf level folder

        # Constants
        self.GRID_SIZES = [256, 128, 64]
        self.MINI_GRID_SIZE = 10
        self.EXAMPLES_PER_CELL = 100
        self.MAX_IMAGE_SIZE = 512
        self.MAX_JSON_MB = 50

    def load_and_normalize(self):
        """Loads UMAP parquet and normalizes coordinates to [0, 1]."""
        logging.info(f"Loading data from {self.input_path}...")
        df = pd.read_parquet(self.input_path)
        
        # Check required columns
        required = ['x', 'y', 'url']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Input file missing columns. Required: {required}")

        # Normalize x and y to 0-1
        logging.info("Normalizing coordinates...")
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        df['x_norm'] = (df['x'] - x_min) / (x_max - x_min)
        df['y_norm'] = (df['y'] - y_min) / (y_max - y_min)
        
        # Convert to list of dicts for faster iteration
        # Renaming x_norm -> x for the web consumer
        points = df[['url', 'caption', 'x_norm', 'y_norm']].rename(
            columns={'x_norm': 'x', 'y_norm': 'y'}
        ).to_dict('records')
        
        logging.info(f"Loaded {len(points):,} points.")
        return points

    async def _download_image(self, session, url, output_path):
        """Downloads and resizes an image asynchronously."""
        if output_path.exists():
            return True
            
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return False
                
                img_data = await response.read()
                img = Image.open(BytesIO(img_data))
                
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Resize
                img.thumbnail((self.MAX_IMAGE_SIZE, self.MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
                
                # Save as WEBP for web optimization
                img.save(output_path, 'WEBP', quality=85)
                return True
        except Exception:
            return False

    def _find_densest_point(self, cell_points, bounds):
        """Finds the representative point in the densest area of the cell."""
        x_min, x_max, y_min, y_max = bounds
        
        if not cell_points:
            return None
        
        # If few points, just find closest to center
        if len(cell_points) <= 50:
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            return min(cell_points, key=lambda p: (p['x'] - center_x)**2 + (p['y'] - center_y)**2)
        
        # Mini-grid strategy for density
        mini_width = (x_max - x_min) / self.MINI_GRID_SIZE
        mini_height = (y_max - y_min) / self.MINI_GRID_SIZE
        
        mini_cells = {}
        for point in cell_points:
            mini_x = int((point['x'] - x_min) / mini_width)
            mini_y = int((point['y'] - y_min) / mini_height)
            # Clamp
            mini_x = min(mini_x, self.MINI_GRID_SIZE - 1)
            mini_y = min(mini_y, self.MINI_GRID_SIZE - 1)
            
            key = (mini_x, mini_y)
            if key not in mini_cells:
                mini_cells[key] = []
            mini_cells[key].append(point)
        
        # Find densest mini-cell
        densest_key = max(mini_cells.keys(), key=lambda k: len(mini_cells[k]))
        densest_points = mini_cells[densest_key]
        
        # Find center of that mini-cell
        mini_x, mini_y = densest_key
        mini_center_x = x_min + (mini_x + 0.5) * mini_width
        mini_center_y = y_min + (mini_y + 0.5) * mini_height
        
        return min(densest_points, key=lambda p: (p['x'] - mini_center_x)**2 + (p['y'] - mini_center_y)**2)

    async def _process_cell_256(self, session, cell_points, cell_x, cell_y):
        """Processes a single leaf cell (256x256)."""
        grid_size = 256
        cell_width = 1.0 / grid_size
        bounds = (
            cell_x * cell_width, (cell_x + 1) * cell_width,
            cell_y * cell_width, (cell_y + 1) * cell_width
        )
        
        best_point = self._find_densest_point(cell_points, bounds)
        if not best_point:
            return None
        
        # Sort neighbors by distance to best point
        sorted_points = sorted(
            cell_points,
            key=lambda p: (p['x'] - best_point['x'])**2 + (p['y'] - best_point['y'])**2
        )
        
        image_filename = f"{cell_x}_{cell_y}.webp"
        output_path = self.images_dir / "256" / image_filename
        
        # Try downloading best point image, if fails try next best
        img_success = False
        final_point = None
        
        for point in sorted_points[:10]: # Try top 10 candidates
            if await self._download_image(session, point['url'], output_path):
                img_success = True
                final_point = point
                break
        
        if not img_success:
            return None

        # Gather random examples for the UI list
        import random
        examples = sorted_points if len(sorted_points) <= self.EXAMPLES_PER_CELL else random.sample(sorted_points, self.EXAMPLES_PER_CELL)
        examples_clean = [{'url': p['url'], 'caption': p['caption']} for p in examples]

        return {
            'count': len(cell_points),
            'img': f"images/256/{image_filename}", # Relative path for web
            'url': final_point['url'],
            'caption': final_point['caption'],
            'examples': examples_clean
        }

    async def generate_grids(self):
        """Main execution flow."""
        points = self.load_and_normalize()
        
        # --- Level 256 (Base) ---
        logging.info("Processing Grid 256x256...")
        
        # Bucket points
        grid_256_buckets = {}
        for p in points:
            cx = min(int(p['x'] * 256), 255)
            cy = min(int(p['y'] * 256), 255)
            key = (cx, cy)
            if key not in grid_256_buckets:
                grid_256_buckets[key] = []
            grid_256_buckets[key].append(p)

        grid_256_data = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for (cx, cy), cell_points in grid_256_buckets.items():
                tasks.append(self._process_cell_256(session, cell_points, cx, cy))
            
            # Execute with progress bar
            results = await tqdm.gather(*tasks, desc="Downloading & Processing")
            
            for (cx, cy), res in zip(grid_256_buckets.keys(), results):
                if res:
                    grid_256_data[f"{cx},{cy}"] = res

        # --- Aggregate Levels 128 & 64 ---
        grid_128_data = self._aggregate_grid(grid_256_data, 128)
        grid_64_data = self._aggregate_grid(grid_128_data, 64)

        return grid_256_data, grid_128_data, grid_64_data

    def _aggregate_grid(self, lower_grid, target_size):
        """Aggregates a lower grid (e.g., 256) into a higher one (e.g., 128)."""
        logging.info(f"Aggregating to grid {target_size}x{target_size}...")
        upper_grid = {}
        
        for cx in range(target_size):
            for cy in range(target_size):
                # Find children (2x2 block from lower level)
                children = []
                for dx in range(2):
                    for dy in range(2):
                        child_key = f"{cx * 2 + dx},{cy * 2 + dy}"
                        if child_key in lower_grid:
                            children.append(lower_grid[child_key])
                
                if not children:
                    continue
                
                # Pick best child (highest count)
                best_child = max(children, key=lambda c: c['count'])
                total_count = sum(c['count'] for c in children)
                
                # Merge examples
                all_examples = []
                limit = self.EXAMPLES_PER_CELL
                
                for child in children:
                    take_n = math.ceil(limit / len(children))
                    all_examples.extend(child['examples'][:take_n])
                
                upper_grid[f"{cx},{cy}"] = {
                    'count': total_count,
                    'img': best_child['img'], # Reuse image from lower level
                    'url': best_child['url'],
                    'caption': best_child['caption'],
                    'examples': all_examples[:limit]
                }
        return upper_grid

    def save_json_split(self, data, filename_base):
        """Saves JSON, splitting if too large."""
        full_path = self.output_dir / f"{filename_base}.json"
        
        # Sort for consistency
        sorted_items = sorted(data.items())
        data_sorted = dict(sorted_items)
        
        # Serialize to check size
        json_str = json.dumps(data_sorted, separators=(',', ':'))
        size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
        
        if size_mb <= self.MAX_JSON_MB:
            with open(full_path, 'w') as f:
                f.write(json_str)
            logging.info(f"Saved {full_path.name} ({size_mb:.2f} MB)")
        else:
            logging.info(f"{full_path.name} too large ({size_mb:.2f} MB). Splitting...")
            parts = math.ceil(size_mb / self.MAX_JSON_MB)
            chunk_size = math.ceil(len(sorted_items) / parts)
            
            for i in range(parts):
                start = i * chunk_size
                end = start + chunk_size
                part_data = dict(sorted_items[start:end])
                
                part_path = self.output_dir / f"{filename_base}_part{i+1}.json"
                with open(part_path, 'w') as f:
                    json.dump(part_data, f, separators=(',', ':'))
                logging.info(f"  Saved part {i+1}: {part_path.name}")
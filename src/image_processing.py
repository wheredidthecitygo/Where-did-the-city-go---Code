import torch
import clip
import requests
import pandas as pd
import numpy as np
import gc
from PIL import Image
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClipProcessor:
    """
    Handles downloading images and extracting embeddings using OpenAI's CLIP model.
    """

    def __init__(self, model_name="ViT-L/14@336px", device=None):
        """
        Args:
            model_name (str): CLIP model variant.
            device (str): 'cuda' or 'cpu'. If None, detects automatically.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading CLIP model: {model_name} on {self.device}...")
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval() # Set to evaluation mode
        except Exception as e:
            logging.error(f"Failed to load CLIP: {e}")
            raise e

    def download_image(self, url, timeout=10):
        """Downloads an image and converts it to RGB PIL Image."""
        try:
            # User agent prevents some 403 errors
            headers = {'User-Agent': 'Mozilla/5.0 (Research/CityIdentity; +http://github.com)'}
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
        except Exception as e:
            # Silent fail for individual images is expected in large datasets
            return None

    def get_embedding(self, image):
        """Extracts the embedding vector from a PIL Image."""
        try:
            # Preprocess and add batch dimension
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # Move to CPU and flatten to 1D array
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            return None

    def process_directory(self, input_dir, output_dir, batch_save_freq=50):
        """
        Iterates over parquet files in input_dir, processes images, and saves to output_dir.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob("*.parquet"))
        if not files:
            logging.warning(f"No parquet files found in {input_path}")
            return

        logging.info(f"Found {len(files)} files to process.")

        for file_path in files:
            self._process_single_file(file_path, output_path)
            gc.collect() # Force garbage collection after each file

    def _process_single_file(self, file_path, output_path):
        """Internal method to process one parquet file."""
        file_name = file_path.name
        output_file = output_path / f"embeddings_{file_name}"

        if output_file.exists():
            logging.info(f"Skipping {file_name} (already processed).")
            return

        logging.info(f"Processing: {file_name}")
        
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            logging.error(f"Error reading {file_name}: {e}")
            return

        results = []
        valid_count = 0

        # Iterate rows
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {file_name}"):
            url = row.get('url')
            
            if pd.isna(url):
                continue

            # 1. Download
            image = self.download_image(url)
            if image is None:
                continue

            # 2. Embedding
            embedding = self.get_embedding(image)
            
            # Clean up image memory immediately
            del image
            
            if embedding is None:
                continue

            # 3. Construct Result
            # Convert row to dict and add embedding columns
            # Note: This creates columns e0, e1... e767
            result_row = row.to_dict()
            for i, val in enumerate(embedding):
                result_row[f'e{i}'] = val
            
            results.append(result_row)
            valid_count += 1

            # Periodic GC inside loop if file is huge
            if valid_count % 100 == 0:
                gc.collect()

        # Save results
        if results:
            df_results = pd.DataFrame(results)
            df_results.to_parquet(output_file, index=False)
            logging.info(f"Saved {valid_count} processed images to {output_file.name}")
        else:
            logging.warning(f"No valid images processed for {file_name}")

        del df, results
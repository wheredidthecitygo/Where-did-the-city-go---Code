import os
import duckdb
import pandas as pd
from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LaionDataLoader:
    """
    Handles the downloading and initial filtering of LAION-5B dataset parts.
    It downloads Parquet files, filters them by keyword using DuckDB, 
    saves the filtered version, and deletes the original massive file.
    """

    def __init__(self, output_dir, hf_token=None, dataset_subset="multi"):
        """
        Args:
            output_dir (str): Path to save the filtered parquet files.
            hf_token (str): Hugging Face API token.
            dataset_subset (str): Subset of LAION-5B to use ('multi', 'nolang', 'en').
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token = hf_token
        
        # Map simple names to full Repo IDs
        self.repos = {
            "multi": "laion/laion2B-multi",
            "nolang": "laion/laion1B-nolang",
            "en": "laion/laion2B-en"
        }
        
        if dataset_subset not in self.repos:
            raise ValueError(f"Dataset subset must be one of {list(self.repos.keys())}")
            
        self.repo_id = self.repos[dataset_subset]
        self.fs = HfFileSystem(token=self.token)

    def get_parquet_list(self):
        """
        Retrieves the list of .parquet files from the Hugging Face repository.
        Excludes .crc files or other metadata.
        """
        logging.info(f"Fetching file list from repository: {self.repo_id}...")
        
        # List all files in the repository
        files = self.fs.glob(f"{self.repo_id}/*.parquet")
        
        # Clean paths (remove repo_id prefix if present in glob output for download compatibility)
        # HfFileSystem usually returns "repo_id/filename". We need just the filename for hf_hub_download sometimes,
        # but here we just need the list.
        cleaned_files = [f.split('/')[-1] for f in files if not f.endswith('.crc')]
        
        logging.info(f"Found {len(cleaned_files)} parquet files.")
        return cleaned_files

    def filter_and_process(self, keywords, max_files=None):
        """
        Main pipeline: Download -> Filter (DuckDB) -> Save -> Delete Original.
        
        Args:
            keywords (list): List of strings to search in the caption (e.g., ["Madrid", "madrid"]).
            max_files (int, optional): Limit processing to N files (good for testing).
        """
        files = self.get_parquet_list()
        
        if max_files:
            files = files[:max_files]
            logging.info(f"Limit applied. Processing only first {max_files} files.")

        # Create SQL condition: caption ILIKE '%keyword1%' OR caption ILIKE '%keyword2%'
        # ILIKE is case insensitive in DuckDB
        sql_condition = " OR ".join([f"caption ILIKE '%{k}%'" for k in keywords])

        for filename in tqdm(files, desc="Processing dataset chunks"):
            
            output_filename = f"filtered_{filename}"
            output_path = self.output_dir / output_filename
            
            # Skip if already processed
            if output_path.exists():
                # logging.info(f"Skipping {filename}, already exists.")
                continue

            temp_path = self.output_dir / "temp_chunk.parquet"
            
            try:
                # 1. Download
                # We download to a temp specific path to avoid cluttering
                local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    repo_type="dataset",
                    token=self.token,
                    local_dir=self.output_dir,
                    local_dir_use_symlinks=False
                )
                
                # 2. Filter using DuckDB (Zero-copy, very fast)
                # We query the downloaded file directly
                query = f"""
                    SELECT * 
                    FROM read_parquet('{local_path}') 
                    WHERE {sql_condition}
                """
                
                df_filtered = duckdb.query(query).to_df()
                
                # 3. Save if matches found
                if not df_filtered.empty:
                    df_filtered.to_parquet(output_path, index=False)
                    logging.info(f"Found {len(df_filtered)} matches in {filename}. Saved.")
                
                # 4. Clean up (Delete the massive original file)
                if os.path.exists(local_path):
                    os.remove(local_path)
                    
            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                # Ensure cleanup even on error
                if os.path.exists(str(self.output_dir / filename)):
                    os.remove(str(self.output_dir / filename))
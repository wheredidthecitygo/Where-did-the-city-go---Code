import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetadataCleaner:
    """
    Filters metadata parquet files by removing rows that contain specific exclusion keywords.
    Useful for removing noise (e.g., filtering out 'football' from a city dataset).
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_dataset(self, exclude_keywords):
        """
        Iterates through all parquet files and removes rows containing any of the exclude_keywords.
        
        Args:
            exclude_keywords (list): List of strings. If a caption contains any of these, the row is dropped.
        """
        files = list(self.input_dir.glob("*.parquet"))
        
        if not files:
            logging.warning(f"No parquet files found in {self.input_dir}")
            return

        total_removed = 0
        total_rows = 0

        # Create regex pattern for faster filtering: "word1|word2|word3"
        # We escape special characters just in case
        import re
        if not exclude_keywords:
            logging.warning("No keywords provided. Just copying files...")
            pattern = None
        else:
            pattern = "|".join([re.escape(word.lower()) for word in exclude_keywords])

        for file_path in tqdm(files, desc="Cleaning metadata"):
            try:
                df = pd.read_parquet(file_path)
                original_count = len(df)
                total_rows += original_count

                if pattern:
                    # Filter: Keep rows that DO NOT contain the pattern
                    # case=False makes it case insensitive
                    mask = df['caption'].astype(str).str.contains(pattern, case=False, na=False)
                    df_clean = df[~mask] # The tilde (~) inverts the mask
                else:
                    df_clean = df

                removed_count = original_count - len(df_clean)
                total_removed += removed_count

                # Save only if there is data left
                if not df_clean.empty:
                    output_file = self.output_dir / file_path.name
                    df_clean.to_parquet(output_file, index=False)
                
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}")

        logging.info(f"--- CLEANING REPORT ---")
        logging.info(f"Total rows processed: {total_rows}")
        logging.info(f"Rows removed: {total_removed}")
        logging.info(f"Rows remaining: {total_rows - total_removed}")
        logging.info(f"Cleaned data saved to: {self.output_dir}")
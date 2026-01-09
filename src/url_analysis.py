import pandas as pd
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UrlAnalyzer:
    """
    Analyzes the source domains of the images in the dataset.
    Extracts the root domain (e.g., 'flickr.com') from URLs.
    """

    def __init__(self):
        self.domain_counts = Counter()

    def extract_domain(self, url):
        """Extracts the domain from a URL string robustly."""
        if not isinstance(url, str):
            return None
        
        try:
            # Quick cleanup
            if '://' not in url:
                url = 'http://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # Remove 'www.' prefix
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Remove port number if present
            if ':' in domain:
                domain = domain.split(':')[0]
                
            return domain
        except:
            return None

    def process_file(self, parquet_path):
        """Reads a parquet file and counts domains from the 'url' column."""
        try:
            # We only need the URL column
            df = pd.read_parquet(parquet_path, columns=['url'])
            
            if df.empty:
                return

            # Apply extraction
            domains = df['url'].apply(self.extract_domain).dropna()
            
            # Update counter
            self.domain_counts.update(domains)
            
        except Exception as e:
            logging.error(f"Error processing {parquet_path.name}: {e}")

    def save_results(self, output_dir, top_n=1000):
        """Saves the top N domains to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.domain_counts:
            logging.warning("No domains found to save.")
            return

        # Save Top N
        df_top = pd.DataFrame(
            self.domain_counts.most_common(top_n),
            columns=['domain', 'count']
        )
        
        output_file = output_dir / "top_domains.csv"
        df_top.to_csv(output_file, index=False)
        logging.info(f"Saved top {top_n} domains to {output_file}")
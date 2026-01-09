import re
import pandas as pd
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextAnalyzer:
    """
    Analyzes text data from LAION-5B captions.
    Calculates frequencies for words, bigrams, and trigrams, applying specific
    filtering rules for the city name.
    """

    # Combined Spanish and English stopwords
    DEFAULT_STOPWORDS = {
        # Spanish
        'de', 'la', 'el', 'en', 'a', 'y', 'que', 'es', 'por', 'un', 'para',
        'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'mas', 'pero', 'sus',
        'le', 'ya', 'o', 'este', 'sí', 'si', 'porque', 'esta', 'entre', 'cuando',
        'muy', 'sin', 'sobre', 'también', 'tambien', 'me', 'hasta', 'hay', 'donde',
        'han', 'quien', 'están', 'estan', 'desde', 'todo', 'nos', 'durante',
        'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante',
        'ellos', 'e', 'esto', 'mí', 'mi', 'antes', 'algunos', 'qué', 'que', 'unos', 'yo',
        'del', 'las', 'los', 'se', 'está', 'esta', 'son', 'ser', 'fue', 'ha',
        # English
        'the', 'of', 'and', 'in', 'to', 'for', 'is', 'on', 'that', 'by',
        'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from',
        'at', 'as', 'your', 'all', 'have', 'new', 'more', 'an', 'was', 'we',
        'will', 'can', 'us', 'about', 'if', 'my', 'has', 'but', 'our', 'one',
        'other', 'do', 'may', 'down', 'side', 'been', 'now', 'find', 'any', 'these', 'each',
        'their', 'there', 'which', 'she', 'him', 'his', 'her', 'would', 'make', 'them',
        'its', 'into', 'out', 'up', 'so', 'what', 'than', 'some', 'could', 'only', 'between',
    }

    def __init__(self, city_name):
        """
        Args:
            city_name (str): The name of the city (to exclude from 1-gram counts).
        """
        self.city_name = city_name.lower().strip()
        self.stopwords = self.DEFAULT_STOPWORDS
        self.regex_clean = re.compile(r'[^\w\s]|[\d]') # Remove punctuation and numbers
        
        # Initialize counters
        self.word_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

    def clean_and_tokenize(self, text):
        """Cleans text and returns a list of valid words."""
        if not isinstance(text, str):
            return []
        
        text = text.lower()
        text = self.regex_clean.sub(' ', text)
        words = text.split()
        # Filter short words and stopwords
        return [w for w in words if len(w) > 2 and w not in self.stopwords]

    def generate_ngrams(self, words, n):
        """Generates tuples of n-grams."""
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]

    def process_file(self, parquet_path):
        """
        Reads a parquet file and updates the counters.
        """
        try:
            # Load only caption column
            df = pd.read_parquet(parquet_path, columns=['caption'])
            
            for caption in df['caption'].dropna():
                words = self.clean_and_tokenize(caption)
                if not words:
                    continue

                # 1. Word Analysis (Exclude City Name)
                # We filter out the city name here because it's obvious it exists in the dataset
                filtered_words = [w for w in words if w != self.city_name]
                self.word_counts.update(filtered_words)

                # 2. N-Gram Analysis (Include City Name)
                # We keep the city name here to find context like "madrid streets", "downtown madrid"
                self.bigram_counts.update(self.generate_ngrams(words, 2))
                self.trigram_counts.update(self.generate_ngrams(words, 3))
                
        except Exception as e:
            logging.error(f"Error processing file {parquet_path}: {e}")

    def save_results(self, output_dir, top_n=500):
        """Saves the top N results to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Helper to save counter
        def save_counter(counter, filename, col_names):
            if not counter:
                return
            
            data = counter.most_common(top_n)
            # Flatten tuples for n-grams if necessary
            formatted_data = []
            for item, freq in data:
                text = " ".join(item) if isinstance(item, tuple) else item
                formatted_data.append({col_names[0]: text, col_names[1]: freq})
            
            pd.DataFrame(formatted_data).to_csv(output_dir / filename, index=False)
            logging.info(f"Saved {filename}")

        save_counter(self.word_counts, "top_words.csv", ["word", "frequency"])
        save_counter(self.bigram_counts, "top_bigrams.csv", ["bigram", "frequency"])
        save_counter(self.trigram_counts, "top_trigrams.csv", ["trigram", "frequency"])
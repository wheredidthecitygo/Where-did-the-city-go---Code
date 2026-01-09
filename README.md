# Where did the city go?

This repository contains a pipeline to research the digital identity of cities by analyzing the RE-LAION-5B dataset. The tool allows researchers to download, filter, analyze, and visualize massive amounts of image-text data associated with specific city names.

The workflow creates a bridge between large-scale AI datasets and qualitative research workshops by exporting visualized data to a web interface and the Miro collaboration platform. Pipeline Architecture

## Pipeline


1. **Data Acquisition (01)**: Downloads metadata from Hugging Face and filters rows by city keyword.

2. **Text Analysis (02):** Analyzes N-grams to understand the vocabulary context.

3. **Metadata Cleaning (03):** Negative filtering to remove noise (e.g., removing "football" from "Madrid").

4. **URL Analysis (04):** Analyzes the source domains of the images to detect biases.

5. **Image Embedding (05):** Downloads images temporarily and extracts CLIP embeddings.

6. **Dimensionality Reduction (06):** Projects high-dimensional embeddings (768D) into 2D space using UMAP.

7. **Web Export (07):** Generates a hierarchical grid of images and JSON files for web visualization.

8. **Collaborative Workshop (08):** Uploads the visualized map to Miro for participatory research.


---


## Installation

1. Clone the repository:

```
git clone <repository_url> 
cd city-identity-explorer
```

2. Install dependencies: Make sure you have Python installed, then run: code Bash

```
pip install -r requirements.txt
```

## Usage Step 1: Data Acquisition

Download the dataset metadata and filter rows containing your city's name.

```
python 01_download_and_filter.py  
--token "hf_YourTokenHere"  
--keywords "Madrid" "madrid"  
--subset multi
```

Options:

```
--token: (Required) Your Hugging Face API token.

--keywords: (Required) Space-separated words to search for.

--subset: Which LAION dataset to use (multi, en, or nolang). Defaults to multi.

--limit: Process only N files (e.g., --limit 5). Use for testing.
```

## Step 2: Text Analysis

Analyze the vocabulary, bigrams, and trigrams. This excludes the city name from single-word counts but keeps it for N-grams (e.g., "downtown Madrid").

```
python 02_analyze_text.py --city "Madrid"
```

Output: Creates data/analysis/ containing CSV files with top words, bigrams, and trigrams. 
## Step 3: Clean Metadata (Negative Filtering)

Remove unwanted topics found in Step 2 (e.g., "football" photos when searching for "Madrid").

```
python 03_clean_metadata.py --exclude "football" "soccer" "jersey" "stadium"
```

Output: Creates data/metadata_cleaned/ containing only the relevant rows. 
## Step 4: URL & Domain Analysis

Identify the source domains of the images (e.g., stock photo sites, news agencies) to understand dataset bias.

```
python 04_analyze_urls.py
```

Output: Creates data/analysis/top_domains.csv. 
## Step 5: Image Processing (CLIP Embeddings)

Download images and extract their "digital meaning" (embeddings) using the CLIP model. Note: This process is time-consuming. A GPU is recommended.

```
python 05_process_images.py
```


Options:

```
--input_dir: Source directory (default: data/metadata_cleaned).

--model: CLIP model version (default: ViT-L/14@336px).
```

## Step 6: Dimensionality Reduction (UMAP)

Project the 768-dimensional embeddings into a 2D map. Optimization: Automatically uses NVIDIA GPU (cuml) if available.

```
python 06_reduce_dimensionality.py --neighbors 25 --dist 0.1
```

Output: Creates data/projection/umap_projection.parquet (Metadata + X/Y coordinates). 
## Step 7: Web Export & Visualization

Prepare data for the web interface. This creates a hierarchical grid (levels 64, 128, 256), selects representative images, and downloads thumbnails.

```
python 07_export_for_web.py
```

Output: Creates data/export/ containing:

```
images/256/: Folder with thumbnails.

grid_64.json, grid_128.json, grid_256.json: Frontend data files.
```

## Step 8: Upload to Miro

Upload the map to a Miro board. Images in dense clusters appear larger, creating a "visual heatmap".

```
python 08_upload_to_miro.py  
--token "YourMiroToken"  
--board "YourBoardID"  
--base_width 400
```

Options:

```
--limit: Upload only N images (useful for testing).

--base_width: Width in pixels for the densest images.

--spacing: Distance between grid cells on the board.
```
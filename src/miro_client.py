import requests
import json
import logging
import time
from pathlib import Path
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MiroClient:
    """
    Handles uploading images and creating widgets on a Miro board.
    """

    def __init__(self, access_token, board_id):
        self.base_url = f"https://api.miro.com/v2/boards/{board_id}"
        self.headers = {
            'Authorization': f'Bearer {access_token}',
            'Accept': 'application/json'
        }

    def upload_image_from_file(self, file_path, x, y, width, title=None):
        """
        Uploads a local image file to the board.
        """
        url = f"{self.base_url}/images"
        
        try:
            # Prepare image data
            with open(file_path, 'rb') as f:
                img_data = f.read()

            # Create JSON metadata
            data_json = {
                "position": {"x": x, "y": y, "origin": "center"},
                "geometry": {"width": int(width)}
            }
            if title:
                data_json["data"] = {"title": title}

            # Multipart upload
            files = {
                'resource': ('image.webp', img_data, 'image/webp'),
                'data': (None, json.dumps(data_json), 'application/json')
            }
            
            # Request
            response = requests.post(url, headers=self.headers, files=files)
            
            # Rate limiting handling
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                logging.warning(f"Rate limited. Waiting {retry_after}s...")
                time.sleep(retry_after)
                return self.upload_image_from_file(file_path, x, y, width, title)

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logging.error(f"Failed to upload {file_path}: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                logging.error(f"API Response: {response.text}")
            return None

    def create_caption_text(self, text, x, y, width):
        """Creates a text widget below the image."""
        url = f"{self.base_url}/texts"
        
        payload = {
            "data": {
                "content": f"<p>{text}</p>"
            },
            "position": {
                "x": x,
                "y": y,
                "origin": "center"
            },
            "geometry": {
                "width": width
            },
            "style": {
                "fontSize": 10,
                "textAlign": "center"
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Failed to create text: {e}")
            return None
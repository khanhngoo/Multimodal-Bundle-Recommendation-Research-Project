#!/usr/bin/env python3
"""
Image Download System for MM Attribute Extraction
Downloads images based on item_id_map.json and item_data.txt
Creates metadata file for tracking and VLM processing
"""

import json
import csv
import os
import sys
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import logging

# Configuration
MAX_WORKERS = 10  # Number of parallel downloads
BATCH_SIZE = 1000  # Log progress every N items
TIMEOUT = 30  # Request timeout in seconds
RETRY_COUNT = 3  # Number of retries for failed downloads

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ImageDownloader:
    def __init__(self, base_dir='./'):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)
        
        self.item_id_map_path = self.base_dir / 'item_id_map.json'
        self.item_data_path = self.base_dir / 'item_data.txt'
        self.metadata_path = self.base_dir / 'image_metadata.csv'
        
        self.stats = {
            'total': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0
        }
    
    def load_item_id_map(self):
        """Load the item ID to index mapping"""
        logger.info(f"Loading item ID map from {self.item_id_map_path}")
        with open(self.item_id_map_path, 'r') as f:
            item_id_map = json.load(f)
        logger.info(f"Loaded {len(item_id_map)} item IDs")
        return item_id_map
    
    def parse_item_data(self, item_id_map):
        """Parse item_data.txt and build a lookup table"""
        logger.info(f"Parsing item data from {self.item_data_path}")
        item_data = {}
        
        with open(self.item_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 50000 == 0:
                    logger.info(f"Processed {line_num} lines...")
                
                try:
                    parts = line.strip().split(',', 3)  # Split into max 4 parts
                    if len(parts) >= 3:
                        item_id = parts[0].strip()
                        image_url = parts[2].strip()
                        description = parts[3].strip() if len(parts) > 3 else ''
                        
                        # Only keep items that are in our item_id_map
                        if item_id in item_id_map:
                            index = item_id_map[item_id]
                            item_data[index] = {
                                'item_id': item_id,
                                'image_url': image_url,
                                'description': description
                            }
                except Exception as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
        
        logger.info(f"Found {len(item_data)} items with valid data")
        return item_data
    
    def get_image_extension(self, url):
        """Extract file extension from URL"""
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check for common image extensions
        for ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
            if ext in path:
                return ext
        
        # Default to .jpg if no extension found
        return '.jpg'
    
    def download_image(self, index, item_id, image_url, description):
        """Download a single image with retry logic"""
        # Handle protocol-relative URLs (starting with //)
        if image_url.startswith('//'):
            image_url = 'https:' + image_url
        
        ext = self.get_image_extension(image_url)
        image_path = self.images_dir / f"{index}{ext}"
        
        # Skip if already exists
        if image_path.exists():
            self.stats['skipped'] += 1
            return {
                'index': index,
                'item_id': item_id,
                'image_path': str(image_path),
                'image_url': image_url,
                'description': description,
                'status': 'skipped'
            }
        
        # Try to download with retries
        for attempt in range(RETRY_COUNT):
            try:
                response = requests.get(
                    image_url,
                    timeout=TIMEOUT,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                response.raise_for_status()
                
                # Validate image
                img = Image.open(BytesIO(response.content))
                img.verify()  # Check if image is valid
                
                # Re-open and save (verify() closes the file)
                img = Image.open(BytesIO(response.content))
                img.save(image_path)
                
                self.stats['downloaded'] += 1
                return {
                    'index': index,
                    'item_id': item_id,
                    'image_path': str(image_path),
                    'image_url': image_url,
                    'description': description,
                    'status': 'success',
                    'width': img.width,
                    'height': img.height
                }
                
            except Exception as e:
                if attempt == RETRY_COUNT - 1:
                    logger.error(f"Failed to download {index} after {RETRY_COUNT} attempts: {e}")
                    self.stats['failed'] += 1
                    return {
                        'index': index,
                        'item_id': item_id,
                        'image_path': '',
                        'image_url': image_url,
                        'description': description,
                        'status': 'failed',
                        'error': str(e)
                    }
                else:
                    time.sleep(1)  # Wait before retry
        
        return None
    
    def create_metadata_csv(self, results):
        """Create metadata CSV file from download results"""
        logger.info(f"Creating metadata file: {self.metadata_path}")
        
        with open(self.metadata_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['index', 'original_item_id', 'image_path', 'image_url', 
                         'description', 'status', 'width', 'height', 'error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in sorted(results, key=lambda x: x['index']):
                writer.writerow({
                    'index': result['index'],
                    'original_item_id': result['item_id'],
                    'image_path': result['image_path'],
                    'image_url': result['image_url'],
                    'description': result.get('description', ''),
                    'status': result['status'],
                    'width': result.get('width', ''),
                    'height': result.get('height', ''),
                    'error': result.get('error', '')
                })
        
        logger.info(f"Metadata file created: {self.metadata_path}")
    
    def run(self):
        """Main execution method"""
        start_time = time.time()
        
        # Load data
        item_id_map = self.load_item_id_map()
        item_data = self.parse_item_data(item_id_map)
        
        self.stats['total'] = len(item_data)
        logger.info(f"Starting download of {self.stats['total']} images...")
        logger.info(f"Using {MAX_WORKERS} parallel workers")
        
        results = []
        
        # Download images in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(
                    self.download_image,
                    index,
                    data['item_id'],
                    data['image_url'],
                    data['description']
                ): index
                for index, data in item_data.items()
            }
            
            # Process completed downloads
            completed = 0
            for future in as_completed(future_to_item):
                result = future.result()
                if result:
                    results.append(result)
                
                completed += 1
                if completed % BATCH_SIZE == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    remaining = (self.stats['total'] - completed) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {completed}/{self.stats['total']} "
                        f"({completed/self.stats['total']*100:.1f}%) - "
                        f"Downloaded: {self.stats['downloaded']}, "
                        f"Skipped: {self.stats['skipped']}, "
                        f"Failed: {self.stats['failed']} - "
                        f"ETA: {remaining/60:.1f} min"
                    )
        
        # Create metadata file
        self.create_metadata_csv(results)
        
        # Final statistics
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Download Complete!")
        logger.info(f"Total items: {self.stats['total']}")
        logger.info(f"Successfully downloaded: {self.stats['downloaded']}")
        logger.info(f"Skipped (already exists): {self.stats['skipped']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total time: {elapsed/60:.2f} minutes")
        logger.info(f"Average rate: {self.stats['total']/elapsed:.2f} items/sec")
        logger.info("=" * 60)
        logger.info(f"Images saved to: {self.images_dir}")
        logger.info(f"Metadata saved to: {self.metadata_path}")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Image Download System for VLM Attribute Extraction")
    print("=" * 60)
    print()
    
    # Determine the script directory
    script_dir = Path(__file__).parent
    
    downloader = ImageDownloader(base_dir=script_dir)
    
    try:
        downloader.run()
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()


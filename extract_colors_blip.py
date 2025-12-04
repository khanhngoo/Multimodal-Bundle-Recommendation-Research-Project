#!/usr/bin/env python3
"""
Main Color Extraction using BLIP VLM
Extracts the main color from images using BLIP's Visual Question Answering capability
Integrates with existing image metadata
"""

import json
import csv
import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering
from tqdm import tqdm
import logging
from collections import Counter

# Configuration
BATCH_LOG_INTERVAL = 100  # Log progress every N images
QUESTION = "What is the main color in this image?"  # Question to ask BLIP

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('color_extraction_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ColorExtractor:
    def __init__(self, base_dir='./'):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / 'images'
        self.metadata_path = self.base_dir / 'image_metadata.csv'
        self.output_path = self.base_dir / 'image_colors.json'
        self.output_csv_path = self.base_dir / 'image_colors.csv'
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.processor = None
        self.model = None
        
        self.stats = {
            'total': 0,
            'processed': 0,
            'failed': 0
        }
    
    def load_model(self):
        """Load BLIP VQA model"""
        model_name = "Salesforce/blip-vqa-base"
        logger.info(f"Loading BLIP model: {model_name}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            logger.info("Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def extract_color(self, image_path, question=QUESTION):
        """
        Extract the main color from an image using BLIP VQA.
        
        Args:
            image_path: Path to the image file
            question: The question to ask about the image
        
        Returns:
            The predicted main color as a string, or None if failed
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            inputs = self.processor(
                images=image, 
                text=question, 
                return_tensors="pt"
            ).to(
                self.device, 
                torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=20)
            
            # Decode the answer
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return answer.strip()
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None
    
    def get_image_files(self):
        """Get all image files from the images directory"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
        
        return sorted(image_files)
    
    def load_existing_metadata(self):
        """Load existing image metadata if available"""
        if not self.metadata_path.exists():
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            return {}
        
        logger.info(f"Loading metadata from: {self.metadata_path}")
        metadata = {}
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    index = row.get('index', '')
                    if index:
                        metadata[index] = row
            
            logger.info(f"Loaded metadata for {len(metadata)} images")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
        
        return metadata
    
    def process_images(self):
        """Process all images and extract main colors"""
        start_time = time.time()
        
        # Get image files
        image_files = self.get_image_files()
        self.stats['total'] = len(image_files)
        
        if self.stats['total'] == 0:
            logger.error(f"No images found in {self.images_dir}")
            return {}
        
        logger.info(f"Found {self.stats['total']} images to process")
        
        # Load existing metadata
        metadata = self.load_existing_metadata()
        
        # Dictionary to store results
        color_results = {}
        
        # Process each image with progress bar
        for image_path in tqdm(image_files, desc="Extracting colors"):
            image_id = image_path.stem  # Get filename without extension
            
            # Extract main color
            main_color = self.extract_color(image_path)
            
            if main_color:
                # Get metadata for this image if available
                meta = metadata.get(image_id, {})
                
                color_results[image_id] = {
                    "image_path": str(image_path),
                    "main_color": main_color,
                    "original_item_id": meta.get('original_item_id', ''),
                    "description": meta.get('description', ''),
                    "status": "success"
                }
                self.stats['processed'] += 1
            else:
                color_results[image_id] = {
                    "image_path": str(image_path),
                    "main_color": "",
                    "status": "failed"
                }
                self.stats['failed'] += 1
            
            # Periodic logging
            completed = self.stats['processed'] + self.stats['failed']
            if completed % BATCH_LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (self.stats['total'] - completed) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {completed}/{self.stats['total']} "
                    f"({completed/self.stats['total']*100:.1f}%) - "
                    f"Processed: {self.stats['processed']}, "
                    f"Failed: {self.stats['failed']} - "
                    f"ETA: {remaining/60:.1f} min"
                )
        
        return color_results
    
    def analyze_colors(self, color_results):
        """Analyze color distribution"""
        logger.info("\n" + "=" * 60)
        logger.info("Color Distribution Analysis")
        logger.info("=" * 60)
        
        # Extract colors (case-insensitive)
        colors = [
            data['main_color'].lower() 
            for data in color_results.values() 
            if data.get('main_color')
        ]
        
        if not colors:
            logger.warning("No colors to analyze")
            return
        
        # Count occurrences
        color_counts = Counter(colors)
        
        # Display top colors
        logger.info("\nTop 15 Most Common Colors:")
        logger.info("-" * 60)
        for color, count in color_counts.most_common(15):
            percentage = (count / len(colors)) * 100
            logger.info(f"{color:20s}: {count:5d} ({percentage:5.1f}%)")
        
        logger.info(f"\nTotal unique colors: {len(color_counts)}")
    
    def save_results(self, color_results):
        """Save results to JSON and CSV files"""
        # Save to JSON
        logger.info(f"\nSaving results to {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(color_results, f, indent=2, ensure_ascii=False)
        
        # Save to CSV
        logger.info(f"Saving results to {self.output_csv_path}")
        with open(self.output_csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'image_id', 
                'main_color', 
                'image_path', 
                'original_item_id', 
                'description',
                'status'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for image_id, data in sorted(color_results.items()):
                writer.writerow({
                    'image_id': image_id,
                    'main_color': data.get('main_color', ''),
                    'image_path': data.get('image_path', ''),
                    'original_item_id': data.get('original_item_id', ''),
                    'description': data.get('description', ''),
                    'status': data.get('status', 'unknown')
                })
        
        logger.info(f"Results saved successfully!")
    
    def run(self):
        """Main execution method"""
        logger.info("=" * 60)
        logger.info("Main Color Extraction using BLIP VLM")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Load model
        if not self.load_model():
            logger.error("Failed to load model. Exiting.")
            return False
        
        # Process images
        logger.info("\nStarting color extraction...")
        color_results = self.process_images()
        
        if not color_results:
            logger.error("No results to save. Exiting.")
            return False
        
        # Analyze results
        self.analyze_colors(color_results)
        
        # Save results
        self.save_results(color_results)
        
        # Final statistics
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info("Extraction Complete!")
        logger.info(f"Total images: {self.stats['total']}")
        logger.info(f"Successfully processed: {self.stats['processed']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total time: {elapsed/60:.2f} minutes")
        logger.info(f"Average rate: {self.stats['total']/elapsed:.2f} images/sec")
        logger.info("=" * 60)
        logger.info(f"JSON output: {self.output_path}")
        logger.info(f"CSV output: {self.output_csv_path}")
        logger.info("=" * 60)
        
        return True


def main():
    """Main entry point"""
    print("=" * 60)
    print("Main Color Extraction using BLIP VLM")
    print("=" * 60)
    print()
    
    # Determine the script directory
    script_dir = Path(__file__).parent
    
    extractor = ColorExtractor(base_dir=script_dir)
    
    try:
        success = extractor.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nExtraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()


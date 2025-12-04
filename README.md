# VLM Attribute/Color Extraction for Multimodal Bundle Recommendation

## Overview
This project focuses on extracting visual attributes (specifically main colors) from fashion items using Vision-Language Models (VLMs). It involves downloading a specific subset of images from the POG iFashion dataset and processing them using the BLIP model to identify dominant colors.

## How to Run

1.  **Install Dependencies and data**:
    First, ensure you have the required packages installed:
    ```bash
    pip install -r requirements.txt
    ```
    Then download item_data.txt dataset, copy to the project folder and unzip via this link: https://drive.google.com/file/d/1UsAu9XxdTQUDopPgXpmA0M6FRbzDn5V0/view?usp=sharing

2.  **Download Images**:
    If the image data in the `images/` folder is missing or incomplete, you can download them again using:
    ```bash
    python download_images.py
    ```

3.  **Run the Process**:
    You can run the full extraction process in two ways:
    
    -   **Python Script**: Run the script directly to extract colors.
        ```bash
        python extract_colors_blip.py
        ```
    -   **Jupyter Notebook**: Use the notebook to view the full sequential process and visualizations.
        ```bash
        jupyter notebook extract_main_color.ipynb
        ```



## File Descriptions

-   **`item_data.txt`**: This file is from the [POG iFashion dataset](https://github.com/wenyuer/POG). It contains **1,973,199** lines of data. Each line includes the item ID, an unknown field, image link, and a description (in Chinese).
-   **`item_id_map.json`**: Contains the mapping of item IDs that are used in the [MultiCBR training set](https://github.com/HappyPointer/MultiCBR). This filters the massive POG dataset down to the specific items relevant for this project.
-   **`download_images.py`**: A script used to download images. It matches item IDs from `item_id_map.json` to entries in `item_data.txt` and downloads the corresponding images using the links provided.
-   **`image_metadata.csv`**: A CSV file generated during the download process containing metadata for each image. Columns include:
    -   `index`: The index from the ID map.
    -   `original_item_id`: The original ID from the POG dataset.
    -   `image_path`: Local path to the downloaded image.
    -   `image_url`: Source URL of the image.
    -   `description`: Item description.
    -   `status`: Download status (success, skipped, failed).
    -   `width`/`height`: Image dimensions.
    -   `error`: Error message if download failed.
-   **`extract_colors_blip.py`**: The main script that uses the BLIP VQA model to ask "What is the main color in this image?" for each downloaded image.

## How It Works

1.  **Data Matching**: The system first reads `item_id_map.json` to get the list of target items. It then scans the massive `item_data.txt` to find the corresponding image URLs and descriptions for these IDs.
2.  **Image Downloading**: Using `download_images.py`, the images are downloaded to the `images/` directory. Metadata is logged to `image_metadata.csv`.
3.  **Color Extraction**: The `extract_colors_blip.py` script loads the BLIP VQA model (Salesforce/blip-vqa-base). It iterates through the downloaded images and prompts the model with the question "What is the main color in this image?". The model's textual output is saved as the extracted color.

## Statistics

Based on the download logs (`download_log.txt`) and metadata (`image_metadata.csv`):

-   **Total Items**: 16,500
-   **Successfully Downloaded**: 15,883
-   **Skipped (Already Existed)**: 608
-   **Failed**: 9

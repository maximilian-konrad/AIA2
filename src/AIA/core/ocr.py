"""
OCR module for extracting text from images using pytesseract and identifying language with langdetect.
"""

# Imports
from tqdm import tqdm
from PIL import Image
import pytesseract
import os
import platform
import traceback
import yaml
from langdetect import detect, LangDetectException

def get_ocr_text(df_images):
    """
    Extract text from images using pytesseract and identify the language.
    
    :param df_images: DataFrame containing image filenames
    :return: DataFrame with OCR results (ocrHasText, ocrText, and ocrLanguage columns)
    """

    # Load the full configuration directly from params.yaml
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'params.yaml'), 'r') as file:
            full_config = yaml.safe_load(file)
        print("Loaded params.yaml directly")
    except Exception as e:
        print(f"Error loading params.yaml directly: {e}")
        full_config = {}

    # Check if the system is Windows
    if platform.system() == "Windows":
        try:
            tesseract_path = full_config.get("features", {}).get("extract_text_ocr", {}).get("parameters", {}).get("windows_path_to_tesseract")
            if tesseract_path:
                print(f"Windows system detected. Using Tesseract path: {tesseract_path}")
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
        except Exception as e:
            print(f"Error accessing Tesseract configuration: {e}")
            traceback.print_exc()

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    # Initialize new columns
    df['ocrHasText'] = False
    df['ocrText'] = ""
    df['ocrLanguage'] = ""
    
    # Iterate over all image paths
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        try:
            # Open the image with PIL
            image = Image.open(image_path)
            
            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(image).strip()
            
            # Determine if text was found
            has_text = len(text) > 0
            
            # Store OCR text result in DataFrame
            df.loc[idx, 'ocrHasText'] = has_text
            df.loc[idx, 'ocrText'] = text
            
        except Exception as e:
            # Handle errors gracefully for OCR text
            df.loc[idx, 'ocrHasText'] = "ERROR"
            df.loc[idx, 'ocrText'] = f"Error: {str(e)}"
        
        # Detect language if text is present
        if has_text:
            try:
                language = detect(text)
            except LangDetectException:
                language = "unknown"
            except Exception as e:
                df.loc[idx, 'ocrHasText'] = "ERROR"
                language = f"Error: {str(e)}"
        else:
            language = ""
        
        # Store language result in DataFrame
        df.loc[idx, 'ocrLanguage'] = language
    
    return df
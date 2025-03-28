"""
OCR module for extracting text from images using pytesseract and identifying language with langdetect.
"""

# Imports
from tqdm import tqdm
from PIL import Image
import pytesseract
import platform
import traceback
from langdetect import detect, LangDetectException
import os

def get_ocr_text(self, df_images):
    """
    Extract text from images using pytesseract and identify the language.
    
    :param self: AIA object
    :param df_images: DataFrame containing image filenames
    :return: DataFrame with OCR results (ocrHasText, ocrText, and ocrLanguage columns)
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Initialize tesseract for Windows
        if platform.system() == "Windows":
            try:
                tesseract_path = self.config.get("features", {}).get("get_ocr_text", {}).get("parameters", {}).get("windows_path_to_tesseract")
                if tesseract_path:
                    print(f"Windows system detected. Using Tesseract path: {tesseract_path}")
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
            except Exception as e:
                error = f"Error configuring Tesseract: {str(e)}"
                print(error)
                traceback.print_exc()
                return df

        # Initialize new columns
        df['ocrHasText'] = False
        df['ocrText'] = ""
        df['ocrLanguage'] = ""
        
        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_ocr'] = "File not found"
                    continue

                # Open the image with PIL
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_ocr'] = f"Image load error: {str(e)}"
                    continue

                # Perform OCR using pytesseract
                try:
                    text = pytesseract.image_to_string(image).strip()
                    has_text = len(text) > 0
                    
                    # Store OCR text result
                    df.loc[idx, 'ocrHasText'] = has_text
                    df.loc[idx, 'ocrText'] = text

                    # Detect language if text is present
                    if has_text:
                        try:
                            language = detect(text)
                        except LangDetectException:
                            language = "unknown"
                        except Exception as e:
                            language = "error"
                            df.loc[idx, 'error_ocr_lang'] = f"Language detection error: {str(e)}"
                    else:
                        language = ""

                    # Store language result
                    df.loc[idx, 'ocrLanguage'] = language

                except Exception as e:
                    error = f"OCR processing error: {str(e)}"
                    print(f"Error processing OCR for {image_path}: {error}")
                    df.loc[idx, 'error_ocr'] = error
                    df.loc[idx, 'ocrHasText'] = False
                    df.loc[idx, 'ocrText'] = ""
                    df.loc[idx, 'ocrLanguage'] = ""
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_ocr'] = error
                df.loc[idx, 'ocrHasText'] = False
                df.loc[idx, 'ocrText'] = ""
                df.loc[idx, 'ocrLanguage'] = ""
                continue

        return df

    except Exception as e:
        print(f"Error in OCR setup: {str(e)}")
        return df
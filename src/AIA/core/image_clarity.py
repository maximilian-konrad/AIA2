import cv2
import numpy as np
from tqdm import tqdm
import os

def calculate_image_clarity(self, df_images):
    """
    Calculate the clarity score for each image by measuring the proportion of high-brightness pixels.
    The score represents the percentage of pixels with brightness values between 0.7 and 1.0
    (after normalizing to 0-1 range). Higher scores indicate brighter, clearer images.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added 'clarity' column containing scores between 0 and 1
            where 1 means all pixels are in the high-brightness range
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"Warning: File not found: {image_path}")
                continue

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Failed to load image: {image_path}")
                continue
            
            # Convert to grayscale to calculate brightness
            if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
                brightness_values = image / 255.0  # Scale to 0–1
            else:  # Color image
                # Convert to grayscale to get brightness
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                brightness_values = grayscale_image / 255.0  # Scale to 0–1

            # Calculate the portion of pixels with brightness in the range 0.7–1
            clarity_mask = (brightness_values >= 0.7) & (brightness_values <= 1.0)
            clarity_score = np.sum(clarity_mask) / brightness_values.size  # Proportion of high-brightness pixels
            
            df.loc[idx, 'clarity'] = clarity_score

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_clarity'] = error
            continue

    return df
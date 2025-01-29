import cv2
import numpy as np
from tqdm import tqdm

def calculate_contrast_of_brightness(df_images):
    """
    Calculate the contrast of brightness for each image by computing the standard deviation
    of pixel intensities. For color images, the RGB channels are averaged first.
    Higher values indicate more contrast in brightness levels across the image.

    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added 'contrast_of_brightness' column containing the standard deviation
            of pixel intensities
    :raises ValueError: If an image file cannot be found or loaded
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found. Please check the file path.")
        
        # Check if the image is grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
            intensity_values = image
        else:  # Color image, use average method for pixel intensity
            intensity_values = np.mean(image, axis=2)
        
        # Calculate the standard deviation of pixel intensity (contrast of brightness)
        contrast_of_brightness = np.std(intensity_values)
        
        df.loc[idx, 'contrast_of_brightness'] = contrast_of_brightness

    return df
import cv2
import numpy as np
from tqdm import tqdm

def calculate_hue_proportions(df_images):
    """
    Calculate the proportions of warm and cold hues in images using HSV color space.
    Warm hues are defined as those outside 30-110° in HSV (reds, oranges, yellows),
    while cold hues are those within 30-110° (greens, blues).

    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added columns:
            - hues_warm: ratio of warm to cold pixels (inf if no cold pixels)
            - hues_cold: ratio of cold to warm pixels (inf if no warm pixels)
    :raises ValueError: If an image file cannot be found or loaded
    :note: Higher ratios indicate stronger presence of that temperature range
           (warm or cold) in the image
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found. Please check the file path.")
        
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract the hue channel
        hue_channel = hsv_image[:, :, 0]
        
        # Define masks for cold and warm hues
        cold_hue_mask = (hue_channel >= 30) & (hue_channel <= 110)
        warm_hue_mask = ~cold_hue_mask  # Invert cold hue mask to get warm hue mask
        
        # Count the number of pixels in each range
        cold_pixel_count = np.sum(cold_hue_mask)
        warm_pixel_count = np.sum(warm_hue_mask)
        
        # Calculate proportions
        if cold_pixel_count == 0:
            warm_to_cold_ratio = float('inf')  # Avoid division by zero
        else:
            warm_to_cold_ratio = warm_pixel_count / cold_pixel_count

        if warm_pixel_count == 0:
            cold_to_warm_ratio = float('inf')  # Avoid division by zero
        else:
            cold_to_warm_ratio = cold_pixel_count / warm_pixel_count
        
        df.loc[idx, 'hues_warm'] = warm_to_cold_ratio
        df.loc[idx, 'hues_cold'] = cold_to_warm_ratio

    return df
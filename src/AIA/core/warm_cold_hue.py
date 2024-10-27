import cv2
import numpy as np

def calculate_hue_proportions(image_path):
    features = {}
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
    
    features['Warm Hues'] = warm_to_cold_ratio
    features['Cold Hues'] = cold_to_warm_ratio

    return features
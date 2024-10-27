import cv2
import numpy as np

def calculate_image_clarity(image_path):
    features = {}
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Please check the file path.")
    
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
    
    features['Image Clarity'] = clarity_score

    return features
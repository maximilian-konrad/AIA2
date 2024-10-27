import cv2
import numpy as np

def calculate_contrast_of_brightness(image_path):
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
    
    return contrast_of_brightness

# Example usage
image_path = '2.jpg'  # Replace with your image path
contrast = calculate_contrast_of_brightness(image_path)
print(f"Contrast of Brightness: {contrast}")



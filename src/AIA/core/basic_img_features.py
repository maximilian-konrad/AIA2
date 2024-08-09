import cv2
from PIL import Image
import numpy as np
import os
from scipy.stats import entropy
from skimage import measure

def extract_basic_image_features(image_path):
    """
    Extract basic features from the image.

    :param image_path: Path to the image file.
    :return: A dictionary containing basic image features.
    """
    features = {}

    # Load the image using cv2
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Image dimensions
    features['height'] = image.shape[0]
    features['width'] = image.shape[1]

    # Image size in kB
    features['size_kb'] = os.path.getsize(image_path) / 1024

    # RGB means
    features['r_mean'] = np.mean(image_rgb[:, :, 0])
    features['g_mean'] = np.mean(image_rgb[:, :, 1])
    features['b_mean'] = np.mean(image_rgb[:, :, 2])

    # Convert RGB to HSV using cv2
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    features['hue_mean'] = np.mean(hsv_image[:, :, 0])
    features['saturation_mean'] = np.mean(hsv_image[:, :, 1])
    features['brightness_mean'] = np.mean(hsv_image[:, :, 2])

    # Convert to Greyscale using cv2
    grayscale_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    features['greyscale_mean'] = np.mean(grayscale_image)
    
    # Calculate Shannon Entropy
    features['shannon_entropy'] = measure.shannon_entropy(grayscale_image)

    return features


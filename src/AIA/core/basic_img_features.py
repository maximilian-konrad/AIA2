import cv2
from PIL import Image
import numpy as np
import os
from scipy.stats import entropy
from skimage import measure
from tqdm import tqdm

def extract_basic_image_features(df_images):
    """
    Extract basic features from the images and add them as columns to the DataFrame.
    Features include: dimensions (height, width), file size, color statistics (RGB means),
    HSV color space metrics, grayscale mean, and Shannon entropy.

    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns:
        - height: image height in pixels
        - width: image width in pixels
        - size_kb: file size in kilobytes
        - r_mean, g_mean, b_mean: average values for RGB channels
        - hue_mean, saturation_mean, brightness_mean: average values in HSV color space
        - greyscale_mean: average intensity in grayscale
        - shannon_entropy: measure of image complexity/information content
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        # Load the image using cv2
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Image dimensions
        df.loc[idx, 'height'] = image.shape[0]
        df.loc[idx, 'width'] = image.shape[1]

        # Image size in kB
        df.loc[idx, 'size_kb'] = os.path.getsize(image_path) / 1024

        # RGB means
        df.loc[idx, 'r_mean'] = np.mean(image_rgb[:, :, 0])
        df.loc[idx, 'g_mean'] = np.mean(image_rgb[:, :, 1])
        df.loc[idx, 'b_mean'] = np.mean(image_rgb[:, :, 2])

        # Convert RGB to HSV using cv2
        hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        df.loc[idx, 'hue_mean'] = np.mean(hsv_image[:, :, 0])
        df.loc[idx, 'saturation_mean'] = np.mean(hsv_image[:, :, 1])
        df.loc[idx, 'brightness_mean'] = np.mean(hsv_image[:, :, 2])

        # Convert to Greyscale using cv2
        grayscale_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        df.loc[idx, 'greyscale_mean'] = np.mean(grayscale_image)
        
        # Calculate Shannon Entropy
        df.loc[idx, 'shannon_entropy'] = measure.shannon_entropy(grayscale_image)

    return df


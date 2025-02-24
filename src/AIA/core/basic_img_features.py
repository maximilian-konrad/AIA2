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
        - hueMean, saturationMean, brightness_mean: average values in HSV color space
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
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # File specific features
        df.loc[idx, 'fileName'] = os.path.splitext(os.path.basename(image_path))[0]
        df.loc[idx, 'fileType'] = os.path.splitext(image_path)[1][1:].lower()
        df.loc[idx, 'fileSize'] = os.path.getsize(image_path) / 1024
        df.loc[idx, 'fileCreationTime'] = os.path.getctime(image_path)

        # # Image meta data
        # df.loc[idx, 'lens'] = image.get('Lens', 'NA')
        # df.loc[idx, 'focalLength'] = image.get('FocalLength', 'NA')
        # df.loc[idx, 'aperture'] = image.get('Aperture', 'NA')
        # df.loc[idx, 'exposureTime'] = image.get('ExposureTime', 'NA')
        # df.loc[idx, 'ISO'] = image.get('ISO', 'NA')
        # df.loc[idx, 'shutterSpeed'] = image.get('ShutterSpeedValue', 'NA')
        # df.loc[idx, 'whiteBalance'] = image.get('WhiteBalance', 'NA')
        # df.loc[idx, 'flash'] = image.get('Flash', 'NA')
        # df.loc[idx, 'meteringMode'] = image.get('MeteringMode', 'NA')
        # df.loc[idx, 'exposureProgram'] = image.get('ExposureProgram', 'NA')

        # Image dimensions
        df.loc[idx, 'height'] = image.shape[0]
        df.loc[idx, 'width'] = image.shape[1]
        df.loc[idx, 'aspectRatio'] = df.loc[idx, 'width'] / df.loc[idx, 'height']

        # Color channels
        df.loc[idx, 'rMean'] = np.mean(image_rgb[:, :, 0])
        df.loc[idx, 'rStd'] = np.std(image_rgb[:, :, 0])
        df.loc[idx, 'gMean'] = np.mean(image_rgb[:, :, 1])
        df.loc[idx, 'gStd'] = np.std(image_rgb[:, :, 1])
        df.loc[idx, 'bMean'] = np.mean(image_rgb[:, :, 2])
        df.loc[idx, 'bStd'] = np.std(image_rgb[:, :, 2])

        # HSV channels
        df.loc[idx, 'hueMean'] = np.mean(image_hsv[:, :, 0])
        df.loc[idx, 'hueStd'] = np.std(image_hsv[:, :, 0])
        # df.loc[idx, 'saturationMean'] = np.mean(image_hsv[:, :, 1])
        # df.loc[idx, 'saturationStd'] = np.std(image_hsv[:, :, 1])
        # df.loc[idx, 'brightnessMean'] = np.mean(image_hsv[:, :, 2])
        # df.loc[idx, 'brightnessStd'] = np.std(image_hsv[:, :, 2])

        # # Grayscale
        df.loc[idx, 'greyscaleMean'] = np.mean(image_gray)
        df.loc[idx, 'greyscaleStd'] = np.std(image_gray)

        # # Shannon entropy
        df.loc[idx, 'shannonEntropy'] = measure.shannon_entropy(image_gray)

    return df

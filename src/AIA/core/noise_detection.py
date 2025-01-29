import cv2
import numpy as np
import math
from scipy.signal import convolve2d
from tqdm import tqdm

def estimate_noise(df_images):
    """
    Estimate the noise level in images using a Laplacian kernel convolution method.
    The function calculates a sigma value that represents the amount of noise,
    where values above 10 indicate significant noise presence in the image.

    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added 'noise' column containing the sigma values
            where higher values (>10) indicate noisier images
    :note: Uses grayscale conversion and 3x3 Laplacian kernel for noise estimation
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
         img = cv2.imread(image_path) 
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

         H, W = gray.shape

         M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

         sigma = np.sum(np.sum(np.absolute(convolve2d(gray, M))))
         sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

         # Value more than 10 indicates a noisy image
         df.loc[idx, 'noise']  = sigma

    return df

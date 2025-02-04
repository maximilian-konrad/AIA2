"""
Here, all image features are implemented according to the specifications of Lan Luo's Management Science paper: Can Consumer-Posted Photos Serve as a Leading Indicator of Restaurant Survival? Evidence from Yelp
DOI: https://doi.org/10.1287/mnsc.2022.4359

Implementation details are saved in an Excel accessible via: https://tumde-my.sharepoint.com/:x:/g/personal/maximilian_konrad_tum_de/Eaxb2FNmiq5IrlhbbCYYcJ0BTIWDtR9A_HrEtSwsgXkKpg?e=7Q0tzM

Each feature will be an individual function.
1. Color Features:
   - brightness
   - saturation
   - contrast
   - clarity
   - warmHue
   - colorfulness

2. Composition Features:
   - diagonalDominance
   - ruleOfThirds
   - physicalVisualBalance
   - colorVisualBalance

3. Figure-ground Relationship Features:
   - sizeDifference
   - colorSifference
   - textureDifference
   - depthOfField
"""


import cv2
from tqdm import tqdm
import numpy as np


def get_color_features(df_images):
    """
    Computes color-related features for images, including:
      - brightness: Mean brightness, normalized to [0,1]
      - saturation: Mean saturation, normalized to [0,1]
      - contrast: Brightness contrast, defined as the standard deviation of the V channel in HSV, normalized to [0,1]
      - clarity: Proportion of pixels in the V channel (normalized) that have values greater than 0.7
      - warmHue: Proportion of warm-hued pixels (H values in HSV <30 or >150)
      - colorfulness: Colorfulness score based on Hasler and Suesstrunk (2003), normalized to [0,1]

    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Store raw colorfulness values for later normalization
    raw_colorfulness_list = []

    # Initialize new columns with NaN
    df['brightness'] = np.nan
    df['saturation'] = np.nan
    df['contrast'] = np.nan
    df['clarity'] = np.nan
    df['warmHue'] = np.nan
    df['colorfulness'] = np.nan

    # Iterate over all image paths
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            # Skip this image if loading fails (leave NaN values)
            continue

        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # Compute brightness: Mean of the V channel, normalized to [0,1]
        brightness_val = np.mean(V / 255.0)

        # Compute saturation: Mean of the S channel, normalized to [0,1]
        saturation_val = np.mean(S / 255.0)

        # Compute contrast: Standard deviation of the V channel, normalized to [0,1]
        # Maximum possible std(V) is 0.5 after normalization, so we multiply by 2
        contrast_val = np.std(V / 255.0) * 2

        # Compute clarity: Proportion of pixels in V channel (normalized) that are greater than 0.7
        clarity_val = np.mean((V / 255.0) > 0.7)

        # Compute warm hue: Proportion of warm-colored pixels (H < 70 or H > 160)
        warm_mask = (H < 70) | (H > 160)
        warmHue_val = np.sum(warm_mask) / warm_mask.size

        # Compute colorfulness based on Hasler and Suesstrunk (2003)
        # Extract BGR channels and convert to float
        B = image[:, :, 0].astype("float")
        G = image[:, :, 1].astype("float")
        R = image[:, :, 2].astype("float")
        # Compute RG and YB components
        rg = R - G
        yb = 0.5 * (R + G) - B
        # Compute standard deviations and means
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        # Compute raw colorfulness metric
        colorfulness_raw = np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        # Store the raw value for later normalization
        raw_colorfulness_list.append(colorfulness_raw)

        # Store computed features in DataFrame
        df.loc[idx, 'brightness'] = brightness_val
        df.loc[idx, 'saturation'] = saturation_val
        df.loc[idx, 'contrast'] = contrast_val
        df.loc[idx, 'clarity'] = clarity_val
        df.loc[idx, 'warmHue'] = warmHue_val
        df.loc[idx, 'colorfulness'] = colorfulness_raw

    # Normalize colorfulness across all images to [0,1]
    raw_colorfulness = df['colorfulness'].astype('float').values
    min_c = np.nanmin(raw_colorfulness)
    max_c = np.nanmax(raw_colorfulness)
    if max_c - min_c == 0:
        # If all images have the same colorfulness value, set all to 0
        df['colorfulness'] = 0.0
    else:
        df['colorfulness'] = (raw_colorfulness - min_c) / (max_c - min_c)

    return df
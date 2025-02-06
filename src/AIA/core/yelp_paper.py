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


def get_composition_features(df_images):
    """
    Computes composition features for each image according to the specifications.

    For each image, the following features are computed:
      - diagonalDominance: The normalized (inverted) minimum distance from the salient center to the two image diagonals.
      - ruleOfThirds: The normalized (inverted) minimum distance from the salient center to the four intersections of a 3×3 grid.
      - physicalVisualBalance:
            • physicalVisualBalance_vertical: 1 minus the normalized vertical distance between the salient center and the image center.
            • physicalVisualBalance_horizontal: 1 minus the normalized horizontal distance between the salient center and the image center.
            • physicalVisualBalance (average): The average of the vertical and horizontal scores.
      - colorVisualBalance:
            • colorVisualBalance_vertical: 1 minus the normalized average Euclidean color distance between top and bottom symmetric pixels.
            • colorVisualBalance_horizontal: 1 minus the normalized average Euclidean color distance between left and right symmetric pixels.
            • colorVisualBalance (average): The average of the vertical and horizontal scores.

    :param df_images: DataFrame containing a 'filename' column with paths to image files.
    :return: DataFrame with added feature columns:
             - diagonalDominance
             - ruleOfThirds
             - physicalVisualBalance_vertical, physicalVisualBalance_horizontal, physicalVisualBalance (average)
             - colorVisualBalance_vertical, colorVisualBalance_horizontal, colorVisualBalance (average)
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Initialize composition feature columns
    df['diagonalDominance'] = np.nan
    df['ruleOfThirds'] = np.nan
    df['physicalVisualBalanceVertical'] = np.nan
    df['physicalVisualBalanceHorizontal'] = np.nan
    df['physicalVisualBalanceMean'] = np.nan
    df['colorVisualBalanceVertical'] = np.nan
    df['colorVisualBalanceHorizontal'] = np.nan
    df['colorVisualBalanceMean'] = np.nan

    # Create a saliency detector using OpenCV's StaticSaliencySpectralResidual
    saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

    # Maximum possible color difference in RGB space (Euclidean distance)
    max_color_distance = np.sqrt(255 ** 2 + 255 ** 2 + 255 ** 2)  # ≈441.67

    # Iterate over each image in the DataFrame
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        # Load the image using cv2
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Get image dimensions (height and width)
        H, W = image.shape[:2]

        # Compute saliency map
        success, saliencyMap = saliency_detector.computeSaliency(image)
        if not success or saliencyMap is None:
            saliencyMap = np.ones((H, W), dtype="float32")
        else:
            saliencyMap = saliencyMap.squeeze()
            if saliencyMap.ndim != 2:
                saliencyMap = saliencyMap[:, :, 0]

        # Compute the weighted (salient) center
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        total_saliency = np.sum(saliencyMap)
        if total_saliency > 0:
            cx = np.sum(X * saliencyMap) / total_saliency
            cy = np.sum(Y * saliencyMap) / total_saliency
        else:
            cx, cy = W / 2.0, H / 2.0

        # 1. Diagonal Dominance
        d1 = np.abs(H * cx - W * cy) / np.sqrt(W ** 2 + H ** 2)  # Distance to diagonal from top-left to bottom-right
        d2 = np.abs(H * cx + W * cy - H * W) / np.sqrt(
            W ** 2 + H ** 2)  # Distance to diagonal from top-right to bottom-left
        d_min = min(d1, d2)
        D_max = (W * H) / np.sqrt(W ** 2 + H ** 2)  # Maximum possible distance to a diagonal
        diagonalDominance = 1 - (d_min / D_max)
        diagonalDominance = np.clip(diagonalDominance, 0, 1)

        # 2. Rule of Thirds
        intersections = [(W / 3.0, H / 3.0), (2 * W / 3.0, H / 3.0),
                         (W / 3.0, 2 * H / 3.0), (2 * W / 3.0, 2 * H / 3.0)]
        distances = [np.sqrt((cx - x) ** 2 + (cy - y) ** 2) for (x, y) in intersections]
        min_distance = min(distances)
        max_distance = np.sqrt(W ** 2 + H ** 2) / 3.0  # Normalization factor
        ruleOfThirds = 1 - (min_distance / max_distance)
        ruleOfThirds = np.clip(ruleOfThirds, 0, 1)

        # 3. Physical Visual Balance
        vertical_balance = 1 - (np.abs(cy - H / 2.0) / (H / 2.0)) if H > 0 else 0
        horizontal_balance = 1 - (np.abs(cx - W / 2.0) / (W / 2.0)) if W > 0 else 0
        physicalVisualBalance_average = (vertical_balance + horizontal_balance) / 2.0

        # 4. Color Visual Balance
        # Vertical color balance: compare top and bottom halves
        h_half = H // 2
        if h_half > 0:
            top_half = image[0:h_half, :, :].astype("float32")
            bottom_half = image[H - h_half:H, :, :].astype("float32")
            diff_v = np.sqrt(np.sum((top_half - bottom_half) ** 2, axis=2))
            avg_diff_v = np.mean(diff_v)
            vertical_color_balance = 1 - (avg_diff_v / max_color_distance)
            vertical_color_balance = np.clip(vertical_color_balance, 0, 1)
        else:
            vertical_color_balance = 1

        # Horizontal color balance: compare left and right halves (mirror the right half)
        w_half = W // 2
        if w_half > 0:
            left_half = image[:, 0:w_half, :].astype("float32")
            right_half = image[:, W - w_half:W, :].astype("float32")[:, ::-1, :]
            diff_h = np.sqrt(np.sum((left_half - right_half) ** 2, axis=2))
            avg_diff_h = np.mean(diff_h)
            horizontal_color_balance = 1 - (avg_diff_h / max_color_distance)
            horizontal_color_balance = np.clip(horizontal_color_balance, 0, 1)
        else:
            horizontal_color_balance = 1

        colorVisualBalance_average = (vertical_color_balance + horizontal_color_balance) / 2.0

        # Save features into DataFrame
        df.loc[idx, 'diagonalDominance'] = diagonalDominance
        df.loc[idx, 'ruleOfThirds'] = ruleOfThirds
        df.loc[idx, 'physicalVisualBalanceVertical'] = vertical_balance
        df.loc[idx, 'physicalVisualBalanceHorizontal'] = horizontal_balance
        df.loc[idx, 'physicalVisualBalanceMean'] = physicalVisualBalance_average
        df.loc[idx, 'colorVisualBalanceVertical'] = vertical_color_balance
        df.loc[idx, 'colorVisualBalanceHorizontal'] = horizontal_color_balance
        df.loc[idx, 'colorVisualBalanceMean'] = colorVisualBalance_average

    return df
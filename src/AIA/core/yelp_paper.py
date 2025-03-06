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
   - colorDifference
   - textureDifference
   - depthOfField
"""

import cv2
from tqdm import tqdm
import numpy as np
import pywt

def get_color_features(self, df_images):
    """
    Computes color-related features for images, including:
      - brightness: Mean brightness, normalized to [0,1]
      - saturation: Mean saturation, normalized to [0,1]
      - contrast: Brightness contrast, defined as the standard deviation of the V channel in HSV, normalized to [0,1]
      - clarity: Proportion of pixels in the V channel (normalized) that have values greater than 0.7
      - warmHue: Proportion of warm-hued pixels (H values in HSV <30 or >150)
      - colorfulness: Colorfulness score based on Hasler and Suesstrunk (2003), normalized to [0,1]

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Get parameters
    color_config = self.config.get("get_color_features", {})
    clarity_threshold = color_config.get("clarity_threshold", 0.7)
    warmHue_threshold_lower = color_config.get("warmHue_threshold_lower", 70)
    warmHue_threshold_upper = color_config.get("warmHue_threshold_upper", 160)
    
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Store raw colorfulness values for later normalization
    raw_colorfulness_list = []

    # Initialize new columns with NaN
    df['brightnessMean'] = np.nan
    df['saturationMean'] = np.nan
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
        clarity_val = np.mean((V / 255.0) > clarity_threshold)

        # Compute warm hue: Proportion of warm-colored pixels (H < 70 or H > 160)
        warm_mask = (H < warmHue_threshold_lower) | (H > warmHue_threshold_upper)
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
        df.loc[idx, 'brightnessMean'] = brightness_val
        df.loc[idx, 'saturationMean'] = saturation_val
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

def get_composition_features(self, df_images):
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

    :param self: AIA object
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

def get_figure_ground_relationship_features(self, df_images):
    """
    Computes figure-ground relationship features for each image according to the specifications.
    
    For each image, the following features are computed:
      - sizeDifference: The absolute difference between the number of figure pixels and background pixels,
                        normalized by the total number of pixels.
      - colorDifference: The Euclidean distance between the average RGB vector of the figure and that of the background,
                         normalized to [0, 1] (using 441.67 as the maximum difference).
      - textureDifference: The absolute difference between the edge densities (using Canny) of the figure and background,
                           normalized to [0, 1].
      - depthOfField: Computed for each HSV dimension (hue, saturation, value) as follows:
            • The image is divided into 16 equal regions.
            • For each HSV channel, the high-frequency (detail) coefficients are computed using a Daubechies wavelet (pywt.dwt2).
            • The score is defined as the sum of absolute detail coefficients in the center four regions divided by the sum
              of absolute detail coefficients over all 16 regions.
            A higher score indicates a lower depth of field.
    
    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files.
    :return: DataFrame with added feature columns:
             - sizeDifference
             - colorDifference
             - textureDifference
             - depthOfFieldHue, depthOfFieldSaturation, depthOfFieldValue
    """

    # Get parameters
    fg_config = self.config.get("get_figure_ground_relationship_features", {})
    saliency_threshold = fg_config.get("saliency_threshold", 0.5)
    canny_edge_low_threshold = fg_config.get("canny_edge_low_threshold", 100)
    canny_edge_high_threshold = fg_config.get("canny_edge_high_threshold", 200)
    
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    # Initialize feature columns using camelCase naming
    df['sizeDifference'] = np.nan
    df['colorDifference'] = np.nan
    df['textureDifference'] = np.nan
    df['depthOfFieldHue'] = np.nan
    df['depthOfFieldSaturation'] = np.nan
    df['depthOfFieldValue'] = np.nan

    # Create a saliency detector for figure-ground segmentation
    saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    
    # Maximum possible color difference in RGB (Euclidean distance)
    max_color_distance = np.sqrt(255**2 + 255**2 + 255**2)  # ≈441.67

    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        image = cv2.imread(image_path)
        if image is None:
            continue

        H, W = image.shape[:2]
        total_pixels = H * W
        
        # Compute saliency map and threshold to segment figure (salient) vs background
        success, saliencyMap = saliency_detector.computeSaliency(image)
        if not success or saliencyMap is None:
            saliencyMap = np.ones((H, W), dtype="float32")
        else:
            saliencyMap = saliencyMap.squeeze()
            if saliencyMap.ndim != 2:
                saliencyMap = saliencyMap[:, :, 0]
        # Normalize saliency map to [0,1] if not already
        saliencyMap = cv2.normalize(saliencyMap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        _, figureMask = cv2.threshold(saliencyMap, saliency_threshold, 1, cv2.THRESH_BINARY)
        figureMask = figureMask.astype(np.uint8)
        backgroundMask = 1 - figureMask
        
        # 1. Size Difference
        figure_pixels = np.sum(figureMask, dtype=np.int64)
        background_pixels = np.sum(backgroundMask, dtype=np.int64)
        sizeDifference = np.abs(figure_pixels - background_pixels) / float(total_pixels)
        
        # 2. Color Difference
        # Compute average RGB for figure and background; if a region is empty, use zeros.
        figure_pixels_indices = np.where(figureMask == 1)
        background_pixels_indices = np.where(backgroundMask == 1)
        if figure_pixels > 0:
            avgRGB_figure = np.mean(image[figure_pixels_indices], axis=0)
        else:
            avgRGB_figure = np.zeros(3)
        if background_pixels > 0:
            avgRGB_background = np.mean(image[background_pixels_indices], axis=0)
        else:
            avgRGB_background = np.zeros(3)
        color_diff = np.linalg.norm(avgRGB_figure - avgRGB_background)
        colorDifference = np.clip(color_diff / max_color_distance, 0, 1)
        
        # 3. Texture Difference
        # Use Canny edge detection on grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_edge_low_threshold, canny_edge_high_threshold)
        # Compute edge density for figure and background
        if figure_pixels > 0:
            edge_density_figure = np.sum(edges[figure_pixels_indices] > 0) / float(figure_pixels)
        else:
            edge_density_figure = 0
        if background_pixels > 0:
            edge_density_background = np.sum(edges[background_pixels_indices] > 0) / float(background_pixels)
        else:
            edge_density_background = 0
        textureDifference = np.clip(np.abs(edge_density_figure - edge_density_background), 0, 1)
        
        # 4. Depth of Field (for each HSV channel)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
        # Divide the image into a 4x4 grid (16 regions)
        grid_rows, grid_cols = 4, 4
        region_h = H // grid_rows
        region_w = W // grid_cols
        
        # Define indices for center 4 regions (positions (1,1), (1,2), (2,1), (2,2) in 0-indexing)
        center_regions = [(1,1), (1,2), (2,1), (2,2)]
        
        # Initialize sums for each channel: hue, saturation, value
        total_detail = {'hue': 0, 'sat': 0, 'val': 0}
        center_detail = {'hue': 0, 'sat': 0, 'val': 0}
        
        # For each region in the 4x4 grid
        for i in range(grid_rows):
            for j in range(grid_cols):
                y0 = i * region_h
                x0 = j * region_w
                # Make sure to include remaining pixels for the last row/column
                y1 = H if i == grid_rows - 1 else (i+1)*region_h
                x1 = W if j == grid_cols - 1 else (j+1)*region_w
                
                region = hsv[y0:y1, x0:x1, :]
                # For each channel, compute the sum of absolute high-frequency coefficients using a Daubechies wavelet.
                # We perform a single-level 2D DWT.
                for idx_channel, key in enumerate(['hue', 'sat', 'val']):
                    coeffs2 = pywt.dwt2(region[:, :, idx_channel], 'db1')
                    # coeffs2 returns (LL, (LH, HL, HH)); we sum absolute values of high-frequency coefficients.
                    (_, (LH, HL, HH)) = coeffs2
                    detail_sum = np.sum(np.abs(LH)) + np.sum(np.abs(HL)) + np.sum(np.abs(HH))
                    total_detail[key] += detail_sum
                    if (i, j) in center_regions:
                        center_detail[key] += detail_sum
        
        # For each channel, compute depth-of-field score as center_detail/total_detail.
        # Avoid division by zero.
        dof_hue = center_detail['hue'] / total_detail['hue'] if total_detail['hue'] != 0 else 0
        dof_sat = center_detail['sat'] / total_detail['sat'] if total_detail['sat'] != 0 else 0
        dof_val = center_detail['val'] / total_detail['val'] if total_detail['val'] != 0 else 0

        # Save features into DataFrame (clipped to [0, 1] where applicable)
        df.loc[idx, 'sizeDifference'] = sizeDifference
        df.loc[idx, 'colorDifference'] = colorDifference
        df.loc[idx, 'textureDifference'] = textureDifference
        df.loc[idx, 'depthOfFieldHue'] = np.clip(dof_hue, 0, 1)
        df.loc[idx, 'depthOfFieldSaturation'] = np.clip(dof_sat, 0, 1)
        df.loc[idx, 'depthOfFieldValue'] = np.clip(dof_val, 0, 1)
        
    return df
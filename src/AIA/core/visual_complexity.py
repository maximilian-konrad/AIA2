import numpy as np
import cv2
from skimage import io, filters
from skimage.measure import regionprops, label
from scipy.optimize import curve_fit
from tqdm import tqdm
import os

def visual_complexity(self, df_images):
    """
    Calculate the visual complexity of each image by counting the number of regions in the binary image.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added 'visualComplexity' column containing the number of regions
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters from config
        threshold = self.config.get('visual_complexity', {}).get('threshold', 25000)
        
        # Initialize column
        df['visualComplexity'] = np.nan
        
        for idx, input_image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(input_image_path):
                    print(f"Warning: File not found: {input_image_path}")
                    df.loc[idx, 'error_complexity'] = "File not found"
                    continue

                # Load image
                try:
                    img = io.imread(input_image_path)
                except Exception as e:
                    print(f"Warning: Failed to load image: {input_image_path}")
                    df.loc[idx, 'error_complexity'] = f"Image load error: {str(e)}"
                    continue

                # Process image
                try:
                    # Convert to grayscale if needed
                    if len(img.shape) > 2:
                        img = np.mean(img, axis=2).astype(np.uint8)

                    # Calculate adaptive threshold
                    thresh = filters.threshold_local(img, block_size=35)
                    binary_img = img > thresh
                    rp_tot = binary_img.shape[0] * binary_img.shape[1]
                    labeled_img = label(binary_img)
                    regions = regionprops(labeled_img)
                    threshold_val = rp_tot / threshold
                    r_spt = sum(1 for region in regions if region.area > threshold_val)

                    df.loc[idx, 'visualComplexity'] = r_spt

                except Exception as e:
                    error = f"Complexity calculation error: {str(e)}"
                    print(f"Error calculating complexity for {input_image_path}: {error}")
                    df.loc[idx, 'error_complexity'] = error
                    continue

            except Exception as e:
                error = f"Error processing {input_image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_complexity'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in visual complexity setup: {str(e)}")
        return df

def self_similarity(self, df_images):
    """
    This function calculates the self-similarity of each image using the power spectrum of the Fourier transform.
    The closer the slope of the power spectrum to -2, the more self-similar the image is.
    The slope is mapped to a similarity score between 0 and 1 using a Gaussian function.
    A score of 1 indicates perfect self-similarity and a score of 0 indicates no self-similarity.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Initialize column
        df['selfSimilarity'] = np.nan

        # Define linear fit function for curve fitting
        def linear_fit(x, a, b):
            return a * x + b

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_similarity'] = "File not found"
                    continue

                # Load image
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Failed to load image: {image_path}")
                        df.loc[idx, 'error_similarity'] = "Image load failed"
                        continue
                except Exception as e:
                    print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_similarity'] = f"Image load error: {str(e)}"
                    continue

                # Calculate self-similarity
                try:
                    # Compute Fourier transform and power spectrum
                    f_transform = np.fft.fft2(img)
                    f_transform_shifted = np.fft.fftshift(f_transform)
                    power_spectrum = np.abs(f_transform_shifted) ** 2

                    # Calculate radial profile
                    h, w = power_spectrum.shape
                    y, x = np.indices((h, w))
                    center = (h // 2, w // 2)
                    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(np.int32)
                    radial_mean = np.bincount(r.ravel(), weights=power_spectrum.ravel()) / np.bincount(r.ravel())

                    # Prepare data for fitting
                    valid = (radial_mean > 0) & (np.arange(len(radial_mean)) > 0)
                    freqs = np.arange(len(radial_mean))[valid]
                    power = radial_mean[valid]
                    log_freqs = np.log(freqs)
                    log_power = np.log(power)

                    # Fit curve and calculate similarity score
                    slope, intercept = curve_fit(linear_fit, log_freqs, log_power)[0]
                    similarity_score = np.exp(-0.5 * abs(slope + 2))

                    df.loc[idx, 'selfSimilarity'] = similarity_score

                except Exception as e:
                    error = f"Similarity calculation error: {str(e)}"
                    print(f"Error calculating similarity for {image_path}: {error}")
                    df.loc[idx, 'error_similarity'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_similarity'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in self-similarity setup: {str(e)}")
        return df
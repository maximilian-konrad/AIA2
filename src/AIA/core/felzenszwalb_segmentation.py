"""
Image segmentation using Felzenszwalb algorithm from scikit-image.
"""

# Imports
from skimage.segmentation import felzenszwalb
from skimage.io import imread
from skimage.color import label2rgb
from tqdm import tqdm
import numpy as np
import os

def felzenszwalb_segmentation(self, df_images): 
    """
    This function performs Felzenszwalb segmentation on the images in the input DataFrame.
    The function adds a new column to the DataFrame with the segmented images in RGB format.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Get parameters from config if available
        scale = self.config.get("felzenszwalb_segmentation", {}).get("scale", 100)
        sigma = self.config.get("felzenszwalb_segmentation", {}).get("sigma", 0.5)
        min_size = self.config.get("felzenszwalb_segmentation", {}).get("min_size", 50)

        # Initialize new columns
        df['felzenszwalbSegmentationRGB'] = None

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    df.loc[idx, 'error_segmentation'] = "File not found"
                    continue

                # Load image
                try:
                    image = imread(image_path)
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    df.loc[idx, 'error_segmentation'] = f"Image load error: {str(e)}"
                    continue

                # Perform segmentation
                try:
                    segments = felzenszwalb(
                        image, 
                        scale=scale, 
                        sigma=sigma, 
                        min_size=min_size
                    )
                    segmented_image = label2rgb(segments, image, kind='avg')
                    df.loc[idx, 'felzenszwalbSegmentationRGB'] = segmented_image

                except Exception as e:
                    error = f"Segmentation error: {str(e)}"
                    print(f"Error performing segmentation for {image_path}: {error}")
                    df.loc[idx, 'error_segmentation'] = error
                    continue

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_segmentation'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in segmentation setup: {str(e)}")
        return df




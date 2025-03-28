"""
Add general comments here.
"""

# Imports
from tqdm import tqdm
import numpy as np
import os

def perceive_quality(self, df_images):
    """
    Describe what the function does here.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Initialize new columns with NaN
        df['featureName'] = np.nan

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    print(f"Warning: File not found: {image_path}")
                    continue

                # TODO: Implement feature computation here
                feature_value = 0  # Replace with actual computation

                # Store computed features in DataFrame
                df.loc[idx, 'featureName'] = feature_value

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_quality'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in quality perception setup: {str(e)}")
        return df

def perceive_realism(self, df_images):
    """
    Describe what the function does here.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Initialize new columns with NaN
        df['featureName'] = np.nan

        # Iterate over all image paths
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    print(f"Warning: File not found: {image_path}")
                    continue

                # TODO: Implement feature computation here
                feature_value = 0  # Replace with actual computation

                # Store computed features in DataFrame
                df.loc[idx, 'featureName'] = feature_value

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_realism'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in realism perception setup: {str(e)}")
        return df


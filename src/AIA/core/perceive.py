"""
Add general comments here.
"""

# Imports
from tqdm import tqdm
import numpy as np

def perceive_quality(self, df_images): # TODO: Implement this function
    """
    Describe what the function does here.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Initialize new columns with NaN
    df['featureName'] = np.nan

    # Iterate over all image paths
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        # Implement feature computation here.

        # Store computed features in DataFrame
        df.loc[idx, 'featureName'] = feature_value

    return df

def perceive_realism(self, df_images): # TODO: Implement this function
    """
    Describe what the function does here.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Initialize new columns with NaN
    df['featureName'] = np.nan

    # Iterate over all image paths
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        # Implement feature computation here.

        # Store computed features in DataFrame
        df.loc[idx, 'featureName'] = feature_value

    return df


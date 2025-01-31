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
    Dolo Lorem Ipsum, Dolo Lorem Ipsum, Dolo Lorem Ipsum, Dolo Lorem Ipsum, Dolo Lorem Ipsum

    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added feature columns:
         - brightness
         - saturation
         - contrast
         - clarity
         - warmHue
         - colorfulness
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
      # Load the image using cv2
      image = cv2.imread(image_path)

      ###

      # Room for additional computations

      ###

      df.loc[idx, 'brightness'] = np.nan# Save brightness of image here
      df.loc[idx, 'saturation'] = np.nan# Save saturation of image here
      df.loc[idx, 'contrast'] = np.nan# Save contrast of image here
      df.loc[idx, 'clarity'] = np.nan# Save clarity of image here
      df.loc[idx, 'warmHue'] = np.nan# Save warmHue of image here
      df.loc[idx, 'colorfulness'] = np.nan# Save colorfulness of image here

    return df
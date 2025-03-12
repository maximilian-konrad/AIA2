"""
Add general comments here.
"""

# Imports
from skimage.segmentation import felzenszwalb
from skimage.io import imread
from skimage.color import label2rgb

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

    # Initialize new columns with NaN
    df['felzenszwalbSegmentationRGB'] = np.nan

    # Iterate over all image paths
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        image = imread(image_path)
        segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        segmented_image = label2rgb(segments, image, kind='avg')
        df.loc[idx, 'felzenszwalbSegmentationRGB'] = segmented_image

    return df




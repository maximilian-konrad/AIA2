import cv2
from tqdm import tqdm

def extract_blur_value(self, df_images):
    """
    Calculate the blur value for each image using Laplacian variance.
    A lower value (< 100) indicates a blurry image, while higher values indicate sharper images.

    :param self: AIA object
    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added 'blur' column containing the Laplacian variance score
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filename'])):

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Set threshold at 100. Value below 100 indicates a blurry image
        df.loc[idx, 'blur'] = cv2.Laplacian(gray, cv2.CV_64F).var()

    return df
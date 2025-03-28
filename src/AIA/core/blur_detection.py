import cv2
from tqdm import tqdm
import os

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
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                if self.verbose: print(f"Warning: File not found: {image_path}")
                continue

            # Load the image using cv2
            image = cv2.imread(image_path)
            if image is None:
                if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Set threshold at 100. Value below 100 indicates a blurry image
            df.loc[idx, 'blur'] = cv2.Laplacian(gray, cv2.CV_64F).var()

        except Exception as e:
            error = f"Error processing {image_path}: {str(e)}"
            print(error)
            df.loc[idx, 'error_blur_detection'] = error
            continue

    return df
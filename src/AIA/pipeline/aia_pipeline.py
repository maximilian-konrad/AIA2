# Standard library imports
import os  # Operating system dependent functionality
import sys  # Access to system-specific parameters and functions
import time

# Third-party library imports
from tqdm import tqdm  # Progress bar for loops
import pandas as pd  # Data manipulation and analysis

# Project-specific imports
from ..core.basic_img_features import extract_basic_image_features  # Function to extract basic features from images
from ..core.nima_neural_image_assessment import neural_image_assessment  # Function to extract NIMA neural image assessment
from ..core.blur_detection import extract_blur_value # Function to extract blur value
from ..core.noise_detection import estimate_noise # Function to extract noise value
from ..core.contrast_of_brightness import calculate_contrast_of_brightness # Function to calculate contrast of brightness
from ..core.image_clarity import calculate_image_clarity # Function to calculate image clarity
from ..core.warm_cold_hue import calculate_hue_proportions # Function to calculate warm hue proportion and cold hue proportion
from ..core.salient_region_features import calculate_salient_region_features # Function to calculate Diagonal Dominance, Rule of Thirds, Visual Balance Intensity, Visual Balance Color 
from ..core.get_coco_labels import detect_coco_labels_yolo11 # Function to calculate Diagonal Dominance, Rule of Thirds, Visual Balance Intensity, Visual Balance Color 
from ..core.yelp_paper import get_color_features # Function to calculate Diagonal Dominance, Rule of Thirds, Visual Balance Intensity, Visual Balance Color 

class AIA:
    """
    Automated Image Analysis (AIA) pipeline class.

    This class encapsulates all functionalities needed to process images, 
    extract features, and manage the image analysis pipeline.
    """

    def __init__(self, config):
        """
        Initialize the AIA pipeline with a configuration dictionary.

        :param config: A dictionary containing configuration parameters such as:
            - 'output_dir' (str): Directory to save the output results.
            - 'resize_percent' (int): Percentage to resize the images (default: 100).
            - 'evaluate_brisque' (bool): Whether to evaluate BRISQUE score (default: False).
            - 'evaluate_sharpness' (bool): Whether to evaluate image sharpness (default: False).
        """
        
        # Store configuration and set up output directory
        self.config = config
        self.output_dir = config.get('output_dir', None)
        
        # If no output directory is specified, use a default path
        if self.output_dir is None:
            self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../outputs/'))
            os.makedirs(self.output_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Optional configuration parameters with default values
        self.resize_percent = config.get('resize_percent', 100)
        self.evaluate_brisque = config.get('evaluate_brisque', False)
        self.evaluate_sharpness = config.get('evaluate_sharpness', False)
        
        print(f"Initialized AIA pipeline with config: {config}")

    def process_batch(self, image_files):
        """
        Process a batch of images from a list of file paths.

        :param image_files: A list of file paths for the images to be processed.
        :return: A DataFrame containing the features extracted from each image.
        """
        
        # Initialize a dataframe with a single column filename and is filled with all image_files from image_files
        df_images = pd.DataFrame({'filename': image_files})
        df_out = df_images.copy(deep=True)

        # List of feature extractor functions
        feature_extractors = [
            extract_basic_image_features,
            # neural_image_assessment,
            # extract_blur_value,
            # estimate_noise,
            # calculate_contrast_of_brightness,
            # calculate_image_clarity,
            # calculate_hue_proportions,
            # calculate_salient_region_features,
            # detect_coco_labels_yolo11,
            get_color_features
        ]

        print(f"Processing batch of n={len(df_images)} images")
        # Iterate over each function specified in feature extractor
        for func in feature_extractors:
            print(f"Processing {func.__name__}()")
            tic = time.perf_counter()
            # Execute function
            df_temp = func(df_images)
            # Append results to dataframe
            df_out = df_out.merge(df_temp, on='filename', how='left')
            toc = time.perf_counter()
            print(f"Time for {func.__name__}(): {toc - tic:.4f} seconds")

        return df_out
   
    def save_results(self, df_results, output_path=None):
        """
        Save the processed results to an Excel file.

        :param output_path: The path to save the results to. If not provided,
                            the results will be saved in the default output directory.
        """
        
        # Save the DataFrame to an Excel file, either to the specified path or to the default location
        if output_path is not None:
            df_results.to_excel(output_path, index=False)
        else:
            output_path = os.path.join(self.output_dir, 'results.xlsx')
            df_results.to_excel(output_path, index=False)
        
        print(f"Results saved to: {output_path}")
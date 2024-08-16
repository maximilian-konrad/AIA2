# Standard library imports
import os  # Operating system dependent functionality
import sys  # Access to system-specific parameters and functions

# Third-party library imports
from tqdm import tqdm  # Progress bar for loops
import pandas as pd  # Data manipulation and analysis

# Project-specific imports
from ..core.basic_img_features import extract_basic_image_features  # Function to extract basic features from images


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

    def process_image(self, image_path):
        """
        Process an individual image using the AIA pipeline.

        :param image_path: The path to the image to process.
        :return: A dictionary containing the extracted features of the image.
        """
        
        # Initialize a dictionary to hold the features, starting with the image path
        features = {
            'image_path': image_path
        }

        # Extract basic image features (e.g., color histograms, entropy) and add them to the features dictionary
        features.update(extract_basic_image_features(image_path))

        # TODO: Implement additional image processing steps, if needed

        return features  # Return the dictionary of extracted features
    
    def process_batch(self, img_dir):
        """
        Process a batch of images located in a specified directory.

        :param img_dir: The directory containing the images to be processed.
        :return: A list of dictionaries, each containing the features extracted from one image.
        """
        
        print(f"Processing batch of images from: {img_dir}")

        results = []  # Initialize an empty list to store the results
        
        # Get a list of all image files in the directory (only PNG, JPG, JPEG formats)
        image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Iterate over each image file with a progress bar
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.process_image(image_path)  # Process the image
            results.append(result)  # Append the result to the list

        self.results = results  # Store the results as an instance variable
        return self.results  # Return the list of results

    
    def save_results(self, output_path=None):
        """
        Save the processed results to an Excel file.

        :param output_path: The path to save the results to. If not provided,
                            the results will be saved in the default output directory.
        """
        
        # Convert the results (a list of dictionaries) into a pandas DataFrame
        df = pd.DataFrame(self.results)
        
        # Save the DataFrame to an Excel file, either to the specified path or to the default location
        if output_path is not None:
            df.to_excel(output_path, index=False)
        else:
            output_path = os.path.join(self.output_dir, 'results.xlsx')
            df.to_excel(output_path, index=False)
        
        print(f"Results saved to: {output_path}")
import os
from tqdm import tqdm
from ..core.basic_img_features import extract_basic_image_features
import pandas as pd
import sys

class AIA:
    def __init__(self, config):
        """
        Initialize the AIA pipeline with a configuration dictionary.
        
        :param config: A dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get('output_dir',None)
        if self.output_dir is None:
            self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../outputs/'))
            os.makedirs(self.output_dir, exist_ok=True)

        self.resize_percent = config.get('resize_percent', 100)
        self.evaluate_brisque = config.get('evaluate_brisque', False)
        self.evaluate_sharpness = config.get('evaluate_sharpness', False)
        print(f"Initialized AIA pipeline with config: {config}")

    def process_image(self, image_path):
        """
        Process an image using the AIA pipeline.
        
        :param image_path: The path to the image to process
        """

        features = {
            'image_path': image_path
        }

        # Extract basic image features
        features.update(extract_basic_image_features(image_path))

        #TODO - Add more image processing steps here

        return features
    
    def process_batch(self, img_dir):
        """
        Process a batch of images using the AIA pipeline.
        """
        print(f"Processing batch of images from: {img_dir}")

        results = []
        
        # Get a list of all files in the directory
        image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Iterate over each image file using tqdm for progress tracking
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.process_image(image_path)
            results.append(result)

        self.results = results
        return self.results
    
    def save_results(self,output_path=None):
        """
        Save the results to an Excel file.
        :param output_path: The path to save the results to.
        """
        df = pd.DataFrame(self.results)
        if output_path is not None:
            df.to_excel(output_path, index=False)
        else:
            output_path = os.path.join(self.output_dir, 'results.xlsx')
            df.to_excel(output_path, index=False)
        
        print(f"Results saved to: {output_path}")

  
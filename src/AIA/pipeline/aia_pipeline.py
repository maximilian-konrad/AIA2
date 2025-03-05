# Standard library imports
import os  # Operating system dependent functionality
import sys  # Access to system-specific parameters and functions
import time
import torch

# Third-party library imports
from tqdm import tqdm  # Progress bar for loops
import pandas as pd  # Data manipulation and analysis
from fpdf import FPDF

# Project-specific imports
from ..utils.helper_functions import load_config
from ..core.basic_img_features import extract_basic_image_features  # Function to extract basic features from images
from ..core.blur_detection import extract_blur_value # Function to extract blur value
from ..core.noise_detection import estimate_noise # Function to extract noise value
from ..core.contrast_of_brightness import calculate_contrast_of_brightness # Function to calculate contrast of brightness
from ..core.image_clarity import calculate_image_clarity # Function to calculate image clarity
from ..core.warm_cold_hue import calculate_hue_proportions # Function to calculate warm hue proportion and cold hue proportion
from ..core.salient_region_features import calculate_salient_region_features # Function to calculate Diagonal Dominance, Rule of Thirds, Visual Balance Intensity, Visual Balance Color 
from ..core.yolo import predict_coco_labels_yolo11, predict_imagenet_classes_yolo11 # Function to calculate Diagonal Dominance, Rule of Thirds, Visual Balance Intensity, Visual Balance Color 
from ..core.yelp_paper import get_color_features, get_composition_features, get_figure_ground_relationship_features # Function to calculate Diagonal Dominance, Rule of Thirds, Visual Balance Intensity, Visual Balance Color
from ..core.nima_idealo import calculate_aesthetic_scores # Function to calculate Diagonal Dominance, Rule of Thirds, Visual Balance Intensity, Visual Balance Color
from ..core.ocr import get_ocr_text # Function to extract text from images using OCR
from ..core.ov_object_detection import detect_objects # Function to detect objects in images
from ..core.visual_complexity import visual_complexity # Function to detect objects in images

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
            - 'include_summary' (bool): Whether to include summary statistics (default: False).
        """
        
        # Store configuration and set up output directory
        self.config = config
        self.full_config = load_config()
        self.output_dir = config.get('output_dir', None)
        
        # If no output directory is specified, use a default path
        if self.output_dir is None:
            self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../outputs/'))
            os.makedirs(self.output_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Optional configuration parameters with default values
        self.cuda_availability = torch.cuda.is_available()
        self.resize_percent = config.get('resize_percent', 100)
        self.evaluate_brisque = config.get('evaluate_brisque', False)
        self.evaluate_sharpness = config.get('evaluate_sharpness', False)
        self.incl_summary_stats = config.get('incl_summary_stats', False)
        
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
            extract_blur_value,
            estimate_noise,
            calculate_contrast_of_brightness,
            calculate_image_clarity,
            calculate_hue_proportions,
            calculate_salient_region_features,
            # predict_coco_labels_yolo11, # TODO: fix this
            predict_imagenet_classes_yolo11, 
            # get_color_features, # TODO: fix saliency issue
            get_composition_features,
            get_figure_ground_relationship_features,
            get_ocr_text,
            calculate_aesthetic_scores,
            detect_objects,
            visual_complexity
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
   
    def save_results(self, df_results, output_path = None):
        """
        Save the processed results to an Excel file.

        :param output_path: The path to save the results to. If not provided,
                            the results will be saved in the default output directory.
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'results.xlsx')

        # Create an Excel writer object
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Check if the DataFrame exceeds Excel's column limit (16,384 columns)
            if df_results.shape[1] > 16384:
                # Create a message DataFrame to inform the user
                message_df = pd.DataFrame({
                    'Message': ['The results contain more than 16,384 columns, which exceeds Excel\'s limit.',
                               'Please refer to the CSV file for the complete results.']
                })
                message_df.to_excel(writer, sheet_name='Raw Data', index=False)
            else:
                # Save the main results to the first sheet if within Excel's limits
                df_results.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Also save as CSV with UTF-8 encoding
            csv_path = os.path.splitext(output_path)[0] + '.csv'
            df_results.to_csv(csv_path, index=False, encoding='utf-8')
            
            # If summary statistics are requested, create and save them
            if self.incl_summary_stats:
                # Calculate summary statistics for numeric columns only
                numeric_cols = df_results.select_dtypes(include=['int64', 'float64']).columns
                
                # Only calculate stats if there are numeric columns
                if len(numeric_cols) > 0:
                    summary_stats = df_results[numeric_cols].agg([
                        'count',
                        'mean',
                        'std',
                        'min',
                        'max'
                    ])
                else:
                    summary_stats = pd.DataFrame()

                # Calculate count and percentage for binary columns
                binary_cols = df_results.select_dtypes(include=['bool']).columns
                
                # Only calculate stats if there are binary columns
                if len(binary_cols) > 0:
                    binary_stats = df_results[binary_cols].agg(['sum', 'count'])
                    binary_stats.loc['share'] = binary_stats.loc['sum'] / binary_stats.loc['count']
                    
                    # Combine numeric and binary statistics if both exist
                    if not summary_stats.empty:
                        combined_stats = pd.concat([summary_stats.transpose(), binary_stats.transpose()], axis=0)
                    else:
                        combined_stats = binary_stats.transpose()
                else:
                    # If no binary columns, just use the numeric stats
                    combined_stats = summary_stats.transpose() if not summary_stats.empty else pd.DataFrame()
                
                # Only save statistics if we have any
                if not combined_stats.empty:
                    combined_stats.to_excel(writer, sheet_name='Summary Statistics')
        
        print(f"Results saved to: {output_path}")

    def generate_pdf_from_excel(self, excel_path, output_pdf_path):
        # Load the Excel file
        xls = pd.ExcelFile(os.path.join(self.output_dir, excel_path))
        df = pd.read_excel(xls, sheet_name='Raw Data')

        # Initialize PDF
        pdf = FPDF(orientation='L', unit='mm', format='letter')
        # Get page dimensions (letter size in landscape)
        page_width = 279.4  # mm
        page_height = 215.9  # mm
        margin = 10  # mm
        gap = 5  # mm gap between left and right halves
        half_width = (page_width - 2 * margin - gap) / 2  # Adjusted to account for gap

        # Iterate over each feature
        features = df.columns[1:]  # Assuming first column is filename
        for feature in features:
            pdf.add_page()

            # Set title
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt=f"Feature: {feature}", ln=True, align='C')

            # Find min and max images for the feature
            min_image = df.loc[df[feature].idxmin()]['filename']
            max_image = df.loc[df[feature].idxmax()]['filename']

            def scale_image(img_path):
                """Helper function to calculate scaled dimensions"""
                if not os.path.exists(img_path):
                    return None, None
                
                from PIL import Image
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    img_aspect = img_w / img_h
                    
                    # Calculate dimensions to fit half page width and full height
                    width_based = half_width
                    height_based = page_height - 40  # Account for margins and text
                    
                    # Calculate both possible dimensions
                    if width_based / img_aspect <= height_based:
                        # Width is the limiting factor
                        return width_based, width_based / img_aspect
                    else:
                        # Height is the limiting factor
                        return height_based * img_aspect, height_based

            # Add image labels on same line
            pdf.set_font("Arial", size=10)
            min_value = df.loc[df[feature].idxmin()][feature]
            max_value = df.loc[df[feature].idxmax()][feature]
            
            # Calculate widths for left and right text cells
            left_text = f"Min Image: {os.path.basename(min_image)} (Value: {min_value:.4f})"
            right_text = f"Max Image: {os.path.basename(max_image)} (Value: {max_value:.4f})"
            
            # Print both texts on same line, both left-aligned
            pdf.cell(half_width + margin, 10, txt=left_text, ln=0, align='L')
            pdf.cell(half_width + margin, 10, txt=right_text, ln=1, align='L')

            # Store the Y position after text for both images to ensure alignment
            image_start_y = pdf.get_y()

            # Add images
            if os.path.exists(min_image):
                w, h = scale_image(min_image)
                if w and h:
                    # Center in left half
                    x = margin + (half_width - w) / 2
                    y = image_start_y + (page_height - image_start_y - h) / 2
                    pdf.image(min_image, x=x, y=y, w=w, h=h)

            if os.path.exists(max_image):
                w, h = scale_image(max_image)
                if w and h:
                    # Center in right half (with gap)
                    x = margin + half_width + gap + (half_width - w) / 2
                    y = image_start_y + (page_height - image_start_y - h) / 2
                    pdf.image(max_image, x=x, y=y, w=w, h=h)

            # Add footer text
            pdf.set_y(2)  # Move to 20mm from bottom
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, txt="Images analyzed with AIA2", ln=True, align='C')

        # Save the PDF
        pdf.output(os.path.join(self.output_dir, output_pdf_path))
        print(f"PDF saved to: {os.path.join(self.output_dir, output_pdf_path)}")

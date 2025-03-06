# General imports
from fpdf import FPDF
import gc  
import os
import pandas as pd
import time
import torch

# Project-specific imports
from ..core.basic_img_features import extract_basic_image_features
from ..core.blur_detection import extract_blur_value
from ..core.contrast_of_brightness import calculate_contrast_of_brightness
from ..core.image_clarity import calculate_image_clarity
from ..core.nima_idealo import calculate_aesthetic_scores
from ..core.noise_detection import estimate_noise
from ..core.ocr import get_ocr_text
from ..core.ov_object_detection import detect_objects
from ..core.salient_region_features import calculate_salient_region_features
from ..core.visual_complexity import visual_complexity
from ..core.warm_cold_hue import calculate_hue_proportions
from ..core.yelp_paper import get_color_features, get_composition_features, get_figure_ground_relationship_features
from ..core.yolo import predict_coco_labels_yolo11, predict_imagenet_classes_yolo11
from ..utils.helper_functions import load_config

class AIA:
    """
    Automated Image Analysis (AIA) pipeline class.

    This class encapsulates all functionalities needed to process images, 
    extract features, and manage the image analysis pipeline.
    """

    def __init__(self, config_path):
        """
        Initialize the AIA pipeline with a configuration dictionary.

        :param config_path: The path to the configuration file.
        """

        # Get parameters
        self.config = load_config(config_path)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.input_dir = self.config.get("general", {}).get("input_dir")
        self.output_dir = self.config.get("general", {}).get("output_dir")
        self.output_dir = os.path.join(self.output_dir , self.timestamp+ "\\")
        os.makedirs(self.output_dir, exist_ok=True)
        self.verbose = self.config.get("general", {}).get("verbose", True)
        self.incl_summary_stats = self.config.get("general", {}).get("summary_stats", {}).get("active", True)

        self.cuda_availability = torch.cuda.is_available()
        self.device = "cuda" if self.cuda_availability else "cpu"
        # Console outputs
        if self.cuda_availability:
            print(f"### Using GPU (CUDA) ###")
        else:
            print(f"### Using CPU ###")

    def process_batch(self):
        """
        Process a batch of images from a list of file paths.

        :param self: AIA object
        :return: A DataFrame containing the features extracted from each image.
        """

        # Identify all images in input directory
        image_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.ppm', '.pgm', '.pbm', '.gif', '.hdr', '.exr'))]

        # Initialize dataframe with all image files
        df_images = pd.DataFrame({'filename': image_files})
        df_out = df_images.copy(deep=True)

        # Create a dataframe of feature extractor functions with active status and time tracking
        df_logs = pd.DataFrame({
            'functions': [
                extract_basic_image_features,
                extract_blur_value,
                estimate_noise,
                calculate_contrast_of_brightness,
                calculate_image_clarity,
                calculate_hue_proportions,
                calculate_salient_region_features,
                predict_coco_labels_yolo11,
                predict_imagenet_classes_yolo11,
                get_color_features,
                get_composition_features,
                get_figure_ground_relationship_features,
                get_ocr_text,
                calculate_aesthetic_scores,
                detect_objects,
                visual_complexity
            ]
        })
        df_logs['active'] = None
        df_logs['seconds_needed'] = None

        # Identify which fucntions should be processed
        for idx, row in df_logs.iterrows():
            func_name = row['functions'].__name__
            df_logs.at[idx, 'active'] = self.config.get('features', {}).get(func_name, {}).get('active', False)

        print(f"### Starting batch of n={len(df_images)} images ###")
        # Iterate over each function in the feature_extractors_df dataframe
        for idx, row in df_logs.iterrows():
            func = row['functions']
            if row['active']:
                print(f"### Processing {func.__name__}() ###")
                tic = time.perf_counter()
                # Execute function
                df_temp = func(self, df_images)
                # Append results to dataframe
                df_out = df_out.merge(df_temp, on='filename', how='left')
                toc = time.perf_counter()
                print(f"{toc - tic:.4f} seconds needed")

                # Save time needed
                df_logs.at[idx, 'seconds_needed'] = toc - tic

                # Run GC and empty cuda cache
                if self.cuda_availability:
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"### Finished batch of n={len(df_images)} images ###")

        return df_out, df_logs

    def save_results(self, df_results, output_dir = None):
        """
        Save the processed results to an Excel file.

        :param output_dir: The directory to save the results to. If not provided,
                            the results will be saved in the default output directory.
        """

        if output_dir is None:
            output_dir = self.output_dir
        
        # Save as CSV
        df_results.to_csv(output_dir + f'{self.timestamp}_results.csv', index=False, encoding='utf-8')

        # Save as XLSX
        with pd.ExcelWriter(output_dir + f'{self.timestamp}_results.xlsx', engine='openpyxl') as writer:
            # Check if the DataFrame exceeds Excel's column limit (16,384 columns)
            if df_results.shape[1] > 16384:
                message_df = pd.DataFrame({
                    'Message': ['The results contain more than 16,384 columns, which exceeds Excel\'s limit.',
                               'Please refer to the CSV file for the complete results.']
                })
                message_df.to_excel(writer, sheet_name='Raw Data', index=False)
            else:
                # Save results
                df_results.to_excel(writer, sheet_name='Raw Data', index=False)
            

            
            # Save summary statistics
            if self.incl_summary_stats:
                # Numeric columns
                numeric_cols = df_results.select_dtypes(include=['int64', 'float64']).columns
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

                # Binary columns
                binary_cols = df_results.select_dtypes(include=['bool']).columns
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
        
        print(f"### Results saved to: {output_dir} with filename: ###")
        print(f"### - {f'{self.timestamp}_results.csv'} ###")
        print(f"### - {f'{self.timestamp}_results.xlsx'} ###")

    def save_logs(self, df_logger, output_dir = None):
        """
        Save the logs to an Excel & CSV file.

        :param output_dir: The directory to save the logs to. If not provided,
                            the results will be saved in the default output directory.
        """

        if output_dir is None:
            output_dir = self.output_dir

        # Save as CSV
        df_logger.to_csv(output_dir + f'{self.timestamp}_logs.csv', index=False, encoding='utf-8')

        # Save as XLSX
        with pd.ExcelWriter(output_dir + f'{self.timestamp}_logs.xlsx', engine='openpyxl') as writer:
            df_logger.to_excel(writer, sheet_name='Raw Data', index=False)
        
        print(f"### Logs saved to: {output_dir} with filename: ###")
        print(f"### - {f'{self.timestamp}_logs.csv'} ###")
        print(f"### - {f'{self.timestamp}_logs.xlsx'} ###")

    def generate_pdf_from_excel(self, excel_path = None, output_pdf_path = None):
        """
        Generate a PDF from an Excel file.

        :param excel_path: The path to the Excel file.
        :param output_pdf_path: The path to save the PDF file.
        """        
        
        if excel_path is None:
            excel_path = f'{self.timestamp}_results.xlsx'
        print(f"### Excel not specified. Using data from current run: {excel_path} ###")
        if output_pdf_path is None:
            output_pdf_path = f'{self.timestamp}_exemplary_images.pdf'

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
            # Check if values are numeric before formatting with .4f
            if isinstance(min_value, (int, float)):
                left_text = f"Min Image: {os.path.basename(min_image)} (Value: {min_value:.4f})"
            else:
                left_text = f"Min Image: {os.path.basename(min_image)} (Value: {min_value})"
                
            if isinstance(max_value, (int, float)):
                right_text = f"Max Image: {os.path.basename(max_image)} (Value: {max_value:.4f})"
            else:
                right_text = f"Max Image: {os.path.basename(max_image)} (Value: {max_value})"
            
            # Print both texts on same line, both left-aligned
            pdf.cell(half_width + margin, 10, txt=left_text, ln=0, align='L')
            pdf.cell(half_width + margin, 10, txt=right_text, ln=1, align='L')

            # Store the Y position after text for both images to ensure alignment
            image_start_y = pdf.get_y()

            # Add images
            if os.path.exists(min_image):
                # TODO: Implement webp support
                w, h = scale_image(min_image)
                if w and h:
                    # Center in left half
                    x = margin + (half_width - w) / 2
                    y = image_start_y + (page_height - image_start_y - h) / 2
                    pdf.image(min_image, x=x, y=y, w=w, h=h)

            if os.path.exists(max_image):
                # TODO: Implement webp support
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
        print(f"### PDF saved to: {os.path.join(self.output_dir, output_pdf_path)} ###")
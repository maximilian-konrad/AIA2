# General imports
from fpdf import FPDF
import gc  
import os
import pandas as pd
import time
import torch
import random

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
from ..core.visual_complexity import visual_complexity, self_similarity
from ..core.warm_cold_hue import calculate_hue_proportions
from ..core.yelp_paper import get_color_features, get_composition_features, get_figure_ground_relationship_features
from ..core.yolo import predict_coco_labels_yolo11, predict_imagenet_classes_yolo11
from ..core.describe import describe_blip, describe_llm
from ..core.felzenszwalb_segmentation import felzenszwalb_segmentation
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
        random.seed(0)

        # Get parameters
        self.config = load_config(config_path)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(f'Timestamp: {self.timestamp}')
        self.input_dir = self.config.get("general", {}).get("input_dir")
        self.output_dir = self.config.get("general", {}).get("output_dir")
        self.output_dir = os.path.join(self.output_dir , self.timestamp+ "\\")
        os.makedirs(self.output_dir, exist_ok=True)
        self.verbose = self.config.get("general", {}).get("verbose", True)
        self.debug_mode = self.config.get("general", {}).get("debug_mode")
        self.debug_img_cnt = self.config.get("general", {}).get("debug_image_count")
        self.incl_summary_stats = self.config.get("general", {}).get("summary_stats", {}).get("active", True)
        self.multi_processing = self.config.get("general", {}).get("multi_processing", {}).get("active", False)
        self.num_processes = self.config.get("general", {}).get("multi_processing", {}).get("num_processes", 4)

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

        # Collect all images from input_dir
        image_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.ppm', '.pgm', '.pbm', '.gif', '.hdr', '.exr'))]
        if self.debug_mode and len(image_files) > self.debug_img_cnt:
            image_files = random.sample(image_files, self.debug_img_cnt)
        image_files.sort()

        # Initialize dataframe with all image files
        df_images = pd.DataFrame({'filename': image_files})
        df_out = df_images.copy(deep=True)

        # Create a dataframe of feature extractor functions with active status and time tracking
        df_logs = pd.DataFrame({
            'functions': [
                extract_basic_image_features,
                describe_blip,
                describe_llm,
                get_ocr_text,
                extract_blur_value,
                estimate_noise,
                calculate_contrast_of_brightness,
                calculate_image_clarity,
                calculate_hue_proportions,
                calculate_salient_region_features,
                get_color_features,
                get_composition_features,
                get_figure_ground_relationship_features,
                calculate_aesthetic_scores,
                visual_complexity,
                self_similarity,
                felzenszwalb_segmentation,
                detect_objects,
                predict_coco_labels_yolo11,
                predict_imagenet_classes_yolo11,
            ]
        })
        df_logs['active'] = None
        df_logs['seconds_needed'] = None

        # Save interim df_out and df_logs
        self.save_results(df_out)
        self.save_logs(df_logs)

        # Identify which functions should be processed
        for idx, row in df_logs.iterrows():
            func_name = row['functions'].__name__
            df_logs.at[idx, 'active'] = self.config.get('features', {}).get(func_name, {}).get('active', False)

        print(f"### Starting batch of n={len(df_images)} images ###")
        # Iterate over each function in the feature_extractors_df dataframe
        for idx, row in df_logs.iterrows():

            # Flush cache
            gc.collect()
            if self.cuda_availability:
                torch.cuda.empty_cache()

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

                # Save interim df_temp and df_logs
                self.save_results(df_temp)
                self.save_logs(df_logs)

                # Run GC and empty cuda cache
                if self.cuda_availability:
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"### Finished batch of n={len(df_images)} images ###")

        # Output final results as Excel files
        self._csv_to_xlsx(
            csv_path=self.output_dir + f'{self.timestamp}_results.csv', 
            xlsx_path=self.output_dir + f'{self.timestamp}_results.xlsx')
        self._csv_to_xlsx(
            csv_path=self.output_dir + f'{self.timestamp}_logs.csv', 
            xlsx_path=self.output_dir + f'{self.timestamp}_logs.xlsx')
        print(f"### Final Excel versions saved to: {self.output_dir} ###")
        
        return df_out, df_logs

    def save_logs(self, df_logger, output_dir=None):
        """
        Save the logs to CSV file. If file already exists,
        merge the new logs with existing data.

        :param df_logger: DataFrame containing logs to save
        :param output_dir: The directory to save the logs to. If not provided,
                          the results will be saved in the default output directory.
        """
        if output_dir is None:
            output_dir = self.output_dir

        csv_path = output_dir + f'{self.timestamp}_logs.csv'

        # Create a copy of the dataframe to avoid modifying the original
        df_to_save = df_logger.copy()

        # Check if the functions column contains actual functions or strings
        if df_to_save['functions'].dtype == 'O' and callable(df_to_save['functions'].iloc[0]):
            # Convert function objects to names only when saving
            df_to_save['functions'] = df_to_save['functions'].apply(lambda x: x.__name__ if x else None)

        # Check if file already exists
        if os.path.exists(csv_path):
            # Load existing data
            df_saved = pd.read_csv(csv_path)
            
            # Update existing rows and append new ones
            for idx, row in df_to_save.iterrows():
                func_name = row['functions']
                mask = df_saved['functions'] == func_name
                
                if mask.any():
                    # Update existing row
                    for col in df_to_save.columns:
                        if col != 'functions' and pd.notna(row[col]):
                            df_saved.loc[mask, col] = row[col]
                else:
                    # Append new row
                    df_saved = pd.concat([df_saved, pd.DataFrame([row])], ignore_index=True)

            # Save updated data
            df_saved.to_csv(csv_path, index=False, encoding='utf-8')
        else:
            # Initial save
            df_to_save.to_csv(csv_path, index=False, encoding='utf-8')

    def save_results(self, df_results, output_dir=None):
        """
        Save the processed results to CSV file. If file already exists,
        merge the new results with existing data.

        :param df_results: DataFrame containing results to save
        :param output_dir: The directory to save the results to. If not provided,
                          the results will be saved in the default output directory.
        """
        if output_dir is None:
            output_dir = self.output_dir

        csv_path = output_dir + f'{self.timestamp}_results.csv'

        # Check if file already exists
        if os.path.exists(csv_path):
            # Load existing data
            df_saved = pd.read_csv(csv_path)
            
            # Merge new results with existing data
            df_merged = df_saved.merge(df_results, on='filename', how='left')
            
            # Save updated CSV
            df_merged.to_csv(csv_path, index=False, encoding='utf-8')
        else:
            # Initial save
            df_results.to_csv(csv_path, index=False, encoding='utf-8')

    def _csv_to_xlsx(self, csv_path, xlsx_path):
        """
        Helper method to convert a CSV file to an Excel file.
        """
        # Handle results Excel file
        if os.path.exists(csv_path):
            
            # Open file
            df = pd.read_csv(csv_path)

            # Handle logs and results
            if csv_path.endswith('_logs.csv'):
                with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Raw Data', index=False)

            elif csv_path.endswith('_results.csv'):
                with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                    # Check Excel column limit
                    if df.shape[1] > 16384:
                        message_df = pd.DataFrame({
                            'Message': ['The results contain more than 16,384 columns, which exceeds Excel\'s limit.',
                                    'Please refer to the CSV file for the complete results.']
                        })
                        message_df.to_excel(writer, sheet_name='Raw Data', index=False)
                    else:
                        df.to_excel(writer, sheet_name='Raw Data', index=False)

                    # Add summary statistics
                    if self.incl_summary_stats:
                        self._save_summary_stats(df, writer)

    def _save_summary_stats(self, df, writer):
        """
        Helper method to calculate and save summary statistics.
        
        :param df: DataFrame to calculate statistics for
        :param writer: ExcelWriter object to save to
        """
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            summary_stats = df[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
        else:
            summary_stats = pd.DataFrame()

        # Binary columns
        binary_cols = df.select_dtypes(include=['bool']).columns
        if len(binary_cols) > 0:
            binary_stats = df[binary_cols].agg(['sum', 'count'])
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
import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from ..utils.helper_functions import download_weights

def calculate_aesthetic_scores(self, df_images):
    """
    Calculates the aesthetic scores for a list of images using a MobileNet-based NIMA model.
    
    For each image, the following steps are performed:
      - Load and resize the image to 224x224.
      - Convert the image from BGR to RGB and normalize pixel values to the range [0, 1].
      - The pre-trained MobileNet-based NIMA model predicts a probability distribution over 10 score bins.
      - The aesthetic score is computed as the weighted sum: sum_{i=1}^{10} (i * p_i),
        where p_i is the predicted probability for score bin i.
    
    :param self: AIA object
    :param df_images: DataFrame with a column 'filename' containing paths to image files.
    :return: DataFrame with an additional column 'nima_score' containing the aesthetic scores.
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    try:
        # Get parameters
        weight_filename = self.config.get("features", {}).get("calculate_aesthetic_scores", {}).get("parameters", {}).get("weight_filename")
        weight_url = self.config.get("features", {}).get("calculate_aesthetic_scores", {}).get("parameters", {}).get("weight_url")

        # Download weights if needed
        try:
            download_weights(
                weight_filename=weight_filename,
                weight_url=weight_url
            )
        except Exception as e:
            print(f"Error downloading weights: {str(e)}")
            return df

        # Load model
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'weights', weight_filename)
            
            base_model = tf.keras.applications.MobileNet(
                input_shape=(224, 224, 3),
                include_top=False,
                pooling='avg',
                weights=None
            )
            x = base_model.output
            x = tf.keras.layers.Dropout(0.75)(x)
            x = tf.keras.layers.Dense(10, activation='softmax')(x)
            model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
            model.load_weights(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return df

        # Process images
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Read and process image
                image = cv2.imread(image_path)
                if image is None:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Preprocess image
                image_resized = cv2.resize(image, (224, 224))
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                image_norm = image_rgb.astype("float32") / 255.0
                image_input = np.expand_dims(image_norm, axis=0)

                # Predict aesthetic score
                predictions = model.predict(image_input, verbose=False)
                p = predictions[0]
                aesthetic_score = sum((i + 1) * p[i] for i in range(len(p)))
                df.loc[idx, 'nima_score'] = aesthetic_score

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_nima'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in NIMA setup: {str(e)}")
        return df
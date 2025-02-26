import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from ..utils.helper_functions import download_weights

def calculate_aesthetic_scores(df_images):
    """
    Calculates the aesthetic scores for a list of images using a MobileNet-based NIMA model.
    
    For each image, the following steps are performed:
      - Load and resize the image to 224x224.
      - Convert the image from BGR to RGB and normalize pixel values to the range [0, 1].
      - The pre-trained MobileNet-based NIMA model predicts a probability distribution over 10 score bins.
      - The aesthetic score is computed as the weighted sum: sum_{i=1}^{10} (i * p_i),
        where p_i is the predicted probability for score bin i.
    
    :param df_images: DataFrame with a column 'filename' containing paths to image files.
    :return: DataFrame with an additional column 'nima_score' containing the aesthetic scores.
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    # Define the weight filename and download URL
    weight_filename = "weights_mobilenet_aesthetic_0.07.hdf5"
    weight_url = "https://github.com/idealo/image-quality-assessment/blob/master/models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5?raw=true"
    
    # If the weight file is not present in the current directory, download it
    if not os.path.exists(weight_filename):
        print("Weights not found. Downloading them from the repository and saving to", weight_filename)
        download_weights(weight_filename, weight_url)
    
    # Construct the absolute path for the model weights (assuming they are stored in a 'weights' folder one level up)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'weights', weight_filename)
    print(f"model_path: {model_path}")
    
    # Construct the MobileNet-based NIMA model within the function
    base_model = tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling='avg',
        weights=None  # Using custom-trained weights instead of ImageNet weights
    )
    x = base_model.output
    x = tf.keras.layers.Dropout(0.75)(x)
    # Output layer with softmax activation to produce a probability distribution over score bins
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    
    # Load weights into the model
    model.load_weights(model_path)
    
    # Process each image in the DataFrame
    for idx, image_path in enumerate(tqdm(df_images['filename'])):
        try:
            # Read the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image not found or unable to load")
            
            # Resize the image to the expected input size (224x224)
            image_resized = cv2.resize(image, (224, 224))
            
            # Convert the image from BGR to RGB and normalize to [0, 1]
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_norm = image_rgb.astype("float32") / 255.0
            
            # Expand dimensions to create a batch of one image
            image_input = np.expand_dims(image_norm, axis=0)
            
            # Predict the probability distribution over score bins (expected shape: [1, 10])
            predictions = model.predict(image_input)
            p = predictions[0]
            
            # Compute the weighted sum as the aesthetic score: sum_{i=1}^{10} (i * p_i)
            aesthetic_score = sum((i + 1) * p[i] for i in range(len(p)))
            df.loc[idx, 'nima_score'] = aesthetic_score
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            df.loc[idx, 'nima_score'] = f"Error: {str(e)}"
    
    return df
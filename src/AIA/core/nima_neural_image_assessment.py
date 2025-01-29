import numpy as np
import tensorflow as tf
import torch
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from ..utils.score_utils import mean_score, std_score

def neural_image_assessment(df_images):
    """
    Assess image aesthetic quality using Neural Image Assessment (NIMA) model based on
    InceptionResNetV2. The model predicts aesthetic ratings distribution and returns
    the mean score and standard deviation for each image.

    :param df_images: DataFrame containing a 'filename' column with paths to image files
    :return: DataFrame with added columns:
            - nima_mean: mean aesthetic score (typically 1-10, higher is better)
            - nima_std: standard deviation of the predicted score distribution
    :note: Automatically uses GPU if CUDA is available, falls back to CPU if not
    """
    # Ensure reproducibility if non-deterministic Tensorflow operations are used
    tf.random.set_seed(37)
    np.random.seed(37)

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Check if CUDA GPU is available
    device = '/GPU:0' if torch.cuda.is_available() else '/CPU:0'
    print(f"Using device: {device}")

    with tf.device(device):
        base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
        x = Dense(10, activation='softmax')(base_model.output)
        model = Model(base_model.input, x)

        # Process each image in the dataframe
        for idx, image_path in enumerate(tqdm(df_images['filename'])):          
            try:
                img = load_img(image_path)
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                scores = model.predict(x, batch_size=1, verbose=0)[0]
                
                df.loc[idx, 'nima_mean']  = mean_score(scores)
                df.loc[idx, 'nima_std']  = std_score(scores)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                df.loc[idx, 'nima_mean'] = f"Error: {str(e)}"
                df.loc[idx, 'nima_std'] = f"Error: {str(e)}"
       
    return df

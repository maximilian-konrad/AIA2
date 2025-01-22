import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


from ..utils.score_utils import mean_score, std_score

def neural_image_assessment(image_path):
    features = {}
    with tf.device('/CPU:0'):
        base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg')
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)

        model = Model(base_model.input, x)

        img = load_img(image_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        std = std_score(scores)

        score_str = "{:.3f} +- ({:.3f})".format(mean, std)

        features['NIMA Score'] = score_str

        return features


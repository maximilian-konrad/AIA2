import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
from ..utils.helper_functions import download_weights

def get_coco_labels(df_images):
    """
    Detects COCO labels in a list of images.

    :param image_pats: Path to image file.
    :return: A list of dictionaries, each containing COCO labels and their presence for each image.
    """

    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    # Check if weights are doenloaded already, otherwise download them
    download_weights(weight_filename = 'yolov3.cfg', weight_url = 'https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg')
    download_weights(weight_filename = 'yolov3.weights', weight_url = 'https://pjreddie.com/media/files/yolov3.weights')
    download_weights(weight_filename = 'coco.names', weight_url = 'https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names')

    # Load the pre-trained model and COCO labels
    net = cv.dnn.readNetFromDarknet('../AIA/weights/yolov3.cfg', '../AIA/weights/yolov3.weights')
    
    # Use CUDA if available, otherwise fallback to CPU
    if cv.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Load COCO labels
    with open('../AIA/weights/coco.names', 'r') as f:
        classes = ['coco_' + line.strip() for line in f.readlines()]

    # Initialize columns for each class
    for label in classes:
        df[label] = False

    # Iterate over all images using enumerate on the DataFrame column
    for idx, image_path in enumerate(tqdm(df_images['filename'])):

        # Load image
        img = cv.imread(image_path)
        if img is None:
            print(f"Could not read image {image_path}")
            return

        # Prepare the image for the model
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Get output layer names
        ln = net.getLayerNames()
        output_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # Forward pass
        outputs = net.forward(output_layers)

        # Analyze the outputs
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    df.at[idx, classes[class_id]] = True

    return df

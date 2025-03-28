import cv2 as cv
import numpy as np
from tqdm import tqdm
from ..utils.helper_functions import download_weights
from ultralytics import YOLO
import pandas as pd
import os

def predict_imagenet_classes_yolo11(self, df_images):
    """
    Predicts ImageNet classes in a list of images using YOLO11 classification model.

    :param self: AIA object
    :param df_images: DataFrame containing image filenames.
    :return: A DataFrame containing ImageNet labels and their prediction probabilities for each image.
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()
    
    # Dictionary to collect all class probabilities before creating DataFrame
    all_probs = {}
    
    try:
        # Check if weights are downloaded already, otherwise download them
        download_weights(
            weight_filename='yolo11n-cls.pt', 
            weight_url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt'
        )

        # Load the YOLO11 classification model
        model = YOLO('../AIA/weights/yolo11n-cls.pt').to(self.device)
        
        # Initialize progress bar
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Load image
                img = cv.imread(image_path)
                if img is None:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Perform inference with the classification model
                results = model(img, verbose=False)
                
                # Process classification results
                for result in results:
                    # Get the probs attribute which contains probabilities for all classes
                    probs = result.probs
                    
                    # Add all class probabilities to our temporary dictionary
                    for i, prob in enumerate(probs.data.tolist()):
                        class_name = result.names[i]
                        column_name = f"imagenet_{class_name}"
                        
                        if column_name not in all_probs:
                            all_probs[column_name] = [0.0] * len(df_images)
                        
                        all_probs[column_name][idx] = prob

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_yolo11_imagenet'] = error
                continue
        
        # Create a DataFrame from the collected probabilities and join with original DataFrame
        probs_df = pd.DataFrame(all_probs)
        result_df = pd.concat([df, probs_df], axis=1)
        return result_df

    except Exception as e:
        print(f"Error in YOLO11 ImageNet classification setup: {str(e)}")
        return df

def predict_coco_labels_yolo11(self, df_images):
    """
    Predicts COCO labels in a list of images.

    :param self: AIA object
    :param df_images: DataFrame containing image filenames.
    :return: A DataFrame containing COCO labels and their prediction probabilities for each image.
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Check if weights are downloaded already, otherwise download them
        download_weights(
            weight_filename='yolov11n.pt', 
            weight_url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
        )

        # Load the YOLOv11 model
        model = YOLO('../AIA/weights/yolov11n.pt')

        # Load COCO labels
        try:
            with open('../AIA/weights/coco.names', 'r') as f:
                classes = ['coco_' + line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading COCO labels: {str(e)}")
            return df

        # Initialize columns for each class
        for label in classes:
            df[label] = 0.0

        # Iterate over all images
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Load image
                img = cv.imread(image_path)
                if img is None:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Perform inference
                results = model(img, verbose=False)

                # Analyze the outputs
                for result in results:
                    for detection in result.boxes:
                        class_id = int(detection.cls)
                        confidence = float(detection.conf)
                        if confidence > df.at[idx, classes[class_id]]:
                            df.at[idx, classes[class_id]] = confidence

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_yolo11_coco'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in YOLO11 COCO detection setup: {str(e)}")
        return df

def predict_coco_labels_yolo_v3(self, df_images):
    """
    Predicts COCO labels in a list of images using YOLOv3.

    :param self: AIA object
    :param df_images: DataFrame containing image filenames.
    :return: A DataFrame containing COCO labels and their prediction probabilities for each image.
    """
    # Create a copy of the input DataFrame to store results
    df = df_images.copy()

    try:
        # Check if weights are downloaded already, otherwise download them
        download_weights(
            weight_filename='yolov3.cfg', 
            weight_url='https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg'
        )
        download_weights(
            weight_filename='yolov3.weights', 
            weight_url='https://pjreddie.com/media/files/yolov3.weights'
        )
        download_weights(
            weight_filename='coco.names', 
            weight_url='https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names'
        )

        # Load the pre-trained model
        net = cv.dnn.readNetFromDarknet('../AIA/weights/yolov3.cfg', '../AIA/weights/yolov3.weights')
        
        # Use CUDA if available
        if cv.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # Load COCO labels
        try:
            with open('../AIA/weights/coco.names', 'r') as f:
                classes = ['coco_' + line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading COCO labels: {str(e)}")
            return df

        # Initialize columns for each class
        for label in classes:
            df[label] = 0.0

        # Get output layer names
        ln = net.getLayerNames()
        output_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # Process images
        for idx, image_path in enumerate(tqdm(df_images['filename'])):
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    if self.verbose: print(f"Warning: File not found: {image_path}")
                    continue

                # Load image
                img = cv.imread(image_path)
                if img is None:
                    if self.verbose: print(f"Warning: Failed to load image: {image_path}")
                    continue

                # Prepare the image for the model
                blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)

                # Forward pass
                outputs = net.forward(output_layers)

                # Analyze the outputs
                for out in outputs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = float(scores[class_id])
                        if confidence > df.at[idx, classes[class_id]]:
                            df.at[idx, classes[class_id]] = confidence

            except Exception as e:
                error = f"Error processing {image_path}: {str(e)}"
                print(error)
                df.loc[idx, 'error_yolov3'] = error
                continue

        return df

    except Exception as e:
        print(f"Error in YOLOv3 setup: {str(e)}")
        return df

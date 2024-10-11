import cv2

def extract_blur_value(image_path):
    features = {}
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set threshold at 100. Value below 100 indicates a blurry image
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()

    features['Blur Value'] = blur_value

    return features
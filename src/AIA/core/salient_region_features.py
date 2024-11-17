import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

def calculate_salient_region_features(image_path):
    """
    Calculate specified attributes of an image:
    1. Diagonal Dominance
    2. Rule of Thirds (ROT)
    3. Visual Balance Intensity
    4. Visual Balance Color

    """

    # Load the DeepLabV3 pre-trained model for saliency detection
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Load the image and preprocess
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Perform saliency detection
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    predictions = torch.argmax(output, dim=0).cpu().numpy()

    # Identify key region (saliency mask)
    saliency_mask = (predictions > 0).astype(np.uint8)

    # Calculate centroid of the salient region
    coords = np.column_stack(np.where(saliency_mask > 0))
    if len(coords) == 0:  # Handle case where no salient region is detected
        features = {
            "Diagonal Dominance": 0,
            "Rule of Thirds (ROT)": 0,
            "Visual Balance Intensity": 0,
            "Visual Balance Color": 0
        }
        return features

    centroid = np.mean(coords, axis=0)  # Centroid as (row, col)
    cy, cx = centroid  # y, x coordinates

    # Diagonal Dominance
    diag1_distance = abs(cy - cx)  # Distance from top-left to bottom-right diagonal
    diag2_distance = abs(cy - (width - cx))  # Distance from top-right to bottom-left diagonal
    diag_dominance = -min(diag1_distance, diag2_distance)

    # Rule of Thirds (ROT)
    thirds_x = [width / 3, 2 * width / 3]
    thirds_y = [height / 3, 2 * height / 3]
    intersections = [(x, y) for x in thirds_x for y in thirds_y]
    distances = [np.sqrt((cx - x) ** 2 + (cy - y) ** 2) for x, y in intersections]
    rot = -min(distances)

    # Visual Balance Intensity
    half_width = width // 2
    left_half = saliency_mask[:, :half_width]
    right_half = saliency_mask[:, half_width:width]

    # Ensure right_half matches the size of left_half
    if right_half.shape[1] > left_half.shape[1]:
        right_half = right_half[:, :left_half.shape[1]]

    left_coords = np.column_stack(np.where(left_half > 0))
    right_coords = np.column_stack(np.where(right_half > 0))

    if len(left_coords) > 0:
        left_centroid = np.mean(left_coords, axis=0)
        left_distance = abs(left_centroid[1] - half_width / 2)
    else:
        left_distance = 0

    if len(right_coords) > 0:
        right_centroid = np.mean(right_coords, axis=0)
        right_distance = abs(right_centroid[1] - half_width / 2)
    else:
        right_distance = 0

    max_distance = max(left_distance, right_distance)
    if max_distance > 0:
        balance_intensity = -abs(left_distance - right_distance) / max_distance
    else:
        balance_intensity = 0

    # Visual Balance Color
    image_np = np.array(image).astype(np.float32)
    left_half_color = image_np[:, :half_width, :]
    right_half_color = image_np[:, half_width:width, :]

    # Ensure right_half_color matches the size of left_half_color
    if right_half_color.shape[1] > left_half_color.shape[1]:
        right_half_color = right_half_color[:, :left_half_color.shape[1], :]

    right_half_color_flipped = right_half_color[:, ::-1, :]  # Flip right half horizontally

    color_diff = np.sqrt(np.sum((left_half_color - right_half_color_flipped) ** 2, axis=2))
    balance_color = -np.mean(color_diff)

    # Combine features into a dictionary
    features = {
        "Diagonal Dominance": diag_dominance,
        "Rule of Thirds (ROT)": rot,
        "Visual Balance Intensity": balance_intensity,
        "Visual Balance Color": balance_color,
    }

    return features

# Example usage
image_path = "image_6.jpg"  # Replace with your image path
features = calculate_salient_region_features(image_path)
print(features)
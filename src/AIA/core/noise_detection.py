import cv2
import numpy as np
import math
from scipy.signal import convolve2d

def estimate_noise(image_path):
     features = {}
     img = cv2.imread(image_path) 
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

     H, W = gray.shape

     M = [[1, -2, 1],
     [-2, 4, -2],
     [1, -2, 1]]

     sigma = np.sum(np.sum(np.absolute(convolve2d(gray, M))))
     sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

     # Value more than 10 indicates a noisy image
     features["Noise Value"] = sigma

     return features

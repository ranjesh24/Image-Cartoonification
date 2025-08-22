import cv2
import numpy as np


def kmeans_quantize(image, k=10):

    if image.dtype != np.uint8:
        image = np.uint8(image)
    
    
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 15,cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    quantized_pixels = centers[labels.flatten()]
    quantized_image = quantized_pixels.reshape(image.shape)
    
    return quantized_image
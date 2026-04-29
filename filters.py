import cv2
import numpy as np


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)


def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)


def brightness(image, value=30):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)


def contrast(image, value=1.5):
    return cv2.convertScaleAbs(image, alpha=value, beta=0)


def sharpen(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ])
    return cv2.filter2D(image, -1, kernel)
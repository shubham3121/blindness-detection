import os
import cv2 as cv


def read_image(directory, img_name):
    """
    Returns the image as numpy array
    """
    img_path = os.path.join(directory, img_name)
    image = cv.imread(img_path)
    return image

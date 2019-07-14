import os
from PIL import Image


def read_image(directory, img_name):
    """
    Returns the image as numpy array
    """
    img_path = os.path.join(directory, img_name)
    image = Image.open(img_path)
    return image

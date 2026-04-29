from PIL import Image
import numpy as np


def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return np.array(image)


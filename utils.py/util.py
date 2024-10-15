import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image):
    """
    Preprocesses the uploaded image:
    1. Resizes the image to 32x32.
    2. Normalizes the pixel values to [0, 1].
    3. Expands dimensions to match the input shape expected by the model.
    """
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

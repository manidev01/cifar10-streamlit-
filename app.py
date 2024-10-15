import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import utils

# Load the trained model
model = tf.keras.models.load_model('model/cifar10_model.h5')

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit app title
st.title("CIFAR-10 Image Classification")

st.write("""
Upload an image from CIFAR-10 categories such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.
The model will classify the image and show the prediction along with the confidence level.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    
    # Preprocess the image
    img_array = utils.preprocess_image(image)
    
    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Display prediction and confidence
    st.write(f"Prediction: {class_names[np.argmax(score)]}")
    st.write(f"Confidence: {100 * np.max(score):.2f}%")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the .keras model
MODEL_PATH = "distracted-21-1.00.keras"
model = load_model(MODEL_PATH)

# Define the Streamlit app
st.title("Driver Distraction Detection")
st.write("Upload an image to classify driver distraction.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing (example, modify as per your model)
    img_size = (224, 224)  # Replace with your model's input size
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0  # Normalize if required
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image_array)
    class_label = np.argmax(prediction, axis=1)

    # Display the result
    st.write("Prediction:", class_label[0])

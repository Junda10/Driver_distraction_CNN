
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
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
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize image to (64, 64)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    prediction = model.predict(img_array)
    class_label = np.argmax(prediction, axis=1)

    # Display prediction
    st.write("Prediction:", class_label[0])

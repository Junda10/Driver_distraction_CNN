
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

    # Preprocessing 
    img_size = (224, 224)  # Replace with your model's input size
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0  # Normalize if required
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

# Pass through preprocessing layers if required
if hasattr(model, 'layers') and 'Flatten' not in str(model.layers[0]):
    prediction = model.predict(image_array)
else:
    # Flatten the image manually if required by the model
    flattened_image = image_array.reshape(1, -1)  # Flatten to match the expected input shape
    prediction = model.predict(flattened_image)

class_label = np.argmax(prediction, axis=1)
st.write("Prediction:", class_label[0])

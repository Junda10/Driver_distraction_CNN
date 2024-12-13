import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "distracted-21-1.00.keras"
model = load_model(MODEL_PATH)

# Define class labels
class_labels = [
    "Normal driving",
    "Texting - right",
    "Talking on the phone - right",
    "Texting - left",
    "Talking on the phone - left",
    "Operating the radio",
    "Drinking",
    "Reaching behind",
    "Hair and makeup",
    "Talking to passenger",
]

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
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  # Get the index of the highest probability
    predicted_label = class_labels[predicted_class]  # Map index to class label

    # Display prediction
    st.write(f"Prediction: {predicted_label}")
    st.write("Class Probabilities:")
    for i, label in enumerate(class_labels):
        st.write(f"{label}: {predictions[0][i]:.2f}")

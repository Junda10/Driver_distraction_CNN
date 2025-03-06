import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2  # Still needed for image processing
from PIL import Image

# Define class labels
class_labels = [
    "c0 - Normal driving",
    "c1 - Texting - right",
    "c2 - Talking on the phone - right",
    "c3 - Texting - left",
    "c4 - Talking on the phone - left",
    "c5 - Operating the radio",
    "c6 - Drinking",
    "c7 - Reaching behind",
    "c8 - Hair and makeup",
    "c9 - Talking to passenger",
]

# Load trained CNN model
@st.cache_resource
def load_model():
    model = torch.load("best_model_CNN_96.76.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Streamlit UI
st.title("Driver Distraction Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to Tensor
    image_tensor = preprocess_image(image)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    st.write(f"**Predicted Class:** {class_labels[predicted_class.item()]}")

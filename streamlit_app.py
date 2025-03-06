import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Define class labels
class_labels = {
    0: "c0 - Normal driving",
    1: "c1 - Texting - right",
    2: "c2 - Talking on the phone - right",
    3: "c3 - Texting - left",
    4: "c4 - Talking on the phone - left",
    5: "c5 - Operating the radio",
    6: "c6 - Drinking",
    7: "c7 - Reaching behind",
    8: "c8 - Hair and makeup",
    9: "c9 - Talking to passenger"
}

# Load model
@st.cache_resource
def load_model():
    model = torch.load("best_model_CNN_96.76.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Image Preprocessing Function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match model input size
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🚗 Driver Distraction Detection")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image and predict
    with st.spinner("🔍 Analyzing..."):
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted_class = torch.max(outputs, 1)
            predicted_label = class_labels[predicted_class.item()]

    # Show result
    st.success(f"🚨 Predicted Distraction: **{predicted_label}**")

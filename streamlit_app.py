import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
from PIL import Image
import torchvision.transforms as transforms

# ----------------------- Class Labels -----------------------
class_labels = {
    0: "Normal Driving",
    1: "Texting - Right Hand",
    2: "Talking on Phone - Right Hand",
    3: "Texting - Left Hand",
    4: "Talking on Phone - Left Hand",
    5: "Operating the Radio",
    6: "Drinking",
    7: "Reaching Behind",
    8: "Hair and Makeup",
    9: "Talking to Passenger"
}

# ----------------------- Load Models -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CNN Model
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten features
        return x

# Load trained CNN model
cnn_model = ImprovedCNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load("best_model_CNN_96.76.pth", map_location=device))
cnn_model.eval()

# Load trained SVM model
svm_model = joblib.load("best_svm_classifier.pkl")

# ----------------------- Streamlit App -----------------------
st.title("🚗 Driver Distraction Detection (CNN + SVM)")

st.markdown("Upload an image, and the hybrid CNN-SVM model will classify the driver's action.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Extract CNN Features
    with torch.no_grad():
        features = cnn_model(img_tensor)
    features = features.cpu().numpy().reshape(1, -1)

    # SVM Prediction
    prediction = svm_model.predict(features)[0]

    # Display Prediction
    st.subheader("🔍 Prediction:")
    st.write(f"**Detected Activity:** {class_labels[prediction]}")

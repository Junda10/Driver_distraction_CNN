import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define CNN Model (same architecture as trained model)
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):  # 10 classes for driver distraction
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

        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Load the trained model
@st.cache_resource
def load_model():
    model = ImprovedCNN(num_classes=10)  # Ensure it matches the trained model
    model.load_state_dict(torch.load("best_model_CNN_96.76.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Load model
model = load_model()

# Class Labels for Driver Distraction Detection
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

# Streamlit UI
st.title("🚗 Driver Distraction Detection")
st.write("Upload an image, and the model will classify the driver's activity.")

# Image Preprocessing Function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# File Upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Add "Start Classification" Button
    if st.button("🚀 Start Classification"):
        # Preprocess and Predict
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            prediction = model(image_tensor)
            predicted_class_idx = torch.argmax(prediction).item()
            predicted_label = class_labels[predicted_class_idx]

        # Display Result
        st.markdown(f"### 🏆 Predicted Activity: **{predicted_label}**")

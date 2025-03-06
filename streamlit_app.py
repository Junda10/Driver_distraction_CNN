import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

# Define class labels with c0 to c9
class_labels = [
    "c0: Normal driving",
    "c1: Texting - right",
    "c2: Talking on the phone - right",
    "c3: Texting - left",
    "c4: Talking on the phone - left",
    "c5: Operating the radio",
    "c6: Drinking",
    "c7: Reaching behind",
    "c8: Hair and makeup",
    "c9: Talking to passenger",
]

# Load trained CNN model
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

# Load model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🚗 Driver Distraction Detection")
st.write("Real-time monitoring using CNN-based model")

# OpenCV Webcam Capture
cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Start webcam feed
if st.button("Start Live Tracking"):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        # Convert frame to PIL Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        # Preprocess image
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        # Predict distraction class
        with torch.no_grad():
            output = model(image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, predicted_class].item() * 100

        # Display results
        prediction_text = f"**Prediction:** {class_labels[predicted_class]} ({confidence:.2f}%)"
        frame_placeholder.image(image, channels="RGB", use_column_width=True)
        prediction_placeholder.markdown(prediction_text)

        # Alert if distraction detected
        if predicted_class != 0:
            st.warning(f"⚠️ Alert: {class_labels[predicted_class]} detected!")

        # Stop when user presses "Stop"
        if st.button("Stop"):
            break

cap.release()
st.write("Live tracking stopped.")

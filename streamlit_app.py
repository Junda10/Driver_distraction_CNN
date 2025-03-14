import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import joblib
from PIL import Image

# ----------------------- Define CNN Feature Extractor -----------------------
# CNN Model
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

# ----------------------- Load Pretrained Models -----------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained CNN as feature extractor
    cnn_model = ImprovedCNN(num_classes=10).to(device)
    cnn_model.load_state_dict(torch.load("best_model_CNN_96.76.pth", map_location=device))
    cnn_model.eval()  # Set to evaluation mode

    # Load trained SVM classifier
    svm_model = joblib.load("best_svm_classifier.pkl")

    return cnn_model, svm_model, device

# Load models
cnn_model, svm_model, device = load_models()

# ----------------------- Define Image Preprocessing -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

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

# ----------------------- Streamlit UI -----------------------
st.title("🚗 Driver Distraction Detection - Hybrid CNN-SVM")
st.write("Upload an image, and the model will classify the driver's activity.")

# File Upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Start Classification"):
        # Preprocess Image
        image_tensor = preprocess_image(image)

        # Extract CNN Features
        with torch.no_grad():
            cnn_features = cnn_model(image_tensor)  # Extract features
        cnn_features = cnn_features.cpu().numpy()  # Convert to NumPy array

        # Predict with SVM
        predicted_class_idx = svm_model.predict(cnn_features)[0]
        predicted_label = class_labels[predicted_class_idx]

        # Display Prediction
        st.markdown(f"### 🏆 Predicted Activity: **{predicted_label}**")


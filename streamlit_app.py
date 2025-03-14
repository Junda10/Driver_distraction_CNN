import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib

# -------------- Load Trained Models --------------
# Define CNN Model
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

# Load Pretrained CNN Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = ImprovedCNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load("best_model_CNN_96.76.pth", map_location=device))
cnn_model.eval()

# Remove last layer for feature extraction
feature_extractor = nn.Sequential(*list(cnn_model.children())[:-1]).to(device)

# Load Trained SVM Model
svm_model = joblib.load("best_svm_classifier.pkl")

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts PIL Image to [0,1] range tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------- Streamlit App --------------
st.title("🖼️ CNN-SVM Image Classification App 0315")
st.write("Upload an image to classify using the hybrid CNN-SVM model.")
st.write("class_labels = 0: Normal Driving")
st.write(    "1: Texting - Right Hand")
st.write(    "2: Talking on Phone - Right Hand")
st.write(    "3: Texting - Left Hand",)
st.write(    "4: Talking on Phone - Left Hand",)
st.write(    "5: Operating the Radio",)
st.write(    "6: Drinking",)
st.write(    "7: Reaching Behind",)
st.write(    "8: Hair and Makeup")
st.write(    "9: Talking to Passenger")


uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Extract Features
    with torch.no_grad():
        features = feature_extractor(image)
    features = features.view(features.size(0), -1).cpu().numpy()  # Flatten

    # SVM Prediction
    prediction = svm_model.predict(features)[0]

    # Display Prediction
    st.subheader("🔍 Prediction:")
    st.write(f"**Detected Activity:** {class_labels[prediction]}")

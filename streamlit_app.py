import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import joblib
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO

# ----------------------- Model Definitions & Loading -----------------------

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

# Load YOLO Model
yolo_model = YOLO("yolov8m.pt")

# Class Labels for Behavior Classification
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

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --------------------- Video Functions ---------------------

def process_frame(frame):
    """Process the frame for YOLO detection and CNN+SVM classification"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect persons using YOLO
    results = yolo_model(image_rgb)
    person_boxes = []
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls == 0:  # Person class
                person_boxes.append(box)
    
    if person_boxes:
        best_box = max(person_boxes, key=lambda b: b.conf.item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        person_crop = image_rgb[y1:y2, x1:x2]
        if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
            tensor = transform(person_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                features = feature_extractor(tensor)
            features = features.view(features.size(0), -1).cpu().numpy()
            prediction = svm_model.predict(features)[0]
            detected_label = class_labels[prediction]
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            cv2.putText(frame, detected_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        detected_label = "No person detected"
    
    return frame

# -------------------- Streamlit App --------------------

st.title("Driver Behavior Monitoring System")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Main Page", "Live Tracking"])

if page == "Main Page":
    st.header("Project Overview")
    st.write("""
        **Driver Behavior Monitoring System**  
        This system uses YOLOv8 for person detection combined with a custom CNN-SVM pipeline to classify driver behaviors in real-time.
        The system detects unsafe driving behaviors such as texting, talking on the phone, operating the radio, etc.
    """)

elif page == "Live Tracking":
    st.header("Live Tracking")
    st.write("This mode uses your webcam to detect and classify driver behavior in real-time.")
    
    # WebRTC streamer for live video input
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed_img = process_frame(img)
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
    
    webrtc_streamer(
        key="driver-monitoring",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

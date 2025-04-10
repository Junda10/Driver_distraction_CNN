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
import io
import av
import tempfile

# ---------------------- Load Trained Models ----------------------
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1)
        return self.fc_layers(x)

CLASS_LABELS = {
    0:"Normal Driving",1:"Texting - Right Hand",2:"Talking on Phone - Right Hand",
    3:"Texting - Left Hand",4:"Talking on Phone - Left Hand",5:"Operating the Radio",
    6:"Drinking",7:"Reaching Behind",8:"Hair and Makeup",9:"Talking to Passenger"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = ImprovedCNN(num_classes=10).to(device)
cnn_model.load_state_dict(torch.load("best_model_CNN_96.76.pth", map_location=device))
cnn_model.eval()
feature_extractor = nn.Sequential(*list(cnn_model.children())[:-1]).to(device)
svm_model = joblib.load("best_svm_classifier.pkl")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

yolo_model = YOLO("yolov8m.pt")

# ------------------ Frame Processing ------------------
def process_frame_and_label(frame):
    """
    Detects a person using YOLO and draws a bounding box,
    but uses the full image (instead of the crop) for classification.
    Returns (annotated_frame, predicted_label).
    """
    # Convert the image to RGB for YOLO
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model(rgb)
    person_detected = False

    # Use YOLO for person detection and draw the bounding box
    for res in results:
        for box in res.boxes:
            if int(box.cls.item()) == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                person_detected = True
                break  # Only use the first detected person
        if person_detected:
            break

    # Use the full image for classification
    # Convert full image (rgb) to a PIL Image
    pil_image = Image.fromarray(rgb)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(tensor)
    features = features.view(features.size(0), -1).cpu().numpy()
    prediction = svm_model.predict(features)[0]
    label = CLASS_LABELS[prediction]

    # Draw the predicted label on the full frame
    color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
    cv2.putText(frame, label, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Status: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame, label

# ------------------ Streamlit App ------------------
st.title("Driver Behavior Monitoring System")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Page", ["Main Page", "Live Tracking", "Video Detection"])

if page == "Main Page":
    st.header("Overview")
    st.write("""
      **Driver Behavior Monitoring**  
      YOLOv8 for person detection + CNN‑SVM for classifying 10 behaviors.
    """)

elif page == "Live Tracking":
    st.header("Live Tracking")
    st.write("Webcam → real‑time behavior classification")

    class Proc(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame):
            img = frame.to_ndarray(format="bgr24")
            ann, _ = process_frame_and_label(img)
            return av.VideoFrame.from_ndarray(ann, format="bgr24")

    webrtc_streamer(
        key="live",
        video_processor_factory=Proc,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

elif page == "Video Detection":
    st.header("Video Detection")
    st.subheader("Upload a video (mp4/mov/avi)")
    vid = st.file_uploader("", type=["mp4", "mov", "avi"])
    if vid:
        # Write to temp file for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        org_ph = st.empty()
        ann_ph = st.empty()
        log_ph = st.empty()

        log = []
        last_sec = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) // 1000)
            if sec > last_sec:
                ann_frame, label = process_frame_and_label(frame.copy())
                log.append({"Time (s)": sec, "Activity": label})
                last_sec = sec
            else:
                ann_frame = ann_frame.copy()

            org_ph.image(frame, channels="BGR", caption=f"Original @ {sec}s")
            ann_ph.image(ann_frame, channels="BGR", caption=f"Annotated @ {sec}s")
            log_ph.table(log)

        cap.release()
        tfile.close()

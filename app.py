import streamlit as st
import cv2
import tempfile
import torch
import math
import cvzone
import numpy as np
import pandas as pd
import difflib
from ultralytics import YOLO
from paddleocr import PaddleOCR
from image_to_text import predict_number_plate

st.set_page_config(page_title="Helmet & Number Plate Detection", layout="wide")
st.title("🚦Helmet and Number Plate Detection")

# Load YOLO model and OCR
model = YOLO("runs/detect/train2/weights/best.pt")
device = torch.device("cpu")
classNames = ["with helmet", "without helmet", "rider", "number plate"]
ocr = PaddleOCR(use_angle_cls=True, lang='en')

mode = st.sidebar.selectbox("Select Input Mode", ["Upload Image", "Upload Video", "Webcam"])

# Store unique and high-confidence OCR results
all_detected_texts = {}

def update_detected_texts(vehicle_number, ocr_conf):
    similar_found = False
    to_replace = None

    for existing_text in all_detected_texts:
        similarity = difflib.SequenceMatcher(None, vehicle_number, existing_text).ratio()
        if similarity >= 0.8:
            similar_found = True
            if ocr_conf > all_detected_texts[existing_text]:
                to_replace = existing_text
            break

    if similar_found:
        if to_replace:
            del all_detected_texts[to_replace]
            all_detected_texts[vehicle_number] = ocr_conf
    else:
        all_detected_texts[vehicle_number] = ocr_conf

# ----------------- Detection Logic -----------------
def run_detection(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device="cpu")
    
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        li = dict()
        rider_box = []

        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
        try:
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
        except:
            pass

        for box in new_boxes:
            cls = int(box[5])
            if classNames[cls] == "rider":
                x1, y1, x2, y2 = map(int, box[:4])
                rider_box.append((x1, y1, x2, y2))

        for j, box in enumerate(new_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = math.ceil((box[4] * 100)) / 100
            cls = int(box[5])

            if classNames[cls] in ["without helmet", "rider", "number plate"] and conf >= 0.45:
                for idx, rider in enumerate(rider_box):
                    if x1 + 10 >= rider[0] and y1 + 10 >= rider[1] and x2 <= rider[2] and y2 <= rider[3]:
                        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15, rt=5, colorR=(255, 0, 0))
                        cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10),
                                           scale=1.5, offset=10, thickness=2,
                                           colorT=(39, 40, 41), colorR=(248, 222, 34))
                        li.setdefault(f"rider{idx}", [])
                        li[f"rider{idx}"].append(classNames[cls])

                        if classNames[cls] == "number plate":
                            crop = img[y1:y2, x1:x2]
                            np_coords = (x1, y1)

        for key, items in li.items():
            if len(set(items)) == 3:
                try:
                    vehicle_number, ocr_conf = predict_number_plate(crop, ocr)
                    if vehicle_number and ocr_conf >= 0.6:
                        update_detected_texts(vehicle_number, ocr_conf)
                        cvzone.putTextRect(img, f"{vehicle_number} {round(ocr_conf * 100, 2)}%",
                                           (np_coords[0], np_coords[1] - 50),
                                           scale=1.5, offset=10, thickness=2,
                                           colorT=(0, 0, 0), colorR=(255, 255, 102))
                except Exception as e:
                    st.sidebar.error(f"OCR Error: {e}")
    return img

# ---------------- Image Mode ----------------
if mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        output = run_detection(img)
        st.image(output, channels="BGR")

# ---------------- Video Mode ----------------
elif mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1
            progress_bar.progress(min(processed / total_frames, 1.0))
            output = run_detection(frame)
            stframe.image(output, channels="BGR")
        cap.release()

# ---------------- Webcam Mode ----------------
elif mode == "Webcam":
    run = st.checkbox("Start Webcam")
    stframe = st.empty()
    if run:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output = run_detection(frame)
            stframe.image(output, channels="BGR")
        cap.release()

# ---------------- Sidebar Output ----------------
if all_detected_texts:
    st.sidebar.markdown("### ✅ Number Plates Detected:")
    for text in sorted(all_detected_texts.keys()):
        st.sidebar.markdown(f"- `{text}`")

    df = pd.DataFrame(sorted(all_detected_texts.keys()), columns=["Number Plate"])
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download as CSV", data=csv, file_name="detected_plates.csv", mime='text/csv')

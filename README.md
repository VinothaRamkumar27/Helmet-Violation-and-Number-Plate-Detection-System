# 🚦 Helmet and Number Plate Detection

This project is a real-time detection system that identifies motorbike riders, checks for helmet usage, and extracts vehicle number plates using YOLOv8, PaddleOCR, and Streamlit.

---

## 📌 Features

- Detects:
  -  Rider
  -  With Helmet
  -  Without Helmet
  -  Number Plates
- Uses YOLOv8 for object detection
- Uses PaddleOCR for number plate text recognition
- Supports:
  -  Image upload
  -  Video upload
  -  Live webcam feed
- Highlights violations (rider without helmet)
- Downloads all detected number plates in CSV format

---

## 🧠 Technologies Used

| Component        | Tech Stack                                     |
|------------------|------------------------------------------------|
| **Language**     | Python                                         |
| **Framework**    | [Streamlit](https://streamlit.io)              |
| **Object Detection** | [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) |
| **OCR**          | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| **UI & Drawing** | OpenCV, cvzone                                 |
| **Others**       | tempfile, torch, difflib, numpy, pandas        |

---

## 🖥️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/helmet-numberplate-detection.git
cd helmet-numberplate-detection
```
### 2. Install Dependencies

Ensure Python 3.8+ is installed. Then run:

```bash
pip install -r requirements.txt
```
### 3. Run the App
```bash
streamlit run app.py
```

## 📸 Output
1. Image ![Alt Text](https://github.com/VinothaRamkumar27/Helmet-Violation-and-Number-Plate-Detection-System/blob/c217fd1d212484f88aa0a7735e51931af441d7a6/Sample%20Outputs/image.png)

2. Video  ![Alt Text](https://github.com/VinothaRamkumar27/Helmet-Violation-and-Number-Plate-Detection-System/blob/7d295011fabbc5b64ddb455e5e553adc4e00d74c/Sample%20Outputs/video.png)

3. Webcam ![Alt Text](https://github.com/VinothaRamkumar27/Helmet-Violation-and-Number-Plate-Detection-System/blob/7d295011fabbc5b64ddb455e5e553adc4e00d74c/Sample%20Outputs/webcam.png)




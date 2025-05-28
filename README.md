
# 🧠 Head Detection using YOLOv8 🚶‍♂️🚶‍♀️

This project implements a custom-trained YOLOv8 model for **head detection** in video frames. The model can count people in a video by detecting heads, even in crowded scenes.

---

## 📁 Project Structure

```
.
├── data.yaml                # YOLOv8 data config
├── yolov8s.pt              # Trained YOLOv8 model
├── Head Detection.ipynb    # Colab notebook for training/inference
├── inference.py            # Script for running inference on video
├── output_video.mp4        # Resultant video with detections (optional)
├── runs/                   # YOLOv8 training results
├── train/ test/ valid/     # Dataset folders
├── README.dataset.txt      # Dataset info
├── README.roboflow.txt     # Roboflow export details
```

---

## 🧪 Dataset

- The dataset was created using [Roboflow](https://roboflow.com/).
- Includes images with head annotations for object detection.
- YOLO format was used for labels.

### Example Labels:
```
<cls_id> <x_center> <y_center> <width> <height>
```

---

## 🚀 How to Run Inference

### 1. Python Script (`inference.py`)
```bash
python inference.py --model yolov8s.pt --source input_video.mp4
```

### 2. Jupyter Notebook
- Open `Head Detection.ipynb`
- Run all cells to test the model on images or video.

---

## 🧠 Model

- **Model**: YOLOv8-small (`yolov8s`)
- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Trained using Google Colab with Roboflow dataset

---

## 📦 Requirements

Install dependencies:
```bash
pip install ultralytics opencv-python
```

---

## 📊 Example Results

- Supports counting people in surveillance or crowd footage
- Works in real-time video processing

---

## 📃 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## ✍️ Author

Made with ❤️ by Rahul Agarwal

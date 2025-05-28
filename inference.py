import cv2
from ultralytics import YOLO
import os

MODEL_PATH = '/Users/rahulagarwal/Downloads/best.pt'
VIDEO_PATH = '/Users/rahulagarwal/Downloads/3521318-uhd_3840_2160_30fps.mp4'
OUTPUT_PATH = '/Users/rahulagarwal/Developer/ml/output_video.mp4'
CONFIDENCE_THRESHOLD = 0.3

print("Loading model from best.pt...")
model = YOLO(MODEL_PATH)

# ------------------------------
# 🎥 LOAD VIDEO
# ------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ------------------------------
# 🚀 PROCESS FRAME BY FRAME
# ------------------------------
print("🔍 Running inference on video...")
frame_number = 0
total_persons_detected = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    person_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{result.names[cls_id]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            person_count += 1

    total_persons_detected += person_count

    # Display count on frame
    cv2.putText(frame, f'No. of Persons: {person_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save frame
    out.write(frame)

    # Optional display
    cv2.imshow('Head Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# ✅ CLEANUP
# ------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n Inference complete.")
print(f" Output video saved at: {OUTPUT_PATH}")
print(f" Total frames processed: {frame_number}")
print(f" Total person detections across all frames: {total_persons_detected}")
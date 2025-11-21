import cv2
import os
import random
import numpy as np
from ultralytics import YOLO

from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deepsort_utils import DeepSortWrapper

print("Imports done.")


yolo_model_path = "trained_models/yolov8m-tuned.pt"
model = YOLO(yolo_model_path)

deepsort = DeepSortWrapper(
    model_filename='trained_models/mars-small128.pb',
    max_cosine_distance=0.4,
    nn_budget=None
)

print("YOLO model and Deep SORT wrapper initialized.")




#IMG_W = 1920
#IMG_H = 1080

IMG_W = 960
IMG_H = 540

# Open input video
#cap = cv2.VideoCapture("videos/video_20937_20230920_133523.mp4")
#cap = cv2.VideoCapture("videos/video_634_20230920_133523.mp4")
cap = cv2.VideoCapture("videos/video_20936_20230920_133523.mp4")
if not cap.isOpened():
    print("Could not open video.")
    raise SystemExit

detection_threshold = 0.8

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Class mapping for your dataset (label_name -> class_index).
class_names = model.names

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break

    # Run YOLO model on the frame
    results = model.predict(source=frame, conf=0.5, verbose=False)[0]
    # Process detections
    detections = []
    for r in results:
        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box
            if conf >= detection_threshold:
                class_name = class_names[int(class_id)]
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(class_id)])
    
    # draw bounding boxes on the frame base on YOLO detections
    
    #for det in all_detections:
    #    x1, y1, x2, y2, score, class_id = det
    #    color = (0, 255, 0)
    #    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #    label = f"{class_names[class_id]}: {score:.2f}"
    #    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    

    # Update Deep SORT with detections
    deepsort.update(frame, detections)

    # Draw bounding boxes and text
    for track in deepsort.tracks:
        x1, y1, x2, y2 = track.bbox
        track_id = track.track_id

        cls_id = track.det_class if hasattr(track, "det_class") else None
        cls_name = class_names[cls_id] if cls_id is not None else "unknown"
        label = f"ID {track_id} - {cls_name}"

        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("YOLOv8 + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

# Release resources
cap.release()

cv2.destroyAllWindows()


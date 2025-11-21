import cv2
import os
import yaml
import json
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


IMG_W, IMG_H = 1920, 1080
#IMG_W, IMG_H = 960, 540

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


cv2.namedWindow("Map", cv2.WINDOW_NORMAL)
camera = '20936'

### GET HOMOGRAPHY DATA FROM JSON ############################################
json_path = 'calibration/zzzzzzzzzzzzzzzz.json'
with open(json_path, "r") as f:
    json_data = json.load(f)

print(f"Loaded JSON data from: {json_path}")

# Get homography matrix from JSON
if json_data.get("homography", {}).get("computed", False):
    H = np.array(json_data["homography"]["matrix"], dtype=np.float32).reshape(3, 3)
    print("Homography matrix loaded from JSON:")
    print(H)
else:
    print("No computed homography found in JSON file!")
    raise SystemExit

# Get JGW data for coordinate transformation
jgw_path = f'calibration/map_{camera}.jgw'
JGW = {}
with open(jgw_path, 'r') as f:
    lines = f.readlines()
for letter, line in zip('ADBECF', lines):
    JGW[letter] = float(line.strip())
print(f"JGW data: {JGW}")

### GOT HOMOGRAPHY DATA ###################################################


map_path = f'calibration/map_{camera}.jpeg'
map_img = cv2.imread(map_path, 1)
cv2.imshow("Map", map_img)
cv2.resizeWindow("Map", IMG_W, IMG_H)

# Process video frames
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (IMG_W, IMG_H))
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
    map_img_copy = map_img.copy()
    for track in deepsort.tracks:
        x1, y1, x2, y2 = track.bbox
        base_x = int(x1 + (x2 - x1) / 2)
        base_y = int(y2)

        track_id = track.track_id

        cls_id = track.det_class if hasattr(track, "det_class") else None
        cls_name = class_names[cls_id] if cls_id is not None else "unknown"
        label = f"ID {track_id} - {cls_name}"

        color = (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.circle(frame, (base_x,base_y), 30, (0, 255, 0), -1)
       
        # Apply homography directly to the point (no undistortion needed)
        point = np.array([[[base_x, base_y]]], dtype=np.float32)
        projected_points = cv2.perspectiveTransform(point, H)
        rounded_points = np.round(projected_points).astype(int)
        px, py = rounded_points[0][0]
        lon = round(JGW['A'] * px + JGW['B'] * py + JGW['C'], 6)
        lat = round(JGW['D'] * px + JGW['E'] * py + JGW['F'], 6)
        print(f'id: {track_id} cls: {cls_name} geo point: {lat},{lon}')



        # cv2.circle(img, center, radius, color, thickness)
        cv2.circle(map_img_copy, (px,py), 30, (0, 255, 0), -1)
        # cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)
        cv2.putText(map_img_copy, label, (px + 30, py - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    # Show the frame with detections
    cv2.imshow("YOLOv8 + DeepSORT", frame)
    cv2.imshow("Map", map_img_copy)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

# Release resources
cap.release()

cv2.destroyAllWindows()


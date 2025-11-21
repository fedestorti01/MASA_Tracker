import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
# Assicurati di avere installato le librerie necessarie:

import torch
print(torch.cuda.is_available())  # Deve restituire: True
print(torch.cuda.get_device_name(0))  # Nome della tua GPU (es. NVIDIA RTX 3060)

yolo_model_path = "trained_models/yolov8m-tuned.pt"
model = YOLO(yolo_model_path)

class_names = model.names

# Inizializza DeepSORT con embedder mobilenet (necessita tensorflow)
# embedder Choice of ['mobilenet', 'torchreid', 'clip_RN50', 'clip_RN101', 'clip_RN50x4', 'clip_RN50x16', 'clip_ViT-B/32', 'clip_ViT-B/16']
tracker = DeepSort(max_age=20, max_cosine_distance=0.5, embedder="mobilenet", embedder_gpu=True)


# Video input/output

#cap = cv2.VideoCapture("videos/video_20937_20230920_133523.mp4")
cap = cv2.VideoCapture("videos/video_634_20230920_133523.mp4")
#cap = cv2.VideoCapture("videos/video_20936_20230920_133523.mp4")


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = model.predict(source=frame, conf=0.5, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        w = x2 - x1
        h = y2 - y1
        detections.append(([x1, y1, w, h], conf, cls_id))  # xywh, conf, class_id

    # Tracciamento con DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        if hasattr(track, "to_ltrb"):
            l, t, r, b = track.to_ltrb()
        else:
            continue  # salta se to_ltrb non disponibile
        
    
        cls_id = track.det_class if hasattr(track, "det_class") else None
        cls_name = class_names[cls_id] if cls_id is not None else "unknown"

        label = f"ID {track_id} - {cls_name}"
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

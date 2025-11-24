
<img src='docs/output.png' style="width: 100%; max-width: 100%;">

# MASA Video Tracker

This repository contains a python only implementation of object detection (yolo) and tracking (DeepSORT) for MASA cameras.
It basically contains two main programs:
 * **calibration/findHmography.py** is used to calibrate cameras ans move from pixel coordinates to GPS coordinates.
 * **main.py** is the actual code processing video feed and running object detection + tracking + coordinate tranformation (homography)

## calibration

See specific documentation:
- [Calibration User Guide](docs/calibration/README_homography.md) - Guide for using calibraiton
- [QGIS User Guide](docs/calibration/README_qgis.md) - Guide for using QGIS to obtain map jgw and jpeg images
- [Modena camera location](https://www.google.com/maps/d/u/0/viewer?mid=1ktbvuJqdWIlVCGKmfdD8C_4u03SPmNdD&ll=44.65693649151485%2C10.930347761740586&z=18)

## main tracker

### How to run it

```bash
python3 main.py -camera 637 -rtsp rtsp://172.25.0.5:8554/c637 -bytetrack -gui
```

## Agruments:

Arguments:

-camera [cam id] is the number of camera
-rtsp [url] is the rtsp url (if not specified, uses video file from videos/ folder)
-gui if you want the gui to show results
-duration [seconds] duration of processing in seconds (default: 30, use 0 for infinite)
-save save performance plots to plots/ folder after processing
-yolo_model_path [path] path to YOLO model (default: trained_models/yolov8m-tuned.pt)
-deepsort_model_path [path] path to DeepSORT ReID model (default: trained_models/mars-small128.pb)

Tracking algorithms:

-deepsort use DeepSORT tracking (Deep Learning ReID + Kalman + Hungarian matching)
-kalman use Kalman-only tracking (faster alternative, motion-based only)
-bytetrack use ByteTrack tracking (YOLOv8 built-in, robust to occlusions)
-botsort use BotSORT tracking (YOLOv8 built-in, ByteTrack + ReID + camera motion compensation)

## Output
The system generates:

JSON output (real-time on console):

json[
  [1, "person", 44.656936, 10.930347],
  [2, "car", 44.657120, 10.930562]
]
Format: [track_id, class_name, latitude, longitude]

Performance plots (when using -save flag):

fps_<model>_<tracker>.png - FPS over time
memory_<model>_<tracker>.png - RAM usage
accuracy_<model>_<tracker>.png - YOLO confidence per track


GUI visualization (when using -gui flag):

Split-screen: video | georeferenced map
Bounding boxes with tracking IDs
Real-time projection on map

## Extra

- [KalmanWrapper Usage Guide](docs/KalmanWrapper_Guide.md) - Guide for using Kalman-only tracking (faster alternative to DeepSORT)

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
python3 main.py -camera 637 -rtsp rtsp://172.25.0.5:8554/c637
```

Agruments:

* -camera [cam id] is the numer of camera
* -rtsp [url] is the rtsp url
* -gui if you want the gui to show results
* -no_tracking if you want yolo without tracking
* -kalman_only if you want kalman filter tracking (insteat of DeepSORT) for speed
* -yolo_model_path [path]
* -deepsort_model_path [path]


## Extra

- [KalmanWrapper Usage Guide](docs/KalmanWrapper_Guide.md) - Guide for using Kalman-only tracking (faster alternative to DeepSORT)
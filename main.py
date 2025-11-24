import argparse
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil
from ultralytics import YOLO
from deepsort_utils import DeepSortWrapper
from deepsort_utils_kalman import KalmanWrapper
from bytetrack_utils import ByteTrackWrapper
from botsort_utils import BotSORTWrapper

DETECTION_THRESHOLD = 0.8
DISPLAY_W = 1920
DISPLAY_H = 1080
SIZE_PERF_WINDOW = 1800

@dataclass
class Config:
    camera: str
    rtsp_url: str
    gui: bool
    tracking_mode: str
    yolo_model_path: str
    deepsort_model_path: str
    detection_threshold: float = DETECTION_THRESHOLD
    duration: int = 30  # Durata processing in secondi

class CalibrationData:
    def __init__(self, camera: str):
        self.camera = camera
        self.homography_matrix = None
        self.jgw_data = {}
        self._load_calibration()

    def _load_calibration(self):
        ### Carica matrice di omografia da JSON
        json_path = f'calibration/calib_{self.camera}.json'

        try:
            with open(json_path, "r") as f:
                json_data = json.load(f)

            if not json_data.get("homography", {}).get("computed", False):
                raise ValueError("Homography non computata")

            self.homography_matrix = np.array(
                json_data["homography"]["matrix"],
                dtype=np.float32
            ).reshape(3, 3)

        except FileNotFoundError:
            print(f"File di calibrazione non trovato: {json_path}")
            raise SystemExit(1)

        ### Carica JGW
        jgw_path = f'calibration/map_{self.camera}.jgw'
        try:
            with open(jgw_path, 'r') as f:
                lines = f.readlines()

            for letter, line in zip('ADBECF', lines):
                self.jgw_data[letter] = float(line.strip())

        except FileNotFoundError:
            print(f"JGW file non trovato: {jgw_path}")  # FIX: aggiunto f-string
            raise SystemExit(1)

    def project_to_map(self, x: int, y: int) -> Tuple[float, float]:
        ### Applica omografia per trasformare coordinate pixel in pixel su mappa
        point = np.array([[[x, y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, self.homography_matrix)
        px, py = np.round(projected[0][0]).astype(int)

        ### Converte pixel in coordinate geografiche
        lon = round(self.jgw_data['A'] * px + self.jgw_data['B'] * py + self.jgw_data['C'], 6)
        lat = round(self.jgw_data['D'] * px + self.jgw_data['E'] * py + self.jgw_data['F'], 6)
        return lat, lon, (px, py)

class VideoProcessor:
    ### Gestione automatica di apertura e chiusura video
    def __init__(self, source: str):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Apertura video non riuscita: {source}")

    def __enter__(self):
        return self.cap

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
        cv2.destroyAllWindows()

    def properties(self) -> Dict[str, int]:
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS))
        }

def parse_arguments() -> argparse.Namespace:
    ### Lettura dei parametri da linea di comando
    parser = argparse.ArgumentParser(description="Multi-algorithm Object Tracking")
    parser.add_argument("-camera", type=str, default="20936", help="Name of the camera")
    parser.add_argument("-rtsp", type=str, default="", help="RTSP URL (if not set, use video file)")
    parser.add_argument("-gui", action="store_true", help="Enable GUI")
    parser.add_argument("-deepsort", action="store_true", help="Use DeepSORT tracking")
    parser.add_argument("-kalman", action="store_true", help="Use Kalman-only tracking")
    parser.add_argument("-bytetrack", action="store_true", help="Use ByteTrack (YOLOv8 built-in)")
    parser.add_argument("-botsort", action="store_true", help="Use BotSORT (YOLOv8 built-in)")
    parser.add_argument("-yolo_model_path", type=str, default="trained_models/yolov8m-tuned.pt", help="Path to YOLO model")
    parser.add_argument("-deepsort_model_path", type=str, default="trained_models/mars-small128.pb", help="Path to DeepSORT model")
    parser.add_argument("-duration", type=int, default=30, help="Durata processing in secondi (0 = infinito)")
    parser.add_argument("-save", action="store_true", help="Salva i grafici dopo averli visualizzati")
    return parser.parse_args()

def tracking_mode(args: argparse.Namespace) -> str:
    ### Indica quale algoritmo usare in base ai flag
    if args.deepsort:
        return "deepsort"
    elif args.bytetrack:
        return "bytetrack"
    elif args.kalman:
        return "kalman"
    elif args.botsort:
        return "botsort"
    else:
        print("Use: -deepsort, -kalman, -bytetrack, -botsort")
        raise SystemExit(1)

def initialize_tracker(config: Config):

    if config.tracking_mode == "deepsort":
        print("Inizializzazione DeepSORT tracker...")
        return DeepSortWrapper(
            model_filename=config.deepsort_model_path,
            max_cosine_distance=0.4,
            nn_budget=None
        )
    elif config.tracking_mode == "kalman":
        print("Inizializzazione Kalman-only tracker...")
        return KalmanWrapper(
            max_iou_distance=0.7,
            max_age=30,
            n_init=3
        )
    elif config.tracking_mode == "bytetrack":
        print("Inizializzazione ByteTrack tracker...")
        return ByteTrackWrapper(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8
        )
    elif config.tracking_mode == "botsort":
        print("Inizializzazione BotSORT tracker...")
        return BotSORTWrapper(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            with_reid=True
        )
    else:
        raise ValueError(f"Modalità tracking non supportata: {config.tracking_mode}")

def get_video_source(camera: str, rtsp_url: str) -> str:
    if rtsp_url:
        return rtsp_url

    video_files = [f for f in os.listdir("videos") if f.startswith(f"video_{camera}")]
    if not video_files:
        raise FileNotFoundError(f"File video non trovato per camera {camera}")

    return os.path.join("videos", video_files[0])

def get_camera_resolution(camera: str) -> Tuple[int, int]:
    img_path = f'calibration/camera_{camera}.jpg'

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Immagine camera non esistente {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine: {img_path}")

    return img.shape[1], img.shape[0]  # width, height

def process_yolo_detections(
        results,
        detection_threshold: float,
        class_names: Dict[int, str]
) -> Tuple[List, Dict]:
    detections = []
    yolo_conf_map = {}

    for r in results:
        ### Estrae detection da YOLO
        for box in r.boxes.data.tolist():
            if len(box) == 6:
                x1, y1, x2, y2, conf, class_id = box
            elif len(box) == 7:
                x1, y1, x2, y2, conf, class_id, track_id = box
            else:
                print(f"Formato di valore inatteso: {len(box)}")
                continue

            if conf >= detection_threshold:
                bbox = (int(x1), int(y1), int(x2), int(y2))
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(class_id)])
                yolo_conf_map[bbox] = float(conf)

    return detections, yolo_conf_map

def update_tracker(tracker, tracking_mode: str, frame, detections, results):
    if tracking_mode in ["deepsort", "kalman"]:
        tracker.update(frame, detections)
        return tracker.tracks

    elif tracking_mode in ["bytetrack", "botsort"]:
        tracker.update(frame, detections, yolo_results=results)
        return tracker.get_tracks()

    return []

def extract_track_info(track, tracking_mode: str, class_names: Dict, yolo_conf_map: Dict):
    # Tutti i wrapper ora ritornano oggetti Track con attributi uniformi
    if not hasattr(track, 'bbox') or not hasattr(track, 'track_id'):
        return None

    x1, y1, x2, y2 = track.bbox
    track_id = track.track_id
    cls_id = track.det_class if hasattr(track, "det_class") else None
    conf = track.confidence if hasattr(track, "confidence") else yolo_conf_map.get(
        (int(x1), int(y1), int(x2), int(y2)), DETECTION_THRESHOLD
    )

    base_x = int(x1 + (x2 - x1) / 2)
    base_y = int(y2)
    cls_name = class_names.get(int(cls_id), "unknown") if cls_id is not None else "unknown"

    return {
        'bbox': (int(x1), int(y1), int(x2), int(y2)),
        'base_point': (base_x, base_y),
        'track_id': track_id,
        'class_name': cls_name,
        'confidence': conf
    }

def draw_detection(frame, map_img, track_info: Dict, projected_point: Tuple, label: str):
    ### Disegni su video e mappa
    color = (0, 255, 0)
    x1, y1, x2, y2 = track_info['bbox']
    base_x, base_y = track_info['base_point']
    px, py = projected_point

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.circle(frame, (base_x, base_y), 5, (0, 255, 0), -1)

    cv2.circle(map_img, (px, py), 30, color, -1)
    cv2.putText(map_img, label, (px + 30, py - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

def create_dual_display(
        frame: np.ndarray,
        map_img: np.ndarray,
        window_width: int = DISPLAY_W,
        window_height: int = DISPLAY_H
) -> np.ndarray:
    ### Creo visualizzazione video|mappa
    half_width = window_width // 2

    frame_resized = cv2.resize(frame, (half_width, window_height))
    map_resized = cv2.resize(map_img, (half_width, window_height))

    combined = np.hstack([frame_resized, map_resized])

    cv2.line(combined, (half_width, 0), (half_width, window_height),
             (255, 255, 255), 2)

    cv2.putText(combined, "Video", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(combined, "Mappa", (half_width + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    return combined

def plot_metric(data: List, config: Dict, model_name: str, tracking_mode: str, save_plots: bool = False):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(config['x'], config['y'],
            marker=config.get('marker', 'o'),
            linestyle='-',
            color=config['color'],
            label=config['label'])

    if config.get('mean'):
        mean_val = np.mean(config['y'])
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f"Media: {mean_val:.2f}{config.get('unit', '')}")

    ax.set_title(f"{config['title']} – Modello: {model_name}", fontsize=14)
    ax.set_xlabel(config['xlabel'], fontsize=12)
    ax.set_ylabel(config['ylabel'], fontsize=12)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Salva il grafico se richiesto
    if save_plots:
        metric_type = config.get('save_name', 'metric')
        filename = f"plots/{metric_type}_{model_name}_{tracking_mode}.png"
        fig.savefig(filename)
        print(f"Grafico salvato: {filename}")
    return fig

def generate_performance_plots(
        fps_data: deque,
        memory_data: deque,
        track_confidences: Dict,
        model_path: str,
        class_names: Dict,
        timestamps: deque,
        tracking_mode: str,
        save_plots: bool = False):

    model_name = os.path.basename(model_path).replace('.pt', '')
    figures = []

    # Crea directory per i grafici se necessario
    if save_plots:
        os.makedirs('plots', exist_ok=True)

    # Accuracy YOLO
    if track_confidences:
        track_ids = list(track_confidences.keys())
        confidences = [conf for (_, conf) in track_confidences.values()]
        class_labels = [cls for (cls, _) in track_confidences.values()]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(track_ids, confidences, color='skyblue')

        for tid, conf, cls in zip(track_ids, confidences, class_labels):
            ax.text(tid, conf + 0.02, f"{cls}\n{conf:.2f}", ha='center', fontsize=9)

        mean_conf = np.mean(confidences)
        ax.axhline(mean_conf, color='red', linestyle='--', linewidth=2,
                   label=f"Media: {mean_conf:.2f}")

        ax.set_title(f"Accuracy YOLO su tracciamento – Modello: {model_name}", fontsize=14)
        ax.set_xlabel("ID Tracciamento", fontsize=12)
        ax.set_ylabel("Accuracy YOLO", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        if save_plots:
            filename = f"plots/accuracy_{model_name}_{tracking_mode}.png"
            fig.savefig(filename)
            print(f"Grafico salvato: {filename}")

        figures.append(fig)
    else:
        print("Nessun oggetto trovato: Grafico accuracy non generato")

    # Grafico FPS
    if fps_data:
        fps_values = list(fps_data)
        time_indices = list(timestamps)

        fig = plot_metric(fps_values, {
            'x': time_indices,
            'y': fps_values,
            'color': 'blue',
            'label': 'FPS per frame',
            'mean': True,
            'title': 'Andamento FPS',
            'xlabel': 'Secondi',
            'ylabel': 'FPS',
            'save_name': 'fps'
        }, model_name, tracking_mode, save_plots)
        figures.append(fig)
    else:
        print("Nessun dato FPS: Grafico FPS non generato")

    # Grafico utilizzo memoria RAM
    if memory_data:
        mem_values = list(memory_data)
        time_indices = list(timestamps)

        fig = plot_metric(mem_values, {
            'x': time_indices,
            'y': mem_values,
            'marker': 's',
            'color': 'purple',
            'label': 'Memoria (MB)',
            'mean': True,
            'unit': ' MB',
            'title': 'Andamento utilizzo RAM',
            'xlabel': 'Tempo (secondi)',
            'ylabel': 'Memoria utilizzata (MB)',
            'save_name': 'memory'
        }, model_name, tracking_mode, save_plots)
        figures.append(fig)

    plt.show(block=True)

    for fig in figures:
        plt.close(fig)

def main():
    args = parse_arguments()

    config = Config(
        camera=args.camera,
        rtsp_url=args.rtsp,
        gui=args.gui,
        tracking_mode=tracking_mode(args),
        yolo_model_path=args.yolo_model_path,
        deepsort_model_path=args.deepsort_model_path,
        duration=args.duration
    )

    print(f"Tracking selezionato: {config.tracking_mode.upper()}")

    # Inizializzo componenti
    IMG_W, IMG_H = get_camera_resolution(config.camera)
    calibration = CalibrationData(config.camera)
    tracker = initialize_tracker(config)
    model = YOLO(config.yolo_model_path)
    class_names = model.names

    video_source = get_video_source(config.camera, config.rtsp_url)

    # Carica mappa e GUI
    map_img = None
    if config.gui:
        map_path = f'calibration/map_{config.camera}.jpeg'
        map_img = cv2.imread(map_path, 1)
        if map_img is None:
            print(f"Impossibile caricare l'immagine della mappa: {map_path}")
            config.gui = False
        else:
            cv2.namedWindow("Video + Map", cv2.WINDOW_NORMAL)

    # Performance tracking
    track_confidences = defaultdict(float)
    fps_data = deque(maxlen=SIZE_PERF_WINDOW)
    memory_data = deque(maxlen=SIZE_PERF_WINDOW)
    timestamps = deque(maxlen=SIZE_PERF_WINDOW)

    # Main processing loop
    try:
        with VideoProcessor(video_source) as cap:
            print("Inizio processing su video...")

            start_process_time = time.time()

            while True:
                time_process = time.time() - start_process_time

                if config.duration > 0 and time_process >= config.duration:
                    print(f"Durata {config.duration} secondi raggiunta. Elaborazione grafici...")
                    break

                start_time = time.time()
                ret, frame = cap.read()

                if not ret or frame is None:
                    print("Fine processing video")
                    break

                frame = cv2.resize(frame, (IMG_W, IMG_H))

                # Process con YOLO
                if config.tracking_mode in ["bytetrack", "botsort"]:
                    results = model.track(
                        source=frame,
                        conf=0.6,
                        persist=True,
                        tracker=f"{config.tracking_mode}.yaml",
                        verbose=False
                    )[0]
                else:
                    results = model.predict(source=frame, conf=0.5, verbose=False)[0]

                # Process detections
                detections, yolo_conf_map = process_yolo_detections(
                    results,
                    config.detection_threshold,
                    class_names
                )

                # Aggiornamento tracker
                tracked_objects = update_tracker(
                    tracker,
                    config.tracking_mode,
                    frame,
                    detections,
                    results
                )

                if config.gui:
                    map_img_copy = map_img.copy()

                frame_detections = []

                for track in tracked_objects:
                    track_info = extract_track_info(
                        track,
                        config.tracking_mode,
                        class_names,
                        yolo_conf_map
                    )

                    if track_info is None:
                        continue

                    # Coordinate della mappa
                    lat, lon, (px, py) = calibration.project_to_map(*track_info['base_point'])

                    frame_detections.append((
                        track_info['track_id'],
                        track_info['class_name'],
                        lat,
                        lon
                    ))

                    if track_info['track_id'] not in track_confidences:
                        track_confidences[track_info['track_id']] = (
                            track_info['class_name'],
                            track_info['confidence']
                        )

                    if config.gui:
                        label = f"ID {track_info['track_id']} - {track_info['class_name']}"
                        draw_detection(
                            frame,
                            map_img_copy,
                            track_info,
                            (px, py),
                            label
                        )

                # Output detections (per MQTT o log)
                if frame_detections:
                    print(json.dumps(frame_detections, indent=2))

                # Calcolo delle performance
                end_time = time.time()
                fps = 1.0 / (end_time - start_time + 1e-8)
                memory = psutil.Process().memory_info().rss / (1024 * 1024)
                fps_data.append(fps)
                memory_data.append(memory)
                current_time = time.time() - start_process_time
                timestamps.append(current_time)

                # Display GUI
                if config.gui:
                    dual_display = create_dual_display(frame, map_img_copy)
                    cv2.imshow("Video + Map", dual_display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        print("Interruzione manuale con CTRL+C")

    finally:
        # Genera grafici
        print("Generazione dei grafici delle performance...")
        generate_performance_plots(
            fps_data,
            memory_data,
            track_confidences,
            config.yolo_model_path,
            class_names,
            timestamps,
            tracking_mode=config.tracking_mode,
            save_plots=args.save
        )
        plt.ioff()

if __name__ == "__main__":
    main()
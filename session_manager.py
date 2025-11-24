import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class SessionConfig:
    camera: str
    tracking_mode: str
    yolo_model_path: str
    deepsort_model_path: str
    detection_threshold: float
    duration: int
    rtsp_url: str
    gui: bool
    start_time: str
    video_source: str
    resolution: tuple  # (width, height)


class SessionManager:
    def __init__(self, config: SessionConfig, base_dir: str = "results"):
        self.config = config
        self.base_dir = base_dir

        # Genera nome sessione con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{config.tracking_mode}_camera{config.camera}_{timestamp}"

        # Crea directory sessione
        self.session_dir = os.path.join(base_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

        # Path dei file
        self.config_path = os.path.join(self.session_dir, "config.json")
        self.metrics_path = os.path.join(self.session_dir, "metrics.csv")
        self.tracks_path = os.path.join(self.session_dir, "tracks.csv")
        self.plots_dir = os.path.join(self.session_dir, "plots")

        # Buffer per scrittura batch
        self.metrics_buffer = []
        self.tracks_buffer = []
        self.buffer_size = 30  # Scrivi ogni 30 frame

        # Inizializza i file
        self._initialize_files()

    def _initialize_files(self):
        # Salva configurazione in JSON
        with open(self.config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        print(f"✓ Config salvata: {self.config_path}")

        # Crea CSV metriche con header
        with open(self.metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'fps',
                'memory_mb',
                'num_tracks',
                'frame_number'
            ])
        print(f"✓ Metrics CSV creato: {self.metrics_path}")

        # Crea CSV tracks con header
        with open(self.tracks_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_number',
                'timestamp',
                'track_id',
                'class_name',
                'confidence',
                'bbox_x1',
                'bbox_y1',
                'bbox_x2',
                'bbox_y2',
                'center_x',
                'center_y',
                'base_x',
                'base_y',
                'latitude',
                'longitude',
                'map_px',
                'map_py'
            ])
        print(f" Tracks CSV creato: {self.tracks_path}")

        # Crea directory per i grafici
        os.makedirs(self.plots_dir, exist_ok=True)

    def add_metrics(self, timestamp: float, fps: float, memory_mb: float,
                    num_tracks: int, frame_number: int):
        self.metrics_buffer.append([
            round(timestamp, 3),
            round(fps, 2),
            round(memory_mb, 2),
            num_tracks,
            frame_number
        ])

        # Avviso se il buffer è pieno
        if len(self.metrics_buffer) >= self.buffer_size:
            self._flush_metrics()

    def add_track(self, frame_number: int, timestamp: float, track_info: Dict):
        x1, y1, x2, y2 = track_info['bbox']
        base_x, base_y = track_info['base_point']

        self.tracks_buffer.append([
            frame_number,
            round(timestamp, 3),
            track_info['track_id'],
            track_info['class_name'],
            round(track_info['confidence'], 4),
            x1, y1, x2, y2,
            int((x1 + x2) / 2),  # center_x
            int((y1 + y2) / 2),  # center_y
            base_x,
            base_y,
            track_info.get('latitude', 0.0),
            track_info.get('longitude', 0.0),
            track_info.get('map_px', 0),
            track_info.get('map_py', 0)
        ])

        # Avviso se il buffer è pieno
        if len(self.tracks_buffer) >= self.buffer_size * 5:
            self._flush_tracks()

    def _flush_metrics(self):
        if not self.metrics_buffer:
            return

        with open(self.metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.metrics_buffer)

        self.metrics_buffer.clear()

    def _flush_tracks(self):
        if not self.tracks_buffer:
            return

        with open(self.tracks_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.tracks_buffer)

        self.tracks_buffer.clear()

    def finalize(self):
        self._flush_metrics()
        self._flush_tracks()
        print(f"\n Sessione completata: {self.session_dir}")

    def get_session_dir(self) -> str:
        return self.session_dir

    def get_plots_dir(self) -> str:
        return self.plots_dir
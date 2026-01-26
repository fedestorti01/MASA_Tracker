import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import xml.etree.ElementTree as ET

@dataclass
class SessionConfig:
    camera: str
    tracking_mode: str
    yolo_model_path: str
    deepsort_model_path: str
    detection_threshold: float
    yolo_conf_threshold: float
    iou_threshold: float
    duration: int
    rtsp_url: str
    gui: bool
    start_time: str
    video_source: str
    resolution: tuple  # (width, height)

class TrackingMetrics:
    def __init__(self):
        # Strutture dati per tracking
        self.ground_truth_tracks = defaultdict(set)
        self.predicted_tracks = defaultdict(set)
        self.track_matches = defaultdict(dict)

        self.id_mappings = defaultdict(set)

        self.total_gt = 0
        self.total_pred = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.id_switches = 0

        self.idfp = 0
        self.idfn = 0
        self.idtp = 0

    def compute_iou(self, bbox1: Tuple[int, int, int, int],
                    bbox2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def update_frame(self, frame_number: int,
                     predictions: List[Dict],
                     ground_truth: List[Dict],
                     iou_threshold: float = 0.4):

        if not ground_truth:
            raise ValueError(
                f"Ground truth mancante per frame {frame_number}. "
                f"Non chiamare update_frame se il frame non ha annotazioni GT."
            )

        pred_ids = {p['track_id']: p for p in predictions}
        gt_ids = {g['track_id']: g for g in ground_truth}

        matches = {}
        matched_gt = set()

        # Matching prediction -> ground truth con IoU
        for pred_id, pred in pred_ids.items():
            best_iou = 0
            best_gt_id = None

            for gt_id, gt in gt_ids.items():
                if gt_id in matched_gt:
                    continue

                iou = self.compute_iou(pred['bbox'], gt['bbox'])

                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_id = gt_id

            if best_gt_id is not None:
                matches[pred_id] = best_gt_id
                matched_gt.add(best_gt_id)

        # Calcolo TP, FP, FN
        tp = len(matches)
        fp = len(pred_ids) - tp
        fn = len(gt_ids) - tp

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_gt += len(gt_ids)
        self.total_pred += len(pred_ids)

        # Calcolo ID switches
        for pred_id, gt_id in matches.items():
            prev_gt_ids = self.id_mappings.get(pred_id, set())

            if prev_gt_ids and gt_id not in prev_gt_ids:
                self.id_switches += 1

            self.id_mappings[pred_id].add(gt_id)

        # Metriche per IDF1
        self.idtp += tp
        self.idfp += fp
        self.idfn += fn

        # Tracking della continuità
        self.predicted_tracks[frame_number] = set(pred_ids.keys())
        self.ground_truth_tracks[frame_number] = set(gt_ids.keys())
        self.track_matches[frame_number] = matches

    def get_metrics(self) -> Dict[str, float]:
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0

        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0

        idf1_denominator = 2 * self.idtp + self.idfp + self.idfn
        idf1 = (2 * self.idtp) / idf1_denominator if idf1_denominator > 0 else 0.0

        return {
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'idf1': round(idf1, 4),
            'id_switches': self.id_switches,
            'total_true_positives': self.total_tp,
            'total_false_positives': self.total_fp,
            'total_false_negatives': self.total_fn,
            'total_ground_truth': self.total_gt,
            'total_predictions': self.total_pred
        }


class SessionManager:
    def __init__(self, config: SessionConfig, base_dir: str = "results", gt_xml_path: str = None):
        self.config = config
        self.base_dir = base_dir
        self.gt_annotations = self._load_ground_truth(gt_xml_path) if gt_xml_path else None

        if gt_xml_path and not self.gt_annotations:
            raise RuntimeError(
                f"ERRORE: Impossibile caricare Ground Truth da {gt_xml_path}. "
                f"Le metriche di tracking non possono essere calcolate."
            )

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
        self.tracking_metrics_path = os.path.join(self.session_dir, "tracking_metrics.json")
        self.plots_dir = os.path.join(self.session_dir, "plots")

        # Buffer per scrittura batch
        self.metrics_buffer = []
        self.tracks_buffer = []
        self.buffer_size = 30

        self.tracking_metrics = TrackingMetrics()

        self.current_frame_tracks = []

        # Contatori per diagnostica
        self.frames_with_gt = 0
        self.frames_without_gt = 0

        self._initialize_files()

    def _load_ground_truth(self, xml_path: str):
        if not xml_path or not os.path.exists(xml_path):
            print(f"\n{'!' * 70}")
            print(f"ATTENZIONE: Ground Truth non trovato!")
            print(f"   Path: {xml_path}")
            print(f"   Esiste: {os.path.exists(xml_path) if xml_path else 'N/A'}")
            print(f"Le metriche di tracking NON saranno affidabili!")
            print(f"{'!' * 70}\n")
            return None

        print(f"\n{'─' * 70}")
        print(f"Caricamento Ground Truth da: {xml_path}")

        try:
            gt_by_frame = {}
            total_annotations = 0
            unique_track_ids = set()

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Parsing XML
            for track in root.findall('track'):
                gt_id = int(track.get('id'))
                unique_track_ids.add(gt_id)

                for box in track.findall('box'):
                    frame = int(box.get('frame'))
                    bbox = (
                        int(float(box.get('xtl'))),
                        int(float(box.get('ytl'))),
                        int(float(box.get('xbr'))),
                        int(float(box.get('ybr')))
                    )

                    if frame not in gt_by_frame:
                        gt_by_frame[frame] = []

                    gt_by_frame[frame].append({
                        'track_id': gt_id,
                        'bbox': bbox
                    })
                    total_annotations += 1

            # Report caricamento
            if gt_by_frame:
                frame_ids = sorted(gt_by_frame.keys())
                print(f"Ground Truth caricato con successo!")
                print(f"  • Frame annotati:        {len(gt_by_frame)}")
                print(f"  • Totale annotazioni:    {total_annotations}")
                print(f"  • Track ID unici:        {len(unique_track_ids)}")
                print(f"  • Media obj/frame:       {total_annotations / len(gt_by_frame):.2f}")
                print(f"  • Frame range:           {min(frame_ids)} → {max(frame_ids)}")
                print(f"{'─' * 70}\n")
            else:
                print(f"XML caricato ma nessuna annotazione trovata!")
                print(f"{'─' * 70}\n")
                return None

            return gt_by_frame

        except Exception as e:
            print(f"\n{'!' * 70}")
            print(f"ERRORE nel parsing del Ground Truth XML:")
            print(f"   {str(e)}")
            print(f"{'!' * 70}\n")
            return None

    def _initialize_files(self):
        # Salva configurazione in JSON (ora include nuovi parametri)
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
        print(f"✓ Tracks CSV creato: {self.tracks_path}")

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

        self.current_frame_tracks.append({
            'track_id': track_info['track_id'],
            'bbox': track_info['bbox'],
            'class_name': track_info['class_name']
        })

        if len(self.tracks_buffer) >= self.buffer_size * 5:
            self._flush_tracks()

    def finalize_frame(self, frame_number: int):
        # Se non abbiamo GT caricato, non possiamo calcolare metriche
        if not self.gt_annotations:
            print(f"⚠️  Frame {frame_number}: Nessun GT disponibile - skip metriche")
            self.current_frame_tracks = []
            return

        # Cerca GT per questo frame specifico
        real_gt = self.gt_annotations.get(frame_number)

        if real_gt is None:
            self.frames_without_gt += 1

            # Log solo ogni 50 frame per non spammare
            if self.frames_without_gt % 50 == 1:
                print(f"Frame {frame_number}: Non presente nel GT (totale skippati: {self.frames_without_gt})")

            self.current_frame_tracks = []
            return

        # Frame ha GT - procediamo con il calcolo
        self.frames_with_gt += 1

        if self.current_frame_tracks:
            try:
                # Usa IOU threshold configurabile (default 0.4)
                self.tracking_metrics.update_frame(
                    frame_number=frame_number,
                    predictions=self.current_frame_tracks,
                    ground_truth=real_gt,
                    iou_threshold=self.config.iou_threshold
                )
            except ValueError as e:
                print(f"Errore nel calcolo metriche frame {frame_number}: {e}")

        self.current_frame_tracks = []

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

    def _save_tracking_metrics(self):
        tracking_metrics = self.tracking_metrics.get_metrics()
        performance_stats = self._calculate_performance_stats()

        all_metrics = {
            "tracking_metrics": {
                "recall": tracking_metrics['recall'],
                "precision": tracking_metrics['precision'],
                "idf1": tracking_metrics['idf1'],
                "id_switches": tracking_metrics['id_switches'],
                "total_true_positives": tracking_metrics['total_true_positives'],
                "total_false_positives": tracking_metrics['total_false_positives'],
                "total_false_negatives": tracking_metrics['total_false_negatives'],
                "total_ground_truth": tracking_metrics['total_ground_truth'],
                "total_predictions": tracking_metrics['total_predictions']
            },
            "performance_metrics": performance_stats,
            "session_info": {
                "camera": self.config.camera,
                "tracking_mode": self.config.tracking_mode,
                "yolo_model": self.config.yolo_model_path,
                "detection_threshold": self.config.detection_threshold,
                "yolo_conf_threshold": self.config.yolo_conf_threshold,
                "iou_threshold": self.config.iou_threshold,
                "duration_seconds": self.config.duration,
                "start_time": self.config.start_time,
                "video_source": self.config.video_source,
                "resolution": self.config.resolution,
                "frames_with_gt": self.frames_with_gt,
                "frames_without_gt": self.frames_without_gt
            }
        }

        # Salva in JSON
        with open(self.tracking_metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n{'=' * 70}")
        print("METRICHE DI TRACKING")
        print(f"{'=' * 70}")
        print(f"Recall:              {tracking_metrics['recall']:.4f}")
        print(f"Precision:           {tracking_metrics['precision']:.4f}")
        print(f"IDF1:                {tracking_metrics['idf1']:.4f}")
        print(f"ID Switches:         {tracking_metrics['id_switches']}")
        print(f"{'-' * 70}")
        print(f"True Positives:      {tracking_metrics['total_true_positives']}")
        print(f"False Positives:     {tracking_metrics['total_false_positives']}")
        print(f"False Negatives:     {tracking_metrics['total_false_negatives']}")
        print(f"Total Ground Truth:  {tracking_metrics['total_ground_truth']}")
        print(f"Total Predictions:   {tracking_metrics['total_predictions']}")

        print(f"\n{'=' * 70}")
        print("CONFIGURAZIONE")
        print(f"{'=' * 70}")
        print(f"Detection Threshold: {self.config.detection_threshold}")
        print(f"YOLO Conf Threshold: {self.config.yolo_conf_threshold}")
        print(f"IOU Threshold:       {self.config.iou_threshold}")

        print(f"\n{'=' * 70}")
        print("METRICHE DI PERFORMANCE")
        print(f"{'=' * 70}")
        print(f"FPS medio:           {performance_stats['avg_fps']:.2f}")
        print(f"FPS min:             {performance_stats['min_fps']:.2f}")
        print(f"FPS max:             {performance_stats['max_fps']:.2f}")
        print(f"Memoria media (MB):  {performance_stats['avg_memory_mb']:.2f}")
        print(f"Memoria max (MB):    {performance_stats['max_memory_mb']:.2f}")
        print(f"Track medi/frame:    {performance_stats['avg_tracks']:.2f}")
        print(f"Track max/frame:     {performance_stats['max_tracks']}")
        print(f"Totale frame:        {performance_stats['total_frames']}")

        print(f"\n{'=' * 70}")
        print("COPERTURA GROUND TRUTH")
        print(f"{'=' * 70}")
        print(f"Frame con GT:        {self.frames_with_gt}")
        print(f"Frame senza GT:      {self.frames_without_gt}")

        total_frames = self.frames_with_gt + self.frames_without_gt
        if total_frames > 0:
            coverage = (self.frames_with_gt / total_frames) * 100
            print(f"Copertura GT:        {coverage:.1f}%")

            if coverage < 50:
                print(f"\nATTENZIONE: Copertura GT bassa (<50%)!")
                print(f"Le metriche potrebbero non essere rappresentative.")

        print(f"{'=' * 70}\n")
        print(f"Tutte le metriche salvate in: {self.tracking_metrics_path}")

    def _calculate_performance_stats(self) -> Dict[str, float]:
        stats = {
            'avg_fps': 0.0,
            'min_fps': 0.0,
            'max_fps': 0.0,
            'avg_memory_mb': 0.0,
            'max_memory_mb': 0.0,
            'avg_tracks': 0.0,
            'max_tracks': 0,
            'total_frames': 0
        }

        if not os.path.exists(self.metrics_path):
            return stats

        try:
            fps_values = []
            memory_values = []
            track_values = []

            with open(self.metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fps_values.append(float(row['fps']))
                    memory_values.append(float(row['memory_mb']))
                    track_values.append(int(row['num_tracks']))

            if fps_values:
                stats['avg_fps'] = sum(fps_values) / len(fps_values)
                stats['min_fps'] = min(fps_values)
                stats['max_fps'] = max(fps_values)

            if memory_values:
                stats['avg_memory_mb'] = sum(memory_values) / len(memory_values)
                stats['max_memory_mb'] = max(memory_values)

            if track_values:
                stats['avg_tracks'] = sum(track_values) / len(track_values)
                stats['max_tracks'] = max(track_values)

            stats['total_frames'] = len(fps_values)

        except Exception as e:
            print(f"Errore nel calcolo delle statistiche di performance: {e}")

        return stats

    def finalize(self):
        self._flush_metrics()
        self._flush_tracks()
        self._save_tracking_metrics()
        print(f"Sessione completata: {self.session_dir}")

    def get_session_dir(self) -> str:
        return self.session_dir

    def get_plots_dir(self) -> str:
        return self.plots_dir

    def get_tracking_metrics(self) -> Dict[str, float]:
        return self.tracking_metrics.get_metrics()
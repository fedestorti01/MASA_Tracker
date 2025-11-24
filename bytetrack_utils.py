import numpy as np
from typing import List, Tuple, Optional

class Track:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int],
                 det_class: int, confidence: float):
        self.track_id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.det_class = det_class
        self.confidence = confidence
        self.is_confirmed = True
        self.time_since_update = 0

class ByteTrackWrapper:
    def __init__(self,
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30):

        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.tracks = []

        print(f"ByteTrack inizializzato con parametri:")
        print(f"  - track_thresh: {track_thresh}")
        print(f"  - track_buffer: {track_buffer}")
        print(f"  - match_thresh: {match_thresh}")

    def update(self, frame: np.ndarray, detections: List, yolo_results=None) -> List[Track]:

        self.tracks = []

        if yolo_results is not None and hasattr(yolo_results, 'boxes'):
            # Usa i risultati diretti da YOLO se disponibili
            for box in yolo_results.boxes:
                if box.id is not None:  # ByteTrack assegna ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    track_id = int(box.id[0])

                    track = Track(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        det_class=cls_id,
                        confidence=conf
                    )
                    self.tracks.append(track)
        else:
            for i, det in enumerate(detections):
                if len(det) >= 6:
                    x1, y1, x2, y2, conf, cls_id = det[:6]
                    track_id = det[6] if len(det) > 6 else i

                    track = Track(
                        track_id=int(track_id),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        det_class=int(cls_id),
                        confidence=float(conf)
                    )
                    self.tracks.append(track)
        return self.tracks

    def get_tracks(self) -> List[Track]:
        return self.tracks

    def reset(self):
        self.tracks = []
        print("ByteTrack tracker resettato")

def create_bytetrack_config(
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8
) -> dict:

    return {
        'track_thresh': track_thresh,
        'track_buffer': track_buffer,
        'match_thresh': match_thresh,
        'frame_rate': 30
    }
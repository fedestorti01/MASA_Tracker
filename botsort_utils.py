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

class BotSORTWrapper:
    def __init__(self,
                 track_high_thresh: float = 0.5,
                 track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.6,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 proximity_thresh: float = 0.5,
                 appearance_thresh: float = 0.25,
                 with_reid: bool = True):

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        self.tracks = []

        print(f"BotSORT inizializzato con parametri:")
        print(f"  - track_high_thresh: {track_high_thresh}")
        print(f"  - track_low_thresh: {track_low_thresh}")
        print(f"  - new_track_thresh: {new_track_thresh}")
        print(f"  - track_buffer: {track_buffer}")
        print(f"  - match_thresh: {match_thresh}")
        print(f"  - with_reid: {with_reid}")

    def update(self, frame: np.ndarray, detections: List, yolo_results=None) -> List[Track]:
        self.tracks = []

        if yolo_results is not None and hasattr(yolo_results, 'boxes'):
            for box in yolo_results.boxes:
                if box.id is not None:  # BotSORT assegna ID
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

                    # Applica filtro basato su confidenza
                    if conf >= self.track_low_thresh:
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
        print("BotSORT tracker resettato")

    def get_feature_dim(self) -> int:
        return 512 if self.with_reid else 0

def create_botsort_config(
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        with_reid: bool = True
) -> dict:

    return {
        'track_high_thresh': track_high_thresh,
        'track_low_thresh': track_low_thresh,
        'new_track_thresh': new_track_thresh,
        'track_buffer': track_buffer,
        'match_thresh': match_thresh,
        'proximity_thresh': 0.5,
        'appearance_thresh': 0.25,
        'with_reid': with_reid
    }
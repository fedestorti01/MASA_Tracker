
import numpy as np
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

class Track:
    def __init__(self, track_id, bbox, det_class=0):
        self.track_id = track_id
        self.bbox = bbox
        self.det_class = det_class


class DeepSortWrapper:
    def __init__(self, model_filename='model_data/mars-small128.pb', max_cosine_distance=0.4, nn_budget=None):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        self.tracks = []

    def update(self, frame, detections):

        # Step 1: If no detections, run a predict-update cycle with an empty list.
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self._update_tracks()
            return

        # Step 2: Convert [x1, y1, x2, y2] to [x, y, w, h] for Deep SORT
        bboxes = np.array([d[:4] for d in detections])
        scores = [d[4] for d in detections]
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
        classes = [d[5] for d in detections]

        # Step 3: Generate appearance features for each bounding box
        features = self.encoder(frame, bboxes)

        # Step 4: Wrap everything in Deep SORT's Detection objects
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id], classes[bbox_id]))

        # Step 5: Predict and update the tracker
        self.tracker.predict()
        self.tracker.update(dets)
        self._update_tracks()

    def _update_tracks(self):
        active_tracks = []
        for track in self.tracker.tracks:
            #if not track.is_confirmed() or track.time_since_update > 10:
            #    continue
            bbox = track.to_tlbr()  # returns [x1, y1, x2, y2]
            track_id = track.track_id
            track_det_class = track.det_class
            active_tracks.append(Track(track_id, bbox, track_det_class))

        self.tracks = active_tracks
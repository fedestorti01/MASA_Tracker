import numpy as np
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection

class Track:
    def __init__(self, track_id, bbox, det_class=0):
        self.track_id = track_id
        self.bbox = bbox
        self.det_class = det_class


class NoNeuralNetworkDistanceMetric:
    """
    Custom distance metric that bypasses neural network matching entirely.
    Always returns a high cost for appearance-based matching, forcing the tracker
    to rely only on IoU matching.
    """
    
    def __init__(self, matching_threshold=1.0, budget=None):
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Dummy implementation - we don't store features"""
        pass

    def distance(self, features, targets):
        """Return high cost matrix to disable appearance matching"""
        # Return high costs so appearance matching is effectively disabled
        cost_matrix = np.full((len(targets), len(features)), self.matching_threshold)
        return cost_matrix


class KalmanWrapper:
    """
    Kalman-only tracking wrapper that uses Deep SORT's Kalman filter and IoU matching
    without neural network features. This is essentially SORT (Simple Online and Realtime Tracking).
    
    This wrapper:
    - Uses Kalman filter for motion prediction
    - Uses IoU (Intersection over Union) for data association
    - Does NOT use neural network features for appearance matching
    - Is much faster than full DeepSORT
    """
    
    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3):
        """
        Initialize Kalman-only tracker
        
        Parameters:
        -----------
        max_iou_distance : float
            Maximum IoU distance for matching detections to tracks.
            Lower values = more strict matching (default: 0.7)
        max_age : int  
            Maximum number of frames to keep a track without detection.
            Higher values = tracks survive longer without detections (default: 30)
        n_init : int
            Number of consecutive detections needed to confirm a track.
            Higher values = fewer false tracks but slower track initialization (default: 3)
        """
        # Use our custom metric that disables neural network matching
        metric = NoNeuralNetworkDistanceMetric(matching_threshold=1.0, budget=None)
        
        self.tracker = DeepSortTracker(
            metric=metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init
        )
        self.tracks = []

    def update(self, frame, detections):
        """
        Update tracker with new detections
        
        Parameters:
        -----------
        frame : np.ndarray
            Current video frame (not used in Kalman-only version)
        detections : list
            List of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        
        # Step 1: If no detections, run a predict-update cycle with an empty list
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self._update_tracks()
            return

        # Step 2: Convert [x1, y1, x2, y2] to [x, y, w, h] format for Deep SORT
        bboxes = np.array([d[:4] for d in detections])
        scores = [d[4] for d in detections]
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]  # Convert to [x, y, w, h]
        classes = [d[5] for d in detections]

        # Step 3: Create simple dummy appearance features
        # We use small non-zero values to avoid division by zero in cosine distance
        # These features will be ignored due to our custom distance metric
        features = np.ones((len(bboxes), 128)) * 0.001  # Small non-zero values

        # Step 4: Wrap everything in Deep SORT's Detection objects
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(
                bbox, 
                scores[bbox_id], 
                features[bbox_id], 
                classes[bbox_id]
            ))

        # Step 5: Predict and update the tracker
        # The Kalman filter predicts where existing tracks should be
        self.tracker.predict()
        
        # Update with new detections using IoU matching only
        self.tracker.update(dets)
        
        # Update our simplified track representation
        self._update_tracks()

    def _update_tracks(self):
        """
        Update internal track representation from Deep SORT tracker
        """
        active_tracks = []
        for track in self.tracker.tracks:
            # Optionally filter tracks based on confirmation status
            # Uncomment the line below to only include confirmed tracks
            # if not track.is_confirmed() or track.time_since_update > 10:
            #     continue
            
            bbox = track.to_tlbr()  # Convert to [x1, y1, x2, y2] format
            track_id = track.track_id
            track_det_class = track.det_class if hasattr(track, 'det_class') else 0
            
            active_tracks.append(Track(track_id, bbox, track_det_class))

        self.tracks = active_tracks

    def get_track_count(self):
        """Return number of active tracks"""
        return len(self.tracks)
    
    def get_confirmed_tracks(self):
        """Return only confirmed tracks"""
        confirmed_tracks = []
        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_tlbr()
                track_id = track.track_id
                track_det_class = track.det_class if hasattr(track, 'det_class') else 0
                confirmed_tracks.append(Track(track_id, bbox, track_det_class))
        return confirmed_tracks

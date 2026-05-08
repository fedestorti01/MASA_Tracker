import cv2
import numpy as np
import config_roi

class TrafficAnalyzer:
    def __init__(self):
        # Conversione lista di punti ROI
        self.roi_points = np.array(config_roi.ROI_VIDEO, dtype=np.int32)

        # Soglie decisionali per il numero di veicoli nel ROI
        self.threshold_intense = 5  # Da 5 auto in su: Traffico Intenso
        self.threshold_sustained = 2  # Da 2 a 4 auto: Traffico Sostenuto

    def _count_vehicles_in_roi(self, tracked_objects):
        count = 0
        ids_in_roi = set()
        class_map = {0: 'car', 1: 'bicycle', 2: 'bus', 3: 'person', 4: 'motorbike'}
        allowed_classes = ['car', 'bus', 'motorbike']

        for track in tracked_objects:
            if not hasattr(track, 'bbox'): continue

            cls_id = getattr(track, 'det_class', None)
            cls_name = class_map.get(cls_id, "unknown")
            x1, y1, x2, y2 = track.bbox
            point = (int(x1 + (x2 - x1) / 2), int(y2))
            is_inside = cv2.pointPolygonTest(self.roi_points, point, False) >= 0

            if cls_name in allowed_classes and is_inside:
                count += 1
                ids_in_roi.add(getattr(track, 'track_id', None))

        return count, ids_in_roi

    def create_pmv_display(self, messaggio, colore):
        panel = np.zeros((250, 700, 3), dtype=np.uint8)
        cv2.rectangle(panel, (10, 10), (690, 240), (60, 60, 60), 5)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.3
        thickness = 3
        text_size = cv2.getTextSize(messaggio, font, font_scale, thickness)[0]
        text_x = (700 - text_size[0]) // 2
        text_y = (250 + text_size[1]) // 2

        cv2.putText(panel, messaggio, (text_x, text_y), font, font_scale, colore, thickness)
        return panel

    def get_status(self, tracked_objects):
        num_veicoli, ids_in_roi = self._count_vehicles_in_roi(tracked_objects)

        if num_veicoli >= self.threshold_intense:
            messaggio = "CODA IN USCITA"
            colore = (0, 0, 255)
        elif num_veicoli >= self.threshold_sustained:
            messaggio = "TRAFFICO RALLENTATO"
            colore = (0, 255, 255)
        else:
            messaggio = "FLUSSO REGOLARE"
            colore = (0, 255, 0)

        pmv_image = self.create_pmv_display(messaggio, colore)
        return messaggio, colore, pmv_image, ids_in_roi
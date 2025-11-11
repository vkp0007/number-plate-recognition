# vehicle_detector.py — YOLOv11 + SORT vehicle tracking
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import torch

class VehicleDetector:
    def __init__(self, model_path="yolo11n.pt", conf_thresh=0.25):
        """
        Initializes YOLOv11 vehicle detection model with SORT tracker.
        """
        self.model = YOLO(model_path)
        self.tracker = Sort()
        self.conf_thresh = conf_thresh
        self.vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detect_and_track(self, frame):
        """
        Detect and track vehicles in a given frame.
        Returns list of (x1, y1, x2, y2, track_id)
        """
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)
        detections = []

        for r in results:
            if not hasattr(r, "boxes"):
                continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in self.vehicle_ids:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

        dets = np.asarray(detections, dtype=float) if detections else np.empty((0, 5))
        tracks = self.tracker.update(dets)

        tracked = []
        for t in tracks:
            x1, y1, x2, y2, track_id = t
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            tracked.append((x1, y1, x2, y2, int(track_id)))

        return tracked

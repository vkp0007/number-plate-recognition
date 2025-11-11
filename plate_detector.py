# plate_detector.py — YOLOv11-based license plate detector

from ultralytics import YOLO
import cv2
import numpy as np
import torch

class PlateDetector:
    def __init__(self, weights_path="best.pt", conf_thresh=0.25):
        """
        Loads YOLOv11 model for plate detection.
        """
        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detect_plates(self, frame):
        """
        Detect license plates in a given frame.
        Returns list of [x1, y1, x2, y2, conf]
        """
        results = self.model.predict(frame, conf=self.conf_thresh, imgsz=640, verbose=False)
        detections = []
        for r in results:
            if not hasattr(r, "boxes"):
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if (x2 - x1) < 20 or (y2 - y1) < 10:
                    continue
                detections.append([x1, y1, x2, y2, conf])
        return detections

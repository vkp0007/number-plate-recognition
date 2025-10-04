from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path="license_plate_detector.pt"):
        self.model = YOLO(model_path)

    def detect_plates(self, frame):
        """
        Detect license plates with lower confidence threshold
        and better handling of edge cases.
        """
        # Lower confidence to catch more plates
        results = self.model.predict(frame, conf=0.15, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Convert to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Basic validation - skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Skip very small detections (likely noise)
                width = x2 - x1
                height = y2 - y1
                if width < 20 or height < 10:
                    continue

                # Check aspect ratio (plates are typically wider than tall)
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                    # Still include but with lower priority
                    pass

                detections.append([x1, y1, x2, y2, conf, cls])

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x[4], reverse=True)

        return detections
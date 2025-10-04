import numpy as np
from ultralytics import YOLO
from sort.sort import Sort  # tracking


class VehicleDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.tracker = Sort()
        # COCO vehicle IDs: car=2, motorcycle=3, bus=5, truck=7
        self.vehicle_ids = [2, 3, 5, 7]
        self.vehicle_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def detect_and_track(self, frame):
        """
        Detect vehicles with lower confidence threshold
        and better tracking information.
        """
        # Lower confidence to catch more vehicles
        results = self.model.predict(frame, conf=0.2, verbose=False)

        detections = []
        detected_vehicles_info = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_ids:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])

                    # Convert to float for numpy array
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                    detections.append([x1, y1, x2, y2, conf])

                    # Store vehicle info for debugging
                    vehicle_type = self.vehicle_names.get(cls, "unknown")
                    detected_vehicles_info.append({
                        'type': vehicle_type,
                        'conf': conf,
                        'bbox': (x1, y1, x2, y2)
                    })

        # Convert to numpy array for SORT tracker
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

        # Update tracker
        tracks = self.tracker.update(detections)

        # Print summary instead of line-by-line (less verbose)
        if len(detected_vehicles_info) > 0:
            unique_types = {}
            for v in detected_vehicles_info:
                vtype = v['type']
                unique_types[vtype] = unique_types.get(vtype, 0) + 1

            summary = ", ".join([f"{count} {vtype}(s)" for vtype, count in unique_types.items()])
            print(f"[YOLO] Frame: {summary} detected → {len(tracks)} tracked")

        return tracks

    def detect_and_track_verbose(self, frame):
        """
        More verbose version - shows each detection.
        Use this for debugging.
        """
        results = self.model.predict(frame, conf=0.2, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_ids:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    detections.append([x1, y1, x2, y2, conf])

                    vehicle_type = self.vehicle_names.get(cls, "unknown")
                    print(f"[YOLO] {vehicle_type.capitalize()} detected with conf={conf:.2f}")

        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        tracks = self.tracker.update(detections)
        return tracks
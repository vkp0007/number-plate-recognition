# main.py — Full ANPR pipeline (optimized for Colab GPU)
import cv2
import os
import numpy as np
import csv
import time
from concurrent.futures import ThreadPoolExecutor

from vehicle_detector import VehicleDetector
from plate_detector import PlateDetector
from paddle_ocr_utils import read_plate_paddleocr
from ocr_preprocess import preprocess_plate_for_ocr

# ======================================
# CONFIGURATION
# ======================================
VIDEO_PATH = "Untitled design.mp4"
OUTPUT_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "recognized_plates.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_SKIP = 3
DISPLAY_SCALE = 0.6
MAX_OCR_THREADS = 4

print("[INFO] Initializing models...")
vehicle_detector = VehicleDetector("yolo11n.pt", conf_thresh=0.4)
plate_detector = PlateDetector("best.pt", conf_thresh=0.4)
print("[INFO] Models loaded successfully.\n")

vehicle_cache = {}  # vehicle_id: (plate_text, conf)
executor = ThreadPoolExecutor(max_workers=MAX_OCR_THREADS)

# ======================================
# HELPER FUNCTIONS
# ======================================
def draw_label(frame, text, x, y):
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)

def run_ocr_async(plate_crop, vehicle_id):
    plate_crop = preprocess_plate_for_ocr(plate_crop)
    text, conf = read_plate_paddleocr(plate_crop)
    vehicle_cache[vehicle_id] = (text, conf)
    return text, conf

# ======================================
# MAIN LOOP
# ======================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    exit()

print("[INFO] Running ANPR — Press 'q' to stop.\n")
frame_count = 0
cv2.namedWindow("ANPR", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame_small = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    vehicles = vehicle_detector.detect_and_track(frame_small)

    for (x1, y1, x2, y2, vid) in vehicles:
        vehicle_crop = frame_small[y1:y2, x1:x2]
        plate_dets = plate_detector.detect_plates(vehicle_crop)

        for (px1, py1, px2, py2, _) in plate_dets:
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            if vid not in vehicle_cache:
                executor.submit(run_ocr_async, plate_crop, vid)

            plate_text, conf = vehicle_cache.get(vid, ("Reading...", 0.0))
            cv2.rectangle(frame_small, (x1, y1), (x2, y2), (255, 120, 0), 2)
            draw_label(frame_small, f"ID {vid} | {plate_text}", x1, y1)

    cv2.imshow("ANPR", frame_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)

# ======================================
# SAVE RESULTS
# ======================================
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Vehicle_ID", "Plate_Text", "Confidence", "Timestamp"])
    for vid, (text, conf) in vehicle_cache.items():
        writer.writerow([vid, text, f"{conf:.2f}", time.strftime("%H:%M:%S")])

print(f"\n✅ Saved recognized plates to: {CSV_PATH}")
print("[INFO] ANPR processing completed.")

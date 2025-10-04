import cv2
import os
import re
from collections import defaultdict
from vehicle_detector import VehicleDetector
from plate_detector import PlateDetector
from ocr_utils import read_license_plate
import csv

# --- CSV logging setup ---
CSV_PATH = "recognized_plates.csv"
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_ID", "Vehicle_ID", "Plate_Text", "OCR_Conf"])

# ==========================
# CONFIGURATION - RELAXED THRESHOLDS
# ==========================
VIDEO_SOURCE = "Untitled design.mp4"
SAVE_DIR = "debug_plates"
os.makedirs(SAVE_DIR, exist_ok=True)

# Lowered thresholds to catch more plates
VEHICLE_CONF_THRESH = 0.3  # Was 0.5
PLATE_CONF_THRESH = 0.25  # Was 0.4
OCR_CONF_THRESH = 0.35  # Was 0.6 - CRITICAL CHANGE

# More flexible regex - accepts more variations
# Now matches: 2-3 letters, 1-2 digits, 1-3 letters, 3-5 digits
PLATE_REGEX = re.compile(r"^[A-Z]{2,3}\d{1,2}[A-Z]{1,3}\d{3,5}$")

# Track previously seen plates (to avoid duplicates)
seen_plates = defaultdict(int)

# Debug counters
debug_stats = {
    "vehicles_detected": 0,
    "plates_detected": 0,
    "ocr_attempts": 0,
    "ocr_success": 0,
    "regex_failed": 0,
    "low_conf": 0
}

# ==========================
# INITIALIZATION
# ==========================
vehicle_detector = VehicleDetector("yolov8n.pt")
plate_detector = PlateDetector("license_plate_detector.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)
cv2.namedWindow("License Plate Detection + OCR", cv2.WINDOW_NORMAL)

frame_id = 0
save_id = 0


def clean_plate_text(text: str) -> str:
    """Normalize OCR output to uppercase alphanumeric only."""
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    return text


def is_valid_plate_format(text: str) -> bool:
    """
    More lenient validation for Indian plates.
    Accepts various formats but checks basic structure.
    """
    if len(text) < 6 or len(text) > 13:
        return False

    # Count letters and digits
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)

    # Must have at least 2 letters and 3 digits
    if letters < 2 or digits < 3:
        return False

    # Try regex match - if fails, still accept if basic structure is good
    if PLATE_REGEX.match(text):
        return True

    # Fallback: Accept if it starts with letters and has good mix
    if text[0:2].isalpha() and letters >= 3 and digits >= 3:
        return True

    return False


# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  End of video or no frame captured.")
        print(f"\n📊 FINAL STATS:")
        print(f"   Vehicles detected: {debug_stats['vehicles_detected']}")
        print(f"   Plates detected: {debug_stats['plates_detected']}")
        print(f"   OCR attempts: {debug_stats['ocr_attempts']}")
        print(f"   OCR success: {debug_stats['ocr_success']}")
        print(f"   Failed regex: {debug_stats['regex_failed']}")
        print(f"   Low confidence: {debug_stats['low_conf']}")
        break

    frame_id += 1
    if frame_id % 2 != 0:
        continue  # process every other frame for better FPS

    # ---- Step 1: Vehicle detection ----
    vehicles = vehicle_detector.detect_and_track(frame)
    debug_stats["vehicles_detected"] += len(vehicles)

    for (vx1, vy1, vx2, vy2, track_id) in vehicles:
        vx1, vy1, vx2, vy2 = map(int, [vx1, vy1, vx2, vy2])
        car_roi = frame[vy1:vy2, vx1:vx2]

        if car_roi.size == 0:
            continue

        # ---- Step 2: License plate detection ----
        plates = plate_detector.detect_plates(car_roi)
        plates = [p for p in plates if p[4] > PLATE_CONF_THRESH]

        debug_stats["plates_detected"] += len(plates)

        if not plates:
            continue

        for (px1, py1, px2, py2, conf, cls) in plates:
            px1, py1 = vx1 + int(px1), vy1 + int(py1)
            px2, py2 = vx1 + int(px2), vy1 + int(py2)

            if px2 <= px1 or py2 <= py1:
                continue

            plate_crop = frame[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            # ---- Step 3: OCR ----
            debug_stats["ocr_attempts"] += 1
            plate_text, ocr_conf = read_license_plate(plate_crop)
            plate_text = clean_plate_text(plate_text)

            # Debug: Print all OCR results
            if plate_text:
                print(f"[🔍 OCR] Car {int(track_id)} → Raw: '{plate_text}' (conf={ocr_conf:.2f})")

            if not plate_text:
                print(f"[❌ OCR] Car {int(track_id)} → Empty text")
                continue

            if ocr_conf < OCR_CONF_THRESH:
                print(f"[⚠️ LOW_CONF] Car {int(track_id)} → '{plate_text}' conf={ocr_conf:.2f} < {OCR_CONF_THRESH}")
                debug_stats["low_conf"] += 1
                continue

            # Validate plate format (more lenient now)
            if not is_valid_plate_format(plate_text):
                print(f"[⚠️ FORMAT] Car {int(track_id)} → Invalid: '{plate_text}'")
                debug_stats["regex_failed"] += 1
                continue

            # Skip excessive duplicates for same vehicle
            if plate_text in seen_plates and seen_plates[plate_text] > 3:
                continue

            seen_plates[plate_text] += 1
            debug_stats["ocr_success"] += 1

            # ---- Step 4: Save to CSV ----
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([frame_id, int(track_id), plate_text, f"{ocr_conf:.2f}"])

            # ---- Step 5: Draw + Save ----
            save_id += 1
            filename = f"{plate_text}_{frame_id}_{save_id}.jpg"
            path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(path, plate_crop)

            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 3)
            cv2.putText(frame, f"{plate_text} ({ocr_conf:.2f})",
                        (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            print(f"[✅ SUCCESS] Car {int(track_id)} → Plate: {plate_text} (conf={ocr_conf:.2f})")
            print(f"📸 Saved: {path}")

    # ---- Step 6: Display ----
    display = cv2.resize(frame, (960, 540))
    cv2.imshow("License Plate Detection + OCR", display)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

print("\n" + "=" * 50)
print("Processing complete! Check 'recognized_plates.csv' for results.")
print("=" * 50)
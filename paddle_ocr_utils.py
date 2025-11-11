# paddle_ocr_utils.py — PaddleOCR-based plate text recognition
import re
import cv2
import numpy as np
from paddleocr import PaddleOCR

print("[INFO] Loading PaddleOCR (v3.x compatible)...")
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Auto CPU/GPU

def clean_text(text: str) -> str:
    """Remove special characters and normalize uppercase."""
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()

def read_plate_paddleocr(plate_img):
    """
    Recognize license plate text using PaddleOCR.
    Returns (text, confidence)
    """
    if plate_img is None or plate_img.size == 0:
        return "", 0.0

    h, w = plate_img.shape[:2]
    if h < 60:
        scale = max(2.0, 100 / h)
        plate_img = cv2.resize(plate_img, (int(w * scale), int(h * scale)))

    try:
        results = ocr.ocr(plate_img)
    except Exception as e:
        print(f"[WARN] PaddleOCR failed: {e}")
        return "", 0.0

    if not results or not results[0]:
        return "", 0.0

    lines = []
    for det in results[0]:
        if len(det) < 2:
            continue

        text_part = det[1]
        if isinstance(text_part, (list, tuple)):
            if len(text_part) == 2 and isinstance(text_part[1], (int, float)):
                text, conf = text_part
            else:
                text = str(text_part[0])
                conf = 0.8
        else:
            text = str(text_part)
            conf = 0.8

        text = clean_text(text)
        if len(text) >= 3:
            lines.append((text, conf))

    if not lines:
        return "", 0.0

    # Best result = longest valid string * confidence
    text, conf = max(lines, key=lambda x: len(x[0]) * x[1])
    return text, float(conf)

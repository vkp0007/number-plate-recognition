import cv2
import numpy as np
import easyocr
import re

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'], gpu=False)


def preprocess_plate(image):
    """Enhance plate ROI for better OCR with multiple preprocessing strategies."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise and sharpen
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # Adaptive threshold to handle lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )

    # Optional: Morphological operations to strengthen characters
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph


def preprocess_plate_v2(image):
    """Alternative preprocessing - better for different lighting conditions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

    # Otsu's thresholding
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def clean_text(text):
    """Remove unwanted characters and normalize text."""
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    return text


def read_license_plate(image):
    """
    Run OCR on preprocessed plate image with multiple preprocessing attempts.
    Returns (text, confidence)
    """
    # Resize if too small
    h, w = image.shape[:2]
    if h < 50 or w < 150:
        scale = max(50 / h, 150 / w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    all_results = []

    # Try original image first
    try:
        results_orig = reader.readtext(image, detail=1)
        for (bbox, text, conf) in results_orig:
            text = clean_text(text)
            if len(text) >= 4:  # Reduced from 4 to catch more
                all_results.append((text, conf, "original"))
    except Exception as e:
        print(f"[OCR ERROR] Original image: {e}")

    # Try preprocessed version 1
    try:
        processed1 = preprocess_plate(image)
        results1 = reader.readtext(processed1, detail=1)
        for (bbox, text, conf) in results1:
            text = clean_text(text)
            if len(text) >= 4:
                all_results.append((text, conf, "preprocess_v1"))
    except Exception as e:
        print(f"[OCR ERROR] Preprocess v1: {e}")

    # Try preprocessed version 2
    try:
        processed2 = preprocess_plate_v2(image)
        results2 = reader.readtext(processed2, detail=1)
        for (bbox, text, conf) in results2:
            text = clean_text(text)
            if len(text) >= 4:
                all_results.append((text, conf, "preprocess_v2"))
    except Exception as e:
        print(f"[OCR ERROR] Preprocess v2: {e}")

    # Find best result
    best_text = ""
    best_conf = 0.0
    best_method = ""

    for (text, conf, method) in all_results:
        # Prefer longer texts with reasonable confidence
        score = conf * (1 + len(text) * 0.05)  # Slight bonus for longer texts
        if score > best_conf * (1 + len(best_text) * 0.05):
            best_text, best_conf, best_method = text, conf, method

    # Debug: Show all attempts
    if all_results:
        print(f"[OCR DEBUG] All attempts: {[(t, f'{c:.2f}', m) for t, c, m in all_results]}")
        if best_text:
            print(f"[OCR DEBUG] Best: '{best_text}' conf={best_conf:.2f} method={best_method}")

    return best_text, best_conf


def read_license_plate_fast(image):
    """
    Faster version - single preprocessing only.
    Use this if the detailed version is too slow.
    """
    # Resize if too small
    h, w = image.shape[:2]
    if h < 50 or w < 150:
        scale = max(50 / h, 150 / w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    processed = preprocess_plate(image)
    results = reader.readtext(processed, detail=1)

    best_text = ""
    best_conf = 0.0

    for (bbox, text, conf) in results:
        text = clean_text(text)
        if len(text) < 4:  # Skip short junk reads
            continue
        if conf > best_conf:
            best_text, best_conf = text, conf

    return best_text, best_conf
# ocr_preprocess.py — Preprocessing for OCR input enhancement
import cv2
import numpy as np

def preprocess_plate_for_ocr(img):
    """
    Enhanced preprocessing for license plate images before OCR.
    Improves readability by applying denoising, contrast enhancement,
    binarization, deskewing, and morphology.
    """
    if img is None or img.size == 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure black text on white background
    if np.mean(binary > 127) < 0.5:
        binary = cv2.bitwise_not(binary)

    # Deskewing using image moments
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1] if len(coords) > 0 else 0
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Resize for OCR (standard height)
    target_h = 60
    aspect = w / float(h)
    new_w = int(aspect * target_h)
    binary = cv2.resize(binary, (new_w, target_h), interpolation=cv2.INTER_AREA)

    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return processed

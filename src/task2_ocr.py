"""
Task 2 — Student Information Extraction [15 points]
Extracts Name and Registration Number from quiz sheet using OCR.
"""

import cv2
import numpy as np
import re
from dataclasses import dataclass
from typing import Optional

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    _easyocr_reader = None
except ImportError:
    EASYOCR_AVAILABLE = False
    _easyocr_reader = None


@dataclass
class StudentInfo:
    name: str = ""
    reg_no: str = ""

    def to_dict(self):
        return {"name": self.name, "reg_no": self.reg_no}


def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None and EASYOCR_AVAILABLE:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """Clean up a region of interest for better OCR."""
    # Upscale for better OCR
    scale = 2
    roi = cv2.resize(roi, (roi.shape[1] * scale, roi.shape[0] * scale),
                     interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    return thresh


def ocr_region(roi: np.ndarray, expected_type: str = "text") -> str:
    """Run OCR on a preprocessed region. Falls back through available engines."""
    processed = preprocess_roi(roi)

    text = ""

    # Try EasyOCR first (better for handwriting)
    reader = get_easyocr_reader()
    if reader:
        try:
            results = reader.readtext(processed, detail=0, paragraph=True)
            text = " ".join(results).strip()
        except Exception:
            pass

    # Fallback to Tesseract
    if not text and TESSERACT_AVAILABLE:
        try:
            cfg = "--psm 7 -c tessedit_char_blacklist=|" if expected_type == "reg" else "--psm 6"
            text = pytesseract.image_to_string(processed, config=cfg).strip()
        except Exception:
            pass

    return text


def find_field_region(image: np.ndarray, label_pattern: str,
                      search_top_fraction: float = 0.35) -> Optional[np.ndarray]:
    """
    Find a labeled field (e.g., 'Name:') and return the ROI to its right / below.
    Searches in the top portion of the page where student info typically lives.
    """
    h, w = image.shape[:2]
    search_region = image[:int(h * search_top_fraction), :]

    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY) if len(search_region.shape) == 3 else search_region

    # Use Tesseract to locate words if available
    if TESSERACT_AVAILABLE:
        try:
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT,
                                              config="--psm 11")
            texts = data["text"]
            lefts = data["left"]
            tops = data["top"]
            widths = data["width"]
            heights = data["height"]

            for i, t in enumerate(texts):
                if re.search(label_pattern, t, re.IGNORECASE):
                    # ROI: same row, to the right of the label
                    x_start = lefts[i] + widths[i]
                    y_start = max(0, tops[i] - 5)
                    y_end = tops[i] + heights[i] + 10
                    x_end = w
                    roi = search_region[y_start:y_end, x_start:x_end]
                    if roi.size > 0:
                        return roi
        except Exception:
            pass

    return None


def heuristic_split(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback: split top portion horizontally into name / reg areas
    when label detection fails.
    Most quiz formats put Name on the left and Reg No on the right in the top band.
    """
    h, w = image.shape[:2]
    top_band = image[:int(h * 0.25), :]
    left_half = top_band[:, :w // 2]
    right_half = top_band[:, w // 2:]
    return left_half, right_half


def clean_name(raw: str) -> str:
    """Remove OCR noise from a name field."""
    raw = re.sub(r"[^A-Za-z\s\.\-]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw.title()


def clean_reg_no(raw: str) -> str:
    """Normalise registration number (e.g., FA22-BSE-001)."""
    # Keep alphanumerics and common separators
    raw = re.sub(r"[^A-Za-z0-9\-/]", "", raw)
    return raw.upper()


def extract_student_info(image_input) -> StudentInfo:
    """
    Main function: Task 2 entry point.
    Accepts: file path (str) or numpy array.
    Returns: StudentInfo(name, reg_no)
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_input}")
    else:
        image = image_input.copy()

    info = StudentInfo()

    # Attempt label-based detection
    name_roi = find_field_region(image, r"name", search_top_fraction=0.35)
    reg_roi = find_field_region(image, r"reg|registration|roll", search_top_fraction=0.35)

    # Fallback to heuristic split
    if name_roi is None or reg_roi is None:
        left, right = heuristic_split(image)
        if name_roi is None:
            name_roi = left
        if reg_roi is None:
            reg_roi = right

    if name_roi is not None and name_roi.size > 0:
        raw_name = ocr_region(name_roi, "text")
        info.name = clean_name(raw_name)

    if reg_roi is not None and reg_roi.size > 0:
        raw_reg = ocr_region(reg_roi, "reg")
        info.reg_no = clean_reg_no(raw_reg)

    print(f"[Task 2] Student: name='{info.name}', reg_no='{info.reg_no}'")
    return info


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/demo_quiz.jpg"
    result = extract_student_info(path)
    print(json.dumps(result.to_dict(), indent=2))

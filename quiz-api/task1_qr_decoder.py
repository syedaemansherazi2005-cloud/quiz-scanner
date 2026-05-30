"""
Task 1 — QR Code Decoding
Decodes the answer key embedded in a QR code on the quiz sheet.
"""

import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# pyzbar disabled — using OpenCV built-in QR decoder
PYZBAR_OK = False


@dataclass
class AnswerKey:
    quiz_set: str = "A"
    part1: dict = field(default_factory=dict)
    part2: dict = field(default_factory=dict)
    negative_marking: float = 0.0

    def to_dict(self):
        return {
            "quiz_set": self.quiz_set,
            "part1": self.part1,
            "part2": self.part2,
            "negative_marking": self.negative_marking,
        }


def _decode_raw(image: np.ndarray) -> Optional[str]:
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(image)
    return data if data else None


def _preprocess_variants(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    yield gray
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield thresh
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yield clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    yield cv2.filter2D(gray, -1, kernel)


def _try_decode(image: np.ndarray) -> Optional[str]:
    for variant in _preprocess_variants(image):
        for angle in [0, 90, 180, 270]:
            img = variant
            if angle != 0:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
                img = cv2.warpAffine(img, M, (w, h))
            result = _decode_raw(img)
            if result:
                return result
    return None


def _parse_payload(payload: str) -> AnswerKey:
    key = AnswerKey()

    set_match = re.search(r"Set-([A-Z])", payload, re.IGNORECASE)
    if set_match:
        key.quiz_set = set_match.group(1).upper()

    def parse_part(label):
        pattern = rf"{label}:\s*(.*?)(?:\||$)"
        match = re.search(pattern, payload, re.IGNORECASE)
        if not match:
            return {}
        answers = {}
        for q_match in re.finditer(r"Q(\d+)=([A-D])", match.group(1), re.IGNORECASE):
            answers[f"Q{int(q_match.group(1)):02d}"] = q_match.group(2).upper()
        return answers

    key.part1 = parse_part("Part-I")
    key.part2 = parse_part("Part-II")

    neg_match = re.search(r"neg(?:ative)?[:\s]*([\d.]+)", payload, re.IGNORECASE)
    if neg_match:
        key.negative_marking = float(neg_match.group(1))

    return key


def decode_answer_key(image: np.ndarray) -> Optional[AnswerKey]:
    payload = _try_decode(image)
    if not payload:
        return None
    return _parse_payload(payload)


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is None:
        print("Usage: python task1_qr_decoder.py <image_path>")
        sys.exit(1)
    key = decode_answer_key(img)
    print(key.to_dict() if key else "No QR code found")
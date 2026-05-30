"""
Task 1 — QR Code Decoding [15 points]
Detects and decodes QR code from quiz image to extract answer key.
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnswerKey:
    quiz_set: str = ""
    subject: str = ""
    class_name: str = ""
    semester: str = ""
    negative_marking: float = 0.0
    part1: dict = field(default_factory=dict)  # {Q01: 'A', ...}
    part2: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "quiz_set": self.quiz_set,
            "subject": self.subject,
            "class": self.class_name,
            "semester": self.semester,
            "negative_marking": self.negative_marking,
            "part1": self.part1,
            "part2": self.part2,
        }


def preprocess_for_qr(image: np.ndarray) -> list[np.ndarray]:
    """Generate multiple preprocessed variants to maximise QR detection."""
    variants = [image]

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    variants.append(thresh)

    # Sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    variants.append(sharpened)

    # CLAHE (contrast limited adaptive histogram equalization) — good for glare
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants.append(clahe_img)

    return variants


def try_rotations(image: np.ndarray) -> list[np.ndarray]:
    """Try 0, 90, 180, 270 degree rotations."""
    rotations = [image]
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        rotations.append(rotated)
    return rotations


def parse_qr_payload(payload: str) -> AnswerKey:
    """
    Parse QR payload string into AnswerKey.
    Expected format:
    AI Quiz SP2026 Set-C | Part-I: Q1=D Q2=A ... | Part-II: Q1=C Q2=D ...
    Optional: | NegMarking=0.25
    """
    key = AnswerKey()
    payload = payload.strip()

    # Extract set identifier — text before first '|'
    parts = [p.strip() for p in payload.split("|")]
    if parts:
        header = parts[0]
        # Try to extract Set
        set_match = re.search(r"Set[-\s]?([A-Za-z0-9]+)", header, re.IGNORECASE)
        if set_match:
            key.quiz_set = set_match.group(1).upper()
        # Subject / class info
        sem_match = re.search(r"(SP|FA|SU)\s*\d{4}", header, re.IGNORECASE)
        if sem_match:
            key.semester = sem_match.group(0).upper()
        key.subject = header.strip()

    def parse_answers(text: str) -> dict:
        """Parse Q1=D Q2=A ... into {Q01: D, Q02: A, ...}"""
        answers = {}
        matches = re.findall(r"Q(\d+)\s*=\s*([A-Da-d])", text)
        for num, ans in matches:
            q_key = f"Q{int(num):02d}"
            answers[q_key] = ans.upper()
        return answers

    for part in parts[1:]:
        part = part.strip()
        if re.match(r"Part[-\s]?I\s*:", part, re.IGNORECASE) and "II" not in part.upper()[:10]:
            key.part1 = parse_answers(part)
        elif re.match(r"Part[-\s]?II\s*:", part, re.IGNORECASE):
            key.part2 = parse_answers(part)
        elif "negmark" in part.lower() or "negative" in part.lower():
            nm = re.search(r"[\d.]+", part)
            if nm:
                key.negative_marking = float(nm.group())

    return key


def decode_answer_key(image_input) -> Optional[AnswerKey]:
    """
    Main function: Task 1 entry point.
    Accepts: file path (str) or numpy array.
    Returns: AnswerKey or None if QR not found.
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_input}")
    else:
        image = image_input.copy()

    # Resize large images for speed
    max_dim = 1600
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    variants = preprocess_for_qr(image)
    all_images = []
    for v in variants:
        all_images.extend(try_rotations(v))

    for img in all_images:
        decoded_objects = pyzbar.decode(img)
        for obj in decoded_objects:
            if obj.type == "QRCODE":
                payload = obj.data.decode("utf-8", errors="replace")
                answer_key = parse_qr_payload(payload)
                print(f"[Task 1] QR decoded successfully. Set: {answer_key.quiz_set}")
                print(f"  Part-I : {answer_key.part1}")
                print(f"  Part-II: {answer_key.part2}")
                return answer_key

    print("[Task 1] WARNING: No QR code found in image.")
    return None


# ── Demo / test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Generate a synthetic QR for demo purposes
        import qrcode

        payload = (
            "AI Quiz SP2026 Set-C | "
            "Part-I: Q1=D Q2=A Q3=B Q4=A Q5=D Q6=A Q7=A Q8=B | "
            "Part-II: Q1=C Q2=D Q3=D Q4=D Q5=C Q6=C Q7=C Q8=B"
        )
        qr = qrcode.make(payload)
        demo_path = "samples/demo_qr.png"
        qr.save(demo_path)
        print(f"[Demo] QR saved to {demo_path}")
        key = decode_answer_key(demo_path)
    else:
        key = decode_answer_key(sys.argv[1])

    if key:
        import json
        print("\nAnswer Key:")
        print(json.dumps(key.to_dict(), indent=2))

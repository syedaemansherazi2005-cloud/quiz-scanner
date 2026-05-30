"""
Task 3 — Bubble Sheet Reading [20 points]
Detects and reads filled bubbles (A/B/C/D) for Part-I and Part-II.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StudentAnswers:
    part1: dict = field(default_factory=dict)   # {Q01: 'A' | None | 'INVALID'}
    part2: dict = field(default_factory=dict)

    def to_dict(self):
        return {"part1": self.part1, "part2": self.part2}


# ── Image preprocessing ───────────────────────────────────────────────────────

def deskew(image: np.ndarray) -> np.ndarray:
    """Correct moderate tilt using Hough lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return image
    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if abs(angle) < 20:
            angles.append(angle)
    if not angles:
        return image
    median_angle = float(np.median(angles))
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def perspective_correct(image: np.ndarray) -> np.ndarray:
    """Attempt to find and warp the document to a top-down view."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    # Find largest quadrilateral
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return image
    pts = approx.reshape(4, 2).astype(np.float32)
    # Order: top-left, top-right, bottom-right, bottom-left
    rect = order_points(pts)
    w_max = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
    h_max = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))
    dst = np.array([[0, 0], [w_max - 1, 0], [w_max - 1, h_max - 1], [0, h_max - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w_max, h_max))


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ── Grid detection ────────────────────────────────────────────────────────────

def detect_bubble_grid(image: np.ndarray, n_questions: int = 8,
                        n_options: int = 4) -> Optional[list[list[tuple]]]:
    """
    Detect circles in the image and organise them into a grid.
    Returns list of rows, each row is a list of (cx, cy, r) per option.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=60,
        param2=30,
        minRadius=10,
        maxRadius=40,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    # Convert to list of (cx, cy, r)
    circle_list = [(c[0], c[1], c[2]) for c in circles]
    if len(circle_list) < n_questions * n_options:
        return None

    # Sort by y (row), then x (column)
    circle_list.sort(key=lambda c: (c[1], c[0]))

    # Cluster into rows by y-proximity
    rows = []
    current_row = [circle_list[0]]
    for c in circle_list[1:]:
        if abs(c[1] - current_row[-1][1]) < 25:
            current_row.append(c)
        else:
            rows.append(sorted(current_row, key=lambda x: x[0]))
            current_row = [c]
    rows.append(sorted(current_row, key=lambda x: x[0]))

    # Filter rows to exactly n_options circles
    valid_rows = [r for r in rows if len(r) == n_options]

    if len(valid_rows) < n_questions:
        return None

    return valid_rows[:n_questions]


def fill_ratio(image: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Return fraction of dark pixels inside a circle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    total = np.sum(mask > 0)
    if total == 0:
        return 0.0
    # Dark pixels = filled
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    filled = np.sum((thresh > 0) & (mask > 0))
    return filled / total


# ── Fallback: contour-based bubble detection ──────────────────────────────────

def detect_bubbles_contour(image: np.ndarray, n_questions: int = 8,
                             n_options: int = 4) -> Optional[list[list[tuple]]]:
    """Alternative detection using contours (for printed grids without perfect circles)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200 or area > 5000:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        # Circularity check
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri ** 2)
        if circularity > 0.6:
            bubbles.append((int(x), int(y), int(r)))

    if len(bubbles) < n_questions * n_options:
        return None

    bubbles.sort(key=lambda c: (c[1], c[0]))

    rows = []
    current_row = [bubbles[0]]
    for c in bubbles[1:]:
        if abs(c[1] - current_row[-1][1]) < 30:
            current_row.append(c)
        else:
            rows.append(sorted(current_row, key=lambda x: x[0]))
            current_row = [c]
    rows.append(sorted(current_row, key=lambda x: x[0]))

    valid_rows = [r for r in rows if len(r) == n_options]
    if len(valid_rows) < n_questions:
        return None
    return valid_rows[:n_questions]


# ── Answer reading ────────────────────────────────────────────────────────────

OPTION_LABELS = ["A", "B", "C", "D"]
FILL_THRESHOLD = 0.35   # fraction above which a bubble is considered "filled"


def read_grid(image: np.ndarray, grid: list[list[tuple]]) -> dict:
    """Given a detected grid of circle tuples, return {Q01: 'A'|None|'INVALID'}."""
    results = {}
    for q_idx, row in enumerate(grid):
        q_key = f"Q{q_idx + 1:02d}"
        filled = []
        for opt_idx, (cx, cy, r) in enumerate(row):
            ratio = fill_ratio(image, cx, cy, r)
            if ratio >= FILL_THRESHOLD:
                filled.append(OPTION_LABELS[opt_idx])

        if len(filled) == 0:
            results[q_key] = None          # unattempted
        elif len(filled) == 1:
            results[q_key] = filled[0]     # correct
        else:
            results[q_key] = "INVALID"     # multiple filled

    return results


def split_image_for_parts(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Attempt to split the answer grid area into Part-I and Part-II sections.
    Simple approach: bottom half of page is split vertically or horizontally.
    """
    h, w = image.shape[:2]
    # Skip the top 30% (header, student info, QR)
    answer_area = image[int(h * 0.30):, :]
    ah = answer_area.shape[0]
    # Part-I left half, Part-II right half (common layout)
    part1 = answer_area[:, : w // 2]
    part2 = answer_area[:, w // 2 :]
    return part1, part2


def read_bubble_sheet(image_input) -> StudentAnswers:
    """
    Main function: Task 3 entry point.
    Accepts: file path (str) or numpy array.
    Returns: StudentAnswers with part1 and part2 dicts.
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_input}")
    else:
        image = image_input.copy()

    # Step 1: deskew
    image = deskew(image)

    answers = StudentAnswers()

    part1_img, part2_img = split_image_for_parts(image)

    for part_name, part_img, target in [
        ("Part-I", part1_img, "part1"),
        ("Part-II", part2_img, "part2"),
    ]:
        # Try Hough circles first, then contour fallback
        grid = detect_bubble_grid(part_img)
        if grid is None:
            grid = detect_bubbles_contour(part_img)

        if grid is not None:
            result = read_grid(part_img, grid)
            setattr(answers, target, result)
            print(f"[Task 3] {part_name}: {result}")
        else:
            # Fill with None (unattempted) so grading doesn't crash
            fallback = {f"Q{i:02d}": None for i in range(1, 9)}
            setattr(answers, target, fallback)
            print(f"[Task 3] WARNING: Could not detect bubble grid for {part_name}.")

    return answers


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/demo_quiz.jpg"
    result = read_bubble_sheet(path)
    print(json.dumps(result.to_dict(), indent=2))

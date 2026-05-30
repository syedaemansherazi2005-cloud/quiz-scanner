"""
generate_samples.py — Creates synthetic quiz sheet images for testing.
"""

import os
import numpy as np
import cv2

try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    print("WARNING: qrcode library not installed.")

os.makedirs("samples", exist_ok=True)

QR_PAYLOAD = (
    "AI Quiz SP2026 Set-C | "
    "Part-I: Q1=D Q2=A Q3=B Q4=A Q5=D Q6=A Q7=A Q8=B | "
    "Part-II: Q1=C Q2=D Q3=D Q4=D Q5=C Q6=C Q7=C Q8=B"
)

STUDENTS = [
    {"name": "Ali Hassan",    "reg": "FA22-BSE-041",
     "p1": ["D","A","B","A","D","A","A","B"],
     "p2": ["C","D","D","D","C","C","C","B"]},
    {"name": "Sara Ahmed",   "reg": "FA22-BSE-042",
     "p1": ["D","A","C","A","D","B","A","B"],
     "p2": ["C","D","D","A","C","C","C","B"]},
    {"name": "Usman Khan",   "reg": "FA22-BSE-043",
     "p1": ["D","B","B","A","D","A","A","B"],
     "p2": ["C","D","B","D","C","A","C","B"]},
    {"name": "Fatima Malik", "reg": "FA22-BSE-044",
     "p1": ["A","A","B","A","D","A","A","B"],
     "p2": ["C","D","D","D","C","C","C","B"]},
    {"name": "Zain Raza",    "reg": "FA22-BSE-045",
     "p1": ["D","A","B","C","A","A","A","B"],
     "p2": ["C","D","D","D","B","C","C","A"]},
]

OPTIONS = ["A", "B", "C", "D"]

BG     = (255, 255, 255)
BLACK  = (20, 20, 20)
GRAY   = (160, 160, 160)
BUBBLE = (220, 220, 220)
FILLED = (30, 30, 30)
BLUE   = (60, 100, 200)


def draw_quiz_sheet(student: dict, index: int) -> np.ndarray:
    W, H = 900, 1100
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(img, (0, 0), (W, 80), (240, 245, 255), -1)
    cv2.rectangle(img, (0, 80), (W, 81), (180, 180, 200), -1)
    cv2.putText(img, "QUIZ — AI (BSE-4A)", (30, 35), font, .9, BLACK, 2)
    cv2.putText(img, "SP2026   Time: 20 min   Marks: 16   Set-C",
                (30, 65), font, .5, GRAY, 1)

    qr_size = 150
    if QR_AVAILABLE:
        qr = qrcode.QRCode(box_size=4, border=2)
        qr.add_data(QR_PAYLOAD)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_arr = np.array(qr_img.convert("RGB"))
        qr_arr = cv2.resize(qr_arr, (qr_size, qr_size))
        img[10:10+qr_size, W-10-qr_size:W-10] = qr_arr
    else:
        cv2.rectangle(img, (W-10-qr_size, 10), (W-10, 10+qr_size), BLACK, 2)
        cv2.putText(img, "QR CODE",
                    (W-10-qr_size+20, 10+qr_size//2), font, .5, BLACK, 1)

    y0 = 110
    cv2.putText(img, "Name:", (30, y0), font, .55, BLACK, 1)
    cv2.putText(img, student["name"], (130, y0), font, .6, BLUE, 2)
    cv2.line(img, (125, y0+4), (550, y0+4), GRAY, 1)

    cv2.putText(img, "Reg #:", (30, y0+35), font, .55, BLACK, 1)
    cv2.putText(img, student["reg"], (130, y0+35), font, .6, BLUE, 2)
    cv2.line(img, (125, y0+39), (550, y0+39), GRAY, 1)

    cv2.putText(img,
                "Class: BSE-4A    Subject: Artificial Intelligence",
                (30, y0+65), font, .45, GRAY, 1)

    cv2.line(img, (30, 200), (W-30, 200), (200, 200, 200), 1)

    def draw_bubble_section(part_label, answers, x_off, answers_key):
        col_w = 50
        row_h = 52
        bub_r = 16
        y_start = 230

        cv2.putText(img, part_label, (x_off, y_start - 15),
                    font, .65, BLACK, 2)

        for j, opt in enumerate(OPTIONS):
            cv2.putText(img, opt,
                        (x_off + 50 + j * col_w + bub_r - 5, y_start + 10),
                        font, .45, GRAY, 1)

        for i, (student_ans, _correct) in enumerate(
                zip(answers, answers_key)):
            y = y_start + 20 + i * row_h
            cv2.putText(img, f"Q{i+1:02d}", (x_off, y + bub_r),
                        font, .45, BLACK, 1)
            for j, opt in enumerate(OPTIONS):
                cx = x_off + 55 + j * col_w + bub_r
                cy = y + bub_r
                is_filled = (opt == student_ans)
                if is_filled:
                    cv2.circle(img, (cx, cy), bub_r, FILLED, -1)
                else:
                    cv2.circle(img, (cx, cy), bub_r, BUBBLE, -1)
                    cv2.circle(img, (cx, cy), bub_r, GRAY, 1)

    KEY_P1 = ["D","A","B","A","D","A","A","B"]
    KEY_P2 = ["C","D","D","D","C","C","C","B"]

    draw_bubble_section("Part-I  (Questions 1-8)", student["p1"], 30, KEY_P1)
    draw_bubble_section("Part-II (Questions 1-8)", student["p2"], 470, KEY_P2)

    cv2.putText(img,
                "* Fill bubble completely | Use pencil | Do not fold",
                (30, H-30), font, .4, GRAY, 1)

    return img


for i, student in enumerate(STUDENTS):
    sheet = draw_quiz_sheet(student, i)
    out_path = f"samples/student_{i+1:02d}_{student['reg'].replace('-','')}.jpg"
    cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Generated: {out_path}")

print(f"\n✓ {len(STUDENTS)} sample quiz sheets saved to samples/")
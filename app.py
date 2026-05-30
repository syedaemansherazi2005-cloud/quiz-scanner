"""
Flask Web Application — Quiz Scanner & Grading System
Provides a web UI for single-image and batch processing.
"""

import os
import sys
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from flask import (Flask, render_template, request, jsonify,
                   send_file, redirect, url_for, flash)
from werkzeug.utils import secure_filename

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from task1_qr_decoder import decode_answer_key, AnswerKey
from task2_ocr import extract_student_info
from task3_bubble_reader import read_bubble_sheet
from task4_grader import grade_quiz, print_report
from task5_batch import run_batch

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "quiz-scanner-secret-2026")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# In-memory job store for batch progress
batch_jobs = {}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan_single():
    """Process a single uploaded quiz image."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)

    try:
        import cv2
        image = cv2.imread(save_path)
        if image is None:
            return jsonify({"error": "Cannot read image file"}), 400

        # Run pipeline
        answer_key = decode_answer_key(image)
        student_info = extract_student_info(image)
        student_answers = read_bubble_sheet(image)

        fallback_payload = request.form.get("fallback_key", "")
        if answer_key is None and fallback_payload:
            try:
                data = json.loads(fallback_payload)
                answer_key = AnswerKey(**data)
            except Exception:
                pass

        if answer_key is None:
            return jsonify({
                "warning": "QR code not detected. Grading requires an answer key.",
                "student": student_info.to_dict(),
                "answers": student_answers.to_dict(),
                "report": None,
            })

        report = grade_quiz(
            student_answers, answer_key,
            student_name=student_info.name,
            reg_no=student_info.reg_no,
        )

        return jsonify({
            "success": True,
            "student": student_info.to_dict(),
            "answers": student_answers.to_dict(),
            "answer_key": answer_key.to_dict(),
            "report": report.to_dict(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(save_path)
        except Exception:
            pass


@app.route("/batch", methods=["POST"])
def batch_process():
    """Accept multiple images, process in background, return job ID."""
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    quiz_label = request.form.get("quiz_label", "Quiz 1")

    if not files:
        return jsonify({"error": "No files selected"}), 400

    job_id = uuid.uuid4().hex
    batch_dir = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(batch_dir, exist_ok=True)

    saved = []
    for file in files:
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            path = os.path.join(batch_dir, fname)
            file.save(path)
            saved.append(path)

    if not saved:
        return jsonify({"error": "No valid image files"}), 400

    batch_jobs[job_id] = {"status": "running", "total": len(saved), "done": 0, "result_path": None}

    def _run():
        try:
            out = run_batch(batch_dir, quiz_label=quiz_label, output_dir=OUTPUT_FOLDER)
            batch_jobs[job_id]["status"] = "done"
            batch_jobs[job_id]["result_path"] = out
        except Exception as e:
            batch_jobs[job_id]["status"] = "error"
            batch_jobs[job_id]["error"] = str(e)
        finally:
            import shutil
            try:
                shutil.rmtree(batch_dir)
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "total": len(saved)})


@app.route("/batch/status/<job_id>")
def batch_status(job_id):
    job = batch_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/batch/download/<job_id>")
def batch_download(job_id):
    job = batch_jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Report not ready"}), 404
    path = job["result_path"]
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


@app.route("/generate-sample-qr")
def generate_sample_qr():
    """Generate a demo QR image for testing."""
    try:
        import qrcode
        import io
        payload = (
            "AI Quiz SP2026 Set-C | "
            "Part-I: Q1=D Q2=A Q3=B Q4=A Q5=D Q6=A Q7=A Q8=B | "
            "Part-II: Q1=C Q2=D Q3=D Q4=D Q5=C Q6=C Q7=C Q8=B"
        )
        qr = qrcode.make(payload)
        buf = io.BytesIO()
        qr.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png",
                         download_name="sample_answer_key_qr.png", as_attachment=True)
    except ImportError:
        return jsonify({"error": "qrcode library not installed"}), 500


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Quiz Scanner & Grading System")
    print("  http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

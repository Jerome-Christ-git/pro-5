"""Flask web app for CopyrightGuard AI."""
import os
import json
import threading
import uuid
from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, redirect, url_for)

from . import engine

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
OUTPUT_DIR = os.path.join(BASE, "outputs")
REPORT_DIR = os.path.join(BASE, "reports")
for d in (UPLOAD_DIR, OUTPUT_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

JOBS = {}  # job_id -> {"status": ..., "progress": ..., "result": ...}


def process_job(job_id, video_path):
    def cb(msg):
        JOBS[job_id]["progress"] = msg
    try:
        JOBS[job_id]["status"] = "running"
        result = engine.run_pipeline(video_path, BASE, progress_cb=cb)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "No file"}), 400
    job_id = uuid.uuid4().hex[:10]
    ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
    path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    f.save(path)
    JOBS[job_id] = {"status": "queued", "progress": "Queued…"}
    threading.Thread(target=process_job, args=(job_id, path), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(j)


@app.route("/result/<job_id>")
def result(job_id):
    j = JOBS.get(job_id)
    if not j or j.get("status") != "done":
        return redirect(url_for("index"))
    return render_template("result.html", job=j, jid=job_id)


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT_DIR, name, as_attachment=False)


@app.route("/reports/<path:name>")
def reports(name):
    return send_from_directory(REPORT_DIR, name, as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print(f"\n🛡️ CopyrightGuard AI running on port {port}\n")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

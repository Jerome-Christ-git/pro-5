"""
CopyrightGuard AI - core analysis + sanitization engine.
Runs fully offline using open-source AI libraries.
"""
import os
import cv2
import json
import uuid
import shutil
import subprocess
import numpy as np
from datetime import datetime
from PIL import Image
import imagehash
import librosa
import soundfile as sf

from app.brands import COPYRIGHTED_KEYWORDS, LOGO_HINT_WORDS

# Lazy-load EasyOCR (heavy)
_reader = None
def get_ocr():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader


def extract_audio(video_path, out_wav):
    """Use ffmpeg to pull audio track as mono 16k wav."""
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
           "-ar", "16000", "-f", "wav", out_wav]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def analyze_audio(wav_path):
    """
    Detect music-like segments (potential copyrighted music).
    Uses spectral flatness + harmonic ratio via librosa.
    Returns list of (start_sec, end_sec, reason, confidence).
    """
    findings = []
    if not os.path.exists(wav_path):
        return findings
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if len(y) == 0:
        return findings

    hop = 2048
    frame = 4096
    # Harmonic-percussive separation → music tends to be harmonic-rich
    y_harm, y_perc = librosa.effects.hpss(y)
    # RMS per window
    rms_h = librosa.feature.rms(y=y_harm, frame_length=frame, hop_length=hop)[0]
    rms_p = librosa.feature.rms(y=y_perc, frame_length=frame, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms_h)), sr=sr, hop_length=hop)

    # Music heuristic: harmonic energy dominates + low flatness (tonal)
    ratio = rms_h / (rms_p + 1e-6)
    music_mask = (ratio > 1.8) & (flatness < 0.25) & (rms_h > 0.02)

    # Merge contiguous windows into segments
    seg_start = None
    for i, m in enumerate(music_mask):
        t = float(times[i])
        if m and seg_start is None:
            seg_start = t
        elif not m and seg_start is not None:
            if t - seg_start >= 3.0:  # only report >=3s music
                conf = float(min(1.0, (t - seg_start) / 30.0 + 0.4))
                findings.append({
                    "type": "audio",
                    "start": round(seg_start, 2),
                    "end": round(t, 2),
                    "reason": "Music-like audio segment (possible copyrighted track)",
                    "confidence": round(conf, 2),
                })
            seg_start = None
    if seg_start is not None:
        t = float(times[-1])
        if t - seg_start >= 3.0:
            findings.append({
                "type": "audio",
                "start": round(seg_start, 2),
                "end": round(t, 2),
                "reason": "Music-like audio segment (possible copyrighted track)",
                "confidence": 0.6,
            })
    return findings


def sample_frames(video_path, every_sec=1.0):
    """Yield (timestamp, frame_bgr) sampled once per `every_sec`."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * every_sec))
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        yield ts, frame
        idx += step
    cap.release()


def analyze_frames(video_path, progress_cb=None):
    """
    Run OCR + perceptual hashing on sampled frames.
    Returns (visual_findings, frame_blur_map)
    frame_blur_map: {timestamp: [ (x,y,w,h), ... ]} rectangles to blur.
    """
    ocr = get_ocr()
    findings = []
    blur_map = {}
    hashes = []

    frames = list(sample_frames(video_path, every_sec=1.0))
    total = len(frames)
    for i, (ts, frame) in enumerate(frames):
        if progress_cb and i % 3 == 0:
            progress_cb(f"Scanning frame {i+1}/{total} @ {ts:.1f}s")

        # Perceptual hash (for future dataset compare / duplicate detection)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h = imagehash.phash(pil)
        hashes.append((ts, str(h)))

        # OCR — downscale for speed
        small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        scale_x = frame.shape[1] / small.shape[1]
        scale_y = frame.shape[0] / small.shape[0]
        try:
            results = ocr.readtext(small)
        except Exception:
            results = []

        for bbox, text, conf in results:
            if conf < 0.35 or not text.strip():
                continue
            low = text.lower()
            matched = None
            for kw in COPYRIGHTED_KEYWORDS:
                if kw in low:
                    matched = kw
                    break
            if matched is None:
                for h_kw in LOGO_HINT_WORDS:
                    if h_kw.lower() in low:
                        matched = h_kw
                        break
            if matched:
                (x1, y1) = bbox[0]
                (x2, y2) = bbox[2]
                x, y = int(min(x1, x2) * scale_x), int(min(y1, y2) * scale_y)
                w, hh = int(abs(x2 - x1) * scale_x), int(abs(y2 - y1) * scale_y)
                # pad
                pad = 10
                x, y = max(0, x - pad), max(0, y - pad)
                w, hh = w + 2*pad, hh + 2*pad
                blur_map.setdefault(round(ts, 2), []).append((x, y, w, hh))
                findings.append({
                    "type": "visual",
                    "timestamp": round(ts, 2),
                    "text": text.strip(),
                    "match": matched,
                    "confidence": round(float(conf), 2),
                    "reason": f"Brand / copyrighted keyword detected: '{matched}'",
                })
    return findings, blur_map, hashes


def sanitize_video(input_path, output_path, blur_map, mute_segments, progress_cb=None):
    """
    Rebuild the video:
      - Blur rectangles from blur_map (applied for ~1s window around each ts)
      - Mute audio segments in mute_segments (list of (start,end))
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_video = output_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (W, H))

    # Pre-index blur windows: for each ts key, apply to frames within +/-0.5s
    blur_events = sorted(blur_map.items())

    def rects_for_time(t):
        rects = []
        for ts, rs in blur_events:
            if abs(ts - t) <= 0.6:
                rects.extend(rs)
        return rects

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        for (x, y, w, hh) in rects_for_time(t):
            x2, y2 = min(W, x + w), min(H, y + hh)
            x, y = max(0, x), max(0, y)
            if x2 > x and y2 > y:
                roi = frame[y:y2, x:x2]
                if roi.size:
                    frame[y:y2, x:x2] = cv2.GaussianBlur(roi, (51, 51), 30)
        writer.write(frame)
        idx += 1
        if progress_cb and idx % 60 == 0:
            progress_cb(f"Rebuilding video {idx}/{total} frames")
    cap.release()
    writer.release()

    # Extract audio, mute segments, then remux
    tmp_audio = output_path + ".audio.wav"
    tmp_audio_clean = output_path + ".clean.wav"
    try:
        extract_audio(input_path, tmp_audio)
        y, sr = librosa.load(tmp_audio, sr=None, mono=True)
        for (s, e) in mute_segments:
            a = int(s * sr); b = int(e * sr)
            a = max(0, a); b = min(len(y), b)
            if b > a:
                y[a:b] = 0.0
        sf.write(tmp_audio_clean, y, sr)
        has_audio = True
    except Exception:
        has_audio = False

    # Mux with ffmpeg
    if has_audio:
        cmd = ["ffmpeg", "-y", "-i", tmp_video, "-i", tmp_audio_clean,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-c:a", "aac", "-b:a", "160k", "-shortest", output_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", tmp_video,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for f in [tmp_video, tmp_audio, tmp_audio_clean]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass


def compute_risk_score(visual, audio):
    score = 0
    score += min(60, len(visual) * 8)
    score += min(40, sum((f["end"] - f["start"]) for f in audio) * 1.5)
    return int(min(100, score))


def risk_label(score):
    if score >= 70: return ("HIGH", "#ff2e63")
    if score >= 35: return ("MEDIUM", "#ffb400")
    if score >= 10: return ("LOW", "#00d9a6")
    return ("SAFE", "#00d9a6")


def generate_report(job_id, video_name, visual, audio, score, report_path,
                    sanitized_name):
    label, color = risk_label(score)
    rows_v = "".join(
        f"<tr><td>{f['timestamp']}s</td><td>{f['match']}</td>"
        f"<td>{f['text']}</td><td>{int(f['confidence']*100)}%</td></tr>"
        for f in visual) or "<tr><td colspan=4 style='text-align:center;opacity:.6'>None detected 🎉</td></tr>"
    rows_a = "".join(
        f"<tr><td>{f['start']}s → {f['end']}s</td><td>{f['reason']}</td>"
        f"<td>{int(f['confidence']*100)}%</td></tr>"
        for f in audio) or "<tr><td colspan=3 style='text-align:center;opacity:.6'>None detected 🎉</td></tr>"

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Copyright Report — {video_name}</title>
<style>
body{{font-family:'Segoe UI',sans-serif;background:#0d0b1f;color:#eef;margin:0;padding:40px}}
h1{{background:linear-gradient(90deg,#00e0ff,#ff2e97);-webkit-background-clip:text;color:transparent;font-size:38px;margin:0}}
.card{{background:#171432;border-radius:20px;padding:24px;margin:20px 0;box-shadow:0 8px 40px #0007}}
.badge{{display:inline-block;padding:10px 22px;border-radius:30px;font-weight:800;font-size:20px;background:{color};color:#111}}
table{{width:100%;border-collapse:collapse;margin-top:12px}}
th,td{{padding:10px;border-bottom:1px solid #2a2650;text-align:left;font-size:14px}}
th{{color:#9df;text-transform:uppercase;font-size:12px;letter-spacing:1px}}
.meter{{height:22px;background:#2a2650;border-radius:20px;overflow:hidden;margin-top:10px}}
.meter>div{{height:100%;width:{score}%;background:linear-gradient(90deg,#00e0ff,#ff2e97);}}
small{{opacity:.6}}
</style></head><body>
<h1>🛡️ CopyrightGuard AI — Analysis Report</h1>
<small>Job {job_id} • {datetime.now().strftime('%Y-%m-%d %H:%M')} • File: {video_name}</small>
<div class='card'>
  <h2>Overall Risk: <span class='badge'>{label} — {score}/100</span></h2>
  <div class='meter'><div></div></div>
  <p>Sanitized video: <b>{sanitized_name}</b></p>
</div>
<div class='card'>
  <h3>🎞️ Visual / Text / Logo Detections ({len(visual)})</h3>
  <table><tr><th>Time</th><th>Matched</th><th>Detected Text</th><th>Conf</th></tr>{rows_v}</table>
</div>
<div class='card'>
  <h3>🎵 Audio Detections ({len(audio)})</h3>
  <table><tr><th>Range</th><th>Reason</th><th>Conf</th></tr>{rows_a}</table>
</div>
<div class='card'><small>Report generated by CopyrightGuard AI • Free, offline, open-source.
Detection is heuristic — always double-check before publishing.</small></div>
</body></html>"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


def run_pipeline(video_path, work_dir, progress_cb=None):
    job_id = uuid.uuid4().hex[:8]
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(work_dir, "outputs")
    rep_dir = os.path.join(work_dir, "reports")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    sanitized_name = f"{base}_safe_{job_id}.mp4"
    sanitized_path = os.path.join(out_dir, sanitized_name)
    report_name = f"{base}_report_{job_id}.html"
    report_path = os.path.join(rep_dir, report_name)

    if progress_cb: progress_cb("Extracting audio…")
    wav_path = os.path.join(work_dir, f"_tmp_{job_id}.wav")
    try: extract_audio(video_path, wav_path)
    except Exception: pass

    if progress_cb: progress_cb("Analyzing audio for music…")
    audio_findings = analyze_audio(wav_path)

    if progress_cb: progress_cb("Loading AI OCR model…")
    visual_findings, blur_map, hashes = analyze_frames(video_path, progress_cb)

    score = compute_risk_score(visual_findings, audio_findings)

    if progress_cb: progress_cb("Sanitizing video (blurring + muting)…")
    mute_segs = [(f["start"], f["end"]) for f in audio_findings]
    sanitize_video(video_path, sanitized_path, blur_map, mute_segs, progress_cb)

    if progress_cb: progress_cb("Generating report…")
    generate_report(job_id, os.path.basename(video_path),
                    visual_findings, audio_findings, score,
                    report_path, sanitized_name)

    if os.path.exists(wav_path):
        try: os.remove(wav_path)
        except: pass

    label, color = risk_label(score)
    return {
        "job_id": job_id,
        "score": score,
        "label": label,
        "color": color,
        "visual_count": len(visual_findings),
        "audio_count": len(audio_findings),
        "sanitized_file": sanitized_name,
        "report_file": report_name,
        "visual": visual_findings[:50],
        "audio": audio_findings[:50],
    }

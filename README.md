# CopyrightGuard AI 🛡️🎬

An AI-powered Copyright Risk Detection & Video Sanitization System.
Upload a video → the app extracts frames, audio, text (OCR), and visual
fingerprints, detects potential copyright risks (logos, brand text,
copyrighted-sounding music segments), then automatically **sanitizes** the
video (blurs logos/text, mutes risky audio segments) and generates a
detailed HTML report with timestamps.

100% free & offline — no paid APIs, no database, no cloud account required.
Everything runs locally with open-source AI (EasyOCR, OpenCV, Librosa, pHash).

---

## ✨ Features
- 🎞️ Frame extraction + perceptual hashing (pHash) for visual fingerprinting
- 🔤 On-screen text / brand / watermark detection using **EasyOCR** (deep-learning OCR)
- 🎵 Audio analysis with **Librosa** — detects music-like segments vs. speech
- 🏷️ Logo & brand keyword matching against a built-in copyrighted-brand list
- ⏱️ Exact timestamps for every violation
- 📊 Copyright Risk Score (0-100) + full HTML report
- 🧼 Automatic sanitization:
  - Blurs frames containing detected brand text/logos
  - Mutes audio segments flagged as copyrighted music
  - Re-encodes a clean, platform-safe video
- 🎨 Colorful, animated, modern UI (gradient neon theme)

---

## 🖥️ How to Run in VS Code (Windows)

### 1. Prerequisites
- **Python 3.10 or 3.11** → https://www.python.org/downloads/
  (During install, tick ✅ *"Add Python to PATH"*)
- **FFmpeg** (required for audio/video processing) → https://www.gyan.dev/ffmpeg/builds/
  Download "ffmpeg-release-essentials.zip", extract, and add the `bin` folder to your Windows PATH.
  Verify by opening a new terminal and running: `ffmpeg -version`

### 2. Open the project
1. Unzip `CopyrightGuardAI.zip`
2. Open the folder in **VS Code** → `File > Open Folder…`
3. Open a new terminal in VS Code: `Terminal > New Terminal`

### 3. Install dependencies (one-time)
Copy-paste these commands **one by one** in the VS Code terminal:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> First install can take 5–10 minutes (EasyOCR downloads a small AI model on first run).

### 4. Run the app
```powershell
python app\main.py
```

You will see:
```
 * Running on http://127.0.0.1:5000
```

Open that URL in your browser → upload a video → click **Analyze & Sanitize**.

### 5. Next time you run it
Just:
```powershell
venv\Scripts\activate
python app\main.py
```

---

## 📂 Output
- Sanitized video → `app/outputs/`
- HTML report → `app/reports/`
- Both are auto-linked on the result page.

---

## ⚙️ Tech Stack (all free & open-source)
| Purpose | Tool |
|---|---|
| Web framework | Flask 3 |
| Computer Vision | OpenCV 4.10 |
| Deep-learning OCR | EasyOCR (PyTorch) |
| Audio analysis | Librosa |
| Perceptual hashing | ImageHash |
| Video I/O | MoviePy + FFmpeg |

Enjoy safe publishing! 🚀

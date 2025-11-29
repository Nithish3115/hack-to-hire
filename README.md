---
title: RVC Voice Cloner
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# 🎤 RVC Voice Cloner - Hack-to-Hire Project

A web-based application for **Retrieval-based Voice Conversion (RVC)**. This tool allows users to clone voices, train new models, and perform real-time inference using a simple web interface.

**[🔴 Live Demo Link](https://nithish3115-rvc-voice-cloner.hf.space)**  
*(Note: The live demo runs on CPU. Inference takes ~40s. Training is disabled on the free tier.)*

---

## ✨ Features

*   **Voice Conversion (Inference):** Upload an audio file and a voice model to change the speaker's voice.
*   **Custom Training:** Preprocess datasets and train new voice models (requires GPU).
*   **Auto-Setup:** Automatically downloads required base models (Hubert, RMVPE) on first run.
*   **Cloud Ready:** Dockerized and optimized for Hugging Face Spaces.

---

## 🚀 How to Run Locally

### Prerequisites
*   Python 3.10
*   FFmpeg installed (`sudo apt install ffmpeg` or download for Windows)
*   NVIDIA GPU (Recommended) or CPU (Slow)

### 1. Clone the Repository
```bash
git clone https://github.com/Nithish3115/hack-to-hire.git

cd hack-to-hire
```
## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Run the App
``` bash
python app.py
```
*  The app will automatically download necessary models (~800MB) on the first launch.

*  Open your browser at: http://127.0.0.1:5000

## 🧪 How to Test the Live Demo (Step-by-Step)

We have provided sample files in the `test_demo/` folder of this repository so you can test the voice cloning immediately.

### 1. Get the Test Files
Go to the **`test_demo/`** folder in this repository and download these 3 files to your computer:
1.  `sample_audio.wav` (The voice to be converted)
2.  `model.pth` (The target voice model)
3.  `model.index` (The feature index file)

### 2. Go to the Live App
Open the live demo: **[Click Here to Open App](https://nithish3115-rvc-voice-cloner.hf.space)**

### 3. Run Inference
1.  **Upload Audio:** Drag & drop `sample_audio.wav` into the "Upload Audio" box.
2.  **Upload Model:**
    *   Click the "Model (.pth)" upload box and select `model.pth`.
    *   Click the "Index (.index)" upload box and select `model.index`.
3.  **Click "Convert"**.
4.  Wait ~30-40 seconds (Free Tier CPU).
5.  **Listen:** The converted audio will appear in the audio player!

---
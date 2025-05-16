# Real-Time Face Emotion Detection with 3D Fourier Visualization

This Python mini-project captures live video from your webcam, detects facial emotions using the **DeepFace** library, and performs a **2D Fast Fourier Transform (FFT)** on each frame. You can also render a **3D Fourier surface plot**, colored based on the detected emotion.

---

## Features

- Real-time emotion detection using webcam input
- Applies 2D Fourier Transform to each frame
- Press `3` to show 3D visualization of frequency spectrum
- Emotion-specific colormaps for better interpretation:
  - `happy` → `inferno`
  - `sad` → `cool`
  - `angry` → `hot`
  - `surprise` → `spring`
  - `neutral` → `viridis`

---

## Requirements

### Python version

```
Python 3.8 or 3.9
```

### Install the following Python libraries:

```
pip install opencv-python numpy matplotlib deepface
```

## How to run

```
python Face_Emotion_Detector.py
```

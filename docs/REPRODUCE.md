# CAFE - Reproduction Guide

This document gives the ordered command sequence used to reproduce the reported
results, tables, and figures. Run commands from the repository root with the
virtual environment active.

After activating the environment, set the repository root on `PYTHONPATH`:

````powershell
$env:PYTHONPATH = (Get-Location).Path

---

## Prerequisites

1. Python 3.12 virtual environment created and activated (see `README.md`).
2. `detector_reference/` present at the repository root.
3. Detector checkpoint at `models/EfficientNetB4_FFPP/EfficientNetB4_FFPP_bestval.pth`.
4. MediaPipe face-landmarker model at `models/face_landmarks/face_landmarker.task`.
5. FaceForensics++ C23 videos placed under `dataset/fake/` and `dataset/real/`
   (see `README.md` for the required layout).

---

## Core reproduction workflow

### Step 1 — Build the manifest

```powershell
python scripts/build_manifest.py
````

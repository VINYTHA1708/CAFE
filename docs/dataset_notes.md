# Dataset Notes

## Dataset

FaceForensics++ C23 subset used for the CAFE project.

## Selected subset

- Deepfakes: 10 fake videos
- Face2Face: 10 fake videos
- FaceSwap: 10 fake videos
- NeuralTextures: 10 fake videos
- Original videos: 20 matching pristine videos
- Total: 60 videos

The current working subset contains 40 manipulated videos and 20 matching original videos.
The original project plan targeted 80 videos, but only 20 matching original videos are currently included in this working subset.

## Storage layout

The working dataset is stored under `dataset/` using the following structure:

```text
dataset/
+-- fake/
¦   +-- Deepfakes/
¦   +-- Face2Face/
¦   +-- FaceSwap/
¦   +-- NeuralTextures/
+-- real/
```

## Manifest

`data/manifest.csv` contains relative video paths and metadata including label, manipulation method, source ID, frame count, FPS, duration, width, and height.

## Validation

The manifest builder inspected all 60 videos using OpenCV.
All 60 videos were readable and none were skipped for being unreadable or shorter than 150 frames.
The measured mean duration is 17.09 seconds.

## Source

The working C23 files were obtained from a public Kaggle mirror of the FaceForensics++ dataset.
This project does not claim official FaceForensics++ authorization based on the mirror alone.

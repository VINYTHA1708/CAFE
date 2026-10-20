# CAFE — Reproducing the Primary Experiment

This guide identifies the final primary run and its preserved outputs. The
primary experiment is a **single detector-guided condition** (`real`) over the
existing 60-video FaceForensics++ C23 subset. Do not use the historical
three-condition command as a reproduction of these results.

## Prerequisites

Run from the repository root with the project environment active. Required
assets are:

1. Python environment with dependencies from `requirements.txt`.
2. The upstream detector reference at `detector_reference/`.
3. The frozen checkpoint at
   `models/EfficientNetB4_FFPP/EfficientNetB4_FFPP_bestval.pth`.
4. The MediaPipe model at `models/face_landmarks/face_landmarker.task`.
5. The 60 FaceForensics++ C23 videos listed in `data/manifest.csv`.

OpenCV decodes video frames as BGR. CAFE converts frames to RGB before
BlazeFace; saved face crops are RGB for the EfficientNet ImageNet
normalization and MediaPipe `SRGB` landmarker. Keep the default corrected cache
root separate from the legacy `cache/` directory so old crops and results stay
untouched.

## Rebuild corrected preprocessing

From the repository root, create corrected crops and then landmarks:

```bash
python scripts/precompute_frames.py \
  --manifest data/manifest.csv \
  --cache-root cache_rgb_corrected
python scripts/precompute_landmarks.py \
  --manifest data/manifest.csv \
  --cache-root cache_rgb_corrected
```

Both commands rebuild the 60 manifest videos in `cache_rgb_corrected/`.
Existing `cache/` contents are not replaced. Frame sampling, face selection,
landmark extraction, and all experiment settings remain unchanged.

## Primary run

The corrected primary run requires CUDA and uses only the detector-guided
`real` condition:

```bash
python scripts/run_batch_corrected_colab.py \
  --manifest data/manifest.csv \
  --config config.yaml \
  --conditions real \
  --cache-root cache_rgb_corrected \
  --out results/primary_rgb_corrected \
  --batch-size 8 \
  --require-cuda
```

The runner logs its exact command, source commit, worktree status, device, and
cache root. It checkpoints per-video JSON outputs and writes a summary CSV as
runs complete. The experiment configuration is unchanged: detector-guided
candidates, existing corrected canonical eye/lip masks, face-texture mask,
interventions and controls, 20 controls per candidate, the 95th-percentile
threshold, p-value calculation, random seed, and `Δ > τ` support rule. Do not
add `placebo` or `authentic`.

## Primary outputs

The corrected run writes new outputs to `results/primary_rgb_corrected/`:

- `batch_corrected_summary.csv`
- One `<video_id>__real.json` file per manifest video
- A batch log with command, commit, device, and cache provenance

The previous outputs in `results/primary_full/` are preserved historical
artifacts from before this RGB-input correction. Do not replace them or treat
their metrics as the corrected run's results. Recalculate all reported metrics
from the new summary and per-video JSON files after the corrected run
completes.

## Historical outputs and scripts

`scripts/run_batch.py` and existing `results/runs/` files belong to the earlier
three-condition experiment, where `real` used detector-guided candidates,
`placebo` used random candidates, and `authentic` used independently seeded
random candidates. These are historical/superseded and do not reproduce the
final primary analysis.

A later candidate-sharing run used the same detector-guided candidates under
all three condition labels. Because intervention and control scoring was
condition-agnostic, those outputs were numerically identical; they are not
independent experimental arms. Historical code and results are preserved but
are not primary evidence.

The legacy `scripts/make_figures.py` reads the historical JSON files from
`results/runs/` and generates three-condition tables and figures. Its output
must not be cited as the final primary results.

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
5. The FaceForensics++ C23 videos listed in `data/manifest.csv`, plus cached
   preprocessing and landmarks where available.

The primary run used a Colab T4 GPU. The detector is frozen; there is no
training.

## Primary run

Use the corrected runner, the existing manifest/configuration, and only the
detector-guided `real` condition:

```bash
python scripts/run_batch_corrected_colab.py \
  --manifest data/manifest.csv \
  --config config.yaml \
  --conditions real \
  --out results/primary_full \
  --require-cuda
```

The runner uses the corrected canonical eye/lip masks at runtime while
preserving the face-texture mask, intervention/control methods, 20 controls
per candidate, 95th-percentile threshold, p-value calculation, random seed,
and `Δ > τ` support rule. It checkpoints per-video JSON outputs and writes a
summary CSV as runs complete. The command above describes the primary
detector-guided run only; do not add `placebo` or `authentic` when reproducing
the primary results.

## Primary outputs

The final run produced:

- `results/primary_full/batch_corrected_summary.csv`
- One `<video_id>__real.json` file per manifest video under `results/primary_full/`
- A batch log under `results/primary_full/`

The recorded primary run completed all 60 videos with 173 candidate rows and
zero failures. The summary CSV is the flat source for the candidate-level
results; each JSON preserves per-video candidate and control details.

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

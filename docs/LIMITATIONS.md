# CAFE — Known Limitations

---

## 1. Single frozen detector

CAFE is built around one frozen EfficientNet-B4 checkpoint trained on
FaceForensics++ C23.  All candidate generation (Grad-CAM), all intervention
scoring, and all control comparisons use this single model.  Explanations are
therefore specific to this detector's decision boundary and do not generalise
to other architectures or training sets.

## 2. Grad-CAM layer choice

Cue assignment uses Grad-CAM computed at `model.efficientnet._conv_head` (the
final convolutional layer of EfficientNet-B4).  This is a single fixed layer;
multi-layer or guided-backpropagation alternatives are not implemented.  If the
detector relies on features from earlier layers the assigned cue may not
reflect the true attribution.

## 3. Three fixed cue channels

The system recognises exactly three cues: `eye_motion`, `mouth_motion`, and
`face_texture`.  Manipulations that affect other facial regions (e.g. hair,
neck, background bleed-through) cannot be captured.  Under the `real`
condition all candidates in the current experiment were assigned `face_texture`
because the full-face Grad-CAM region dominates for this detector.

## 4. Intervention fidelity

- `eye_motion` and `mouth_motion` interventions replace the region with a
  linear interpolation between the frames immediately before and after the
  interval.  If those boundary frames are unavailable or contain invalid
  landmarks the intervention falls back to returning the unmodified frames.
- `face_texture` blurs the face region with a fixed Gaussian kernel
  (`blur_sigma=8.0`).  This is a coarse perturbation that may affect
  detector scores for reasons unrelated to texture (e.g. edge artefacts).
- All interventions use a fixed `blend_alpha=0.8` and `feather_px=5`.  These
  values are configurable but were not swept in the current experiment.

## 5. MediaPipe landmark failures

Landmark extraction uses MediaPipe FaceLandmarker in IMAGE mode with
`num_faces=1`.  Frames where detection fails are filled with NaN and skipped
during intervention and cue assignment.  Videos with many failed frames may
produce unreliable candidates.

## 6. Small dataset

The experiment uses 60 videos (40 fake, 20 real) from FaceForensics++ C23.
Support rates are therefore estimated from small counts (e.g. 5 supported
videos out of 40 in the `real / fake` cell) and carry high variance.

## 7. Low overall support rate

Under the default configuration (τ at the 95th percentile of 20 mixed
controls) the video-level support rate on fake videos is 12.5% (`real`
condition).  Most videos receive no verified explanation.  This reflects the
conservatism of the control threshold rather than a failure of the
interventions per se, but it limits the practical utility of the system.

## 8. Control set size

Each candidate is compared against 20 controls (2 sham, ~9 interval-shift,
~9 cue-swap, exact split depends on available non-overlapping intervals).  The
permutation p-value `(1 + #{δ_i ≥ δ}) / (1 + m)` has a minimum value of
`1/21 ≈ 0.048` with m = 20, which limits statistical resolution.

## 9. Idempotent batch runner — no re-run on config change

`scripts/run_batch.py` skips any output file that already exists and is valid
JSON.  If `config.yaml` is changed after a partial run, existing results will
not be recomputed.  The output directory must be cleared manually before
re-running with different parameters.

## 10. FFmpeg path is hard-coded

`scripts/check_detector.py` contains a hard-coded absolute path to an FFmpeg
binary.  The re-encode sanity check will fail on any machine where FFmpeg is
not installed at that exact location.

## 11. No report or presentation in the repository

No written report, thesis, or slide deck is stored in this repository or in
the surrounding file system.  Only the source code, cached data, run results,
tables, and figures described in `README.md` are present.

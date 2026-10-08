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

## 3. Cue-assignment bias

The system recognises exactly three cues: `eye_motion`, `mouth_motion`, and
`face_texture`.  Manipulations that affect other facial regions (e.g. hair,
neck, background bleed-through) cannot be captured. All 173 candidates in the
primary experiment were assigned `face_texture`. This is a
candidate-generation/cue-assignment bias and means the experiment cannot fairly
compare the three cue channels. It is not evidence that `face_texture` is
superior.

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

The primary experiment uses 60 videos (40 fake, 20 real) from FaceForensics++
C23. Observed support rates are based on small counts (6/40 fake videos and
0/20 real videos), are descriptive of this subset, and should not be
generalized to all of FaceForensics++.

## 7. Low overall support rate

Under the primary configuration (τ at the 95th percentile of 20 mixed
controls), 6/40 fake videos (15%) received a supported explanation and 34/40
did not; 0/20 genuine videos received a supported explanation. These are
descriptive results. The low support rate is not attributed to threshold
conservatism or intervention failure because this experiment does not isolate
a single cause.

## 8. Control set size

Each candidate is compared against 20 controls (2 sham, ~9 interval-shift,
~9 cue-swap, exact split depends on available non-overlapping intervals).  The
permutation p-value `(1 + #{δ_i ≥ δ}) / (1 + m)` has a minimum value of
`1/21 ≈ 0.047619` with m = 20. All six supported candidates attain this
minimum. It is not, by itself, conventional statistical significance; the
coarse p-value resolution, 173 candidate tests, and small evaluation subset
limit statistical interpretation.

## 9. No temporal localization ground truth

The FF++ evaluation subset used here does not provide temporal manipulation
ground-truth annotations for measuring temporal localization accuracy.
Candidate intervals should not be presented as validated manipulation
boundaries or localization performance.

## 10. Idempotent batch runner — no re-run on config change

`scripts/run_batch.py` skips any output file that already exists and is valid
JSON.  If `config.yaml` is changed after a partial run, existing results will
not be recomputed.  The output directory must be cleared manually before
re-running with different parameters.

## 11. Historical condition outputs

The historical three-condition experiment used different candidate
populations for `real`, `placebo`, and `authentic`. A later candidate-sharing
run reused detector-guided candidates under all three labels; because scoring
was condition-agnostic, those outputs were numerically identical and are not
independent experimental arms. Neither set of three-condition results is the
primary experiment.

## 12. FFmpeg path is hard-coded

`scripts/check_detector.py` contains a hard-coded absolute path to an FFmpeg
binary. The re-encode sanity check may fail on machines where FFmpeg is not
installed at that exact location.

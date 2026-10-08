# CAFE — Method: Module-to-Implementation Mapping

This document maps each report module to its purpose, source file(s), and the
key function or class that implements it.

---

## Module 1 — Video Preprocessing

**Purpose:** Uniformly sample frames from a raw MP4 video, detect and crop the
dominant face in each frame using BlazeFace, and persist the results to a
per-video cache (`cache/<video_id>/`).

| Item | Detail |
|------|--------|
| Source file | `cafe/preprocessing.py` |
| Key functions | `extract_frames`, `detect_and_crop_faces`, `load_or_build_cache` |
| Cache artefacts | `frames.npy`, `faces.npy`, `index.json` |
| Face detector | BlazeFace (`detector_reference/blazeface/`) |
| Frames per video | 32 (configurable via `config.yaml → frames_per_video`) |
| Face crop size | 224 × 224 px |

Landmark extraction is handled separately by `cafe/landmarks.py`
(`extract_landmarks`, `load_or_build_landmark_cache`), which uses the MediaPipe
FaceLandmarker task model to produce 478-point landmark arrays stored as
`cache/<video_id>/landmarks.npy`.

---

## Module 2 — Base Deepfake Detector

**Purpose:** Provide a frozen, inference-only deepfake detector that scores
individual face crops and aggregates them into a per-video probability.

| Item | Detail |
|------|--------|
| Source file | `cafe/detector/base_detector.py` |
| Key class | `BaseDetector` |
| Architecture | EfficientNet-B4 (`detector_reference/architectures/fornet.py`) |
| Checkpoint | `models/EfficientNetB4_FFPP/EfficientNetB4_FFPP_bestval.pth` |
| Training dataset | FaceForensics++ C23 |
| Input normalisation | ImageNet mean/std via `isplutils.utils.get_transformer` |
| Key methods | `score_frames(faces)` → per-frame probabilities; `score_video(faces)` → mean probability; `score_frame_batches(face_batches)` → batch-scored video means; `predict(faces)` → `("REAL"\|"DEEPFAKE", score)` |
| Inference mode | `model.eval()` + `torch.no_grad()`; no fine-tuning |

---

## Module 3 — Candidate Explanation Generation

**Purpose:** Generate detector-guided temporal intervals and assign a facial
cue to each candidate explanation hypothesis.

| Item | Detail |
|------|--------|
| Source file | `cafe/candidates.py` |
| Key functions | `propose_intervals`, `assign_cue`, `generate_candidates`, `generate_placebo_candidates` |
| Interval selection | Top-k non-overlapping windows ranked by mean per-frame detector score (`propose_intervals`) |
| Cue assignment | Grad-CAM overlap with three region masks: `eye_motion` (eyes), `mouth_motion` (mouth), `face_texture` (full face) — implemented in `assign_cue` via `_gradcam_for_face` |
| Grad-CAM layer | `model.efficientnet._conv_head` |
| Cue channels | `eye_motion`, `mouth_motion`, `face_texture` |
| Candidates per video | 6 (configurable via `config.yaml → n_candidates`) |
| Interval length | 8 sampled frames (configurable via `config.yaml → interval_length`) |

### Primary experiment

For each video, the primary experiment scores the original sampled face crops,
selects the top non-overlapping intervals, and assigns each interval the cue
with the greatest Grad-CAM overlap. The resulting detector-guided candidate
list is then counterfactually verified. Corrected canonical MediaPipe eye and
lip masks are used for those cue regions; the full-face mask used by
`face_texture` is unchanged. In the final 60-video primary run, all 173
candidates were assigned `face_texture`. This is a cue-assignment limitation,
not evidence that `face_texture` is superior; the primary results do not
support a comparative claim about the three cue channels.

### Historical candidate conditions

The earlier three-condition experiment used detector-guided candidates for
`real`, randomly sampled cue-interval pairs for `placebo`, and independently
seeded random cue-interval pairs for `authentic`. These different candidate
populations make those outputs historical and superseded for the primary
conclusions. A later candidate-sharing run reused detector-guided candidates
across all three condition labels; because verification was condition-agnostic,
those repeated outputs are not independent experimental arms. Neither run is
the final primary experiment.

---

## Module 4 — Counterfactual Interventions

**Purpose:** Apply a targeted, region-masked perturbation to a specified cue
channel and temporal interval, producing a modified frame array for re-scoring.

| Item | Detail |
|------|--------|
| Source file | `cafe/interventions.py` |
| Key function | `apply_intervention(frames, landmarks, cue, interval, strength, sham)` |
| `eye_motion` | Interpolates eye-region pixels between the frames immediately before and after the interval |
| `mouth_motion` | Interpolates mouth-region pixels between the frames immediately before and after the interval |
| `face_texture` | Gaussian-blurs the full face region within the interval |
| Compositing | Feathered alpha-composite via `_alpha_composite`; `blend_alpha=0.8`, `feather_px=5`, `blur_sigma=8.0` (configurable) |
| Sham control | `_apply_sham` runs the same mask/blend pipeline but composites the frame with an identical copy (no-op) |
| Region masks | Soft convex-hull masks from MediaPipe landmarks via `cafe/landmarks.py → region_mask` |

---

## Module 5 — Control-Based Verification

**Purpose:** Build a null distribution of detector-score effects from matched
controls, derive a threshold τ, and compute a permutation-style p-value.
Support is an operational decision, not a claim of conventional statistical
significance.

| Item | Detail |
|------|--------|
| Source files | `cafe/controls.py`, `cafe/verification.py` |
| Key functions | `generate_controls` (controls.py); `compute_threshold`, `compute_p_value`, `verify_candidate` (verification.py) |
| Control types | `sham` (same interval, identity blend), `interval_shift` (non-overlapping interval, same cue), `cue_swap` (same interval, different cue) |
| Controls per candidate | 20 (configurable via `config.yaml → n_controls`) |
| Threshold τ | 95th percentile of control effects (configurable via `config.yaml → tau_percentile`) |
| p-value formula | `p = (1 + #{δ_i ≥ δ}) / (1 + m)` where m = number of controls |
| Operational support criterion | `δ_candidate > τ`; otherwise abstain |

---

With 20 controls, the current p-value formula has a minimum attainable value
of `1/21 ≈ 0.047619`. That resolution, the small evaluation subset, and
multiple candidate tests limit statistical interpretation. A p-value at this
minimum is not, by itself, evidence of conventional statistical significance.

## Module 6 — Explanation Decision and Rendering

**Purpose:** Aggregate verified candidates into a final decision, render a
human-readable explanation or abstention message, and persist the result.

| Item | Detail |
|------|--------|
| Source files | `cafe/decision.py`, `cafe/pipeline.py`, `cafe/templates.py` |
| Key functions | `decide` (decision.py), `render_explanation` (decision.py), `run_cafe` (pipeline.py) |
| Decision logic | All candidates with `supported=True` are collected and sorted by effect descending; if none are supported the system abstains |
| Output | `CafeResult` dataclass; JSON written to `results/runs/<video_id>.json` |
| Text templates | `VERIFIED_TEMPLATE`, `ABSTENTION_TEMPLATE` in `cafe/templates.py` |

---

## Batch runner and analysis scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_cafe.py` | CLI wrapper for `run_cafe()` — single video |
| `scripts/run_batch_corrected_colab.py` | Corrected runner; the primary run uses `--conditions real` |
| `scripts/run_batch.py` | Historical three-condition runner; does not reproduce the primary experiment |
| `scripts/build_manifest.py` | Builds `data/manifest.csv` from `dataset/` |
| `scripts/precompute_frames.py` | Pre-builds frame/face caches for all manifest videos |
| `scripts/precompute_landmarks.py` | Pre-builds landmark caches for all manifest videos |
| `scripts/make_figures.py` | Generates all metric tables (`results/tables/`) and quantitative figures (`docs/figures/`) |
| `scripts/make_qualitative.py` | Generates qualitative case panels and demo videos |
| `scripts/plot_intervention_examples.py` | Generates `docs/figures/intervention_examples.png` |
| `scripts/check_detector.py` | Re-encode sensitivity sanity study |
| `scripts/verify_video.py` | Standalone verification for one video (development utility) |

The primary run's per-video results and summary are in
`results/primary_full/`. The historical three-condition outputs under
`results/runs/` must not be presented as independent primary conditions.

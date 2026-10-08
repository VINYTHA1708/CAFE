# CAFE — Primary Results

This document reports the final primary experiment: one detector-guided
candidate-generation condition on the 60-video FaceForensics++ C23 evaluation
subset. The run completed on a Colab T4 GPU in approximately 15 minutes 16
seconds. The primary per-video JSONs, batch log, and flat summary are in
`results/primary_full/`; the summary is
`results/primary_full/batch_corrected_summary.csv`.

## Primary setup

| Parameter | Value |
|-----------|-------|
| Dataset | FaceForensics++ C23 subset |
| Videos | 60: 40 fake and 20 real/original |
| Detector | Frozen EfficientNet-B4 FF++ |
| Conditions | One primary condition: `real` (detector-guided candidates) |
| Candidate generation | Detector/frame-score interval ranking and Grad-CAM cue assignment |
| Candidates | 173 total: 116 fake-video candidates and 57 real-video candidates |
| Controls | 20 per candidate: 2 sham, approximately 9 interval-shift and 9 cue-swap |
| Threshold | 95th percentile of the candidate's control effects |
| Operational support rule | Supported iff `Δ > τ`; otherwise abstain |
| Failures | 0 |
| Training | None |

The three available cue channels are `eye_motion`, `mouth_motion`, and
`face_texture`. **All 173 primary candidates were assigned `face_texture`.**
This is a candidate-generation/cue-assignment bias and limitation: these
results cannot fairly compare the three cue channels and do not establish that
`face_texture` is superior.

## Video- and candidate-level support

| Label | Videos | Candidates | Supported candidates | Videos with a supported explanation | Video-level support |
|-------|-------:|-----------:|----------------------:|------------------------------------:|--------------------:|
| Fake | 40 | 116 | 6 | 6 | 15.0% (6/40) |
| Real/original | 20 | 57 | 0 | 0 | 0% (0/20) |
| **Overall** | **60** | **173** | **6** | **6** | — |

Overall candidate-level support was **3.47% (6/173)**.

On this 60-video evaluation subset, CAFE produced a verified explanation for
6/40 fake videos and abstained on all 20 genuine videos. This is the observed
behaviour of an explanation-verification layer, not 15% deepfake-detection
accuracy, and should not be generalized to all of FaceForensics++.

## Observed support by manipulation method

| Manifest method | Videos with supported explanation / videos |
|-----------------|-------------------------------------------:|
| Deepfakes | 1/10 |
| Face2Face | 2/10 |
| FaceSwap | 2/10 |
| NeuralTextures | 1/10 |
| Original | 0/20 |

These are descriptive counts from this subset, not comparative performance
claims about manipulation methods.

## Six supported candidates

Intervals below are original video-frame indices from the primary JSON files.

| Video | Cue | Interval | Δ | τ | p-value |
|-------|-----|----------|---:|---:|--------:|
| Deepfakes_000_003 | `face_texture` | 267–356 | 0.015711 | 0.011659 | 0.047619 |
| Face2Face_000_003 | `face_texture` | 175–243 | 0.001414 | 0 | 0.047619 |
| Face2Face_001_870 | `face_texture` | 194–330 | 0.001806 | 0.000186 | 0.047619 |
| FaceSwap_012_026 | `face_texture` | 82–155 | 0.002522 | 0.001659 | 0.047619 |
| FaceSwap_016_209 | `face_texture` | 10–86 | 0.009843 | 0.009075 | 0.047619 |
| NeuralTextures_015_919 | `face_texture` | 182–281 | 0.000968 | 0.000885 | 0.047619 |

## Candidate effect and threshold summaries

| Statistic | Candidate effect Δ | Threshold τ |
|-----------|-------------------:|------------:|
| Mean | -0.095124 | 0.003644 |
| Median | -0.075958 | 0 |
| 75th percentile | — | 0.002525 |
| 95th percentile | 0.006071 | 0.020077 |
| Maximum | 0.030978 | 0.040921 |

Most candidate effects did not exceed their candidate-specific control
threshold; those candidates were rejected and did not yield an explanation.
This is a descriptive result and is not attributed to a single cause.

## Sham ablation

| Control calculation | Fake candidates supported | Real candidates supported | Original six cases retained |
|---------------------|--------------------------:|---------------------------:|----------------------------|
| Full mixed controls | 6/116 | 0/57 | — |
| Non-sham controls only | 8/116 | 0/57 | Yes, all six |

Removing sham controls increased the number of supported fake candidates in
this ablation. The primary result retains the full mixed-control design; the
ablation is a sensitivity analysis, not a replacement primary analysis.

## Statistical interpretation and scope

The six reported p-values equal `1/21 ≈ 0.047619`, the minimum attainable
under the current p-value calculation with 20 controls. This minimum value
alone does **not** establish conventional statistical significance. The small
control count, 173 tested candidates, and multiple testing limit statistical
interpretation.

The intervals are candidate explanation intervals, not validated temporal
localizations: the evaluation subset does not provide temporal manipulation
ground-truth annotations for measuring localization accuracy. Results are
limited to this 60-video subset and this frozen detector. CAFE is an
explanation-verification layer, not a standalone deepfake detector.

## Secondary Validation and Diagnostic Studies

The repository also contains supplementary validation and diagnostic analyses,
including re-encode sensitivity testing, qualitative intervention and case
visualizations, sham-control ablation, and other diagnostic analyses. These
studies are supplementary validation/diagnostic analyses, not additional
primary experimental conditions.

## Historical/Superseded Three-Condition Experiment

Earlier outputs used three differently generated candidate populations:
`real` used detector-guided candidates, `placebo` used random candidates, and
`authentic` used independently seeded random candidates. Those results are
historical and are not the final primary experiment.

A later candidate-sharing run reused the detector-guided candidate list under
all three labels. Since intervention and control scoring did not vary by
condition, those outputs became numerically identical; they are not three
independent experimental arms and must not be interpreted as such. Historical
code and results remain preserved in the repository, but the three-condition
tables and comparisons are not reported here as primary findings.

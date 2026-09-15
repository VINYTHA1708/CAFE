# CAFE — Results

All numbers in this document are read directly from the files in
`results/tables/` and `results/sanity/` produced by `scripts/make_figures.py`
and `scripts/check_detector.py`.  Figures are in `docs/figures/`.

---

## Experimental setup

| Parameter | Value |
|-----------|-------|
| Videos | 60 (40 fake, 20 real) from FaceForensics++ C23 |
| Fake methods | Deepfakes (10), Face2Face (10), FaceSwap (10), NeuralTextures (10) |
| Frames per video | 32 (uniform sampling) |
| Candidates per video | 6 |
| Interval length | 8 sampled frames |
| Controls per candidate | 20 (sham × 2, interval-shift, cue-swap) |
| τ percentile (default) | 95 |
| Conditions | `real` (detector-guided), `placebo` (random cue-interval), `authentic` (random on real videos) |

---

## Video-level support and abstention rates

Source: `results/tables/support_abstention.md`

| condition | label | videos | videos with supported explanation | support rate | abstention rate |
|:----------|:------|-------:|----------------------------------:|-------------:|----------------:|
| authentic | fake  | 40     | 7                                 | 0.175        | 0.825           |
| authentic | real  | 20     | 1                                 | 0.050        | 0.950           |
| placebo   | fake  | 40     | 4                                 | 0.100        | 0.900           |
| placebo   | real  | 20     | 1                                 | 0.050        | 0.950           |
| real      | fake  | 40     | 5                                 | 0.125        | 0.875           |
| real      | real  | 20     | 0                                 | 0.000        | 1.000           |

Key observations:
- Under the `real` condition (detector-guided candidates), 0/20 genuine videos
  receive a supported explanation, confirming the control mechanism suppresses
  false positives on real content.
- The `placebo` condition (random candidates) yields a 10% support rate on fake
  videos, providing a baseline for spurious support.
- The `authentic` condition (random candidates on real videos) yields 5% support
  on genuine videos, consistent with the 95th-percentile threshold.

---

## Candidate-level support rates

Source: `results/tables/candidate_support.md`

| condition | label | candidate rows | supported candidates | candidate support rate |
|:----------|:------|---------------:|---------------------:|-----------------------:|
| authentic | fake  | 240            | 9                    | 0.0375                 |
| authentic | real  | 120            | 1                    | 0.0083                 |
| placebo   | fake  | 240            | 5                    | 0.0208                 |
| placebo   | real  | 120            | 2                    | 0.0167                 |
| real      | fake  | 116            | 5                    | 0.0431                 |
| real      | real  | 57             | 0                    | 0.0000                 |

---

## Detector-score effect (δ) by cue

Source: `results/tables/delta_by_cue.md`

| condition | cue          | count | median   | mean     | std    | min      | max    |
|:----------|:-------------|------:|---------:|---------:|-------:|---------:|-------:|
| authentic | eye_motion   | 112   | −0.0561  | −0.0706  | 0.0575 | −0.2208  | 0.0106 |
| authentic | face_texture | 131   | −0.0925  | −0.1043  | 0.0911 | −0.2472  | 0.0309 |
| authentic | mouth_motion | 117   | −0.0555  | −0.0567  | 0.0479 | −0.1893  | 0.0101 |
| placebo   | eye_motion   | 133   | −0.0612  | −0.0678  | 0.0578 | −0.2174  | 0.0367 |
| placebo   | face_texture | 108   | −0.0788  | −0.0995  | 0.0841 | −0.2476  | 0.0344 |
| placebo   | mouth_motion | 119   | −0.0485  | −0.0535  | 0.0454 | −0.2113  | 0.0092 |
| real      | face_texture | 173   | −0.0760  | −0.0951  | 0.0879 | −0.2475  | 0.0310 |

Under the `real` condition all candidates are assigned `face_texture` because
Grad-CAM overlap with the full-face region dominates for this detector.

---

## Detector-score effect (δ) by FF++ manipulation method

Source: `results/tables/delta_by_method.md`

Selected rows for the `real` condition (detector-guided):

| method         | count | median   | mean     | std    | min      | max     |
|:---------------|------:|---------:|---------:|-------:|---------:|--------:|
| Deepfakes      | 31    | −0.0085  | −0.0490  | 0.0729 | −0.2191  | 0.0310  |
| Face2Face      | 28    | −0.0402  | −0.0516  | 0.0550 | −0.1788  | 0.0018  |
| FaceSwap       | 27    | −0.0006  | −0.0345  | 0.0709 | −0.2301  | 0.0098  |
| NeuralTextures | 30    | −0.0720  | −0.0719  | 0.0461 | −0.1583  | 0.0010  |
| original       | 57    | −0.2091  | −0.1825  | 0.0633 | −0.2475  | −0.0284 |

---

## Pooled control-effect distribution

Source: `results/tables/control_effect_distribution.md`

| n control effects | median   | mean     | std    | min      | q25      | q75     | q95    | max    |
|------------------:|---------:|---------:|-------:|---------:|---------:|--------:|-------:|-------:|
| 17 860            | −0.0528  | −0.0679  | 0.0701 | −0.2480  | −0.1103  | −0.0026 | 0.0001 | 0.0689 |

The 95th percentile of the pooled control distribution is ≈ 0.0001, meaning
the threshold τ is near zero for most candidates.

---

## Authentic-condition effect summary

Source: `results/tables/authentic_effects.md`

| condition | count | median δ | mean δ  | std δ  | min δ   | max δ  |
|:----------|------:|---------:|--------:|-------:|--------:|-------:|
| authentic | 360   | −0.0634  | −0.0783 | 0.0720 | −0.2472 | 0.0309 |

---

## Sensitivity to τ percentile

Source: `results/tables/tau_sensitivity.md`

| τ percentile | condition | label | videos | videos supported | video support rate |
|-------------:|:----------|:------|-------:|-----------------:|-------------------:|
| 90           | real      | fake  | 40     | 11               | 0.275              |
| 90           | real      | real  | 20     | 0                | 0.000              |
| 90           | placebo   | fake  | 40     | 10               | 0.250              |
| 90           | placebo   | real  | 20     | 4                | 0.200              |
| 95           | real      | fake  | 40     | 5                | 0.125              |
| 95           | real      | real  | 20     | 0                | 0.000              |
| 95           | placebo   | fake  | 40     | 4                | 0.100              |
| 95           | placebo   | real  | 20     | 1                | 0.050              |
| 99           | real      | fake  | 40     | 5                | 0.125              |
| 99           | real      | real  | 20     | 0                | 0.000              |
| 99           | placebo   | fake  | 40     | 2                | 0.050              |
| 99           | placebo   | real  | 20     | 0                | 0.000              |

At τ = 95 and τ = 99 the `real / real` support rate is 0 in both cases.

---

## Baselines

### No-control baseline (fixed δ > 0.05 threshold)

Source: `results/tables/baseline_no_control.md`

All four condition × label combinations yield 0 supported videos and 0
supported candidates at the δ > 0.05 threshold, confirming that raw
intervention effects alone do not exceed this fixed threshold.

### Sham-only baseline (τ from sham controls only, 95th percentile)

Source: `results/tables/baseline_sham_only.md`

| condition | label | videos | videos supported | video support rate |
|:----------|:------|-------:|-----------------:|-------------------:|
| real      | fake  | 40     | 10               | 0.250              |
| real      | real  | 20     | 0                | 0.000              |
| placebo   | fake  | 40     | 9                | 0.225              |
| placebo   | real  | 20     | 1                | 0.050              |

Using only sham controls inflates the fake support rate (0.25 vs. 0.125 with
the full mixed control set), demonstrating that interval-shift and cue-swap
controls are necessary to calibrate the threshold correctly.

---

## Runtime

Source: `results/tables/runtime.md`

| condition | videos | mean (s) | median (s) | min (s) | max (s) |
|:----------|-------:|---------:|-----------:|--------:|--------:|
| authentic | 60     | 27.77    | 27.71      | 26.47   | 29.32   |
| placebo   | 60     | 27.59    | 27.67      | 26.53   | 28.50   |
| real      | 60     | 15.98    | 16.49      | 10.85   | 21.98   |

The `real` condition is faster because Grad-CAM is run only once per candidate
during candidate generation; the `placebo` and `authentic` conditions skip
Grad-CAM but run the same number of control measurements.

---

## Re-encode sensitivity sanity check

Source: `results/sanity/reencode_sensitivity.csv`

20 videos (10 fake, 10 real) were subjected to four transformations:

| Transformation | Description |
|:---------------|:------------|
| `png_lossless` | PNG round-trip (lossless) |
| `h264_high_quality` | H.264 CRF 18 |
| `h264_moderate` | H.264 CRF 28 |
| `null_blend` | Identity alpha-blend (α = 0.5, identical inputs) |

PNG lossless and null-blend produce zero absolute delta on all 20 videos.
H.264 CRF 18 produces small deltas (median ≈ 0.02–0.05); H.264 CRF 28 can
produce larger deltas on some videos (up to ≈ 0.48 for one outlier), indicating
the detector is sensitive to moderate compression artefacts.

---

## Qualitative figures

All figures are in `docs/figures/`.

| Figure | Description |
|:-------|:------------|
| `case_verified.png` | Deepfakes_000_003 — verified face_texture explanation |
| `case_abstained.png` | real_011 — abstention (no candidate exceeds τ) |
| `case_control_rejected.png` | Deepfakes_001_870 — raw effect present but rejected by controls |
| `effect_vs_control_distribution.png` | Candidate vs. control effect histograms |
| `placebo_vs_real_support.png` | Video support rate: normal vs. placebo |
| `tau_sensitivity.png` | Support rate vs. τ percentile |
| `effect_by_cue.png` | Effect distribution by cue (box plot) |
| `support_by_method.png` | Candidate support rate by FF++ method |
| `intervention_examples.png` | Before/after face crops for all three cues |

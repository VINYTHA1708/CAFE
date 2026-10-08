# CAFE - Counterfactually Verified Explanations for Deepfake Video Detection

CAFE is a research system that verifies whether an explanation given for a deepfake detection decision is actually supported by the detector's behaviour.

Given a video and a frozen deepfake detector, CAFE generates candidate explanations consisting of a visual cue and temporal interval, applies counterfactual interventions, compares the resulting detector-score change against a control distribution, and either emits a verified explanation or abstains.

---

## Project Modules

| #   | Module                       | Purpose                                                                 | Source file(s)                             |
| --- | ---------------------------- | ----------------------------------------------------------------------- | ------------------------------------------ |
| 1   | Preprocessing                | Uniform frame sampling, BlazeFace face detection, and face-crop caching | `cafe/preprocessing.py`                    |
| 2   | Base detector                | Frozen EfficientNet-B4 FF++ detector wrapper                            | `cafe/detector/base_detector.py`           |
| 3   | Candidate generation         | Grad-CAM interval ranking and cue assignment                            | `cafe/candidates.py`                       |
| 4   | Counterfactual interventions | Eye-motion, mouth-motion, and face-texture interventions                | `cafe/interventions.py`                    |
| 5   | Control-based verification   | Control generation, threshold estimation, and p-value calculation       | `cafe/controls.py`, `cafe/verification.py` |
| 6   | Explanation decision         | Supported/abstained decision and explanation rendering                  | `cafe/decision.py`, `cafe/pipeline.py`     |

---

## Installation

The project uses Python 3.12.

### 1. Clone the repository

```powershell
git clone https://github.com/VINYTHA1708/CAFE.git
cd CAFE
```

### 2. Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

For Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the pinned dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set the project root on `PYTHONPATH`

Run this from the repository root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Keep this setting active while running the CAFE scripts.

---

## Detector Reference

CAFE uses code from the upstream `polimi-ispl/icpr2020dfdc` repository.

The reference repository is not redistributed in this project. Clone it into the repository root:

```powershell
git clone https://github.com/polimi-ispl/icpr2020dfdc detector_reference
```

The expected structure is:

```text
CAFE/
└── detector_reference/
```

The upstream repository provides the EfficientNet-B4 implementation and the BlazeFace components used by CAFE.

---

## Dataset - FaceForensics++

CAFE uses a 60-video subset of FaceForensics++ C23.

The experimental subset contains:

- 10 Deepfakes videos
- 10 Face2Face videos
- 10 FaceSwap videos
- 10 NeuralTextures videos
- 20 matching original videos

The dataset is **not redistributed** with this repository.

Obtain FaceForensics++ through its official access process and place the required C23 videos under:

```text
dataset/
├── fake/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceSwap/
│   └── NeuralTextures/
└── real/
```

The project expects the selected videos to correspond to the entries in:

```text
data/manifest.csv
```

---

## Detector Weights

CAFE uses a frozen EfficientNet-B4 checkpoint trained on FaceForensics++ C23 from the upstream `polimi-ispl/icpr2020dfdc` repository.

The checkpoint is:

```text
EfficientNetB4_FFPP_bestval.pth
```

Place it at:

```text
models/EfficientNetB4_FFPP/EfficientNetB4_FFPP_bestval.pth
```

The checkpoint used by the project is available from the upstream repository's referenced storage location.

CAFE also requires the MediaPipe Face Landmarker model:

```text
models/face_landmarks/face_landmarker.task
```

---

## Running CAFE

Make sure the virtual environment is active and `PYTHONPATH` has been set from the repository root.

### Build the dataset manifest

```powershell
python scripts/build_manifest.py
```

Expected result for the project subset:

```text
Total videos: 60
Fake videos: 40
Real videos: 20
Skipped: 0
```

### Precompute frames and face crops

```powershell
python scripts/precompute_frames.py
```

The pipeline samples 32 frames uniformly from each video and detects/crops the dominant face using BlazeFace.

### Precompute facial landmarks

```powershell
python scripts/precompute_landmarks.py
```

### Single-video inference

Example:

```powershell
python scripts/run_cafe.py --video dataset/fake/Deepfakes/000_003.mp4
```

The result is written to:

```text
results/runs/<video_id>.json
```

### Final primary experiment

The final primary experiment is one detector-guided condition over the 60-video
FaceForensics++ C23 subset. It generated candidates from the original
detector/frame scores, then verified each candidate using counterfactual
interventions and 20 controls. It completed with 173 candidates and no failed
videos.

The historical three-condition runner (`scripts/run_batch.py`) evaluates
`real`, `placebo`, and `authentic` with different candidate populations. Its
outputs under `results/runs/` are historical and are not the primary results.
The later candidate-sharing run is also not three independent experimental
conditions.

To run the primary detector-guided condition with the corrected masks and
Colab-ready runner:

```powershell
python scripts/run_batch_corrected_colab.py `
  --manifest data/manifest.csv `
  --config config.yaml `
  --conditions real `
  --out results/primary_full
```

The primary per-video JSON files and summary are stored in
`results/primary_full/`, including
`results/primary_full/batch_corrected_summary.csv`. Do not use the old
three-condition command as a reproduction of these primary results.

---

## Reproducing the Quantitative Results

The final primary results are summarized in `docs/RESULTS.md` and in
`results/primary_full/batch_corrected_summary.csv`. The historical
`scripts/make_figures.py` reads the legacy three-condition JSONs from
`results/runs/`; it does not reproduce the final primary analysis.

### Generate qualitative case studies

```powershell
python scripts/make_qualitative.py
```

### Generate intervention examples

```powershell
python scripts/plot_intervention_examples.py
```

For the complete ordered reproduction procedure, see:

```text
docs/REPRODUCE.md
```

---

## Result Locations

| Artifact                | Path                        |
| ----------------------- | --------------------------- |
| Primary per-video JSONs | `results/primary_full/`     |
| Primary batch summary   | `results/primary_full/batch_corrected_summary.csv` |
| Historical run JSONs    | `results/runs/`             |
| Historical tables       | `results/tables/`           |
| Existing figures        | `docs/figures/`             |
| Screenshots             | `docs/screenshots/`         |
| Demo videos             | `results/demo_videos/`      |
| Detector sanity results | `results/sanity/`           |
| Primary execution log   | `results/primary_full/`     |

---

## Experimental Configuration

The main configuration is stored in:

```text
config.yaml
```

Important parameters include:

| Parameter                    |            Value |
| ---------------------------- | ---------------: |
| Frames per video             |               32 |
| Candidate explanations       |                6 |
| Interval length              | 8 sampled frames |
| Controls per candidate       |               20 |
| Default threshold percentile |               95 |
| Random seed                  |               42 |
| Blend alpha                  |              0.8 |
| Feather size                 |             5 px |
| Blur sigma                   |              8.0 |

The three explanation cues are:

```text
eye_motion
mouth_motion
face_texture
```

---

## Core Verification Principle

For a video `V`, the frozen detector produces a base score:

```text
s = D(V)
```

For a candidate explanation consisting of cue `c` and temporal interval `I`, CAFE applies an intervention:

```text
V' = Phi(V, c, I)
```

and measures the detector-score effect:

```text
Delta(c, I) = D(V) - D(V')
```

The candidate effect is compared with a distribution of control effects.

Let the control effects be:

```text
E = {Delta_1, Delta_2, ..., Delta_m}
```

The verification threshold is the selected percentile of the control distribution:

```text
tau = percentile(E, 95)
```

A candidate explanation is operationally supported when:

```text
Delta(c, I) > tau
```

If no tested candidate passes the verification criterion, CAFE abstains instead of presenting an unsupported explanation.

---

## Primary Experimental Results

On this 60-video evaluation subset, CAFE produced a verified explanation for
6/40 fake videos and abstained on all 20 genuine videos. This is explanation
verification behaviour, not standalone deepfake-detection accuracy. All 173
primary candidates were assigned `face_texture`; this candidate-generation /
cue-assignment bias prevents a fair comparison of the three cue channels.

The six supported candidates each have p = 0.047619, the minimum attainable
with 20 controls under the current calculation. This value alone is not
conventional statistical significance; the small control count and multiple
candidate tests limit statistical interpretation.

See `docs/RESULTS.md` and
`results/primary_full/batch_corrected_summary.csv` for the primary results.
Earlier three-condition results and tooling are historical/superseded and are
not evidence from three independent arms of the primary experiment.

---

## Limitations

Important limitations include:

- CAFE uses one frozen EfficientNet-B4 detector.
- Explanations are therefore specific to the behaviour of that detector.
- Candidate generation relies on Grad-CAM from a fixed detector layer.
- Temporal intervals are candidate explanations rather than ground-truth manipulation boundaries.
- The FF++ subset does not provide temporal manipulation-segment annotations suitable for temporal localization accuracy.
- Landmark detection fails on a small number of sampled frames.
- The current intervention operators use approximate region transformations rather than a full physically realistic manipulation model.
- The experiments use a relatively small 60-video subset.
- The system is designed to verify explanations, not to replace the underlying deepfake detector.

See:

```text
docs/LIMITATIONS.md
```

for the detailed limitations.

---

## Reproducibility

The repository contains the code, configuration, documentation, figures, and evaluation scripts required to reproduce the reported experiment, subject to access to the required external dataset, detector reference code, and model files.

The reproduction workflow is documented in:

```text
docs/REPRODUCE.md
```

The implementation mapping is documented in:

```text
docs/METHOD.md
```

---

## Citation and Attribution

### FaceForensics++

```bibtex
@inproceedings{rossler2019faceforensics++,
  title     = {FaceForensics++: Learning to Detect Manipulated Facial Images},
  author    = {Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa
               and Riess, Christian and Thies, Justus and Niessner, Matthias},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV)},
  year      = {2019}
}
```

### Base Detector

CAFE uses the EfficientNet-B4 implementation and pretrained FF++ checkpoint from the upstream `polimi-ispl/icpr2020dfdc` repository.

```bibtex
@INPROCEEDINGS{9412711,
  author={Bonettini, Nicolò and Cannas, Edoardo Daniele and Mandelli, Sara and Bondi, Luca and Bestagini, Paolo and Tubaro, Stefano},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  title={Video Face Manipulation Detection Through Ensemble of CNNs},
  year={2021},
  pages={5012-5019},
  doi={10.1109/ICPR48806.2021.9412711}
}
```

### BlazeFace

The preprocessing stage uses the BlazeFace implementation included in the upstream detector reference repository.

### MediaPipe

The facial landmark stage uses the MediaPipe Face Landmarker task model.

See `CITATION.md` for the complete attribution information.

---

## Project Structure

```text
CAFE/
├── cafe/
│   ├── candidates.py
│   ├── controls.py
│   ├── decision.py
│   ├── interventions.py
│   ├── landmarks.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── verification.py
│   └── detector/
│       └── base_detector.py
│
├── config.yaml
├── data/
│   └── manifest.csv
├── docs/
│   ├── figures/
│   ├── screenshots/
│   ├── LIMITATIONS.md
│   ├── METHOD.md
│   ├── REPRODUCE.md
│   └── RESULTS.md
├── models/
├── notebooks/
├── results/
├── scripts/
├── tests/
├── CITATION.md
├── README.md
└── requirements.txt
```

---

## License and External Resources

The CAFE source code is maintained separately from the external datasets, pretrained detector checkpoint, detector reference repository, and MediaPipe model.

Users are responsible for complying with the applicable licences, access conditions, and research-use restrictions of those external resources.

The FaceForensics++ dataset is not included in this repository.

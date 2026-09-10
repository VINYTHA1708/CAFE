# Phase 8: Detector Re-Encode Sensitivity Study

## Purpose

This study measures how sensitive the frozen EfficientNet-B4 FF++ detector is to ordinary frame round-tripping and null transformations before counterfactual verification experiments.

Twenty videos were evaluated: 10 fake and 10 real. Each video used the same 32 sampled frames as the baseline detector pipeline.

## Transformations

1. Original cached frames — baseline.
2. PNG lossless round-trip.
3. High-quality H.264 re-encoding using CRF 18.
4. Moderate H.264 re-encoding using CRF 28.
5. Null blend using a 0.5/0.5 blend of a face crop with an identical copy.

For each transformation:

|Delta| = |transformed detector score - original detector score|

## Results

| Transformation | Median |Delta| | 95th percentile |Delta| |
|---|---:|---:|
| PNG lossless | 0.000000 | 0.000000 |
| Null blend | 0.000000 | 0.000000 |
| H.264 high quality | 0.015946 | 0.099953 |
| H.264 moderate | 0.093158 | 0.487245 |

## Interpretation

The lossless PNG round-trip produced zero measurable detector-score change, indicating deterministic inference under this round-trip.

The null blend also produced zero measurable effect, as expected because the face crop was blended with an identical copy of itself.

H.264 re-encoding produced substantially larger score changes, with the effect increasing under moderate compression.

Therefore, CAFE verification experiments should operate on in-memory frame arrays and avoid video encoding/decoding inside the detector scoring loop. Modified videos may still be generated separately for visualization or demonstration.

## Files

- results/sanity/reencode_sensitivity.csv
- docs/figures/reencode_sensitivity.png
- scripts/check_detector.py

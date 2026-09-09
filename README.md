# CAFE
## Counterfactually Verified Explanations for Deepfake Video Detection

CAFE is a research project that verifies whether an explanation given for a deepfake detection decision is actually supported by the detector's behavior.

### Project Pipeline

1. Video Preprocessing
2. Base Deepfake Detection
3. Candidate Explanation Generation
4. Counterfactual Intervention
5. Control-Based Verification
6. Explanation Decision

### Dataset

Primary dataset: FaceForensics++ (C23)

A selected subset of fake and corresponding real videos is used for experimentation.

### Project Structure

- src/preprocessing/ - Video preprocessing
- src/detector/ - Deepfake detector
- src/explanations/ - Candidate explanation generation
- src/counterfactual/ - Counterfactual interventions
- src/verification/ - Control-based verification
- src/evaluation/ - Evaluation and analysis
- dataset/ - Local dataset files
- models/ - Model files
- results/ - Experimental results
- logs/ - Experiment logs

### Status

Implementation in progress.

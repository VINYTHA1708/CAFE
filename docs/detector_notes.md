\# CAFE Base Detector Notes



\## Detector



\- Model: EfficientNet-B4

\- Detector type: Frozen pretrained deepfake detector

\- Repository: `polimi-ispl/icpr2020dfdc`

\- Upstream repository directory used by CAFE: `detector\_reference/`

\- Dataset/checkpoint: FaceForensics++ (FF++) C23

\- Checkpoint: `EfficientNetB4\_FFPP\_bestval.pth`



\## Checkpoint



The pretrained checkpoint was downloaded from the checkpoint URL referenced by the

upstream detector repository:



`https://f002.backblazeb2.com/file/icpr2020/EfficientNetB4\_FFPP\_bestval-93aaad84946829e793d1a67ed7e0309b535e2f2395acb4f8d16b92c0616ba8d7.pth`



Local checkpoint:



`models/EfficientNetB4\_FFPP/EfficientNetB4\_FFPP\_bestval.pth`



\## Input preprocessing



\- Input: face crops from Phase 6

\- Face policy: `scale`

\- Detector patch size: `224 × 224`

\- Normalisation:

&#x20; - Mean: `\[0.485, 0.456, 0.406]`

&#x20; - Standard deviation: `\[0.229, 0.224, 0.225]`

\- No random cropping

\- No test-time augmentation

\- Model kept in evaluation mode



The preprocessing transform is taken from the upstream detector implementation.



\## Inference



The model produces a raw logit for each face crop. CAFE applies the sigmoid function

to obtain a deepfake probability in `\[0, 1]`.



For a video, the CAFE video score is the arithmetic mean of the per-frame

probabilities.



\## CAFE wrapper contract



`BaseDetector` exposes:



\- `score\_frames(faces)` → per-frame probabilities

\- `score\_video(faces)` → mean probability

\- `predict(faces)` → `("REAL", score)` or `("DEEPFAKE", score)`



The detector is frozen and is not fine-tuned or retrained by CAFE.



\## Reproducibility



Inference uses:



\- `model.eval()`

\- `torch.no\_grad()`

\- deterministic preprocessing

\- no random cropping

\- no test-time augmentation



The same cached face array was scored twice and produced bit-identical results.



\## Validation



On a 5 fake + 5 real sanity subset:



\- Fake mean score: `0.7832`

\- Real mean score: `0.1577`

\- Separation: `0.6255`



The detector therefore showed clear separation between fake and real videos on

the selected sanity subset.



\## Upstream code status



The upstream detector implementation is kept separate from the CAFE package.

No modifications are made inside the upstream detector code.


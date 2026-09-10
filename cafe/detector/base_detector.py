from pathlib import Path
import sys

import numpy as np
import torch
from scipy.special import expit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = PROJECT_ROOT / "detector_reference"
CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "EfficientNetB4_FFPP"
    / "EfficientNetB4_FFPP_bestval.pth"
)

if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from architectures import fornet
from isplutils import utils


class BaseDetector:
    """Frozen EfficientNet-B4 FF++ detector wrapper."""

    def __init__(self, checkpoint_path=CHECKPOINT):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = fornet.EfficientNetB4().eval().to(self.device)

        state = torch.load(
            str(checkpoint_path),
            map_location="cpu",
        )
        self.model.load_state_dict(state, strict=True)

        self.transformer = utils.get_transformer(
            face_policy="scale",
            patch_size=224,
            net_normalizer=self.model.get_normalizer(),
            train=False,
        )

    def score_frames(self, faces: np.ndarray) -> np.ndarray:
        """Return one deepfake probability for each face crop."""
        if not isinstance(faces, np.ndarray):
            raise TypeError("faces must be a numpy.ndarray")

        if faces.ndim != 4:
            raise ValueError(
                "faces must have shape (N, H, W, C)"
            )

        if len(faces) == 0:
            return np.array([], dtype=np.float32)

        batch = torch.stack(
            [
                self.transformer(image=face)["image"]
                for face in faces
            ]
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(batch).cpu().numpy().flatten()

        return expit(logits).astype(np.float32)

    def score_video(self, faces: np.ndarray) -> float:
        """Return the mean per-frame deepfake probability."""
        scores = self.score_frames(faces)

        if len(scores) == 0:
            raise ValueError("Cannot score a video with no face crops")

        return float(scores.mean())

    def predict(self, faces: np.ndarray) -> tuple[str, float]:
        """Return ('REAL'|'DEEPFAKE', score)."""
        score = self.score_video(faces)
        label = "DEEPFAKE" if score >= 0.5 else "REAL"
        return label, score

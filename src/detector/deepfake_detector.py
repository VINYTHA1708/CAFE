import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.special import expit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTOR_REFERENCE = PROJECT_ROOT / "detector_reference"
CHECKPOINT = PROJECT_ROOT / "models" / "EfficientNetB4_FFPP" / "EfficientNetB4_FFPP_bestval.pth"

sys.path.insert(0, str(DETECTOR_REFERENCE))

from architectures import fornet
from blazeface import BlazeFace, FaceExtractor, VideoReader
from isplutils import utils


class DeepfakeDetector:
    """Frozen EfficientNet-B4 + BlazeFace detector used by CAFE."""

    def __init__(self, checkpoint_path=CHECKPOINT, num_frames=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames

        self.model = fornet.EfficientNetB4().eval().to(self.device)

        state = torch.load(str(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(state, strict=True)

        self.face_detector = BlazeFace().to(self.device)
        self.face_detector.load_weights(
            str(DETECTOR_REFERENCE / "blazeface" / "blazeface.pth")
        )
        self.face_detector.load_anchors(
            str(DETECTOR_REFERENCE / "blazeface" / "anchors.npy")
        )

        self.video_reader = VideoReader(verbose=False)

        self.face_extractor = FaceExtractor(
            video_read_fn=lambda path: self.video_reader.read_frames(
                path, num_frames=self.num_frames
            ),
            facedet=self.face_detector,
        )

        self.transformer = utils.get_transformer(
            face_policy="scale",
            patch_size=224,
            net_normalizer=self.model.get_normalizer(),
            train=False,
        )

    def predict_frames(self, frames):
        """Return one raw detector logit for each in-memory BGR frame."""

        if not frames:
            return np.array([], dtype=np.float32)

        faces = []
        for frame in frames:
            result = self.face_extractor.process_image(img=frame)

            if not result["faces"]:
                continue

            # FaceExtractor sorts faces by descending confidence.
            faces.append(result["faces"][0])

        if not faces:
            raise ValueError("No faces detected in supplied frames")

        batch = torch.stack(
            [self.transformer(image=face)["image"] for face in faces]
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(batch).cpu().numpy().flatten()

        return logits

    def predict_video(self, video_path):
        """Return the detector's video-level score."""

        result = self.face_extractor.process_videos(
            input_dir=str(Path(video_path).parent),
            filenames=[Path(video_path).name],
            video_idxs=[0],
        )

        if not result:
            raise ValueError(f"No frames/faces detected in: {video_path}")

        faces = [
            frame["faces"][0]
            for frame in result
            if len(frame["faces"]) > 0
        ]

        if not faces:
            raise ValueError(f"No faces detected in: {video_path}")

        batch = torch.stack(
            [self.transformer(image=face)["image"] for face in faces]
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(batch).cpu().numpy().flatten()

        video_score = float(expit(logits.mean()))

        return {
            "score": video_score,
            "logits": logits,
            "num_frames_scored": len(logits),
        }

from pathlib import Path
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANDMARK_MODEL = PROJECT_ROOT / "models" / "face_landmarks" / "face_landmarker.task"


def _create_landmarker():
    base_options = python.BaseOptions(
        model_asset_path=str(LANDMARK_MODEL)
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_landmarks(faces):
    """
    Extract 478 MediaPipe face landmarks from cached face crops.

    Returns:
        landmarks: float32 array of shape (N, 478, 3).
                   Failed frames are filled with NaN values.
        success: boolean array of shape (N,).
    """
    faces = np.asarray(faces)

    if faces.ndim != 4 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape (N,H,W,3)")

    n_frames = len(faces)
    landmarks = np.full((n_frames, 478, 3), np.nan, dtype=np.float32)
    success = np.zeros(n_frames, dtype=bool)

    landmarker = _create_landmarker()

    for i, face in enumerate(faces):
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=face
        )

        result = landmarker.detect(image)

        if not result.face_landmarks:
            continue

        points = result.face_landmarks[0]

        if len(points) != 478:
            continue

        landmarks[i] = np.asarray(
            [[p.x, p.y, p.z] for p in points],
            dtype=np.float32
        )
        success[i] = True

    return landmarks, success


def region_mask(landmarks, region, feather_px=5):
    """
    Create a soft spatial mask from face landmarks.

    Supported regions:
        eyes
        mouth
        face
    """
    landmarks = np.asarray(landmarks)

    if landmarks.shape != (478, 3):
        raise ValueError("landmarks must have shape (478,3)")

    if not np.isfinite(landmarks).all():
        raise ValueError("landmarks contain invalid values")

    h = w = 224

    points = landmarks[:, :2].copy()
    points[:, 0] *= w
    points[:, 1] *= h

    if region == "eyes":
        indices = list(range(33, 133)) + list(range(263, 362))
    elif region == "mouth":
        indices = list(range(61, 88)) + list(range(178, 195)) + list(range(308, 318))
    elif region == "face":
        indices = list(range(478))
    else:
        raise ValueError(f"Unknown region: {region}")

    selected = points[indices]

    mask = np.zeros((h, w), dtype=np.float32)

    import cv2

    hull = cv2.convexHull(
        np.round(selected).astype(np.int32)
    )

    cv2.fillConvexPoly(mask, hull, 1.0)

    if feather_px > 0:
        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=float(feather_px),
            sigmaY=float(feather_px)
        )

    max_value = float(mask.max())
    if max_value > 0:
        mask /= max_value

    return np.clip(mask, 0.0, 1.0).astype(np.float32)
from pathlib import Path
import json
import numpy as np

from cafe.landmarks import extract_landmarks
from cafe.utils.paths import get_cache_path


def load_or_build_landmark_cache(video_id):
    cache_dir = get_cache_path() / video_id
    faces_path = cache_dir / "faces.npy"
    landmarks_path = cache_dir / "landmarks.npy"
    index_path = cache_dir / "index.json"

    if not faces_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Preprocessed cache not found for {video_id}"
        )

    if landmarks_path.exists():
        return np.load(landmarks_path)

    faces = np.load(faces_path)

    landmarks, success = extract_landmarks(faces)

    np.save(landmarks_path, landmarks)

    index = json.loads(index_path.read_text(encoding="utf-8"))

    for i, ok in enumerate(success):
        index[i]["landmark_found"] = bool(ok)

    index_path.write_text(
        json.dumps(index, indent=2),
        encoding="utf-8"
    )

    return landmarks

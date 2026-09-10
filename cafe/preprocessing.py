from pathlib import Path
import json
import cv2
import numpy as np
import sys

from cafe.utils.config import load_config
from cafe.utils.paths import get_cache_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "detector_reference"
if str(REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(REFERENCE_ROOT))

from blazeface import BlazeFace, FaceExtractor


def extract_frames(video_path, n_frames=None):
    """Uniformly sample frames while preserving original indices and timestamps."""
    config = load_config()
    if n_frames is None:
        n_frames = config["frames_per_video"]

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total_frames <= 0 or fps <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: {video_path}")

    n_samples = min(n_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

    frames = []
    original_indices = []
    timestamps = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()
        if not success:
            cap.release()
            raise ValueError(f"Could not read frame {index}: {video_path}")

        frames.append(frame)
        original_indices.append(int(index))
        timestamps.append(float(index / fps))

    cap.release()
    return frames, original_indices, timestamps


def _create_face_extractor():
    face_detector = BlazeFace()
    face_detector.load_weights(str(REFERENCE_ROOT / "blazeface" / "blazeface.pth"))
    face_detector.load_anchors(str(REFERENCE_ROOT / "blazeface" / "anchors.npy"))
    return FaceExtractor(
        video_read_fn=lambda path: [],
        facedet=face_detector,
    )


def detect_and_crop_faces(frames):
    """Detect the highest-confidence face in each frame."""
    config = load_config()
    face_size = 224
    extractor = _create_face_extractor()

    faces = []
    metadata = []

    for frame in frames:
        result = extractor.process_image(img=frame)

        if not result["faces"]:
            faces.append(np.zeros((face_size, face_size, 3), dtype=np.uint8))
            metadata.append({"bbox": None, "face_found": False})
            continue

        face = result["faces"][0]
        face = cv2.resize(face, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
        faces.append(face.astype(np.uint8))

        detection = result["detections"][0]
        detection = result["detections"][0]
        bbox = (
            int(detection[1]),
            int(detection[0]),
            int(detection[3]),
            int(detection[2]),
        )
        metadata.append({"bbox": bbox, "face_found": True})

    return faces, metadata


def build_frame_index(original_indices, timestamps, metadata):
    return [
        {
            "sampled_position": i,
            "original_frame_index": original_indices[i],
            "timestamp_sec": timestamps[i],
            "bbox": metadata[i]["bbox"],
            "face_found": metadata[i]["face_found"],
        }
        for i in range(len(original_indices))
    ]


def load_or_build_cache(video_id, video_path=None):
    config = load_config()
    cache_dir = get_cache_path() / video_id
    frames_path = cache_dir / "frames.npy"
    faces_path = cache_dir / "faces.npy"
    index_path = cache_dir / "index.json"

    if frames_path.exists() and faces_path.exists() and index_path.exists():
        return {
            "frames": np.load(frames_path),
            "faces": np.load(faces_path),
            "index": json.loads(index_path.read_text(encoding="utf-8")),
        }

    if video_path is None:
        raise ValueError("video_path is required when cache does not exist")

    frames, indices, timestamps = extract_frames(
        video_path, config["frames_per_video"]
    )
    faces, metadata = detect_and_crop_faces(frames)
    index = build_frame_index(indices, timestamps, metadata)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(frames_path, np.asarray(frames, dtype=np.uint8))
    np.save(faces_path, np.asarray(faces, dtype=np.uint8))
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    return {
        "frames": np.asarray(frames, dtype=np.uint8),
        "faces": np.asarray(faces, dtype=np.uint8),
        "index": index,
    }


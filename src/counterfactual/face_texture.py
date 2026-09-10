import cv2
import numpy as np

from src.explanations.face_landmarks import FaceLandmarkDetector
from src.explanations.face_regions import get_region_box
from src.counterfactual.intervention_utils import blend_region, clamp_box


FACE_TEXTURE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
    103, 67, 109
]


def _extract_region(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    return frame[y1:y2, x1:x2]


def _interpolate_region(region_start, region_end, t, target_size):
    """Resize both regions to target_size and linearly interpolate."""
    a = cv2.resize(
        region_start, target_size, interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    b = cv2.resize(
        region_end, target_size, interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    return np.clip(
        a * (1.0 - t) + b * t, 0, 255
    ).astype(np.uint8)


def apply_face_texture(
    frames,
    start_frame,
    end_frame,
    feather_ratio=0.15,
    landmark_detector=None,
):
    """
    Replace inner face-region content in intermediate frames with
    temporal interpolation between the boundary frames.

    Args:
        frames: list of BGR numpy arrays (in-memory, not modified in place).
        start_frame: index of interval start boundary (inclusive).
        end_frame: index of interval end boundary (inclusive).
        feather_ratio: passed to blend_region.
        landmark_detector: optional pre-constructed FaceLandmarkDetector.

    Returns:
        New list of frames with the intervention applied.
    """
    if start_frame < 0 or end_frame >= len(frames) or start_frame >= end_frame:
        raise ValueError(
            f"Invalid interval [{start_frame}, {end_frame}] for {len(frames)} frames."
        )

    own_detector = landmark_detector is None
    if own_detector:
        landmark_detector = FaceLandmarkDetector()

    try:
        frame_start = frames[start_frame]
        frame_end = frames[end_frame]
        h, w = frame_start.shape[:2]

        lm_start = landmark_detector.detect(frame_start)
        lm_end = landmark_detector.detect(frame_end)

        if lm_start is None or lm_end is None:
            raise RuntimeError(
                "Could not detect face landmarks on one or both boundary frames."
            )

        box_s = get_region_box(lm_start, FACE_TEXTURE_INDICES, w, h)
        box_e = get_region_box(lm_end, FACE_TEXTURE_INDICES, w, h)

        region_s = _extract_region(frame_start, box_s)
        region_e = _extract_region(frame_end, box_e)

        x1, y1, x2, y2 = clamp_box(box_s, w, h)
        target_size = (x2 - x1, y2 - y1)

        if target_size[0] < 1 or target_size[1] < 1:
            raise RuntimeError(
                "Face texture bounding box is too small to interpolate."
            )

        interval_len = end_frame - start_frame
        output = [f.copy() for f in frames]

        for idx in range(start_frame + 1, end_frame):
            t = (idx - start_frame) / interval_len

            interpolated = _interpolate_region(
                region_s, region_e, t, target_size
            )

            output[idx] = blend_region(
                output[idx],
                interpolated,
                box_s,
                feather_ratio,
            )

    finally:
        if own_detector:
            landmark_detector.close()

    return output

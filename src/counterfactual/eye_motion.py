import cv2
import numpy as np

from src.explanations.face_landmarks import FaceLandmarkDetector
from src.explanations.face_regions import get_eye_boxes
from src.counterfactual.intervention_utils import blend_region, clamp_box


def _extract_region(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    return frame[y1:y2, x1:x2]


def _interpolate_region(region_start, region_end, t, target_size):
    """Resize both regions to target_size and linearly interpolate."""
    a = cv2.resize(region_start, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    b = cv2.resize(region_end, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.clip(a * (1.0 - t) + b * t, 0, 255).astype(np.uint8)


def apply_eye_motion(
    frames,
    start_frame,
    end_frame,
    eyes=("left_eye", "right_eye"),
    feather_ratio=0.15,
    landmark_detector=None,
):
    """
    Replace eye-region content in frames[start_frame+1 : end_frame] with
    temporal interpolation between the boundary frames.

    Args:
        frames: list of BGR numpy arrays (in-memory, not modified in place).
        start_frame: index of the interval start boundary (inclusive, kept as-is).
        end_frame: index of the interval end boundary (inclusive, kept as-is).
        eyes: which eyes to intervene on; any subset of ("left_eye", "right_eye").
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

        boxes_start = get_eye_boxes(lm_start, w, h)
        boxes_end = get_eye_boxes(lm_end, w, h)

        interval_len = end_frame - start_frame  # denominator for t

        output = [f.copy() for f in frames]

        for idx in range(start_frame + 1, end_frame):
            t = (idx - start_frame) / interval_len
            frame = output[idx]

            for eye in eyes:
                box_s = boxes_start[eye]
                box_e = boxes_end[eye]

                region_s = _extract_region(frame_start, box_s)
                region_e = _extract_region(frame_end, box_e)

                # Use the start-frame box as the target location/size.
                x1, y1, x2, y2 = clamp_box(box_s, w, h)
                target_size = (x2 - x1, y2 - y1)

                if target_size[0] < 1 or target_size[1] < 1:
                    continue

                interpolated = _interpolate_region(region_s, region_e, t, target_size)
                frame = blend_region(frame, interpolated, box_s, feather_ratio)

            output[idx] = frame

    finally:
        if own_detector:
            landmark_detector.close()

    return output

import numpy as np

from src.explanations.face_regions import (
    get_eye_boxes,
    get_mouth_box,
    get_region_box,
    FACE_TEXTURE_INDICES,
)


def _box_center(box):
    x1, y1, x2, y2 = box
    return np.array(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        dtype=np.float32,
    )


def _center_motion(boxes):
    """Return frame-to-frame center displacement."""
    if len(boxes) < 2:
        return np.array([], dtype=np.float32)

    centers = np.array([_box_center(box) for box in boxes])
    return np.linalg.norm(np.diff(centers, axis=0), axis=1)


def generate_candidate_explanations(
    landmarks_per_frame,
    frame_width,
    frame_height,
    min_interval=2,
    motion_threshold=3.0,
):
    """
    Generate candidate facial explanations from landmark motion.

    Each candidate is represented as:
        {
            "cue": str,
            "start_frame": int,
            "end_frame": int,
        }

    Args:
        landmarks_per_frame: list of MediaPipe landmark results or None.
        frame_width: video frame width in pixels.
        frame_height: video frame height in pixels.
        min_interval: minimum candidate interval length.
        motion_threshold: minimum center displacement in pixels.

    Returns:
        List of candidate explanation dictionaries.
    """
    valid = [
        landmarks is not None
        for landmarks in landmarks_per_frame
    ]

    candidates = []

    left_eye_boxes = []
    right_eye_boxes = []
    mouth_boxes = []
    texture_boxes = []

    for landmarks in landmarks_per_frame:
        if landmarks is None:
            left_eye_boxes.append(None)
            right_eye_boxes.append(None)
            mouth_boxes.append(None)
            texture_boxes.append(None)
            continue

        eyes = get_eye_boxes(
            landmarks,
            frame_width,
            frame_height,
        )

        left_eye_boxes.append(eyes["left_eye"])
        right_eye_boxes.append(eyes["right_eye"])

        mouth_boxes.append(
            get_mouth_box(
                landmarks,
                frame_width,
                frame_height,
            )
        )

        texture_boxes.append(
            get_region_box(
                landmarks,
                FACE_TEXTURE_INDICES,
                frame_width,
                frame_height,
            )
        )

    def add_motion_candidates(cue, boxes):
        valid_boxes = [
            box for box in boxes
            if box is not None
        ]

        if len(valid_boxes) < 2:
            return

        previous = None
        start = None

        for frame_idx, box in enumerate(boxes):
            if box is None:
                if start is not None and frame_idx - start >= min_interval:
                    candidates.append(
                        {
                            "cue": cue,
                            "start_frame": start,
                            "end_frame": frame_idx - 1,
                        }
                    )
                start = None
                previous = None
                continue

            if previous is None:
                previous = box
                continue

            displacement = float(
                np.linalg.norm(
                    _box_center(box) - _box_center(previous)
                )
            )

            if displacement >= motion_threshold:
                if start is None:
                    start = frame_idx - 1
            elif start is not None:
                if frame_idx - start >= min_interval:
                    candidates.append(
                        {
                            "cue": cue,
                            "start_frame": start,
                            "end_frame": frame_idx - 1,
                        }
                    )
                start = None

            previous = box

        if start is not None:
            end = len(boxes) - 1
            if end - start >= min_interval:
                candidates.append(
                    {
                        "cue": cue,
                        "start_frame": start,
                        "end_frame": end,
                    }
                )

    add_motion_candidates("left_eye_motion", left_eye_boxes)
    add_motion_candidates("right_eye_motion", right_eye_boxes)
    add_motion_candidates("mouth_motion", mouth_boxes)
    add_motion_candidates("face_texture", texture_boxes)

    return candidates

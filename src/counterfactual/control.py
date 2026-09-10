import cv2
import numpy as np

from src.counterfactual.intervention_utils import blend_region, clamp_box


def get_control_box(frame_shape, face_box, scale=0.3):
    """
    Create a matched control region inside the detected face area.

    The control region is placed in the upper side of the face region,
    providing a nuisance intervention within the detector's face crop.
    """
    height, width = frame_shape[:2]

    x1, y1, x2, y2 = clamp_box(face_box, width, height)

    face_width = x2 - x1
    face_height = y2 - y1

    control_width = max(8, int(face_width * scale))
    control_height = max(8, int(face_height * scale))

    # Place the control region near the upper-left part of the face.
    cx1 = x1 + int(face_width * 0.05)
    cy1 = y1 + int(face_height * 0.05)

    cx2 = min(x2, cx1 + control_width)
    cy2 = min(y2, cy1 + control_height)

    return clamp_box(
        (cx1, cy1, cx2, cy2),
        width,
        height,
    )

def _extract_region(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    return frame[y1:y2, x1:x2]


def _interpolate_region(region_start, region_end, t, target_size):
    """Resize both regions and linearly interpolate them."""
    a = cv2.resize(
        region_start,
        target_size,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    b = cv2.resize(
        region_end,
        target_size,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    return np.clip(
        a * (1.0 - t) + b * t,
        0,
        255,
    ).astype(np.uint8)


def apply_control_intervention(
    frames,
    start_frame,
    end_frame,
    control_box,
    feather_ratio=0.15,
):
    """
    Apply a neutral temporal interpolation to a control region.

    Boundary frames remain unchanged. Only intermediate frames are
    modified.
    """
    if start_frame < 0 or end_frame >= len(frames) or start_frame >= end_frame:
        raise ValueError(
            f"Invalid interval [{start_frame}, {end_frame}] for {len(frames)} frames."
        )

    frame_start = frames[start_frame]
    frame_end = frames[end_frame]

    region_start = _extract_region(frame_start, control_box)
    region_end = _extract_region(frame_end, control_box)

    x1, y1, x2, y2 = clamp_box(
        control_box,
        frame_start.shape[1],
        frame_start.shape[0],
    )

    target_size = (x2 - x1, y2 - y1)

    if target_size[0] < 1 or target_size[1] < 1:
        raise RuntimeError("Control bounding box is too small.")

    interval_len = end_frame - start_frame
    output = [frame.copy() for frame in frames]

    for idx in range(start_frame + 1, end_frame):
        t = (idx - start_frame) / interval_len

        interpolated = _interpolate_region(
            region_start,
            region_end,
            t,
            target_size,
        )

        output[idx] = blend_region(
            output[idx],
            interpolated,
            control_box,
            feather_ratio,
        )

    return output

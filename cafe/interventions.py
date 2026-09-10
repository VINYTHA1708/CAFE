from pathlib import Path
import numpy as np

from cafe.landmarks import region_mask
from cafe.utils.config import load_config


def apply_intervention(frames, landmarks, cue, interval, strength=None):
    """
    Apply a counterfactual intervention and return a modified copy.

    Parameters
    ----------
    frames : np.ndarray
        Frame array with shape (N,H,W,3).
    landmarks : np.ndarray
        Landmark array with shape (N,478,3).
    cue : str
        One of: eye_motion, mouth_motion, face_texture.
    interval : tuple[int, int]
        Inclusive original-frame interval.
    strength : dict, optional
        Intervention parameters. If omitted, values are read from config.yaml.
    """
    frames = np.asarray(frames)
    landmarks = np.asarray(landmarks)

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (N,H,W,3)")

    if landmarks.shape != (len(frames), 478, 3):
        raise ValueError(
            "landmarks must have shape (N,478,3) matching frames"
        )

    if cue not in {"eye_motion", "mouth_motion", "face_texture"}:
        raise ValueError(f"Unknown cue: {cue}")

    if len(interval) != 2:
        raise ValueError("interval must contain (t1,t2)")

    t1, t2 = map(int, interval)

    if t1 > t2:
        raise ValueError("interval start must be <= interval end")

    if t1 < 0 or t2 >= len(frames):
        raise ValueError("interval must be within sampled frame positions")

    config = load_config()

    if strength is None:
        strength = config["intervention"]

    for key in ("blend_alpha", "feather_px", "blur_sigma"):
        if key not in strength:
            raise ValueError(f"Missing intervention parameter: {key}")

    result = frames.copy()

    # Operators will be added next.
    if cue == "eye_motion":
        return _apply_eye_motion(
            result, landmarks, t1, t2, strength
        )

    if cue == "mouth_motion":
        return _apply_mouth_motion(
            result, landmarks, t1, t2, strength
        )

    return _apply_face_texture(
        result, landmarks, t1, t2, strength
    )


def _apply_eye_motion(frames, landmarks, t1, t2, strength):
    raise NotImplementedError


def _apply_mouth_motion(frames, landmarks, t1, t2, strength):
    raise NotImplementedError


def _apply_face_texture(frames, landmarks, t1, t2, strength):
    raise NotImplementedError
from pathlib import Path
import numpy as np
import cv2

from cafe.landmarks import region_mask
from cafe.utils.config import load_config


def _alpha_composite(original, replacement, mask, alpha):
    """
    Feathered alpha composite.

    alpha controls intervention strength.
    mask is a soft [0,1] spatial mask.
    """
    a = np.clip(mask * alpha, 0.0, 1.0)[..., None]

    result = (
        original.astype(np.float32) * (1.0 - a)
        + replacement.astype(np.float32) * a
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_face_texture(frames, landmarks, t1, t2, strength):
    result = frames.copy()

    alpha = float(strength["blend_alpha"])
    feather_px = int(strength["feather_px"])
    blur_sigma = float(strength["blur_sigma"])

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend_alpha must be in [0,1]")

    if feather_px < 0:
        raise ValueError("feather_px must be >= 0")

    if blur_sigma <= 0:
        raise ValueError("blur_sigma must be > 0")

    for t in range(t1, t2 + 1):
        lm = landmarks[t]

        if not np.isfinite(lm).all():
            continue

        mask = region_mask(
            lm,
            "face",
            feather_px=feather_px
        )

        replacement = cv2.GaussianBlur(
            result[t],
            (0, 0),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma
        )

        result[t] = _alpha_composite(
            result[t],
            replacement,
            mask,
            alpha
        )

    return result
def _apply_mouth_motion(frames, landmarks, t1, t2, strength):
    result = frames.copy()

    alpha = float(strength["blend_alpha"])
    feather_px = int(strength["feather_px"])

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend_alpha must be in [0,1]")

    if feather_px < 0:
        raise ValueError("feather_px must be >= 0")

    n = len(frames)

    before = max(0, t1 - 1)
    after = min(n - 1, t2 + 1)

    if not np.isfinite(landmarks[before]).all():
        before = t1

    if not np.isfinite(landmarks[after]).all():
        after = t2

    if not np.isfinite(landmarks[before]).all() or not np.isfinite(landmarks[after]).all():
        return result

    def mouth_box(lm):
        pts = lm[:, :2].copy()
        pts[:, 0] *= frames.shape[2]
        pts[:, 1] *= frames.shape[1]

        indices = list(range(61, 88)) + list(range(178, 195)) + list(range(308, 318))
        selected = pts[indices]

        x0 = max(0, int(np.floor(selected[:, 0].min())))
        x1 = min(frames.shape[2], int(np.ceil(selected[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(selected[:, 1].min())))
        y1 = min(frames.shape[1], int(np.ceil(selected[:, 1].max())) + 1)

        return x0, y0, x1, y1

    bx0, by0, bx1, by1 = mouth_box(landmarks[before])
    ax0, ay0, ax1, ay1 = mouth_box(landmarks[after])

    source_before = frames[before, by0:by1, bx0:bx1]
    source_after = frames[after, ay0:ay1, ax0:ax1]

    for t in range(t1, t2 + 1):
        lm = landmarks[t]

        if not np.isfinite(lm).all():
            continue

        tx0, ty0, tx1, ty1 = mouth_box(lm)

        width = tx1 - tx0
        height = ty1 - ty0

        if width <= 1 or height <= 1:
            continue

        before_resized = cv2.resize(
            source_before,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        after_resized = cv2.resize(
            source_after,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        position = 0.0 if t2 == t1 else (t - t1) / (t2 - t1)

        replacement = (
            before_resized.astype(np.float32) * (1.0 - position)
            + after_resized.astype(np.float32) * position
        ).astype(np.uint8)

        mask = region_mask(
            lm,
            "mouth",
            feather_px=feather_px
        )

        local_mask = mask[ty0:ty1, tx0:tx1]

        original = result[t, ty0:ty1, tx0:tx1]

        result[t, ty0:ty1, tx0:tx1] = _alpha_composite(
            original,
            replacement,
            local_mask,
            alpha
        )

    return result
def _apply_eye_motion(frames, landmarks, t1, t2, strength):
    result = frames.copy()

    alpha = float(strength["blend_alpha"])
    feather_px = int(strength["feather_px"])

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend_alpha must be in [0,1]")

    if feather_px < 0:
        raise ValueError("feather_px must be >= 0")

    n = len(frames)

    before = max(0, t1 - 1)
    after = min(n - 1, t2 + 1)

    if not np.isfinite(landmarks[before]).all():
        before = t1

    if not np.isfinite(landmarks[after]).all():
        after = t2

    if not np.isfinite(landmarks[before]).all() or not np.isfinite(landmarks[after]).all():
        return result

    def eye_box(lm):
        pts = lm[:, :2].copy()
        pts[:, 0] *= frames.shape[2]
        pts[:, 1] *= frames.shape[1]

        indices = list(range(33, 133)) + list(range(263, 362))
        selected = pts[indices]

        x0 = max(0, int(np.floor(selected[:, 0].min())))
        x1 = min(frames.shape[2], int(np.ceil(selected[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(selected[:, 1].min())))
        y1 = min(frames.shape[1], int(np.ceil(selected[:, 1].max())) + 1)

        return x0, y0, x1, y1

    bx0, by0, bx1, by1 = eye_box(landmarks[before])
    ax0, ay0, ax1, ay1 = eye_box(landmarks[after])

    source_before = frames[before, by0:by1, bx0:bx1]
    source_after = frames[after, ay0:ay1, ax0:ax1]

    for t in range(t1, t2 + 1):
        lm = landmarks[t]

        if not np.isfinite(lm).all():
            continue

        tx0, ty0, tx1, ty1 = eye_box(lm)

        width = tx1 - tx0
        height = ty1 - ty0

        if width <= 1 or height <= 1:
            continue

        before_resized = cv2.resize(
            source_before,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        after_resized = cv2.resize(
            source_after,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        position = 0.0 if t2 == t1 else (t - t1) / (t2 - t1)

        replacement = (
            before_resized.astype(np.float32) * (1.0 - position)
            + after_resized.astype(np.float32) * position
        ).astype(np.uint8)

        mask = region_mask(
            lm,
            "eyes",
            feather_px=feather_px
        )

        local_mask = mask[ty0:ty1, tx0:tx1]
        original = result[t, ty0:ty1, tx0:tx1]

        result[t, ty0:ty1, tx0:tx1] = _alpha_composite(
            original,
            replacement,
            local_mask,
            alpha
        )

    return result

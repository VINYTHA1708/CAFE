import cv2
import numpy as np


def clamp_box(box, width, height):
    """Clamp (x1, y1, x2, y2) to valid image boundaries."""
    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return x1, y1, x2, y2


def create_feather_mask(height, width, feather_ratio=0.15):
    """Create a soft elliptical mask for natural region blending."""
    mask = np.zeros((height, width), dtype=np.float32)

    center = (width // 2, height // 2)
    axes = (
        max(1, int(width * 0.5)),
        max(1, int(height * 0.5)),
    )

    cv2.ellipse(
        mask,
        center,
        axes,
        0,
        0,
        360,
        1.0,
        -1,
    )

    blur_size = max(3, int(min(height, width) * feather_ratio))
    if blur_size % 2 == 0:
        blur_size += 1

    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask


def blend_region(frame, replacement, box, feather_ratio=0.15):
    """Feather-blend a replacement region into a frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, width, height)

    region_width = x2 - x1
    region_height = y2 - y1

    replacement = cv2.resize(
        replacement,
        (region_width, region_height),
        interpolation=cv2.INTER_LINEAR,
    )

    mask = create_feather_mask(
        region_height,
        region_width,
        feather_ratio,
    )

    mask = mask[..., None]

    original_region = frame[y1:y2, x1:x2].astype(np.float32)
    replacement = replacement.astype(np.float32)

    blended = (
        original_region * (1.0 - mask)
        + replacement * mask
    )

    output = frame.copy()
    output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return output

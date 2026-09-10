import numpy as np


# MediaPipe Face Mesh landmark indices.
# These groups cover the visible left/right eye contours
# and the outer mouth/lip region.
LEFT_EYE_INDICES = [
    33, 133, 160, 159, 158, 157, 173,
    144, 145, 153, 154, 155, 246
]

RIGHT_EYE_INDICES = [
    362, 263, 387, 386, 385, 384, 398,
    373, 374, 380, 381, 382, 466
]

MOUTH_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 409, 270, 269, 267,
    0, 37, 39, 40, 185, 146
]


def landmarks_to_points(landmarks, frame_width, frame_height):
    """Convert normalized MediaPipe landmarks to pixel coordinates."""
    points = np.array(
        [
            [
                int(round(point.x * frame_width)),
                int(round(point.y * frame_height)),
            ]
            for point in landmarks
        ],
        dtype=np.int32,
    )

    return points


def get_region_box(
    landmarks,
    indices,
    frame_width,
    frame_height,
    padding_ratio=0.25,
):
    """Return a padded pixel bounding box for selected landmarks."""
    points = landmarks_to_points(
        landmarks,
        frame_width,
        frame_height,
    )

    selected = points[indices]

    x_min = int(selected[:, 0].min())
    y_min = int(selected[:, 1].min())
    x_max = int(selected[:, 0].max())
    y_max = int(selected[:, 1].max())

    width = max(1, x_max - x_min)
    height = max(1, y_max - y_min)

    pad_x = int(round(width * padding_ratio))
    pad_y = int(round(height * padding_ratio))

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(frame_width, x_max + pad_x)
    y_max = min(frame_height, y_max + pad_y)

    return x_min, y_min, x_max, y_max


def get_eye_boxes(landmarks, frame_width, frame_height):
    """Return left-eye and right-eye bounding boxes."""
    left_box = get_region_box(
        landmarks,
        LEFT_EYE_INDICES,
        frame_width,
        frame_height,
    )

    right_box = get_region_box(
        landmarks,
        RIGHT_EYE_INDICES,
        frame_width,
        frame_height,
    )

    return {
        "left_eye": left_box,
        "right_eye": right_box,
    }


def get_mouth_box(landmarks, frame_width, frame_height):
    """Return the padded mouth bounding box."""
    return get_region_box(
        landmarks,
        MOUTH_INDICES,
        frame_width,
        frame_height,
    )

FACE_TEXTURE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
    103, 67, 109
]

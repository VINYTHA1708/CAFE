import cv2
from pathlib import Path


def get_video_info(video_path):
    """
    Read basic information about a video.

    Returns:
        dict: frame count, FPS, width, height, and duration.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0

    return {
        "path": str(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
    }

def sample_frame_indices(frame_count, num_samples=32):
    """
    Select evenly spaced frame indices from a video.

    Args:
        frame_count: Total number of frames in the video.
        num_samples: Number of frames to sample.

    Returns:
        list: Selected frame indices.
    """
    if frame_count <= 0:
        return []

    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    num_samples = min(num_samples, frame_count)

    indices = [
        int(i * (frame_count - 1) / (num_samples - 1))
        for i in range(num_samples)
    ] if num_samples > 1 else [0]

    return indices

def load_sampled_frames(video_path, num_samples=32):
    """
    Load evenly sampled frames from a video.

    Args:
        video_path: Path to the video file.
        num_samples: Number of frames to load.

    Returns:
        tuple: (frames, frame_indices)
            frames: List of BGR frames as NumPy arrays.
            frame_indices: Corresponding frame numbers.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(frame_count, num_samples)

    frames = []

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = cap.read()

        if not success:
            cap.release()
            raise ValueError(
                f"Could not read frame {index} from video: {video_path}"
            )

        frames.append(frame)

    cap.release()

    return frames, frame_indices
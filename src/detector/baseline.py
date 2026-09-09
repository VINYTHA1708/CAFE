import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.deepfake_detector import DeepfakeDetector


def run_baseline(video_path):
    """Run the frozen detector and return the baseline result."""
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = DeepfakeDetector()
    return detector.predict_video(video_path)


if __name__ == "__main__":
    video = Path("dataset/fake/Deepfakes/000_003.mp4")

    result = run_baseline(video)

    print("Baseline inference: OK")
    print("Video:", video)
    print("Score:", result["score"])
    print("Frames scored:", result["num_frames_scored"])

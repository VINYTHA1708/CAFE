from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANDMARK_MODEL = (
    PROJECT_ROOT
    / "models"
    / "face_landmarks"
    / "face_landmarker.task"
)


class FaceLandmarkDetector:
    """MediaPipe face landmark detector for CAFE."""

    def __init__(self, model_path=LANDMARK_MODEL):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmark model not found: {model_path}"
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

    def detect(self, frame):
        """
        Detect face landmarks from a BGR OpenCV frame.

        Returns:
            list of normalized landmarks, or None if no face is found.
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self.landmarker.detect(image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    def close(self):
        """Release the MediaPipe landmarker."""
        self.landmarker.close()

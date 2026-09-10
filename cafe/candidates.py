import numpy as np
import torch


CUE_REGIONS = {
    "eye_motion": "eyes",
    "mouth_motion": "mouth",
    "face_texture": "face",
}


def propose_intervals(per_frame_scores, interval_length, k):
    """Return top-k non-overlapping intervals in sampled positions."""
    scores = np.asarray(per_frame_scores, dtype=np.float32)

    if scores.ndim != 1:
        raise ValueError("per_frame_scores must be a 1-D array")
    if interval_length < 1:
        raise ValueError("interval_length must be >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")

    n_frames = len(scores)

    if interval_length > n_frames:
        raise ValueError("interval_length cannot exceed number of frames")

    windows = []

    for start in range(n_frames - interval_length + 1):
        end = start + interval_length - 1
        score = float(scores[start:end + 1].mean())
        windows.append((score, start, end))

    windows.sort(key=lambda item: (-item[0], item[1]))

    selected = []

    for score, start, end in windows:
        overlaps = any(
            not (end < selected_start or start > selected_end)
            for _, selected_start, selected_end in selected
        )

        if overlaps:
            continue

        selected.append((score, start, end))

        if len(selected) == k:
            break

    return [
        (start, end, score)
        for score, start, end in selected
    ]


def _gradcam_for_face(detector, face):
    """Compute a normalized Grad-CAM heatmap for one face crop."""
    activations = []
    gradients = []

    model = detector.model
    layer = model.efficientnet._conv_head

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    try:
        x = detector.transformer(image=face)["image"]
        x = x.unsqueeze(0).to(detector.device)

        model.zero_grad(set_to_none=True)

        logit = model(x)
        logit.backward()

        activation = activations[0]
        gradient = gradients[0]

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1)
        cam = torch.relu(cam)

        cam_min = cam.detach().min()
        cam_max = cam.detach().max()

        if float(cam_max - cam_min) > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        return cam.squeeze().detach().cpu().numpy().astype(np.float32)

    finally:
        handle_f.remove()
        handle_b.remove()


def assign_cue(faces, landmarks, interval, detector):
    """Assign the cue with the highest mean Grad-CAM overlap."""
    from cafe.landmarks import region_mask

    t1, t2 = map(int, interval)

    if t1 < 0 or t2 >= len(faces) or t1 > t2:
        raise ValueError("Invalid interval")

    overlap_sums = {
        cue: 0.0
        for cue in CUE_REGIONS
    }

    valid_frames = 0

    for t in range(t1, t2 + 1):
        if not np.isfinite(landmarks[t]).all():
            continue

        cam = _gradcam_for_face(detector, faces[t])

        for cue, region in CUE_REGIONS.items():
            mask = region_mask(
                landmarks[t],
                region,
                feather_px=5,
            )

            overlap_sums[cue] += float(
                (cam * mask).sum() / (cam.sum() + 1e-8)
            )

        valid_frames += 1

    if valid_frames == 0:
        raise ValueError("No valid landmark frames in interval")

    overlaps = {
        cue: value / valid_frames
        for cue, value in overlap_sums.items()
    }

    best_cue = max(overlaps, key=overlaps.get)

    return best_cue, overlaps
def _to_original_interval(frame_index, interval):
    """Convert sampled-position interval to original frame indices."""
    t1, t2 = map(int, interval)

    if t1 < 0 or t2 >= len(frame_index) or t1 > t2:
        raise ValueError("Invalid sampled interval")

    return (
        int(frame_index[t1]["original_frame_index"]),
        int(frame_index[t2]["original_frame_index"]),
    )


def generate_candidates(video_id, config=None):
    """Generate detector-driven candidate explanation hypotheses."""
    import json
    from pathlib import Path

    import numpy as np

    from cafe.detector.base_detector import BaseDetector
    from cafe.utils.config import load_config

    if config is None:
        config = load_config()

    cache_dir = Path(config["paths"]["cache"]) / video_id

    faces = np.load(cache_dir / "faces.npy")
    landmarks = np.load(cache_dir / "landmarks.npy")

    with open(cache_dir / "index.json", "r", encoding="utf-8") as f:
        frame_index = json.load(f)

    detector = BaseDetector()

    per_frame_scores = detector.score_frames(faces)

    proposed = propose_intervals(
        per_frame_scores,
        config["interval_length"],
        config["n_candidates"],
    )

    candidates = []

    for sampled_start, sampled_end, window_score in proposed:
        cue, overlaps = assign_cue(
            faces,
            landmarks,
            (sampled_start, sampled_end),
            detector,
        )

        original_interval = _to_original_interval(
            frame_index,
            (sampled_start, sampled_end),
        )

        candidates.append(
            {
                "cue": cue,
                "interval": original_interval,
                "window_score": float(window_score),
                "cue_overlap": float(overlaps[cue]),
                "cue_overlaps": {
                    key: float(value)
                    for key, value in overlaps.items()
                },
                "sampled_interval": (
                    int(sampled_start),
                    int(sampled_end),
                ),
            }
        )

    return candidates

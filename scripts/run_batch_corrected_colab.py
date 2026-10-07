"""Run the corrected CAFE batch with batched detector inference.

The frozen detector, preprocessing, candidate/control definitions, and
statistical decision rules are reused. Only the canonical landmark masks and
the execution/inference batching are specified here.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cafe import interventions as intervention_module
from cafe import landmarks as landmark_module
from cafe.candidates import (
    assign_cue,
    generate_placebo_candidates,
    propose_intervals,
)
from cafe.controls import apply_control, generate_controls
from cafe.detector.base_detector import BaseDetector
from cafe.interventions import apply_intervention
from cafe.landmarks import load_or_build_landmark_cache
from cafe.preprocessing import load_or_build_cache
from cafe.utils.config import load_config
from cafe.verification import compute_p_value, compute_threshold

CONDITIONS = ("real", "placebo", "authentic")
RIGHT_EYE_INDICES = (
    7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
)
LEFT_EYE_INDICES = (
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
)
MOUTH_INDICES = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178,
    181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321,
    324, 375, 402, 405, 409, 415,
)


def corrected_region_mask(landmarks, region, feather_px=5):
    """Build canonical, feathered eye/lip masks; preserve the original face mask."""
    landmarks = np.asarray(landmarks)
    if landmarks.shape != (478, 3):
        raise ValueError("landmarks must have shape (478,3)")
    if not np.isfinite(landmarks).all():
        raise ValueError("landmarks contain invalid values")

    points = landmarks[:, :2].copy()
    points[:, 0] *= 224
    points[:, 1] *= 224

    if region == "eyes":
        index_groups = (RIGHT_EYE_INDICES, LEFT_EYE_INDICES)
    elif region == "mouth":
        index_groups = (MOUTH_INDICES,)
    elif region == "face":
        index_groups = (tuple(range(478)),)
    else:
        raise ValueError(f"Unknown region: {region}")

    mask = np.zeros((224, 224), dtype=np.float32)
    for indices in index_groups:
        hull = cv2.convexHull(np.round(points[list(indices)]).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)

    if feather_px > 0:
        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=float(feather_px),
            sigmaY=float(feather_px),
        )

    max_value = float(mask.max())
    if max_value > 0:
        mask /= max_value
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def install_corrected_masks():
    """Install corrected masks into both consumers without editing production files."""
    landmark_module.region_mask = corrected_region_mask
    intervention_module.region_mask = corrected_region_mask


class BatchedDetector(BaseDetector):
    """The same frozen model with bounded, batched frame inference."""

    def __init__(self, checkpoint_path=None, batch_size=16):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if checkpoint_path is None:
            super().__init__()
        else:
            super().__init__(checkpoint_path=checkpoint_path)
        self.batch_size = int(batch_size)

    def _score_images(self, images: Iterable[np.ndarray]) -> np.ndarray:
        chunk = []
        device_probabilities = []

        def infer_chunk():
            batch = torch.stack(chunk).to(
                self.device,
                non_blocking=self.device.type == "cuda",
            )
            with torch.inference_mode():
                logits = self.model(batch).reshape(-1)
                probabilities = torch.sigmoid(logits)
            device_probabilities.append(probabilities)
            del batch

        for image in images:
            chunk.append(self.transformer(image=image)["image"])
            if len(chunk) == self.batch_size:
                infer_chunk()
                chunk.clear()

        if chunk:
            infer_chunk()

        if not device_probabilities:
            return np.array([], dtype=np.float32)

        # Keep scores on-device until inference for the complete request ends.
        return (
            torch.cat(device_probabilities)
            .to(device="cpu")
            .numpy()
            .astype(np.float32, copy=False)
        )

    def score_frames(self, faces: np.ndarray) -> np.ndarray:
        if not isinstance(faces, np.ndarray):
            raise TypeError("faces must be a numpy.ndarray")
        if faces.ndim != 4:
            raise ValueError("faces must have shape (N,H,W,C)")
        if len(faces) == 0:
            return np.array([], dtype=np.float32)
        return self._score_images(iter(faces))

    def score_frame_batches(self, face_batches: list[np.ndarray]) -> np.ndarray:
        if not face_batches:
            return np.array([], dtype=np.float32)

        lengths = []
        for faces in face_batches:
            if not isinstance(faces, np.ndarray):
                raise TypeError("each face batch must be a numpy.ndarray")
            if faces.ndim != 4:
                raise ValueError("each face batch must have shape (N,H,W,C)")
            if len(faces) == 0:
                raise ValueError("face batches cannot be empty")
            lengths.append(len(faces))

        scores = self._score_images(
            image
            for faces in face_batches
            for image in faces
        )
        video_scores = []
        offset = 0
        for length in lengths:
            video_scores.append(float(scores[offset:offset + length].mean()))
            offset += length
        return np.asarray(video_scores, dtype=np.float32)


def _to_original_interval(frame_index, interval):
    start, end = map(int, interval)
    if start < 0 or end >= len(frame_index) or start > end:
        raise ValueError("Invalid sampled interval")
    return (
        int(frame_index[start]["original_frame_index"]),
        int(frame_index[end]["original_frame_index"]),
    )


def _generate_real_candidates(video_id, frames, landmarks, frame_index, frame_scores, config, detector):
    """Use the production candidate algorithm with already-loaded/scored inputs."""
    proposed = propose_intervals(
        frame_scores,
        config["interval_length"],
        config["n_candidates"],
    )
    candidates = []
    for sampled_start, sampled_end, window_score in proposed:
        cue, overlaps = assign_cue(
            frames,
            landmarks,
            (sampled_start, sampled_end),
            detector,
        )
        candidates.append({
            "cue": cue,
            "interval": _to_original_interval(
                frame_index,
                (sampled_start, sampled_end),
            ),
            "window_score": float(window_score),
            "cue_overlap": float(overlaps[cue]),
            "cue_overlaps": {key: float(value) for key, value in overlaps.items()},
            "sampled_interval": (int(sampled_start), int(sampled_end)),
        })
    return candidates


def _stable_video_seed(seed, video_id, offset=0):
    return int(seed) + sum((i + 1) * ord(ch) for i, ch in enumerate(video_id)) + offset


def _run_condition(
    video_id,
    label,
    method,
    condition,
    frames,
    landmarks,
    frame_index,
    original_score,
    frame_scores,
    config,
    detector,
):
    started = time.perf_counter()
    if len(frames) == 0:
        return {
            "video_id": video_id,
            "label": label,
            "method": method,
            "condition": condition,
            "status": "no-candidates",
            "candidates": [],
            "timings": {"total_sec": time.perf_counter() - started},
        }

    if condition == "real":
        candidates = _generate_real_candidates(
            video_id, frames, landmarks, frame_index, frame_scores, config, detector
        )
    else:
        seed_offset = 100000 if condition == "authentic" else 0
        rng = random.Random(_stable_video_seed(config["seed"], video_id, seed_offset))
        candidates = generate_placebo_candidates(
            len(frames),
            int(config["n_candidates"]),
            rng,
            config=config,
            frame_index=frame_index,
        )

    if not candidates:
        return {
            "video_id": video_id,
            "label": label,
            "method": method,
            "condition": condition,
            "status": "no-candidates",
            "original_score": original_score,
            "candidates": [],
            "timings": {"total_sec": time.perf_counter() - started},
        }

    records = []
    n_controls = int(config["n_controls"])
    tau_percentile = float(config["tau_percentile"])
    seed = int(config["seed"])

    for candidate_index, candidate in enumerate(candidates):
        candidate_for_controls = dict(candidate)
        candidate_for_controls["interval"] = tuple(
            map(int, candidate["sampled_interval"])
        )
        controls = generate_controls(
            {"n_frames": len(frames)},
            candidate_for_controls,
            n_controls,
            random.Random(seed + candidate_index),
        )
        control_frames = [
            apply_control(frames, landmarks, control)
            for control in controls
        ]
        candidate_frames = apply_intervention(
            frames,
            landmarks,
            candidate["cue"],
            tuple(map(int, candidate["sampled_interval"])),
            strength=config["intervention"],
        )
        # A single bounded inference pass scores controls and candidate in a stable order.
        scores = detector.score_frame_batches(control_frames + [candidate_frames])
        control_effects = [
            float(original_score - float(score))
            for score in scores[:-1]
        ]
        control_records = [
            {
                "control": control,
                "original_score": float(original_score),
                "modified_score": float(score),
                "delta": float(original_score - float(score)),
            }
            for control, score in zip(controls, scores[:-1])
        ]
        modified_score = float(scores[-1])
        delta = float(original_score - modified_score)
        tau = compute_threshold(control_effects, percentile=tau_percentile)
        p_value = compute_p_value(delta, control_effects)
        records.append({
            "candidate": candidate,
            "original_score": float(original_score),
            "modified_score": modified_score,
            "candidate_effect": delta,
            "control_effects": control_effects,
            "tau_percentile": tau_percentile,
            "tau": float(tau),
            "p_value": float(p_value),
            "supported": bool(delta > tau),
            "control_records": control_records,
        })
        del control_frames, candidate_frames

    return {
        "video_id": video_id,
        "label": label,
        "method": method,
        "condition": condition,
        "status": "success",
        "original_score": float(original_score),
        "candidates": records,
        "timings": {"total_sec": time.perf_counter() - started},
    }


def _flatten_summary(record):
    rows = []
    for item in record.get("candidates", []):
        candidate = item["candidate"]
        t1, t2 = map(int, candidate["interval"])
        rows.append({
            "video_id": record["video_id"],
            "label": record["label"],
            "method": record["method"],
            "condition": record["condition"],
            "cue": candidate["cue"],
            "t1": t1,
            "t2": t2,
            "delta": item.get("candidate_effect"),
            "tau": item.get("tau"),
            "p": item.get("p_value"),
            "supported": item.get("supported"),
            "status": record["status"],
        })
    if not rows:
        rows.append({
            "video_id": record["video_id"],
            "label": record["label"],
            "method": record["method"],
            "condition": record["condition"],
            "cue": "",
            "t1": "",
            "t2": "",
            "delta": "",
            "tau": "",
            "p": "",
            "supported": "",
            "status": record["status"],
        })
    return rows


SUMMARY_FIELDS = (
    "video_id", "label", "method", "condition", "cue", "t1", "t2",
    "delta", "tau", "p", "supported", "status",
)


def _atomic_json(path, record):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _is_resumable(path, video_id, condition):
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        record.get("status") != "success"
        or record.get("video_id") != video_id
        or record.get("condition") != condition
        or not isinstance(record.get("candidates"), list)
    ):
        return False
    for candidate in record["candidates"]:
        if (
            len(candidate.get("control_effects", [])) != 20
            or float(candidate.get("tau_percentile", -1)) != 95.0
        ):
            return False
    return True


def _refresh_summary(output_dir, manifest, conditions):
    rows = []
    for _, row in manifest.iterrows():
        video_id = str(row["video_id"])
        for condition in conditions:
            path = output_dir / f"{video_id}__{condition}.json"
            if not path.exists():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("video_id") == video_id and record.get("condition") == condition:
                rows.extend(_flatten_summary(record))
    summary_path = output_dir / "batch_corrected_summary.csv"
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, summary_path)
    return summary_path, len(rows)


def _setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"batch_corrected_colab_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    return log_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/batch_corrected")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is unavailable")

    config = load_config(args.config)
    if (
        tuple(config["cues"]) != ("eye_motion", "mouth_motion", "face_texture")
        or int(config["n_controls"]) != 20
        or float(config["tau_percentile"]) != 95.0
    ):
        raise ValueError("Config does not match the frozen corrected experiment setup")

    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 60 or manifest["video_id"].nunique() != 60:
        raise ValueError("Expected exactly 60 unique videos in the existing manifest")
    if args.videos:
        requested = set(args.videos)
        unknown = requested - set(manifest["video_id"].astype(str))
        if unknown:
            raise ValueError(f"Videos not present in the manifest: {sorted(unknown)}")
        manifest = manifest[manifest["video_id"].astype(str).isin(requested)]

    conditions = tuple(condition for condition in CONDITIONS if condition in args.conditions)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_corrected_masks()
    detector = BatchedDetector(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
    )
    log_path = _setup_logging(output_dir)
    logging.info(
        "device=%s batch_size=%d videos=%d conditions=%s",
        detector.device,
        args.batch_size,
        len(manifest),
        conditions,
    )

    total_units = len(manifest) * len(conditions)
    progress = tqdm(total=total_units, desc="Corrected CAFE", unit="run")
    completed = 0
    failures = []
    started_all = time.perf_counter()

    for _, row in manifest.iterrows():
        video_id = str(row["video_id"])
        video_path = Path(str(row["path"]))
        label = str(row["label"])
        method = str(row["method"])
        pending = []
        for condition in conditions:
            out_path = output_dir / f"{video_id}__{condition}.json"
            if _is_resumable(out_path, video_id, condition):
                completed += 1
                progress.update(1)
                continue
            pending.append((condition, out_path))

        if not pending:
            continue

        try:
            cache = load_or_build_cache(video_id, video_path=video_path)
            frames = cache["faces"]
            frame_index = cache["index"]
            landmarks = load_or_build_landmark_cache(video_id)
            original_score = None
            frame_scores = None
            if len(frames):
                frame_scores = detector.score_frames(frames)
                original_score = float(frame_scores.mean())

            for condition, out_path in pending:
                try:
                    record = _run_condition(
                        video_id,
                        label,
                        method,
                        condition,
                        frames,
                        landmarks,
                        frame_index,
                        original_score,
                        frame_scores,
                        config,
                        detector,
                    )
                    _atomic_json(out_path, record)
                    logging.info(
                        "Completed %s | %s | status=%s candidates=%d elapsed=%.2f",
                        video_id,
                        condition,
                        record["status"],
                        len(record.get("candidates", [])),
                        record["timings"]["total_sec"],
                    )
                except Exception as exc:
                    logging.exception("Failed %s | %s", video_id, condition)
                    failure = {
                        "video_id": video_id,
                        "label": label,
                        "method": method,
                        "condition": condition,
                        "status": "error",
                        "error": repr(exc),
                    }
                    _atomic_json(out_path, failure)
                    failures.append(failure)
                finally:
                    completed += 1
                    _refresh_summary(output_dir, manifest, conditions)
                    elapsed = time.perf_counter() - started_all
                    rate = completed / elapsed if elapsed else 0.0
                    remaining = (total_units - completed) / rate if rate else 0.0
                    progress.update(1)
                    progress.set_postfix(
                        done=f"{completed}/{total_units}",
                        failed=len(failures),
                        eta=f"{remaining / 3600:.1f}h",
                    )
        except Exception as exc:
            logging.exception("Failed to load video %s", video_id)
            for condition, out_path in pending:
                failure = {
                    "video_id": video_id,
                    "label": label,
                    "method": method,
                    "condition": condition,
                    "status": "error",
                    "error": repr(exc),
                }
                _atomic_json(out_path, failure)
                failures.append(failure)
                completed += 1
                progress.update(1)
                _refresh_summary(output_dir, manifest, conditions)

    progress.close()
    summary_path, candidate_rows = _refresh_summary(output_dir, manifest, conditions)
    print(json.dumps({
        "videos_selected": len(manifest),
        "conditions_selected": list(conditions),
        "run_units_selected": total_units,
        "completed_or_resumed": completed,
        "candidate_rows_in_summary": candidate_rows,
        "failures": failures,
        "summary_csv": str(summary_path),
        "log": str(log_path),
        "device": str(detector.device),
    }, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

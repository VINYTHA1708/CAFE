import random


VALID_CUES = ("eye_motion", "mouth_motion", "face_texture")
CONTROL_TYPES = ("sham", "interval_shift", "cue_swap")


def _interval_length(interval):
    t1, t2 = map(int, interval)
    return t2 - t1 + 1


def _non_overlapping_intervals(n_frames, length, claimed, rng):
    claimed_start, claimed_end = map(int, claimed)

    candidates = []
    for start in range(0, n_frames - length + 1):
        end = start + length - 1

        if end < claimed_start or start > claimed_end:
            candidates.append((start, end))

    rng.shuffle(candidates)
    return candidates


def generate_controls(video, candidate, n_controls, rng):
    """
    Generate seeded control specifications for one candidate.

    Parameters
    ----------
    video : dict
        Video metadata. Must contain n_frames.
    candidate : dict
        Candidate specification containing cue, interval and strength.
    n_controls : int
        Number of controls to generate.
    rng : random.Random
        Seeded random-number generator.

    Returns
    -------
    list[dict]
        Control specifications.
    """
    if n_controls < 1:
        raise ValueError("n_controls must be positive")

    n_frames = int(video["n_frames"])
    candidate_cue = candidate["cue"]
    candidate_interval = tuple(map(int, candidate["interval"]))
    candidate_strength = dict(candidate["strength"])

    length = _interval_length(candidate_interval)

    if candidate_cue not in VALID_CUES:
        raise ValueError(f"Unknown candidate cue: {candidate_cue}")

    if candidate_interval[0] < 0 or candidate_interval[1] >= n_frames:
        raise ValueError("Candidate interval is outside video")

    if length > n_frames:
        raise ValueError("Candidate interval is longer than video")

    controls = []

    shifted = _non_overlapping_intervals(
        n_frames,
        length,
        candidate_interval,
        rng
    )

    # Use interval-shift controls whenever possible.
    for interval in shifted:
        controls.append({
            "type": "interval_shift",
            "cue": candidate_cue,
            "interval": interval,
            "strength": dict(candidate_strength),
        })

    # Add cue swaps using the exact candidate interval.
    swap_cues = [cue for cue in VALID_CUES if cue != candidate_cue]
    while len(controls) < n_controls:
        cue = rng.choice(swap_cues)

        controls.append({
            "type": "cue_swap",
            "cue": cue,
            "interval": candidate_interval,
            "strength": dict(candidate_strength),
        })

    # Replace a few controls with sham runs when possible.
    sham_count = min(2, n_controls)
    for i in range(sham_count):
        controls[i] = {
            "type": "sham",
            "cue": candidate_cue,
            "interval": candidate_interval,
            "strength": dict(candidate_strength),
        }

    return controls[:n_controls]


def apply_control(frames, landmarks, spec):
    """
    Apply a generated control through the intervention code path.
    """
    from cafe.interventions import apply_intervention

    return apply_intervention(
        frames,
        landmarks,
        spec["cue"],
        spec["interval"],
        strength=spec["strength"],
        sham=(spec["type"] == "sham"),
    )

# Phase 17: Metrics, baselines, tables and figures
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = Path("results")
RUNS = RESULTS / "runs"
TABLES = RESULTS / "tables"
FIGURES = Path("docs/figures")

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def load_runs():
    records = []
    for path in sorted(RUNS.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def flatten_candidates(records):
    rows = []

    for record in records:
        for item in record.get("candidates", []):
            candidate = item.get("candidate", {})

            rows.append({
                "video_id": record["video_id"],
                "label": record["label"],
                "method": record["method"],
                "condition": record["condition"],
                "status": record.get("status", ""),
                "cue": candidate.get("cue", ""),
                "t1": candidate.get("interval", [None, None])[0],
                "t2": candidate.get("interval", [None, None])[1],
                "delta": item.get("candidate_effect"),
                "tau": item.get("tau"),
                "p": item.get("p_value"),
                "supported": bool(item.get("supported", False)),
                "control_effects": item.get("control_effects", []),
            })

    return pd.DataFrame(rows)


def video_level_support(df, condition):
    sub = df[df["condition"] == condition].copy()

    if sub.empty:
        return pd.DataFrame()

    result = (
        sub.groupby(["video_id", "label"], as_index=False)
        .agg(
            candidates=("video_id", "size"),
            supported=("supported", "sum"),
        )
    )

    result["verified_explanation"] = result["supported"] > 0
    return result


def rate_table(df):
    rows = []

    for condition in sorted(df["condition"].unique()):
        for label in ["fake", "real"]:
            sub = df[
                (df["condition"] == condition) &
                (df["label"] == label)
            ]

            if sub.empty:
                continue

            v = (
                sub.groupby("video_id", as_index=False)
                .agg(
                    candidates=("video_id", "size"),
                    supported=("supported", "sum"),
                )
            )

            v["verified_explanation"] = v["supported"] > 0

            total = len(v)
            supported = int(v["verified_explanation"].sum())

            rows.append({
                "condition": condition,
                "label": label,
                "videos": total,
                "videos_with_supported_explanation": supported,
                "support_rate": supported / total if total else 0,
                "abstention_rate": 1 - supported / total if total else 0,
            })

    return pd.DataFrame(rows)

def candidate_rate_table(df):
    rows = []

    for condition in sorted(df["condition"].unique()):
        for label in ["fake", "real"]:
            sub = df[
                (df["condition"] == condition) &
                (df["label"] == label)
            ]

            if sub.empty:
                continue

            rows.append({
                "condition": condition,
                "label": label,
                "candidate_rows": len(sub),
                "supported_candidates": int(sub["supported"].sum()),
                "candidate_support_rate": (
                    float(sub["supported"].mean())
                    if len(sub) else 0
                ),
            })

    return pd.DataFrame(rows)

def delta_by_cue(df):
    sub = df[df["delta"].notna()].copy()

    return (
        sub.groupby(["condition", "cue"])["delta"]
        .agg(
            count="count",
            median="median",
            mean="mean",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )


def delta_by_method(df):
    sub = df[df["delta"].notna()].copy()

    return (
        sub.groupby(["condition", "method"])["delta"]
        .agg(
            count="count",
            median="median",
            mean="mean",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )


def pooled_control_effects(records):
    values = []

    for record in records:
        for item in record.get("candidates", []):
            values.extend(item.get("control_effects", []))

    return np.asarray(values, dtype=float)


def control_summary(records):
    values = pooled_control_effects(records)

    if len(values) == 0:
        return pd.DataFrame()

    return pd.DataFrame([{
        "n_control_effects": len(values),
        "median": np.median(values),
        "mean": np.mean(values),
        "std": np.std(values),
        "minimum": np.min(values),
        "q25": np.percentile(values, 25),
        "q75": np.percentile(values, 75),
        "q95": np.percentile(values, 95),
        "maximum": np.max(values),
    }])


def tau_sensitivity(df):
    rows = []

    for percentile in [90, 95, 99]:
        for condition in ["real", "placebo"]:
            for label in ["fake", "real"]:
                sub = df[
                    (df["condition"] == condition) &
                    (df["label"] == label)
                ].copy()

                if sub.empty:
                    continue

                supported_candidates = 0
                valid_candidates = 0
                supported_videos = set()

                for _, row in sub.iterrows():
                    controls = np.asarray(
                        row["control_effects"],
                        dtype=float,
                    )

                    if len(controls) == 0 or pd.isna(row["delta"]):
                        continue

                    tau = np.percentile(controls, percentile)
                    valid_candidates += 1

                    if row["delta"] > tau:
                        supported_candidates += 1
                        supported_videos.add(row["video_id"])

                total_videos = sub["video_id"].nunique()

                rows.append({
                    "tau_percentile": percentile,
                    "condition": condition,
                    "label": label,
                    "videos": total_videos,
                    "videos_with_supported_explanation": len(
                        supported_videos
                    ),
                    "video_support_rate": (
                        len(supported_videos) / total_videos
                        if total_videos else 0
                    ),
                    "candidate_rows": valid_candidates,
                    "supported_candidates": supported_candidates,
                    "candidate_support_rate": (
                        supported_candidates / valid_candidates
                        if valid_candidates else 0
                    ),
                })

    return pd.DataFrame(rows)

def authentic_effects(df):
    sub = df[
        (df["condition"] == "authentic") &
        df["delta"].notna()
    ]

    if sub.empty:
        return pd.DataFrame()

    return pd.DataFrame([{
        "condition": "authentic",
        "count": len(sub),
        "median_delta": sub["delta"].median(),
        "mean_delta": sub["delta"].mean(),
        "std_delta": sub["delta"].std(),
        "minimum_delta": sub["delta"].min(),
        "maximum_delta": sub["delta"].max(),
    }])


def runtime_summary(records):
    rows = []

    for record in records:
        runtime = record.get("timings", {}).get("total_sec")

        if runtime is not None:
            rows.append({
                "video_id": record["video_id"],
                "condition": record["condition"],
                "runtime_sec": float(runtime),
            })

    runtime_df = pd.DataFrame(rows)

    if runtime_df.empty:
        return pd.DataFrame()

    return (
        runtime_df.groupby("condition")["runtime_sec"]
        .agg(
            videos="count",
            mean_sec="mean",
            median_sec="median",
            min_sec="min",
            max_sec="max",
        )
        .reset_index()
    )


def no_control_baseline(df, threshold=0.05):
    rows = []

    for condition in ["real", "placebo"]:
        for label in ["fake", "real"]:
            sub = df[
                (df["condition"] == condition) &
                (df["label"] == label)
            ].copy()

            if sub.empty:
                continue

            supported = sub["delta"] > threshold

            video_support = (
                pd.DataFrame({
                    "video_id": sub["video_id"],
                    "supported": supported,
                })
                .groupby("video_id")["supported"]
                .any()
            )

            rows.append({
                "baseline": "no_control",
                "condition": condition,
                "label": label,
                "delta_threshold": threshold,
                "videos": len(video_support),
                "videos_supported": int(video_support.sum()),
                "video_support_rate": float(video_support.mean()),
                "candidate_rows": len(sub),
                "supported_candidates": int(supported.sum()),
                "candidate_support_rate": (
                    float(supported.mean())
                    if len(sub) else 0
                ),
            })

    return pd.DataFrame(rows)

def sham_only_baseline(records, df, percentile=95):
    rows = []

    for condition in ["real", "placebo"]:
        for label in ["fake", "real"]:
            sub = df[
                (df["condition"] == condition) &
                (df["label"] == label)
            ].copy()

            if sub.empty:
                continue

            supported_candidates = 0
            valid_candidates = 0
            supported_videos = set()

            for _, row in sub.iterrows():
                record = next(
                    (
                        r for r in records
                        if r["video_id"] == row["video_id"]
                        and r["condition"] == condition
                    ),
                    None,
                )

                if record is None:
                    continue

                candidate_index = None
                for idx, item in enumerate(record.get("candidates", [])):
                    candidate = item.get("candidate", {})
                    if (
                        candidate.get("cue") == row["cue"] and
                        candidate.get("interval", [None, None])[0] == row["t1"] and
                        candidate.get("interval", [None, None])[1] == row["t2"]
                    ):
                        candidate_index = idx
                        break

                if candidate_index is None:
                    continue

                item = record["candidates"][candidate_index]
                sham_effects = [
                    control.get("delta", 0.0)
                    for control in item.get("control_records", [])
                    if control.get("control", {}).get("type") == "sham"
                ]

                if not sham_effects or pd.isna(row["delta"]):
                    continue

                tau = np.percentile(
                    np.asarray(sham_effects, dtype=float),
                    percentile,
                )

                valid_candidates += 1

                if row["delta"] > tau:
                    supported_candidates += 1
                    supported_videos.add(row["video_id"])

            total_videos = sub["video_id"].nunique()

            rows.append({
                "baseline": "sham_only",
                "condition": condition,
                "label": label,
                "tau_percentile": percentile,
                "videos": total_videos,
                "videos_supported": len(supported_videos),
                "video_support_rate": (
                    len(supported_videos) / total_videos
                    if total_videos else 0
                ),
                "candidate_rows": valid_candidates,
                "supported_candidates": supported_candidates,
                "candidate_support_rate": (
                    supported_candidates / valid_candidates
                    if valid_candidates else 0
                ),
            })

    return pd.DataFrame(rows)

def save_table(df, name):
    csv_path = TABLES / f"{name}.csv"
    md_path = TABLES / f"{name}.md"

    df.to_csv(csv_path, index=False)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print(f"TABLE: {csv_path}")


def make_figures(df, records):
    # 1. Candidate vs control effect distribution
    candidate_effects = df[df["delta"].notna()]["delta"].to_numpy()
    controls = pooled_control_effects(records)

    plt.figure(figsize=(8, 5))
    plt.hist(candidate_effects, bins=40, density=True, alpha=0.6, label="Candidate delta")
    plt.hist(controls, bins=40, density=True, alpha=0.6, label="Control delta")
    plt.xlabel("Detector score effect (delta)")
    plt.ylabel("Density")
    plt.title("Candidate effects vs. control effects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES / "effect_vs_control_distribution.png",
        dpi=200,
    )
    plt.close()

    # 2. Placebo vs real support
    rates = rate_table(df)
    selected = rates[(rates["condition"].isin(["real", "placebo"])) & (rates["label"] == "fake")]

    plt.figure(figsize=(7, 5))
    plt.bar(
        selected["condition"].replace({"real": "Normal CAFE", "placebo": "Placebo"}),
        selected["support_rate"] * 100,
    )
    plt.xlabel("Condition")
    plt.ylabel("Video support rate (%)")
    plt.title("CAFE support rate: normal condition vs. placebo")
    plt.tight_layout()
    plt.savefig(
        FIGURES / "placebo_vs_real_support.png",
        dpi=200,
    )
    plt.close()

    # 3. Tau sensitivity
    sensitivity = tau_sensitivity(df)

    plt.figure(figsize=(8, 5))

    series = [
        ("real", "fake", "real / fake"),
        ("real", "real", "real / genuine"),
        ("placebo", "fake", "placebo / fake"),
        ("placebo", "real", "placebo / genuine"),
    ]

    for condition, label, name in series:
        sub = sensitivity[
            (sensitivity["condition"] == condition) &
            (sensitivity["label"] == label)
        ].sort_values("tau_percentile")

        if not sub.empty:
            plt.plot(
                sub["tau_percentile"],
                sub["video_support_rate"] * 100,
                marker="o",
                label=name,
            )

    plt.xlabel("Control threshold percentile")
    plt.ylabel("Video support rate (%)")
    plt.title("Sensitivity to control threshold percentile")
    plt.xticks([90, 95, 99])
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURES / "tau_sensitivity.png",
        dpi=200,
    )
    plt.close()
    # 4. Effect by cue
    sub = df[df["delta"].notna()].copy()
    cues = [c for c in ["eye_motion", "mouth_motion", "face_texture"]
            if c in sub["cue"].unique()]

    data = [
        sub.loc[sub["cue"] == cue, "delta"].to_numpy()
        for cue in cues
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=cues)
    plt.xlabel("Cue")
    plt.ylabel("Detector score effect (delta)")
    plt.title("Detector effect by candidate cue")
    plt.tight_layout()
    plt.savefig(
        FIGURES / "effect_by_cue.png",
        dpi=200,
    )
    plt.close()

    # 5. Support by manipulation method
    fake_real = df[
        (df["condition"] == "real") &
        (df["label"] == "fake")
    ].copy()

    method_rates = (
        fake_real.groupby("method")["supported"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        method_rates.index,
        method_rates.to_numpy() * 100,
    )
    plt.xlabel("FF++ manipulation method")
    plt.ylabel("Candidate support rate (%)")
    plt.title("Verified candidate support by manipulation method")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        FIGURES / "support_by_method.png",
        dpi=200,
    )
    plt.close()


def main():
    records = load_runs()

    if len(records) != 180:
        raise RuntimeError(
            f"Expected 180 run JSON files, found {len(records)}"
        )

    df = flatten_candidates(records)

    print(f"Loaded run files: {len(records)}")
    print(f"Candidate rows: {len(df)}")
    print(f"Videos: {df['video_id'].nunique()}")

    save_table(rate_table(df), "support_abstention")
    save_table(candidate_rate_table(df), "candidate_support")
    save_table(delta_by_cue(df), "delta_by_cue")
    save_table(delta_by_method(df), "delta_by_method")
    save_table(control_summary(records), "control_effect_distribution")
    save_table(tau_sensitivity(df), "tau_sensitivity")
    save_table(authentic_effects(df), "authentic_effects")
    save_table(runtime_summary(records), "runtime")
    save_table(no_control_baseline(df), "baseline_no_control")
    save_table(sham_only_baseline(records, df), "baseline_sham_only")

    make_figures(df, records)

    print()
    print("=== PHASE 17 COMPLETE ===")
    print("Tables:", TABLES)
    print("Figures:", FIGURES)


if __name__ == "__main__":
    main()









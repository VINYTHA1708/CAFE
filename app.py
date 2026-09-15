"""
CAFE Demo — Streamlit frontend
Run: streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# ── project root on sys.path; pipeline uses relative paths ──────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAFE — Deepfake Explanation Verifier",
    page_icon="",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── layout ── */
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1100px; }
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: #334155; }
section[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }

/* ── page header ── */
.page-header {
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-size: 1.6rem !important;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 0.2rem 0;
}
.page-header p {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
}

/* ── verdict banners ── */
.verdict-block {
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.verdict-deepfake {
    background: #fff1f2;
    border: 1.5px solid #fca5a5;
}
.verdict-real {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
}
.verdict-label {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin: 0 0 0.3rem 0;
}
.verdict-deepfake .verdict-label { color: #b91c1c; }
.verdict-real     .verdict-label { color: #15803d; }
.verdict-score {
    font-size: 0.95rem;
    color: #475569;
    margin: 0;
}

/* ── explanation status banner ── */
.expl-block {
    border-radius: 8px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.75rem;
}
.expl-verified  { background: #f0fdf4; border: 1.5px solid #86efac; }
.expl-abstained { background: #fffbeb; border: 1.5px solid #fcd34d; }
.expl-rejected  { background: #fff7ed; border: 1.5px solid #fdba74; }
.expl-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
}
.expl-verified  .expl-title { color: #15803d; }
.expl-abstained .expl-title { color: #92400e; }
.expl-rejected  .expl-title { color: #9a3412; }
.expl-desc {
    font-size: 0.88rem;
    color: #475569;
    margin: 0;
    line-height: 1.5;
}

/* ── metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.75rem; }
.metric-card {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
}
.metric-card .mc-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}
.metric-card .mc-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1;
}

/* ── section headings ── */
.section-heading {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin: 2rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e2e8f0;
}

/* ── candidate row ── */
.cand-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.cand-cue {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1e293b;
}
.cand-frames {
    font-size: 0.82rem;
    color: #64748b;
}
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
}
.badge-verified  { background: #dcfce7; color: #15803d; }
.badge-abstained { background: #fef9c3; color: #854d0e; }
.badge-rejected  { background: #ffedd5; color: #9a3412; }

/* ── tech table ── */
.tech-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
.tech-table th {
    text-align: left;
    font-weight: 600;
    color: #64748b;
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid #e2e8f0;
    background: #f8fafc;
}
.tech-table td {
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid #f1f5f9;
    color: #1e293b;
    font-family: monospace;
}
.tech-table tr:last-child td { border-bottom: none; }
.pass { color: #15803d; font-weight: 700; }
.fail { color: #b91c1c; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── constants ────────────────────────────────────────────────────────────────
CUE_LABELS = {
    "eye_motion":   "Eye Motion",
    "mouth_motion": "Mouth Motion",
    "face_texture": "Face Texture",
}

EXPL_COPY = {
    "verified": (
        "VERIFIED EXPLANATION",
        "The detector's decision is causally linked to a specific facial region and "
        "time window. Removing that cue measurably reduced the deepfake score beyond "
        "what random interventions produce.",
    ),
    "abstained": (
        "NO EXPLANATION VERIFIED",
        "No candidate explanation produced a score change large enough to exceed the "
        "control threshold. The detector flagged this video, but CAFE cannot confirm "
        "which facial cue is responsible.",
    ),
    "control_rejected": (
        "CONTROL-REJECTED",
        "All candidate interventions reduced the deepfake score by less than or equal "
        "to random control interventions. The explanation candidates are not supported "
        "by the detector's behaviour.",
    ),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _overall_status(candidates: list) -> str:
    if any(c.get("supported") for c in candidates):
        return "verified"
    if all((c.get("candidate_effect") or 0) <= 0 for c in candidates):
        return "control_rejected"
    return "abstained"


def _badge(supported: bool, delta: float) -> str:
    if supported:
        return '<span class="badge badge-verified">VERIFIED</span>'
    if delta <= 0:
        return '<span class="badge badge-rejected">CONTROL-REJECTED</span>'
    return '<span class="badge badge-abstained">ABSTAINED</span>'


def _load_faces(video_id: str) -> np.ndarray | None:
    p = ROOT / "cache" / video_id / "faces.npy"
    return np.load(str(p)) if p.exists() else None


def _metric_card(label: str, value: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-value">{value}</div>'
        f'</div>'
    )


# ── main render ──────────────────────────────────────────────────────────────

def render_result(result: dict) -> None:
    # normalise field names across pipeline / batch JSON schemas
    label = (result.get("label") or result.get("prediction", {}).get("label", "?")).upper()
    score = float(
        result.get("score")
        or result.get("original_score")
        or result.get("prediction", {}).get("score", 0.0)
        or 0.0
    )
    video_id  = result.get("video_id", "unknown")
    candidates = result.get("candidates", [])
    timing     = (result.get("timings") or {}).get("total_sec")
    n_supported = sum(1 for c in candidates if c.get("supported"))
    status      = _overall_status(candidates) if candidates else "abstained"

    # ── 1. Verdict ────────────────────────────────────────────────────────
    is_fake = (label == "DEEPFAKE")
    v_cls   = "verdict-deepfake" if is_fake else "verdict-real"
    v_text  = "DEEPFAKE DETECTED" if is_fake else "REAL VIDEO"
    st.markdown(
        f'<div class="verdict-block {v_cls}">'
        f'<p class="verdict-label">{v_text}</p>'
        f'<p class="verdict-score">Detector confidence score: <strong>{score:.4f}</strong>'
        f' &nbsp;({score:.1%})</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 2. Explanation status ─────────────────────────────────────────────
    expl_cls_map = {
        "verified":        "expl-verified",
        "abstained":       "expl-abstained",
        "control_rejected":"expl-rejected",
    }
    expl_title, expl_desc = EXPL_COPY[status]
    st.markdown(
        f'<div class="expl-block {expl_cls_map[status]}">'
        f'<p class="expl-title">{expl_title}</p>'
        f'<p class="expl-desc">{expl_desc}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── 3. Metric cards ───────────────────────────────────────────────────
    runtime_str = f"{timing:.1f}s" if timing else "—"
    st.markdown(
        '<div class="metric-row">'
        + _metric_card("Detection Score",       f"{score:.4f}")
        + _metric_card("Candidates Tested",     str(len(candidates)))
        + _metric_card("Verified Explanations", str(n_supported))
        + _metric_card("Pipeline Runtime",      runtime_str)
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── 4. Face crops ─────────────────────────────────────────────────────
    faces = _load_faces(video_id)
    if faces is not None and len(faces) > 0:
        st.markdown('<p class="section-heading">Sampled Face Crops</p>', unsafe_allow_html=True)
        n_show = min(8, len(faces))
        cols   = st.columns(n_show)
        for i, col in enumerate(cols):
            idx = i * (len(faces) // n_show)
            col.image(faces[idx], use_container_width=True, caption=f"Frame {idx}")

    # ── 5. Candidate explanations ─────────────────────────────────────────
    if candidates:
        st.markdown('<p class="section-heading">Candidate Explanations</p>', unsafe_allow_html=True)

        for i, rec in enumerate(candidates, start=1):
            cand      = rec.get("candidate", {})
            cue       = cand.get("cue", "?")
            interval  = cand.get("interval", [0, 0])
            sampled   = cand.get("sampled_interval", interval)
            delta     = float(rec.get("candidate_effect") or 0)
            tau       = float(rec.get("tau") or 0)
            p_val     = float(rec.get("p_value") or 1)
            supported = bool(rec.get("supported", False))
            cue_label = CUE_LABELS.get(cue, cue)
            badge_html = _badge(supported, delta)

            expander_label = (
                f"Candidate {i}  —  {cue_label}  |  "
                f"Frames {interval[0]}–{interval[1]}  |  "
                + ("VERIFIED" if supported else ("CONTROL-REJECTED" if delta <= 0 else "ABSTAINED"))
            )

            with st.expander(expander_label, expanded=(i == 1 and supported)):
                # status line
                st.markdown(
                    f'<div class="cand-header">'
                    f'<span class="cand-cue">{cue_label}</span>'
                    f'<span class="cand-frames">Frames {interval[0]}–{interval[1]}'
                    f' &nbsp;(sampled {sampled[0]}–{sampled[1]})</span>'
                    f'{badge_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # two-column layout: table left, chart right
                left, right = st.columns([1, 1], gap="large")

                with left:
                    st.markdown("**Technical Verification Details**")
                    pass_fail_delta = "pass" if supported else "fail"
                    st.markdown(
                        '<table class="tech-table">'
                        "<thead><tr><th>Parameter</th><th>Value</th></tr></thead>"
                        "<tbody>"
                        f"<tr><td>Counterfactual effect (Δ)</td>"
                        f'<td class="{pass_fail_delta}">{delta:+.6f}</td></tr>'
                        f"<tr><td>Control threshold (τ, 95th pct)</td>"
                        f"<td>{tau:.6f}</td></tr>"
                        f"<tr><td>p-value</td>"
                        f"<td>{p_val:.4f}</td></tr>"
                        f"<tr><td>Δ &gt; τ (supported)</td>"
                        f'<td class="{pass_fail_delta}">{"Yes" if supported else "No"}</td></tr>'
                        f"<tr><td>Original score</td>"
                        f"<td>{float(rec.get('original_score') or 0):.6f}</td></tr>"
                        f"<tr><td>Modified score</td>"
                        f"<td>{float(rec.get('modified_score') or 0):.6f}</td></tr>"
                        "</tbody></table>",
                        unsafe_allow_html=True,
                    )

                with right:
                    ctrl_effects = rec.get("control_effects", [])
                    if ctrl_effects:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(4.5, 2.6))
                        fig.patch.set_facecolor("#f8fafc")
                        ax.set_facecolor("#f8fafc")

                        ax.hist(ctrl_effects, bins=12, color="#cbd5e1",
                                edgecolor="#94a3b8", linewidth=0.5, alpha=1.0,
                                label="Control effects")
                        ax.axvline(tau,   color="#f59e0b", linewidth=1.5,
                                   linestyle="--", label=f"τ = {tau:.4f}")
                        ax.axvline(delta, color="#16a34a" if supported else "#dc2626",
                                   linewidth=2.0, label=f"Δ = {delta:.4f}")

                        ax.set_xlabel("Score change (Δ)", fontsize=8, color="#475569")
                        ax.set_ylabel("Count",            fontsize=8, color="#475569")
                        ax.tick_params(labelsize=7, colors="#64748b")
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#e2e8f0")
                        ax.legend(fontsize=7, framealpha=0.9,
                                  edgecolor="#e2e8f0", facecolor="white")
                        ax.set_title("Control distribution vs candidate effect",
                                     fontsize=8, color="#475569", pad=6)
                        fig.tight_layout(pad=1.2)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

    # ── 6. Rendered explanation text ──────────────────────────────────────
    rendered = result.get("rendered_explanation")
    if rendered:
        st.markdown('<p class="section-heading">Full Explanation Text</p>',
                    unsafe_allow_html=True)
        st.code(rendered, language=None)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## CAFE")
    st.markdown(
        "<small>Counterfactually Verified Explanations<br>for Deepfake Video Detection</small>",
        unsafe_allow_html=True,
    )
    st.divider()

    mode = st.radio(
        "Mode",
        ["Upload & Run Pipeline", "Browse Existing Results"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "<small>"
        "<strong>Detector</strong><br>EfficientNet-B4 (FF++ C23)<br><br>"
        "<strong>Explanation cues</strong><br>"
        "Eye motion &nbsp;·&nbsp; Mouth motion &nbsp;·&nbsp; Face texture<br><br>"
        "<strong>Verification threshold</strong><br>"
        "95th-percentile of control distribution"
        "</small>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="page-header">'
    "<h1>CAFE — Deepfake Explanation Verifier</h1>"
    "<p>Counterfactually verified explanations for deepfake video detection &nbsp;·&nbsp; "
    "EfficientNet-B4 detector &nbsp;·&nbsp; FaceForensics++ C23</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── MODE 1: Upload & Run ──────────────────────────────────────────────────────
if mode == "Upload & Run Pipeline":

    uploaded = st.file_uploader(
        "Upload a video file to analyse",
        type=["mp4", "avi", "mov"],
        help="The video must contain a detectable face. Processing takes ~30–90 seconds.",
    )

    if uploaded:
        vid_col, _ = st.columns([1, 1])
        with vid_col:
            st.video(uploaded)

        st.markdown("")
        run_btn = st.button("Run CAFE Pipeline", type="primary", use_container_width=False)

        if run_btn:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            progress = st.progress(0, text="Initialising pipeline…")
            try:
                from cafe.pipeline import run_cafe

                progress.progress(15, text="Preprocessing frames and detecting faces…")
                result_obj = run_cafe(tmp_path, output_dir=ROOT / "results" / "runs")
                progress.progress(100, text="Analysis complete.")

                st.success("Pipeline finished successfully.")
                st.divider()
                render_result(result_obj.to_dict())

            except Exception as exc:
                progress.empty()
                st.error(f"Pipeline error: {exc}")
                st.exception(exc)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    else:
        st.markdown(
            '<div style="background:#f8fafc;border:1px dashed #cbd5e1;border-radius:8px;'
            'padding:2rem;text-align:center;color:#94a3b8;font-size:0.9rem;">'
            "Upload a video file above to begin analysis."
            "</div>",
            unsafe_allow_html=True,
        )

# ── MODE 2: Browse Existing Results ──────────────────────────────────────────
else:
    runs_dir   = ROOT / "results" / "runs"
    json_files = sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []

    if not json_files:
        st.warning("No result files found in results/runs/.")
    else:
        labels = [f.stem for f in json_files]

        filter_col, select_col = st.columns([1, 3])
        with filter_col:
            condition_filter = st.selectbox(
                "Condition", ["all", "authentic", "placebo", "real"]
            )
        with select_col:
            filtered = (
                labels if condition_filter == "all"
                else [l for l in labels if l.endswith(f"__{condition_filter}")]
            )
            if not filtered:
                st.warning("No results match the selected condition.")
                st.stop()
            chosen = st.selectbox("Result", filtered)

        st.divider()

        result_path = runs_dir / f"{chosen}.json"
        data = json.loads(result_path.read_text(encoding="utf-8"))

        # normalise schema differences between batch JSON and pipeline JSON
        if "original_score" in data and "score" not in data:
            data["score"] = data["original_score"]
        if "label" not in data:
            data["label"] = "DEEPFAKE" if (data.get("score") or 0) >= 0.5 else "REAL"

        render_result(data)

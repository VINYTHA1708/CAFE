CUE_HUMAN_NAMES = {
    "eye_motion": "eye motion",
    "mouth_motion": "mouth motion",
    "face_texture": "face texture",
}

VERIFIED_TEMPLATE = (
    "Prediction: {label} (score {s:.3f})\n"
    "Verified explanation: the detector's output depends on {cue_human_name}\n"
    "during frames {t1}-{t2} ({start:.2f}s-{end:.2f}s).\n"
    "Effect {delta:.3f} exceeded the control threshold {tau:.3f} (p = {p:.3f})."
)

ABSTENTION_TEMPLATE = (
    "Prediction: {label} (score {s:.3f})\n"
    "No explanation could be verified. {n} candidate explanations were tested;\n"
    "none produced an effect exceeding the control threshold {tau:.3f}.\n"
    "Tested: {candidate_list}"
)

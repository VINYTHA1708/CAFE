import numpy as np
import matplotlib.pyplot as plt

from cafe.landmarks import load_or_build_landmark_cache
from cafe.interventions import apply_intervention

VIDEO_ID = "Deepfakes_000_003"
frames = np.load(f"cache/{VIDEO_ID}/faces.npy")
landmarks = load_or_build_landmark_cache(VIDEO_ID)

interval = (8, 23)
positions = [8, 12, 16, 20, 23]
cues = ["eye_motion", "mouth_motion", "face_texture"]

fig, axes = plt.subplots(
    len(cues),
    len(positions) * 2,
    figsize=(16, 7)
)

for row, cue in enumerate(cues):
    modified = apply_intervention(
        frames,
        landmarks,
        cue,
        interval
    )

    for col, position in enumerate(positions):
        axes[row, col * 2].imshow(frames[position])
        axes[row, col * 2].set_title(f"Before {position}")

        axes[row, col * 2 + 1].imshow(modified[position])
        axes[row, col * 2 + 1].set_title(f"After {position}")

        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    axes[row, 0].set_ylabel(
        cue,
        rotation=90,
        fontsize=11
    )

fig.suptitle(
    "CAFE Counterfactual Intervention Examples",
    fontsize=14
)

fig.tight_layout()

output = "docs/figures/intervention_examples.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Saved:", output)
print("Interval:", interval)
print("Positions:", positions)

import numpy as np


def compute_counterfactual_effect(original_logits, counterfactual_logits):
    """
    Compute the detector's counterfactual effect in logit space.

    A positive effect means the counterfactual intervention reduced
    the detector's fake score evidence.

    Args:
        original_logits: detector logits for the original frames.
        counterfactual_logits: detector logits after intervention.

    Returns:
        Mean logit effect as a float.
    """
    original = np.asarray(original_logits, dtype=np.float32)
    counterfactual = np.asarray(counterfactual_logits, dtype=np.float32)

    if original.size == 0 or counterfactual.size == 0:
        raise ValueError("Logit arrays cannot be empty.")

    if original.shape != counterfactual.shape:
        raise ValueError(
            "Original and counterfactual logits must have the same shape."
        )

    return float(original.mean() - counterfactual.mean())


def compute_framewise_effect(original_logits, counterfactual_logits):
    """
    Compute frame-wise detector logit effects.

    Returns:
        NumPy array where each value is:
        original_logit - counterfactual_logit
    """
    original = np.asarray(original_logits, dtype=np.float32)
    counterfactual = np.asarray(counterfactual_logits, dtype=np.float32)

    if original.size == 0 or counterfactual.size == 0:
        raise ValueError("Logit arrays cannot be empty.")

    if original.shape != counterfactual.shape:
        raise ValueError(
            "Original and counterfactual logits must have the same shape."
        )

    return original - counterfactual
def compute_threshold(control_effects, margin=0.05):
    """
    Compute the verification threshold from control intervention effects.
    """
    effects = np.asarray(control_effects, dtype=np.float32)

    if effects.size == 0:
        raise ValueError("control_effects cannot be empty")

    if margin < 0:
        raise ValueError("margin must be non-negative")

    return float(effects.mean() + margin)

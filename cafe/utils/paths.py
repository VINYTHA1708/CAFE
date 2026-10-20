from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_dataset_path():
    return PROJECT_ROOT / "dataset"


def get_cache_path(cache_root=None):
    path = (
        Path(cache_root)
        if cache_root is not None
        else PROJECT_ROOT / "cache_rgb_corrected"
    )
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_results_path():
    return PROJECT_ROOT / "results"


def get_models_path():
    return PROJECT_ROOT / "models"

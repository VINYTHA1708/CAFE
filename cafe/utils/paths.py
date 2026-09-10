from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_dataset_path():
    return PROJECT_ROOT / "dataset"


def get_cache_path():
    return PROJECT_ROOT / "cache"


def get_results_path():
    return PROJECT_ROOT / "results"


def get_models_path():
    return PROJECT_ROOT / "models"

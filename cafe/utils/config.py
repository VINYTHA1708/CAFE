from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path=None):
    path = Path(config_path) if config_path else CONFIG_PATH

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

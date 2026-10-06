"""Load assumptions.yaml. Code uses only each parameter's `value`."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS = ROOT / "assumptions.yaml"


def load_assumptions(path: Path = ASSUMPTIONS) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def values(section: str, path: Path = ASSUMPTIONS) -> dict:
    """{param: value} for one section, e.g. values("portfolio")["pd_base"]."""
    return {k: v["value"] for k, v in load_assumptions(path)[section].items()}

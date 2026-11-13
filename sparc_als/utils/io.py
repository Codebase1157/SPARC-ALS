"""I/O helpers: YAML config loading with !include support and
deterministic per-fold path naming."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class _IncludeLoader(yaml.SafeLoader):
    pass


def _include(loader: _IncludeLoader, node: yaml.Node) -> Any:
    path = Path(loader.name).parent / loader.construct_scalar(node)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, _IncludeLoader)


_IncludeLoader.add_constructor("!include", _include)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, _IncludeLoader)


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prediction_dir(corpus: str, experiment: str, seed: int, root: str | Path = "predictions") -> Path:
    """Return a fully qualified predictions directory tagged by seed."""
    d = Path(root) / corpus / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def speaker_logits_path(corpus: str, experiment: str, seed: int, root: str | Path = "predictions") -> Path:
    return prediction_dir(corpus, experiment, seed, root) / f"speaker_logits_seed{seed}.csv"


def utterance_logits_path(corpus: str, experiment: str, seed: int, root: str | Path = "predictions") -> Path:
    return prediction_dir(corpus, experiment, seed, root) / f"utterance_logits_seed{seed}.csv"


def temperature_path(corpus: str, experiment: str, seed: int, root: str | Path = "predictions") -> Path:
    return prediction_dir(corpus, experiment, seed, root) / f"temperature_seed{seed}.json"

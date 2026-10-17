from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("paths", {}).get("root", ".")).resolve()
    for key in ("raw_dir", "processed_dir", "frames_dir", "output_dir"):
        if key in cfg.get("paths", {}):
            p = Path(cfg["paths"][key])
            cfg["paths"][key] = str(p if p.is_absolute() else root / p)
    return cfg

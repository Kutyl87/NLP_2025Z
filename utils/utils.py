from __future__ import annotations
import os
from pathlib import Path
from typing import Dict
from datetime import datetime


def ensure_dirs() -> Dict[str, str]:
    out_root = "data/output"
    plots = f"{out_root}/plots"
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return {"out_root": out_root, "plots": plots}


def add_timestamp_suffix(filename: str | Path) -> str:
    filename = str(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base, ext = os.path.splitext(filename)
    return f"{base}_{timestamp}{ext}"

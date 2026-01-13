from __future__ import annotations
import os
from typing import Dict


def ensure_dirs() -> Dict[str, str]:
    out_root = "data/output"
    plots = f"{out_root}/plots"
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return {"out_root": out_root, "plots": plots}


def allowed_file(filename: str, allowed_extensions: list[str]) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

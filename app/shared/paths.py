"""Well-known application directories."""

from __future__ import annotations

import re
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECTS_DIR = APP_ROOT / "projects"
SETTINGS_DIR = APP_ROOT / "settings"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

_UNSAFE_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*]')


def ensure_app_dirs() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)


def safe_folder_name(name: str) -> str:
    cleaned = _UNSAFE_FOLDER_CHARS.sub("_", name).strip()
    return cleaned or "Unnamed"

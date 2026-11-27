"""Пути и флаги обработки спектров (без хардкода OneDrive)."""
import os
from pathlib import Path

# Корень репозитория: .../ramansPy
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# По умолчанию: raw/ и processed/ рядом с main/
_DEFAULT_INPUT = _PROJECT_ROOT / "data" / "raw"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "processed"

INPUT_DIR = Path(os.environ.get("RAMAN_INPUT_DIR", str(_DEFAULT_INPUT)))
OUTPUT_DIR = Path(os.environ.get("RAMAN_OUTPUT_DIR", str(_DEFAULT_OUTPUT)))

# RAMAN_USE_ORIGIN=0 — только файлы, без OriginPro
USE_ORIGIN = os.environ.get("RAMAN_USE_ORIGIN", "1").strip().lower() not in ("0", "false", "no")

# Длина волны лазера (нм) для оценки L_a по Cançado et al.
LASER_WAVELENGTH_NM = float(os.environ.get("RAMAN_LASER_NM", "532"))


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

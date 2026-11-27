#!/usr/bin/env python3
"""
Типовой сценарий: обработка спектров → (опц.) синтетика → обучение NN → копирование модели в app/.

Переменные окружения см. main/config.py (RAMAN_INPUT_DIR, RAMAN_OUTPUT_DIR, RAMAN_USE_ORIGIN, ...).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent


def run_py(script: Path, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cwd if cwd is not None else script.parent),
        check=True,
        env=env if env is not None else os.environ.copy(),
    )


def cmd_process(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    if args.no_origin:
        env["RAMAN_USE_ORIGIN"] = "0"
    run_py(ROOT / "main" / "main.py", cwd=ROOT / "main", env=env)


def cmd_synthetic(_: argparse.Namespace) -> None:
    run_py(ROOT / "synthetic_data" / "main.py", cwd=ROOT / "synthetic_data")


def cmd_train_direct(_: argparse.Namespace) -> None:
    neyronka_dir = ROOT / "neyronka"
    sys.path.insert(0, str(neyronka_dir))
    import importlib.util

    spec = importlib.util.spec_from_file_location("neyronka_main", neyronka_dir / "main.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.train_model()


def cmd_copy_models(_: argparse.Namespace) -> None:
    out = ROOT / "neyronka" / "output"
    dest = ROOT / "app" / "models"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("graphene_model.keras", "scalers.npz"):
        src = out / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            print(f"Скопировано: {src} -> {dest / name}")
        else:
            print(f"Нет файла: {src}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Конвейер ramansPy")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("process", help="Обработка сырых спектров (main/main.py)")
    sp.add_argument("--no-origin", action="store_true", help="Только файлы, без OriginPro")
    sp.set_defaults(func=cmd_process)

    sub.add_parser("synthetic", help="Генерация синтетических спектров").set_defaults(func=cmd_synthetic)

    sub.add_parser("train", help="Обучение NN (neyronka.train_model)").set_defaults(func=cmd_train_direct)

    sub.add_parser("copy-models", help="Копировать keras и scalers в app/models/").set_defaults(
        func=cmd_copy_models
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

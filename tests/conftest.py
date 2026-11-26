"""Shared pytest setup: import the flat modules from the repo root, render off-screen."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

# Small windows keep the render tests fast; the guardrails do not depend on size.
SMALL_WINDOW = (480, 480)


def png_background_ratio(path: str | Path, background: str = "#FFFFFF") -> float:
    """Background-pixel ratio of a saved PNG (RGB compare, tolerance 4)."""
    from shido.core.validator import background_ratio

    arr = np.asarray(Image.open(path).convert("RGB"))
    return background_ratio(arr, background=background)


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture(autouse=True)
def _output_root_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Default runs (no output_dir) write to <tmp>/出力, never to <repo root>/出力."""
    root = tmp_path / "出力"
    monkeypatch.setenv("CRYSTAL_MCP_OUTPUT_ROOT", str(root))
    return root

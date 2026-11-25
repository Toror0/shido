"""Batch 25: rotation animation (GIF / MP4) through the normal render_single_view pipeline."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from PIL import Image
from pymatgen.core import Lattice, Structure

from shido import server
from shido.analyzer import compute_bonds
from conftest import SMALL_WINDOW
from shido.core import animation
from shido.core.output_paths import describe_file


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _animate(path, fmt="gif", frames=12, axis="up", **kw):
    s = _nacl()
    _, bonds, envs = compute_bonds(s, ["Na-Cl:0:3.0"])
    return animation.render_rotation_animation(
        s, path, camera_dir=[1.0, 0.36, 0.24], view_up=[0.0, 0.0, 1.0], fmt=fmt,
        frames=frames, axis=axis, size=200, render_bonds=bonds, polyhedra_envs=envs,
        window_size=SMALL_WINDOW, caption_text="NaCl", **kw,
    )


def _frame(im: Image.Image, i: int) -> np.ndarray:
    im.seek(i)
    return np.asarray(im.convert("RGB"), dtype=float)


def test_gif_frames_size_and_turning(tmp_path):
    res = _animate(tmp_path / "rotation.gif", frames=12)
    assert res["format"] == "gif" and res["frames"] == 12 and res["axis"] == "up"
    assert 5_000 < res["bytes"] < 2_000_000
    with Image.open(res["path"]) as im:
        assert im.n_frames == 12 and max(im.size) == 200
        first, half = _frame(im, 0), _frame(im, 6)
        quarter = _frame(im, 3)
    # half a turn shows the other side: clearly different from the first frame
    assert np.mean(np.abs(first - half)) > 3.0
    assert np.mean(np.abs(first - quarter)) > 3.0
    # not blank
    assert np.mean(first < 200) > 0.03
    # only the GIF is kept (the still went to a temporary folder)
    assert [p.name for p in tmp_path.iterdir()] == ["rotation.gif"]


def test_axis_and_options_follow(tmp_path):
    res = _animate(tmp_path / "c.gif", frames=4, axis="c", style="space-filling", labels="element")
    with Image.open(res["path"]) as im:
        assert im.n_frames == 4
    with pytest.raises(ValueError):
        animation.normalize_axis("q")
    with pytest.raises(ValueError):
        animation.normalize_animate("avi")
    assert animation.normalize_animate(None) is None
    with pytest.raises(ValueError):
        animation.normalize_frames(1)


def test_mp4_or_gif_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(animation, "_ffmpeg_exe", lambda: None)
    res = _animate(tmp_path / "rotation.mp4", fmt="mp4", frames=4)
    assert res["format"] == "gif" and res["path"].endswith(".gif") and res["warnings"]


def test_cli_and_mcp_expose_animate(tmp_path, monkeypatch):
    parser = server._build_arg_parser()
    opts = server._bulk_options_from_args(parser.parse_args(["demo"]))
    assert "animate" not in opts  # default output unchanged
    args = parser.parse_args(["demo", "--animate", "gif", "--frames", "36", "--animate-axis", "c"])
    opts = server._bulk_options_from_args(args)
    assert (opts["animate"], opts["animate_frames"], opts["animate_axis"]) == ("gif", 36, "c")

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    props = (getattr(tools["render_view"], "input_schema", None)
             or getattr(tools["render_view"], "inputSchema"))["properties"]
    assert {"animate", "animate_frames", "animate_axis"} <= set(props)

    assert "アニメーション" in describe_file("rotation.gif")
    assert "アニメーション" in describe_file("view_custom_rotation.gif")


def test_run_bulk_analysis_writes_rotation_gif(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "render_bulk_views", lambda structure, output_dir, **kw: {})
    real = animation.render_rotation_animation

    def small(*a, **kw):
        kw.update(size=160, window_size=SMALL_WINDOW)
        return real(*a, **kw)

    monkeypatch.setattr(animation, "render_rotation_animation", small)
    res = server.run_bulk_analysis(_nacl(), tmp_path, animate="gif", animate_frames=6)
    gif = tmp_path / "図" / "rotation.gif"
    assert res["animation"]["frames"] == 6 and gif.is_file()

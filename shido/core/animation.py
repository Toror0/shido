"""Rotation animation (VESTA batch 25): GIF / MP4 of the crystal turning around an axis.

The frames are taken by the normal `renderer.render_single_view` pipeline (same style,
range, labels, planes, vectors, isosurfaces ...): `RotationHook` is passed as its
``frame_hook`` and, with the finished VTK scene still open off screen, turns the camera
step by step, takes one screenshot per step, annotates it like the still figure and
shrinks it to ``size`` px. GIF is written with PIL; MP4 needs ffmpeg (imageio-ffmpeg or
an ``ffmpeg`` on PATH), otherwise a GIF is written instead with a warning.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ANIMATE_FORMATS = ("gif", "mp4")
ANIMATE_AXES = ("up", "a", "b", "c", "x", "y", "z")
DEFAULT_FRAMES = 72
DEFAULT_SIZE = 400  # px (long side) of each frame: keeps a 72-frame GIF around 1-2 MB
DEFAULT_FPS = 24
_FRAME_MARGIN = 1.10  # fixed zoom for all frames, with room for the corners while turning


def normalize_animate(fmt: Any) -> str | None:
    """'gif' / 'mp4' (case-insensitive); None / '' / 'none' = no animation."""
    if fmt is None:
        return None
    text = str(fmt).strip().lower().lstrip(".")
    if text in ("", "none", "off", "false"):
        return None
    if text not in ANIMATE_FORMATS:
        raise ValueError(f"animate must be one of {', '.join(ANIMATE_FORMATS)}, got {fmt!r}.")
    return text


def normalize_frames(frames: Any) -> int:
    n = DEFAULT_FRAMES if frames is None else int(frames)
    if not 2 <= n <= 720:
        raise ValueError(f"frames must be 2-720, got {frames!r}.")
    return n


def normalize_axis(axis: Any) -> str:
    text = "up" if axis is None else str(axis).strip().lower()
    if text not in ANIMATE_AXES:
        raise ValueError(f"animate axis must be one of {', '.join(ANIMATE_AXES)}, got {axis!r}.")
    return text


def axis_vector(structure: Any, axis: str, up_unit: np.ndarray) -> np.ndarray:
    """Cartesian unit vector of the rotation axis ('up' = vertical of the first frame)."""
    if axis == "up":
        v = np.asarray(up_unit, dtype=float)
    elif axis in ("a", "b", "c"):
        v = np.asarray(structure.lattice.matrix[("a", "b", "c").index(axis)], dtype=float)
    else:
        v = np.eye(3)[("x", "y", "z").index(axis)]
    return v / max(float(np.linalg.norm(v)), 1e-12)


def rotate(v: np.ndarray, k: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of v about the unit axis k by angle (rad)."""
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def _frame_points(structure: Any, atoms: list) -> tuple[np.ndarray, np.ndarray]:
    """Points (atom centres + unit-cell corners) and their radii that must stay in view."""
    pts = [np.asarray(a[0], dtype=float) for a in atoms]
    rad = [float(a[2]) for a in atoms]
    m = structure.lattice.matrix
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                pts.append(i * m[0] + j * m[1] + k * m[2])
                rad.append(0.0)
    return np.array(pts), np.array(rad)


def _views(cam_unit: np.ndarray, up_unit: np.ndarray, k: np.ndarray, n: int):
    for i in range(n):
        ang = 2.0 * math.pi * i / n
        d = rotate(cam_unit, k, ang)
        u = rotate(up_unit, k, ang)
        u = u - d * float(np.dot(u, d))
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        yield d, u


class RotationHook:
    """``frame_hook`` for `render_single_view`: captures ``frames`` turned views.

    After the render, ``frames_rgb`` holds the annotated, downsized RGB frames and
    `write` saves them. ``axis`` is one of ANIMATE_AXES ('up' = turntable).
    """

    def __init__(self, frames: Any = None, axis: Any = None, size: int | None = None) -> None:
        self.n = normalize_frames(frames)
        self.axis = normalize_axis(axis)
        self.size = int(size or DEFAULT_SIZE)
        self.frames_rgb: list[Image.Image] = []

    def __call__(self, plotter: Any, info: dict[str, Any]) -> None:
        from shido import renderer as R

        structure = info["structure"]
        cam_unit = np.asarray(info["cam_unit"], dtype=float)
        up_unit = np.asarray(info["up_unit"], dtype=float)
        center = np.asarray(info["cell_center"], dtype=float)
        perspective = bool(info["perspective"])
        win_w, win_h = (float(v) for v in info["window_size"])
        k = axis_vector(structure, self.axis, up_unit)
        views = list(_views(cam_unit, up_unit, k, self.n))

        # One zoom for every frame (constant scale bar): the widest of the turned views.
        pts, rad = _frame_points(structure, info["atoms"])
        rel = pts - center
        scale = float(info["parallel_scale"])
        need = 0.0
        for d, u in views:
            r = np.cross(u, d)
            half_w = float(np.max(np.abs(rel @ r) + rad))
            half_h = float(np.max(np.abs(rel @ u) + rad))
            need = max(need, half_h, half_w * win_h / max(win_w, 1.0))
        scale = max(scale, need * _FRAME_MARGIN)
        ratio = scale / max(float(info["parallel_scale"]), 1e-12)
        app = None if perspective else float(info["angstrom_per_pixel"]) * ratio
        cam_dist = float(info["cam_dist"]) * (ratio if perspective else 1.0)

        # only labels of atoms that are drawn (cut-off planes / whole molecules), as in the still
        candidates = (
            R._labels_on_atoms(
                R.atom_label_candidates(
                    structure, info["include_boundary_atoms"], info["labels"], info["frac_range"]
                ),
                info["atoms_drawn"], structure, info["labels"],
            )
            if info["labels"]
            else None
        )
        with tempfile.TemporaryDirectory(prefix="crystal_anim_") as tmp:
            tmp_png = Path(tmp) / "frame.png"
            for d, u in views:
                plotter.camera_position = [(center + d * cam_dist).tolist(), center.tolist(), u.tolist()]
                if not perspective:
                    plotter.camera.parallel_scale = scale
                plotter.renderer.ResetCameraClippingRange()
                plotter.render()  # otherwise screenshot() returns the cached first image
                img = plotter.screenshot(
                    str(tmp_png), transparent_background=info["transparent_background"], return_img=True
                )
                labels = (
                    R._project_visible_labels(
                        plotter, candidates, info["atoms_drawn"], d, u, info["window_size"],
                        (int(img.shape[1]), int(img.shape[0])),
                    )
                    if candidates
                    else None
                )
                final = R._annotate_publication_image(
                    img_path=tmp_png,
                    angstrom_per_pixel=app,
                    orientation_label=None,  # the view direction changes every frame
                    caption_text=info["caption_text"],
                    transparent_background=info["transparent_background"],
                    legend_elements=info["legend_elements"],
                    atom_labels=labels,
                )
                with Image.open(final) as im:
                    frame = R._flatten_white(im.convert("RGBA")).convert("RGB")
                frame.thumbnail((self.size, self.size), Image.LANCZOS)
                self.frames_rgb.append(frame)

    def write(self, output_path: str | Path, fmt: Any = "gif", fps: int = DEFAULT_FPS) -> dict[str, Any]:
        return write_animation(self.frames_rgb, output_path, fmt, fps)


def _ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg  # type: ignore

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return shutil.which("ffmpeg")


def write_animation(
    frames: list[Image.Image], output_path: str | Path, fmt: Any = "gif", fps: int = DEFAULT_FPS
) -> dict[str, Any]:
    """Save frames as GIF (PIL, loops forever) or MP4 (H.264 via ffmpeg).

    Without ffmpeg an MP4 request writes ``<stem>.gif`` instead and says so in ``warnings``.
    Returns {"path", "format", "frames", "fps", "bytes", "warnings"}.
    """
    fmt = normalize_animate(fmt) or "gif"
    try:
        fps_ok = int(fps) >= 1
    except (TypeError, ValueError):
        fps_ok = False
    if not fps_ok:
        raise ValueError(f"fps must be a whole number >= 1, got {fps!r}.")
    if not frames:
        raise ValueError("no frames to write.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if fmt == "mp4":
        exe = _ffmpeg_exe()
        if exe is None:
            msg = "mp4 needs ffmpeg (pip install imageio-ffmpeg); wrote a GIF instead."
            logger.warning(msg)
            warnings.append(msg)
            fmt = "gif"
        else:
            out = out.with_suffix(".mp4")
            w, h = frames[0].size
            w2, h2 = w - w % 2, h - h % 2  # H.264 / yuv420p needs even sides
            with tempfile.TemporaryDirectory(prefix="crystal_mp4_") as tmp:
                for i, f in enumerate(frames):
                    f.crop((0, 0, w2, h2)).save(Path(tmp) / f"f{i:04d}.png")
                subprocess.run(
                    [exe, "-y", "-loglevel", "error", "-framerate", str(int(fps)),
                     "-i", str(Path(tmp) / "f%04d.png"), "-c:v", "libx264",
                     "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
                    check=True,
                )
    if fmt == "gif":
        out = out.with_suffix(".gif")
        # one shared adaptive palette: smaller file and no flicker between frames
        pal = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        quant = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        quant[0].save(
            out, save_all=True, append_images=quant[1:], duration=int(round(1000 / int(fps))),
            loop=0, optimize=False, disposal=1,
        )
    return {
        "path": str(out.resolve()),
        "format": fmt,
        "frames": len(frames),
        "fps": int(fps),
        "bytes": out.stat().st_size,
        "warnings": warnings,
    }


def render_rotation_animation(
    structure: Any,
    output_path: str | Path,
    camera_dir: Any,
    view_up: Any,
    fmt: Any = "gif",
    frames: Any = None,
    axis: Any = None,
    size: int | None = None,
    fps: int = DEFAULT_FPS,
    **render_kwargs: Any,
) -> dict[str, Any]:
    """Render a turning animation with the same settings as `render_single_view`
    (``render_kwargs``: style, render_bonds, labels, frac_range, planes, ...).
    The still image goes to a temporary folder; only the GIF / MP4 is kept."""
    from shido.renderer import render_single_view

    hook = RotationHook(frames=frames, axis=axis, size=size)
    render_kwargs.pop("image_formats", None)
    with tempfile.TemporaryDirectory(prefix="crystal_anim_still_") as tmp:
        render_single_view(
            structure, Path(tmp) / "still.png", camera_dir=camera_dir, view_up=view_up,
            frame_hook=hook, **render_kwargs,
        )
    res = hook.write(output_path, fmt, fps)
    res["axis"] = hook.axis
    return res

"""Headless PyVista 3D crystal structure and slab PNG renderer for publication figures."""

from __future__ import annotations

import base64
import io
import itertools
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pymatgen.core import Element, Structure

from shido.core.output_paths import FIG_DIR  # noqa: E402
from shido.core.species import site_occupancies, warn_partial_occupancy
from shido.core.validator import (
    ScientificValidationError,
    validate_blank_image,
    validate_miller_index,
    validate_structure,
)
from shido.theme_manager import ColorManager, get_color_manager, get_theme, resolve_structure_colors

logger = logging.getLogger("shido")

# Above this many bonds the drawing is limited to nearest-neighbor pairs only.
MAX_RENDER_BONDS: int = 2000
# A bond counts as "nearest" when it is within this fraction of the shortest bond
# of the same element pair.
NEAREST_BOND_TOL: float = 0.05

# Active per-structure color map populated during render calls so composition-ordered
# fallback colors apply seamlessly across spheres, polyhedra, and legends.
_CURRENT_COLOR_MAP: dict[str, str] = {}

# Comprehensive Jmol / CPK element colors (H through Bi + common actinides)
JMOL_COLORS: dict[str, str] = {
    "H": "#E6E6E6",
    "He": "#D9FFFF",
    "Li": "#CC80FF",
    "Be": "#C2FF00",
    "B": "#FFB5B5",
    "C": "#909090",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "Ne": "#B3E3F5",
    "Na": "#AB5CF2",
    "Mg": "#8AFF00",
    "Al": "#BFA6A6",
    "Si": "#F0C8A0",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Ar": "#80D1E3",
    "K": "#8F40D4",
    "Ca": "#3DFF00",
    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Ga": "#C28F8F",
    "Ge": "#668F8F",
    "As": "#BD80E3",
    "Se": "#FFA100",
    "Br": "#A62929",
    "Kr": "#5CB8D1",
    "Rb": "#702EB0",
    "Sr": "#00FF00",
    "Y": "#94FFFF",
    "Zr": "#94E0E0",
    "Nb": "#73C2C9",
    "Mo": "#54B5B5",
    "Tc": "#3B9E9E",
    "Ru": "#248F8F",
    "Rh": "#0A7D8C",
    "Pd": "#006985",
    "Ag": "#C0C0C0",
    "Cd": "#FFD98F",
    "In": "#A67573",
    "Sn": "#668080",
    "Sb": "#9E63B5",
    "Te": "#D47A00",
    "I": "#940094",
    "Xe": "#429EB0",
    "Cs": "#57178F",
    "Ba": "#00C900",
    "La": "#70D4FF",
    "Ce": "#FFFFC7",
    "Pr": "#D9FFC7",
    "Nd": "#C7FFC7",
    "Pm": "#A3FFC7",
    "Sm": "#8FFFC7",
    "Eu": "#61FFC7",
    "Gd": "#45FFC7",
    "Tb": "#30FFC7",
    "Dy": "#1FFFC7",
    "Ho": "#00FF9C",
    "Er": "#00E675",
    "Tm": "#00D452",
    "Yb": "#00BF38",
    "Lu": "#00AB24",
    "Hf": "#4DC2FF",
    "Ta": "#4DA6FF",
    "W": "#2194D6",
    "Re": "#267DAB",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#D0D0E0",
    "Au": "#FFD123",
    "Hg": "#B8B8D0",
    "Tl": "#A6544D",
    "Pb": "#575961",
    "Bi": "#9E4FB5",
    "Th": "#00BAFF",
    "U": "#008FFF",
}

DEFAULT_ELEMENT_COLORS: dict[str, str] = JMOL_COLORS

# Covalent radii in Angstroms (Cordero et al., H through Bi + Th, U)
COVALENT_RADII: dict[str, float] = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Ga": 1.22,
    "Ge": 1.20,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Kr": 1.16,
    "Rb": 2.20,
    "Sr": 1.95,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "In": 1.42,
    "Sn": 1.39,
    "Sb": 1.39,
    "Te": 1.38,
    "I": 1.39,
    "Xe": 1.40,
    "Cs": 2.44,
    "Ba": 2.15,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Pm": 1.99,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "Tl": 1.45,
    "Pb": 1.46,
    "Bi": 1.48,
    "Th": 2.06,
    "U": 1.96,
}

BALL_RADIUS_SCALE: float = 0.40
_BALL_RADIUS_MIN: float = 0.16
_BALL_RADIUS_MAX: float = 1.25
_BOND_RADIUS: float = 0.075
_UNIT_CELL_LINE_RADIUS: float = 0.018
_DPI: int = 300
_FOOTER_HEIGHT: int = 64

# Display styles (VESTA-like). "ball-stick" is the default and draws exactly as before.
#   space-filling: atoms at full radius (radius table x 1.0, no min/max clip), no bonds
#   stick: atoms shrunk to the stick radius, bonds coloured half-and-half by element
#   wireframe: no atom spheres, bonds as thin lines coloured half-and-half by element
#   ellipsoid: displacement ellipsoids (needs ADPs, see render_single_view ``adp``)
RENDER_STYLES: tuple[str, ...] = ("ball-stick", "space-filling", "stick", "wireframe", "ellipsoid")
DEFAULT_RENDER_STYLE: str = "ball-stick"
_STYLE_ALIASES: dict[str, str] = {
    "ball-and-stick": "ball-stick",
    "ballstick": "ball-stick",
    "ball": "ball-stick",
    "space-fill": "space-filling",
    "spacefill": "space-filling",
    "spacefilling": "space-filling",
    "cpk": "space-filling",
    "sticks": "stick",
    "wire": "wireframe",
    "wire-frame": "wireframe",
    "ellipsoids": "ellipsoid",
    "ortep": "ellipsoid",
    "thermal": "ellipsoid",
}
_STICK_RADIUS_FACTOR: float = 2.0  # stick radius = render.bond_radius x this (0.15 A by default)
_WIREFRAME_LINE_WIDTH: float = 2.4  # px at 300 dpi


def normalize_style(style: str | None) -> str:
    """Return the canonical style name ('ball-stick' for None/''); raise ValueError if unknown."""
    if style is None or not str(style).strip():
        return DEFAULT_RENDER_STYLE
    key = str(style).strip().lower().replace("_", "-").replace(" ", "-")
    key = _STYLE_ALIASES.get(key, key)
    if key not in RENDER_STYLES:
        raise ValueError(
            f"Unknown style {style!r}. Choose one of: {', '.join(RENDER_STYLES)}."
        )
    return key


# ---------------------------------------------------------------------------
# Display parameters from settings.py, read at CALL time (not import time) so that
# values changed in the GUI apply without restarting. The module constants above
# are only fallbacks for when settings.py / a key is unavailable.
# ---------------------------------------------------------------------------
def _settings_dict() -> dict[str, Any]:
    try:
        from shido import settings as _settings

        return _settings.load_settings()
    except Exception:
        return {}


def _lookup(cfg: dict[str, Any], path: str, fallback: Any) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return fallback
    return fallback if node is None else node


def _setting(path: str, fallback: Any) -> Any:
    """Read one dotted settings key now, e.g. ``_setting("render.dpi", 300)``."""
    return _lookup(_settings_dict(), path, fallback)


def _float_setting(path: str, fallback: float) -> float:
    try:
        return float(_setting(path, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def _render_dpi() -> int:
    try:
        dpi = int(round(float(_setting("render.dpi", _DPI))))
    except (TypeError, ValueError):
        dpi = _DPI
    return dpi if dpi > 0 else _DPI


def _dpi_pair() -> tuple[int, int]:
    dpi = _render_dpi()
    return (dpi, dpi)


# Pixel sizes in this module (window 1200 px, fonts, line widths, paddings, ...) are
# designed for 300 dpi. They are multiplied by render.dpi / 300 so the printed size
# of a figure stays the same at any dpi (600 dpi = twice the pixels, same inches).
_BASE_DPI: int = 300


def _px_scale() -> float:
    """render.dpi / 300 (1.0 at the default 300 dpi)."""
    return _render_dpi() / float(_BASE_DPI)


def _px(value: float, scale: float | None = None) -> int:
    """A 300-dpi pixel length converted to the current dpi (integer, >= 1)."""
    s = _px_scale() if scale is None else float(scale)
    return max(1, int(round(float(value) * s)))


def _res_scale(resolution: int, scale: float | None = None, cap: int = 72) -> int:
    """Mesh resolution for a 300-dpi design value at the current dpi.

    The polygon outline error is r_px / n^2 and r_px grows with the dpi, so n grows
    with sqrt(dpi / 300) to keep the same sub-pixel smoothness.
    """
    s = _px_scale() if scale is None else float(scale)
    if s <= 1.0:
        return int(resolution)
    return int(min(max(int(round(resolution * math.sqrt(s))), int(resolution)), cap))


def default_radius_params() -> dict[str, Any]:
    """Radius parameters of the shipped defaults (module constants + Cordero covalent radii).

    Figures that use their own hand-tuned sphere / tube sizes (Heusler sublattice
    diagram, c/a thumbnails, ...) multiply those sizes by ``current / default`` so they
    look exactly as before with default settings and follow the settings otherwise.
    """
    return {
        "scale": BALL_RADIUS_SCALE,
        "min": _BALL_RADIUS_MIN,
        "max": _BALL_RADIUS_MAX,
        "table": {},  # empty table -> COVALENT_RADII (same values as the default yaml table)
    }


def _ball_radius_for_occs(
    species_occs: list[tuple[str, float]],
    params: dict[str, Any],
) -> float:
    """Clipped, scaled display radius for a list of (symbol, occupancy) pairs."""
    table = params["table"]
    pos = [(sym, occ) for sym, occ in species_occs if occ > 0.0]
    total = sum(occ for _, occ in pos)
    if total > 0.0:
        radius = sum(_single_element_covalent_radius(sym, table) * occ for sym, occ in pos) / total
    elif species_occs:
        radius = _single_element_covalent_radius(species_occs[0][0], table)
    else:
        radius = 1.2
    if radius <= 0.0:
        radius = 1.2
    lo, hi = float(params["min"]), float(params["max"])
    if hi < lo:
        lo, hi = hi, lo
    return float(np.clip(radius * float(params["scale"]), lo, hi))


def relative_ball_radius(
    base_radius: float,
    species_occs: list[tuple[str, float]] | str,
    params: dict[str, Any] | None = None,
) -> float:
    """Scale a hand-tuned sphere radius by (radius with current settings) / (radius with defaults).

    ``species_occs`` is a symbol or a list of (symbol, occupancy). With the default
    settings this returns ``base_radius`` unchanged; ``render.ball_radius_scale`` /
    ``_min`` / ``_max`` and ``atomic_radii`` change it proportionally. The result is
    clipped to [ball_radius_min, ball_radius_max].
    """
    occs = [(species_occs, 1.0)] if isinstance(species_occs, str) else list(species_occs)
    p = params if params is not None else _radius_params()
    cur = _ball_radius_for_occs(occs, p)
    ref = _ball_radius_for_occs(occs, default_radius_params())
    val = float(base_radius) * (cur / ref if ref > 0.0 else 1.0)
    lo, hi = float(p["min"]), float(p["max"])
    if hi < lo:
        lo, hi = hi, lo
    return float(np.clip(val, lo, hi))


def line_radius_ratio(kind: str) -> float:
    """current / default for ``render.unit_cell_line_radius`` (kind='cell') or ``render.bond_radius`` ('bond')."""
    if kind == "cell":
        cur = _float_setting("render.unit_cell_line_radius", _UNIT_CELL_LINE_RADIUS)
        ref = _UNIT_CELL_LINE_RADIUS
    else:
        cur = _float_setting("render.bond_radius", _BOND_RADIUS)
        ref = _BOND_RADIUS
    cur = max(cur, 1e-4)
    return cur / ref


def _resolve_transparent(value: bool | None) -> bool:
    """Explicit True/False wins; None means 'use render.transparent_background'."""
    if value is None:
        return bool(_setting("render.transparent_background", False))
    return bool(value)


def _radius_params() -> dict[str, Any]:
    """Snapshot of the radius-related settings (one settings read per render call)."""
    cfg = _settings_dict()

    def _f(path: str, fb: float) -> float:
        try:
            return float(_lookup(cfg, path, fb))
        except (TypeError, ValueError):
            return fb

    table = _lookup(cfg, "atomic_radii", {}) or {}
    return {
        "scale": _f("render.ball_radius_scale", BALL_RADIUS_SCALE),
        "min": _f("render.ball_radius_min", _BALL_RADIUS_MIN),
        "max": _f("render.ball_radius_max", _BALL_RADIUS_MAX),
        "table": table if isinstance(table, dict) else {},
    }
_VACANCY_COLOR: str = "#EAEAEA"

_UNIT_CELL_EDGES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((0, 0, 0), (1, 0, 0)),
    ((0, 0, 0), (0, 1, 0)),
    ((0, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 1, 0)),
    ((1, 0, 0), (1, 0, 1)),
    ((0, 1, 0), (1, 1, 0)),
    ((0, 1, 0), (0, 1, 1)),
    ((0, 0, 1), (1, 0, 1)),
    ((0, 0, 1), (0, 1, 1)),
    ((1, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 1, 1)),
)


def _axes_label_font_file() -> str | None:
    """Return a TrueType font path for axis-triad labels (VTK's built-in font draws '[' like '(')."""
    try:
        from matplotlib import font_manager

        path = font_manager.findfont("Arial", fallback_to_default=True)
        return path if path and os.path.isfile(path) else None
    except Exception:
        return None


def add_orientation_axes(plotter: Any, xlabel: str, ylabel: str, zlabel: str, **kwargs: Any) -> Any:
    """Add the corner axis triad with labels drawn in a TrueType font.

    VTK's bundled Arial-like font renders square brackets as round ones, which turns
    crystallographic directions such as ``[100]`` into plane notation ``(100)``.
    ``line_width`` is a 300-dpi value and is scaled by render.dpi / 300.
    """
    if "line_width" in kwargs and kwargs["line_width"] is not None:
        kwargs["line_width"] = float(kwargs["line_width"]) * _px_scale()
    try:
        actor = plotter.add_axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, **kwargs)
    except Exception:
        actor = plotter.add_axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    font_file = _axes_label_font_file()
    if font_file is None:
        return actor
    try:
        import vtk

        marker = actor.GetOrientationMarker() if hasattr(actor, "GetOrientationMarker") else actor
        for cap in (
            marker.GetXAxisCaptionActor2D(),
            marker.GetYAxisCaptionActor2D(),
            marker.GetZAxisCaptionActor2D(),
        ):
            prop = cap.GetCaptionTextProperty()
            prop.SetFontFamily(vtk.VTK_FONT_FILE)
            prop.SetFontFile(font_file)
            prop.ItalicOff()
    except Exception:
        pass
    return actor


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' hex string to an (R, G, B) uint8 tuple."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (128, 128, 128)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _site_species_occupancies(site: Any) -> list[tuple[str, float]]:
    """Sorted (element, occupancy) pairs; reads ``site.species`` only (safe for disorder)."""
    return site_occupancies(site)


def _site_symbol(site: Any) -> str:
    """Extract the primary element symbol from a pymatgen Site or Species."""
    occs = _site_species_occupancies(site)
    return occs[0][0] if occs else str(site)


def _element_color(symbol: str, color_map: dict[str, str] | None = None) -> str:
    """Return theme hex color for an element symbol (active theme + composition fallback)."""
    if color_map and symbol in color_map:
        return color_map[symbol]
    if symbol in _CURRENT_COLOR_MAP:
        return _CURRENT_COLOR_MAP[symbol]
    theme = get_theme()
    theme_elems = theme.get("elements", {})
    if symbol in theme_elems:
        return str(theme_elems[symbol])
    return DEFAULT_ELEMENT_COLORS.get(symbol, "#808080")


def _single_element_covalent_radius(symbol: str, table: dict[str, Any] | None = None) -> float:
    """Return the unscaled display radius (Å): settings ``atomic_radii`` > COVALENT_RADII > pymatgen."""
    if table is None:
        table = _radius_params()["table"]
    val = table.get(symbol)
    if val is not None:
        try:
            fval = float(val)
            if fval > 0.0:
                return fval
        except (TypeError, ValueError):
            pass
    if symbol in COVALENT_RADII:
        return COVALENT_RADII[symbol]
    try:
        elem = Element(symbol)
        if elem.atomic_radius is not None:
            return float(elem.atomic_radius)
        if elem.atomic_radius_calculated is not None:
            return float(elem.atomic_radius_calculated)
    except Exception:
        pass
    return 1.2


def _element_radius(
    site: Any,
    scale: float | None = None,
    params: dict[str, Any] | None = None,
) -> float:
    """Return scaled ball-and-stick display radius (in Angstroms) for a pymatgen site.

    ``scale`` overrides ``render.ball_radius_scale``; ``params`` is a ``_radius_params()``
    snapshot (pass it when sizing many sites to avoid re-reading settings per site).
    """
    p = params if params is not None else _radius_params()
    table = p["table"]
    occs = _site_species_occupancies(site)
    total_occ = sum(occ for _, occ in occs if occ > 0.0)
    if total_occ > 0.0:
        radius = (
            sum(_single_element_covalent_radius(sym, table) * occ for sym, occ in occs if occ > 0.0)
            / total_occ
        )
    else:
        radius = _single_element_covalent_radius(_site_symbol(site), table)

    if radius <= 0.0:
        radius = 1.2

    eff_scale = float(p["scale"]) if scale is None else float(scale)
    lo, hi = float(p["min"]), float(p["max"])
    if hi < lo:
        lo, hi = hi, lo
    return float(np.clip(radius * eff_scale, lo, hi))


def _collect_atoms_to_draw(
    structure: Structure,
    include_boundary_atoms: bool = True,
    tol: float = 1e-3,
    style: str | None = None,
    frac_range: Any = None,
) -> list[tuple[np.ndarray, list[tuple[str, float]], float]]:
    """Return (cart_coords, species_occs, radius) list, optionally mirroring boundary atoms.

    ``frac_range`` [x0, x1, y0, y1, z0, z1] draws all atoms in that fractional box.

    ``style`` sets the radius: ball-stick (default) = scaled/clipped radius,
    space-filling = unscaled radius table value, stick / wireframe = stick radius.
    """
    style = normalize_style(style)
    radius_params = _radius_params()
    if style == "space-filling":
        radius_params = {**radius_params, "scale": 1.0, "min": 0.0, "max": math.inf}
    stick_radius = _stick_radius() if style in ("stick", "wireframe") else None

    site_info: dict[int, tuple[list[tuple[str, float]], float]] = {}
    for idx, site in enumerate(structure):
        site_info[idx] = (
            _site_species_occupancies(site),
            stick_radius if stick_radius is not None else _element_radius(site, params=radius_params),
        )
    return [
        (cart, site_info[idx][0], site_info[idx][1])
        for idx, cart in _drawn_site_images(structure, include_boundary_atoms, tol, frac_range)
    ]


def normalize_frac_range(frac_range: Any) -> list[float] | None:
    """None / empty -> None; otherwise 6 floats x0 x1 y0 y1 z0 z1 (each lo <= hi)."""
    if frac_range is None:
        return None
    vals = [float(v) for v in frac_range]
    if not vals:
        return None
    if len(vals) != 6:
        raise ValueError("range needs 6 numbers: x0 x1 y0 y1 z0 z1.")
    if any(vals[2 * k + 1] < vals[2 * k] for k in range(3)):
        raise ValueError("range: each upper bound must be >= the lower bound.")
    return vals


def _drawn_site_images(
    structure: Structure,
    include_boundary_atoms: bool = True,
    tol: float = 1e-3,
    frac_range: Any = None,
) -> list[tuple[int, np.ndarray]]:
    """(site index, cart coords) of every drawn atom, in drawing order (boundary images
    mirrored, duplicates at the same position dropped). Shared by atoms and labels.

    ``frac_range`` [x0, x1, y0, y1, z0, z1] (VESTA "Boundary") draws every image whose
    fractional coords lie in that box instead (``include_boundary_atoms`` is then ignored).
    """
    lattice = structure.lattice
    out: list[tuple[int, np.ndarray]] = []
    seen: set[tuple[int, int, int]] = set()

    frac_range = normalize_frac_range(frac_range)
    if frac_range is not None:
        from shido.core.cell_transform import atoms_in_range

        for item in atoms_in_range(structure, frac_range):
            cart = np.asarray(
                lattice.get_cartesian_coords(np.asarray(item["frac"], dtype=float)), dtype=float
            )
            key = tuple(np.round(cart * 1000.0).astype(int))
            if key not in seen:
                seen.add(key)
                out.append((int(item["site"]), cart))
        return out

    for idx, site in enumerate(structure):
        if not include_boundary_atoms:
            cart = np.asarray(site.coords, dtype=float)
            key = tuple(np.round(cart * 1000.0).astype(int))
            if key not in seen:
                seen.add(key)
                out.append((idx, cart))
            continue

        frac = np.mod(np.asarray(site.frac_coords, dtype=float), 1.0)
        frac = np.where(np.isclose(frac, 1.0, atol=tol), 0.0, frac)

        shifts_per_axis: list[list[float]] = []
        for coord in frac:
            if coord <= tol:
                shifts_per_axis.append([0.0, 1.0])
            else:
                shifts_per_axis.append([0.0])

        for shift in itertools.product(*shifts_per_axis):
            img_frac = frac + np.asarray(shift, dtype=float)
            cart = np.asarray(lattice.get_cartesian_coords(img_frac), dtype=float)
            key = tuple(np.round(cart * 1000.0).astype(int))
            if key not in seen:
                seen.add(key)
                out.append((idx, cart))

    return out


LABEL_MODES: tuple[str, ...] = ("site", "element")


def normalize_labels(labels: str | None) -> str | None:
    """Return 'site' / 'element', or None for no labels (None, '', 'none', 'off')."""
    if labels is None:
        return None
    key = str(labels).strip().lower()
    if key in ("", "none", "off", "no", "false"):
        return None
    if key not in LABEL_MODES:
        raise ValueError(f"Unknown labels {labels!r}. Choose one of: {', '.join(LABEL_MODES)} (or none).")
    return key


BOND_COLOR_MODES: tuple[str, ...] = ("single", "split")


def normalize_bond_color(bond_color: str | None) -> str | None:
    """Return 'single' / 'split', or None for the style's own default (ball-stick = single)."""
    if bond_color is None or not str(bond_color).strip():
        return None
    key = str(bond_color).strip().lower()
    key = {"one": "single", "mono": "single", "bicolor": "split", "two": "split"}.get(key, key)
    if key not in BOND_COLOR_MODES:
        raise ValueError(f"Unknown bond_color {bond_color!r}. Choose one of: {', '.join(BOND_COLOR_MODES)}.")
    return key


# Non-metal (anion-forming) elements. A structure with none of them (a metal or an
# intermetallic such as Mn2CoGa) gets no coordination polyhedra unless asked for.
POLYHEDRA_ANION_ELEMENTS: frozenset[str] = frozenset({
    "H", "B", "C", "N", "O", "F", "P", "S", "Cl", "As", "Se", "Br", "Te", "I",
})

POLYHEDRA_AUTO_LABEL: str = "自動（陰イオンになる元素があるときだけ）"


def structure_has_anion(structure: Any) -> bool:
    """True if any site holds an element of POLYHEDRA_ANION_ELEMENTS (partial occupancy too)."""
    for site in structure:
        for sp in site.species:
            if str(getattr(sp, "symbol", sp)) in POLYHEDRA_ANION_ELEMENTS:
                return True
    return False


def resolve_show_polyhedra(show_polyhedra: bool | None, structure: Any, polyhedra_center: Any = None) -> bool:
    """Explicit True / False wins. ``None`` (auto, the default): draw polyhedra when the
    structure has an anion-forming element (oxides, halides, ...) or a polyhedron centre
    was chosen; metals and intermetallics get none (the figure stays readable)."""
    if show_polyhedra is not None:
        return bool(show_polyhedra)
    if polyhedra_center:
        return True
    return structure_has_anion(structure)


def normalize_polyhedra_center(center: Any) -> list[str] | None:
    """Element symbols for the polyhedron centres ('Mn', 'Mn,Ga' or a list); None = automatic."""
    if center is None:
        return None
    items = [center] if isinstance(center, str) else list(center)
    out: list[str] = []
    for item in items:
        for part in str(item).replace(" ", ",").split(","):
            part = part.strip()
            if part:
                sym = part[:1].upper() + part[1:].lower()
                if sym not in out:
                    out.append(sym)
    return out or None


def _site_label_text(site: Any, mode: str) -> str:
    """'element': element symbols ('Mn/Ga' for mixed sites); 'site': the CIF site label
    (e.g. 'Ni1'), falling back to the element text when the site has no label."""
    elem = "/".join(sym for sym, _ in _site_species_occupancies(site))
    if mode == "site":
        label = getattr(site, "label", None)
        if label and str(label).strip():
            return str(label).strip()
    return elem


def atom_label_candidates(
    structure: Structure,
    include_boundary_atoms: bool = True,
    labels: str | None = "site",
    frac_range: Any = None,
) -> list[tuple[np.ndarray, str]]:
    """One (cart coords, text) per drawn atom, same order as `_collect_atoms_to_draw`."""
    mode = normalize_labels(labels)
    if mode is None:
        return []
    texts = [_site_label_text(site, mode) for site in structure]
    return [
        (cart, texts[idx])
        for idx, cart in _drawn_site_images(
            structure, include_boundary_atoms, frac_range=frac_range
        )
    ]


def _labels_on_atoms(
    candidates: list[tuple[np.ndarray, str]], atoms: Any, structure: Structure, labels: str | None
) -> list[tuple[np.ndarray, str]]:
    """Labels of the atoms left after cutoff planes / whole molecules (``atoms`` None = all)."""
    if atoms is None:
        return candidates
    from shido.core.molecules import keep_near_atoms

    return [c for c, k in zip(candidates, keep_near_atoms([c[0] for c in candidates], atoms)) if k]


def _replicate_in_range(
    structure: Structure,
    atom_carts: list[np.ndarray],
    frac_range: list[float],
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]] | None,
    polyhedra_envs: list[tuple[int, list[np.ndarray], str]] | None,
    tol: float = 0.02,
) -> tuple[list[tuple[np.ndarray, np.ndarray, str, str]], list[tuple[int, list[np.ndarray], str]]]:
    """Copy unit-cell bonds / polyhedra by lattice translations onto the drawn range.

    A bond is kept when both ends land on drawn atoms; a polyhedron when its centre
    site (``structure[idx]`` + translation) and all its ligands are drawn atoms (so no
    polyhedron sticks out of the range). Duplicates are dropped.
    """
    from scipy.spatial import cKDTree

    if not atom_carts:
        return [], []
    tree = cKDTree(np.asarray(atom_carts, dtype=float))
    mat = np.asarray(structure.lattice.matrix, dtype=float)
    shifts = [
        np.asarray(n, dtype=float) @ mat
        for n in itertools.product(
            *[
                range(int(math.floor(frac_range[2 * k])) - 1, int(math.ceil(frac_range[2 * k + 1])) + 1)
                for k in range(3)
            ]
        )
    ]

    def _on_atom(p: np.ndarray) -> bool:
        return bool(tree.query(p, distance_upper_bound=tol)[0] <= tol)

    bonds_out: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    seen_b: set[tuple] = set()
    for start, end, e1, e2 in render_bonds or []:
        s0 = np.asarray(start, dtype=float)
        e0 = np.asarray(end, dtype=float)
        for t in shifts:
            s, e = s0 + t, e0 + t
            if not (_on_atom(s) and _on_atom(e)):
                continue
            ks = tuple(np.round(s * 100.0).astype(int))
            ke = tuple(np.round(e * 100.0).astype(int))
            key = (min(ks, ke), max(ks, ke))
            if key in seen_b:
                continue
            seen_b.add(key)
            bonds_out.append((s, e, e1, e2))

    polys_out: list[tuple[int, list[np.ndarray], str]] = []
    seen_p: set[tuple] = set()
    for idx, ligands, elem in polyhedra_envs or []:
        center = np.asarray(structure[int(idx)].coords, dtype=float)
        lig = np.asarray(ligands, dtype=float)
        for t in shifts:
            c = center + t
            if not _on_atom(c) or not all(_on_atom(q) for q in lig + t):
                continue
            key = tuple(np.round(c * 100.0).astype(int))
            if key in seen_p:
                continue
            seen_p.add(key)
            polys_out.append((idx, [p for p in lig + t], elem))
    return bonds_out, polys_out


def resolve_view_direction(
    structure: Structure,
    view_dir: list[float] | tuple[float, ...] | None = None,
    view_normal: list[float] | tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Camera direction, view-up and badge label for a [uvw] direction or (hkl) normal.

    [uvw] -> u a + v b + w c; (hkl) -> reciprocal vector h a* + k b* + l c* (the plane
    normal). The camera sits on that side looking back (same as the [100] view, which
    uses camera_dir = a). View-up is c, or b when the direction is (nearly) along c.
    """
    if (view_dir is None) == (view_normal is None):
        raise ValueError("Give exactly one of view_dir [u v w] or view_normal (h k l).")
    vals = list(view_dir if view_dir is not None else view_normal)  # type: ignore[arg-type]
    if len(vals) != 3:
        raise ValueError(f"A view direction needs 3 numbers, got {vals!r}.")
    vec3 = np.asarray([float(v) for v in vals], dtype=float)
    if float(np.linalg.norm(vec3)) < 1e-12:
        raise ValueError("The view direction must not be 0 0 0.")
    lattice = structure.lattice
    if view_dir is not None:
        cam = np.asarray(lattice.get_cartesian_coords(vec3), dtype=float)
        label = "[" + _format_index_triplet(vec3) + "]"
    else:
        cam = np.asarray(
            lattice.reciprocal_lattice_crystallographic.get_cartesian_coords(vec3), dtype=float
        )
        label = "(" + _format_index_triplet(vec3) + ") normal"
    cam = cam / np.linalg.norm(cam)
    _, b_vec, c_vec = (np.asarray(v, dtype=float) for v in lattice.matrix)
    c_unit = c_vec / np.linalg.norm(c_vec)
    up = c_vec if abs(float(np.dot(cam, c_unit))) < 0.99 else b_vec
    return cam, _orthonormal_up(cam, up), label


def _format_index_triplet(vals: np.ndarray) -> str:
    """'110' for small integers, '1 1 0' style with spaces otherwise ('-1' as '-1')."""
    ints = [int(round(v)) for v in vals]
    if all(abs(v - i) < 1e-9 for v, i in zip(vals, ints)):
        if all(0 <= i <= 9 for i in ints):
            return "".join(str(i) for i in ints)
        return " ".join(str(i) for i in ints)
    return " ".join(f"{v:g}" for v in vals)


_SPHERE_RES_MAX: int = 36
_SPHERE_RES_MIN: int = 12


def _sphere_resolution(
    radius: float,
    angstrom_per_pixel: float | None,
    max_res: int = _SPHERE_RES_MAX,
    px_scale: float = 1.0,
) -> int:
    """Pick theta/phi resolution so the polygonal outline stays sub-pixel.

    A regular n-gon of radius r deviates from the circle by r*(1-cos(pi/n)) ~ r*pi^2/(2n^2).
    Keeping that below ~0.35 px gives n >= 3.75*sqrt(r_px). Small on-screen atoms (large
    supercells, overview thumbnails) get coarser meshes; big atoms keep the old 36.
    ``px_scale`` (dpi / 300) raises the cap by sqrt(px_scale) for high-dpi renders.
    """
    max_res = _res_scale(max_res, px_scale)
    if not angstrom_per_pixel or angstrom_per_pixel <= 0.0:
        return int(max_res)
    r_px = max(float(radius) / float(angstrom_per_pixel), 0.0)
    n = int(math.ceil(3.8 * math.sqrt(r_px)))
    return int(min(max(n, _SPHERE_RES_MIN), max_res))


def _merge_spheres(
    centers: list[np.ndarray],
    radii: list[float],
    resolution: int,
) -> pv.PolyData:
    """Build many spheres as ONE PolyData by translating a template mesh.

    Same geometry as ``pv.merge([pv.Sphere(...) ...])`` but avoids one VTK source and
    a long merge per atom (the slow part for supercells).
    """
    probe = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution)
    if probe.n_cells == 0 or len(probe.faces) != 4 * probe.n_cells:
        # Not a pure triangle mesh on this VTK version: fall back to the plain merge.
        spheres = [
            pv.Sphere(radius=float(r), center=c, theta_resolution=resolution, phi_resolution=resolution)
            for c, r in zip(centers, radii)
        ]
        return pv.merge(spheres) if len(spheres) > 1 else spheres[0]

    pts_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    offset = 0
    templates: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for center, radius in zip(centers, radii):
        key = round(float(radius), 6)
        if key not in templates:
            tmpl = pv.Sphere(
                radius=float(radius),
                center=(0.0, 0.0, 0.0),
                theta_resolution=resolution,
                phi_resolution=resolution,
            )
            faces = np.asarray(tmpl.faces, dtype=np.int64).reshape(-1, 4)  # all triangles
            templates[key] = (np.asarray(tmpl.points, dtype=float), faces)
        t_pts, t_faces = templates[key]
        pts_all.append(t_pts + np.asarray(center, dtype=float))
        shifted = t_faces.copy()
        shifted[:, 1:] += offset
        faces_all.append(shifted)
        offset += len(t_pts)
    return pv.PolyData(np.vstack(pts_all), faces=np.concatenate(faces_all).ravel())


def _build_disordered_sphere(
    center: np.ndarray,
    radius: float,
    species_occs: list[tuple[str, float]],
    resolution: int = 32,
) -> pv.PolyData:
    """Build a multi-colored sphere split into latitude bands by occupancy fraction."""
    sphere = pv.Sphere(
        radius=radius,
        center=center,
        theta_resolution=resolution,
        phi_resolution=resolution,
    )
    cell_centers = sphere.cell_centers().points
    # Spherical zone area is linear in z (Archimedes' hat-box theorem),
    # so mapping z_frac in [0, 1] directly matches each occupancy fraction.
    rel_z = np.clip((cell_centers[:, 2] - float(center[2])) / max(radius, 1e-8), -1.0, 1.0)
    z_frac = 0.5 * (rel_z + 1.0)

    bands: list[tuple[float, tuple[int, int, int]]] = []
    total_occ = sum(max(0.0, occ) for _, occ in species_occs)
    if total_occ <= 0.0:
        bands.append((1.0, _hex_to_rgb(_VACANCY_COLOR)))
    else:
        scale = 1.0 if total_occ < 0.995 else (1.0 / total_occ)
        cum = 0.0
        for sym, occ in species_occs:
            if occ <= 0.0:
                continue
            cum += occ * scale
            bands.append((min(cum, 1.0), _hex_to_rgb(_element_color(sym))))
        if total_occ < 0.995:
            bands.append((1.0, _hex_to_rgb(_VACANCY_COLOR)))
        elif bands:
            bands[-1] = (1.0, bands[-1][1])

    rgb = np.empty((sphere.n_cells, 3), dtype=np.uint8)
    prev_bound = -1.0
    for upper, color_rgb in bands:
        mask = (z_frac > prev_bound) & (z_frac <= upper + 1e-9)
        rgb[mask] = color_rgb
        prev_bound = upper

    sphere.cell_data["rgb"] = rgb
    return sphere


def _mesh_is_empty(mesh: Any) -> bool:
    """True for None or a mesh without points/cells (VTK may crash or warn on these)."""
    if mesh is None:
        return True
    try:
        return int(mesh.n_points) == 0 or int(mesh.n_cells) == 0
    except Exception:
        return True


def _safe_add_mesh(plotter: pv.Plotter, mesh: Any, **kwargs: Any) -> bool:
    """``plotter.add_mesh`` that skips empty meshes. Returns True if the mesh was added."""
    if _mesh_is_empty(mesh):
        logger.debug("Skipped an empty mesh (%s).", kwargs.get("color", "no color"))
        return False
    plotter.add_mesh(mesh, **kwargs)
    return True


def limit_render_bonds(
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]],
    max_bonds: int = MAX_RENDER_BONDS,
    nearest_tol: float = NEAREST_BOND_TOL,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """Keep at most ``max_bonds`` bonds; above it keep nearest-neighbor pairs only.

    Step 1: per element pair keep bonds within ``(1 + nearest_tol)`` of that pair's
    shortest bond. Step 2: if still too many, keep the ``max_bonds`` shortest.
    """
    if len(render_bonds) <= max_bonds:
        return list(render_bonds)
    lengths = [
        float(np.linalg.norm(np.asarray(b[1], dtype=float) - np.asarray(b[0], dtype=float)))
        for b in render_bonds
    ]
    pair_min: dict[tuple[str, ...], float] = {}
    for bond, length in zip(render_bonds, lengths):
        if length < 1e-6:
            continue
        key = tuple(sorted((str(bond[2]), str(bond[3]))))
        pair_min[key] = min(pair_min.get(key, math.inf), length)
    kept = [
        (bond, length)
        for bond, length in zip(render_bonds, lengths)
        if length >= 1e-6
        and length
        <= pair_min[tuple(sorted((str(bond[2]), str(bond[3]))))] * (1.0 + nearest_tol) + 1e-6
    ]
    if len(kept) > max_bonds:
        kept.sort(key=lambda t: t[1])
        kept = kept[:max_bonds]
    logger.warning(
        "Dense structure: %d bonds > %d. Drawing nearest-neighbor pairs only (%d bonds).",
        len(render_bonds),
        max_bonds,
        len(kept),
    )
    return [bond for bond, _ in kept]


def _add_unit_cell(
    plotter: pv.Plotter,
    structure: Structure,
    tube_radius: float | None = None,
) -> None:
    """Draw the 12 edges of the unit cell as crisp dark tubes (radius: render.unit_cell_line_radius)."""
    if tube_radius is None:
        tube_radius = _float_setting("render.unit_cell_line_radius", _UNIT_CELL_LINE_RADIUS)
    tube_radius = max(float(tube_radius), 1e-4)
    lattice = structure.lattice
    tubes: list[pv.PolyData] = []
    for f0, f1 in _UNIT_CELL_EDGES:
        p0 = lattice.get_cartesian_coords(f0)
        p1 = lattice.get_cartesian_coords(f1)
        line = pv.Line(p0, p1)
        tubes.append(line.tube(radius=tube_radius, n_sides=12))

    cell_mesh = pv.merge(tubes)
    cell_color = str(get_theme().get("unit_cell", "#2B2D42"))
    _safe_add_mesh(plotter, cell_mesh, color=cell_color, smooth_shading=True)


def _merge_cylinders(
    starts: np.ndarray,
    ends: np.ndarray,
    radius: float,
    resolution: int = 18,
) -> pv.PolyData:
    """Build capped cylinders p1->p2 as ONE PolyData from a transformed template.

    Equivalent to merging ``pv.Cylinder(...)`` per bond (only the roll angle of the
    18-gon about the bond axis differs, which is invisible) but vectorized: the
    per-bond VTK source + merge was the slowest part of rendering supercells.
    """
    tmpl = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=radius,
        height=1.0,
        resolution=resolution,
        capping=True,
    )
    t_pts = np.asarray(tmpl.points, dtype=float)  # axis along z, z in [-0.5, 0.5]
    t_nrm = tmpl.point_data.get("Normals")
    t_faces = np.asarray(tmpl.faces, dtype=np.int64)

    # Mark which entries of the flat VTK face array are vertex counts (not indices).
    is_count = np.zeros(len(t_faces), dtype=bool)
    i = 0
    while i < len(t_faces):
        is_count[i] = True
        i += int(t_faces[i]) + 1

    vec = ends - starts
    lengths = np.linalg.norm(vec, axis=1)
    d = vec / lengths[:, None]
    # Orthonormal frame (u, v, d) per bond.
    helper = np.where(np.abs(d[:, [0]]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    centers = 0.5 * (starts + ends)

    n_b, n_p = len(starts), len(t_pts)
    pts = (
        centers[:, None, :]
        + t_pts[None, :, 0:1] * u[:, None, :]
        + t_pts[None, :, 1:2] * v[:, None, :]
        + (t_pts[None, :, 2:3] * lengths[:, None, None]) * d[:, None, :]
    ).reshape(-1, 3)

    faces = np.tile(t_faces, n_b).reshape(n_b, -1)
    faces[:, ~is_count] += (np.arange(n_b, dtype=np.int64) * n_p)[:, None]
    mesh = pv.PolyData(pts, faces=faces.ravel())

    if t_nrm is not None:
        t_nrm = np.asarray(t_nrm, dtype=float)
        nrm = (
            t_nrm[None, :, 0:1] * u[:, None, :]
            + t_nrm[None, :, 1:2] * v[:, None, :]
            + t_nrm[None, :, 2:3] * d[:, None, :]
        ).reshape(-1, 3)
        mesh.point_data["Normals"] = nrm
    return mesh


def _add_bonds(
    plotter: pv.Plotter,
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]],
    bond_radius: float | None = None,
    max_bond_length: float | None = None,
) -> None:
    """Draw bonds as crisp ball-and-stick cylinders using the active theme bond color.

    ``bond_radius`` defaults to ``render.bond_radius``; bonds longer than
    ``max_bond_length`` (default ``bonds.max_bond_length``, <= 0 disables) are skipped.
    """
    if bond_radius is None:
        bond_radius = _float_setting("render.bond_radius", _BOND_RADIUS)
    bond_radius = max(float(bond_radius), 1e-4)
    if max_bond_length is None:
        max_bond_length = _float_setting("bonds.max_bond_length", 0.0)
    max_len = float(max_bond_length) if max_bond_length and max_bond_length > 0.0 else math.inf
    bond_color = str(get_theme().get("bond", "#B8C0CC"))

    if not render_bonds:
        logger.info("No bonds to draw (isolated atoms or large lattice); bond mesh skipped.")
        return
    render_bonds = limit_render_bonds(render_bonds)

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for p1_raw, p2_raw, _elem1, _elem2 in render_bonds:
        p1 = np.asarray(p1_raw, dtype=float)
        p2 = np.asarray(p2_raw, dtype=float)
        length = float(np.linalg.norm(p2 - p1))
        if length < 1e-6 or length > max_len + 1e-6:
            continue
        starts.append(p1)
        ends.append(p2)

    if starts:
        merged = _merge_cylinders(np.asarray(starts), np.asarray(ends), bond_radius)
        _safe_add_mesh(
            plotter,
            merged,
            color=bond_color,
            smooth_shading=True,
            ambient=0.25,
            diffuse=0.75,
            specular=0.35,
            specular_power=30,
        )


def _add_polyhedra(
    plotter: pv.Plotter,
    structure: Structure,
    polyhedra_envs: list[tuple[int, list[np.ndarray], str]],
    center_elements: list[str] | None = None,
) -> None:
    """Draw translucent convex hull polyhedra for the least frequent element.

    ``center_elements`` (e.g. ['Mn']) picks the centre elements instead of the automatic
    least-frequent choice; centres with fewer than 4 neighbours get no polyhedron.
    """
    if not polyhedra_envs:
        return
    px_scale = _px_scale()

    elem_counts = Counter(_site_symbol(site) for site in structure)
    env_elems = {str(elem) for _, _, elem in polyhedra_envs}
    if not env_elems:
        return

    if center_elements:
        target_elems = {str(e) for e in center_elements} & env_elems
        missing = sorted({str(e) for e in center_elements} - env_elems)
        if missing:
            logger.warning(
                "Polyhedra centre %s: no site with >= 4 bonded neighbours, no polyhedron drawn.",
                ", ".join(missing),
            )
    else:
        min_count = min(elem_counts.get(e, float("inf")) for e in env_elems)
        target_elems = {
            e for e in env_elems if elem_counts.get(e, float("inf")) == min_count
        }

    for _, ligand_coords, center_elem in polyhedra_envs:
        elem_str = str(center_elem)
        if elem_str not in target_elems:
            continue
        if len(ligand_coords) < 4:
            continue

        pts = np.asarray(ligand_coords, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            continue

        try:
            hull = (
                pv.PolyData(pts)
                .delaunay_3d()
                .extract_surface(algorithm="dataset_surface")
            )
            if not _mesh_is_empty(hull):
                color = _element_color(elem_str)
                _safe_add_mesh(
                    plotter,
                    hull,
                    color=color,
                    opacity=0.22,
                    smooth_shading=False,
                    show_edges=True,
                    edge_color=color,
                    line_width=1.0 * px_scale,
                )
        except Exception:
            continue


def _orthonormal_up(cam_dir: np.ndarray, view_up: np.ndarray) -> np.ndarray:
    """Return a unit view_up vector orthogonal to cam_dir."""
    c = cam_dir / np.linalg.norm(cam_dir)
    u = np.asarray(view_up, dtype=float)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-8:
        u = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        u = u / u_norm

    u_orth = u - np.dot(u, c) * c
    if float(np.linalg.norm(u_orth)) < 1e-6:
        fallback = (
            np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(c[1]) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=float)
        )
        u_orth = fallback - np.dot(fallback, c) * c

    return u_orth / np.linalg.norm(u_orth)


def _compute_parallel_framing(
    structure: Structure,
    atoms: list[tuple[np.ndarray, list[tuple[str, float]], float]],
    cam_unit: np.ndarray,
    up_unit: np.ndarray,
    window_size: tuple[int, int],
    cell_slack: float = 2.0,
) -> tuple[np.ndarray, float, float]:
    """Compute (focal_point, parallel_scale, angstrom_per_pixel) for parallel projection.

    The view is framed on the screen-space bounding box of the atoms (radii included),
    not of the whole unit cell, so a slab with a long vacuum gap still fills the image.
    Along a screen axis the cell corners are included only when the cell extent is at
    most ``cell_slack`` x the atom extent (+4 A), i.e. for ordinary bulk cells, where
    the complete cell outline should stay visible.
    """
    lattice = structure.lattice
    cell_center = np.asarray(lattice.get_cartesian_coords([0.5, 0.5, 0.5]), dtype=float)

    right_vec = np.cross(up_unit, cam_unit)
    r_norm = float(np.linalg.norm(right_vec))
    if r_norm < 1e-8:
        right_unit = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right_unit = right_vec / r_norm
    axes = (right_unit, np.asarray(up_unit, dtype=float))

    corners = np.array(
        [lattice.get_cartesian_coords(f) for f in itertools.product((0.0, 1.0), repeat=3)],
        dtype=float,
    )
    cell_lo = [float((corners @ ax).min()) for ax in axes]
    cell_hi = [float((corners @ ax).max()) for ax in axes]

    if atoms:
        pts = np.asarray([a[0] for a in atoms], dtype=float)
        rad = np.asarray([a[2] for a in atoms], dtype=float)
        atom_lo = [float((pts @ ax - rad).min()) for ax in axes]
        atom_hi = [float((pts @ ax + rad).max()) for ax in axes]
    else:
        atom_lo, atom_hi = list(cell_lo), list(cell_hi)

    lo: list[float] = []
    hi: list[float] = []
    for k in range(2):
        atom_ext = atom_hi[k] - atom_lo[k]
        cell_ext = cell_hi[k] - cell_lo[k]
        if cell_ext <= cell_slack * atom_ext + 4.0:
            lo.append(min(atom_lo[k], cell_lo[k]))
            hi.append(max(atom_hi[k], cell_hi[k]))
        else:
            lo.append(atom_lo[k])
            hi.append(atom_hi[k])

    # Focal point: centre of the chosen screen box, at the depth of the cell centre.
    depth = float(np.dot(cell_center, cam_unit))
    center = (
        right_unit * (0.5 * (lo[0] + hi[0]))
        + axes[1] * (0.5 * (lo[1] + hi[1]))
        + cam_unit * depth
    )

    half_w = max(0.5 * (hi[0] - lo[0]), 1.0)
    half_h = max(0.5 * (hi[1] - lo[1]), 1.0)

    aspect = float(window_size[0]) / max(float(window_size[1]), 1.0)
    parallel_scale = max(half_h, half_w / aspect) * 1.18
    angstrom_per_pixel = (2.0 * parallel_scale) / max(float(window_size[1]), 1.0)
    return center, float(parallel_scale), float(angstrom_per_pixel)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a crisp TrueType font (Arial / DejaVuSans), falling back to Pillow default."""
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _select_scale_bar_length(
    img_width: int,
    angstrom_per_pixel: float,
    px_scale: float = 1.0,
) -> tuple[float, int]:
    """Select the cleanest scale bar length (in Angstroms) spanning ~12%-26% of image width.

    ``px_scale`` (dpi / 300) scales the 48 px minimum bar length.
    """
    candidates = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    min_px = 0.12 * img_width
    max_px = 0.26 * img_width
    target_px = 0.18 * img_width

    if angstrom_per_pixel <= 0.0:
        return 2.0, int(round(target_px))

    in_range = [
        (val, val / angstrom_per_pixel)
        for val in candidates
        if min_px <= (val / angstrom_per_pixel) <= max_px
    ]
    if in_range:
        best_val, best_px = min(in_range, key=lambda pair: abs(pair[1] - target_px))
    else:
        best_val, best_px = min(
            ((val, val / angstrom_per_pixel) for val in candidates),
            key=lambda pair: abs(pair[1] - target_px),
        )

    bar_px = int(np.clip(round(best_px), _px(48, px_scale), int(img_width * 0.40)))
    return best_val, bar_px


def _format_scale_label(length_angstrom: float) -> str:
    """Format scale bar length cleanly, e.g. '0.5 Å' or '2 Å'."""
    if abs(length_angstrom - round(length_angstrom)) < 1e-6:
        return f"{int(round(length_angstrom))} Å"
    return f"{length_angstrom:g} Å"


def _draw_centered_footer(
    canvas: Image.Image,
    caption_text: str,
    footer_top: int,
    footer_height: int | None = None,
    px_scale: float | None = None,
) -> None:
    """Draw a clean white publication footer bar with top divider and centered caption.

    Sizes are 300-dpi values times ``px_scale`` (default: render.dpi / 300).
    """
    s = _px_scale() if px_scale is None else float(px_scale)
    if footer_height is None:
        footer_height = _px(_FOOTER_HEIGHT, s)
    draw = ImageDraw.Draw(canvas)
    width = canvas.width
    is_rgba = canvas.mode == "RGBA"
    bg_fill = (255, 255, 255, 240) if is_rgba else (255, 255, 255)
    line_fill = (205, 205, 205, 255) if is_rgba else (205, 205, 205)
    text_fill = (24, 24, 24, 255) if is_rgba else (24, 24, 24)

    draw.rectangle([(0, footer_top), (width, footer_top + footer_height)], fill=bg_fill)
    draw.line([(0, footer_top), (width, footer_top)], fill=line_fill, width=_px(2, s))

    font_size = max(_px(18, s), min(_px(26, s), int(round(width * 0.021))))
    font = _load_font(font_size)
    max_text_w = width - _px(48, s)
    step = _px(2, s)
    while font_size > _px(12, s):
        bbox = draw.textbbox((0, 0), caption_text, font=font)
        if (bbox[2] - bbox[0]) <= max_text_w:
            break
        font_size -= step
        font = _load_font(font_size)

    bbox = draw.textbbox((0, 0), caption_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    tx = (width - text_w) / 2.0 - bbox[0]
    ty = footer_top + (footer_height - text_h) / 2.0 - bbox[1]
    draw.text((tx, ty), caption_text, fill=text_fill, font=font)


def _draw_element_legend(
    draw: ImageDraw.ImageDraw,
    elements: list[str],
    width: int,
    height: int,
    margin: int,
    ps: float,
) -> None:
    """Draw a boxed element legend (colored circle + symbol per row) in the top-right corner.

    Sizes are fractions of the image width (2.5 % font at 1200 px = 30 px, like the shared
    legend below multi-panel figures). Rows wrap into more columns when one column would
    take more than ~60 % of the image height, and the box never leaves the image.
    """
    font_px = max(_px(12, ps), int(round(width * 0.025)))
    font = _load_font(font_px)
    r = max(2, int(round(font_px * 0.40)))  # marker radius
    label_gap = max(2, int(round(font_px * 0.40)))
    row_h = int(round(font_px * 1.45))
    col_gap = int(round(font_px * 0.9))
    pad_x = int(round(font_px * 0.6))
    pad_y = int(round(font_px * 0.45))
    outline_w = max(1, int(round(font_px / 15.0)))

    label_w = max(
        (draw.textbbox((0, 0), e, font=font)[2] - draw.textbbox((0, 0), e, font=font)[0])
        for e in elements
    )
    item_w = 2 * r + label_gap + label_w

    max_rows = max(1, int((height * 0.6 - 2 * pad_y) // row_h))
    n_cols = max(1, math.ceil(len(elements) / max_rows))
    n_rows = math.ceil(len(elements) / n_cols)

    box_w = 2 * pad_x + n_cols * item_w + (n_cols - 1) * col_gap
    box_h = 2 * pad_y + n_rows * row_h
    bx1 = width - margin
    by0 = margin
    bx0 = max(0, bx1 - box_w)
    by1 = min(height - 1, by0 + box_h)

    draw.rounded_rectangle(
        [(bx0, by0), (bx1, by1)],
        radius=_px(10, ps),
        fill=(255, 255, 255, 232),
        outline=(200, 200, 200, 255),
        width=_px(2, ps),
    )
    for i, elem in enumerate(elements):
        col, row = divmod(i, n_rows)
        x = bx0 + pad_x + col * (item_w + col_gap)
        cy = by0 + pad_y + row * row_h + row_h / 2.0
        fill = _hex_to_rgb(_element_color(elem)) + (255,)
        draw.ellipse(
            [(x, cy - r), (x + 2 * r, cy + r)], fill=fill, outline=(26, 26, 26, 255), width=outline_w
        )
        draw.text(
            (x + 2 * r + label_gap, cy), elem, fill=(24, 24, 24, 255), font=font, anchor="lm"
        )


def _annotate_publication_image(
    img_path: str | Path,
    angstrom_per_pixel: float | None,
    orientation_label: str | None = None,
    caption_text: str | None = None,
    transparent_background: bool | None = None,
    legend_elements: list[str] | None = None,
    atom_labels: list[tuple[float, float, str]] | None = None,
) -> Path:
    """Overlay scale bar, orientation badge, element legend, and optional caption footer.

    ``legend_elements`` draws a fixed-size element legend (top-right). It is drawn here
    with PIL instead of VTK's legend box, which scales its text to fill the box (huge
    letters cut off at the right edge for one element).
    """
    transparent_background = _resolve_transparent(transparent_background)
    path = Path(img_path)
    with Image.open(path) as src:
        base = src.convert("RGBA")

    ps = _px_scale()
    footer_h = _px(_FOOTER_HEIGHT, ps)
    width, height = base.size
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    margin = max(_px(22, ps), int(round(width * 0.022)))

    # 0. Atom labels (drawn first, under the scale bar / badge / legend boxes)
    if atom_labels:
        _draw_atom_labels(draw, atom_labels, width, ps)

    # 1. Scale bar (bottom-right; omitted for perspective views)
    if angstrom_per_pixel is not None:
        bar_angstrom, bar_px = _select_scale_bar_length(width, angstrom_per_pixel, px_scale=ps)
        scale_label = _format_scale_label(bar_angstrom)
        scale_font = _load_font(max(_px(18, ps), int(round(width * 0.022))))
        lbl_bbox = draw.textbbox((0, 0), scale_label, font=scale_font)
        lbl_w = lbl_bbox[2] - lbl_bbox[0]
        lbl_h = lbl_bbox[3] - lbl_bbox[1]

        pad_x = _px(18, ps)
        pad_y = _px(12, ps)
        bar_h = max(_px(6, ps), int(round(height * 0.006)))
        gap = _px(8, ps)
        box_w = max(bar_px, lbl_w) + 2 * pad_x
        box_h = lbl_h + gap + bar_h + 2 * pad_y
        margin = max(_px(22, ps), int(round(width * 0.022)))

        bx1 = width - margin
        bx0 = bx1 - box_w
        by1 = height - margin
        by0 = by1 - box_h

        draw.rounded_rectangle(
            [(bx0, by0), (bx1, by1)],
            radius=_px(10, ps),
            fill=(255, 255, 255, 232),
            outline=(200, 200, 200, 255),
            width=_px(2, ps),
        )

        bar_x0 = bx0 + (box_w - bar_px) / 2.0
        bar_x1 = bar_x0 + bar_px
        bar_y1 = by1 - pad_y
        bar_y0 = bar_y1 - bar_h
        draw.rectangle([(bar_x0, bar_y0), (bar_x1, bar_y1)], fill=(17, 17, 17, 255))
        # Crisp vertical end ticks
        tick_h = bar_h + _px(6, ps)
        tick_w = _px(2, ps)
        draw.rectangle([(bar_x0, bar_y1 - tick_h), (bar_x0 + tick_w, bar_y1)], fill=(17, 17, 17, 255))
        draw.rectangle([(bar_x1 - tick_w, bar_y1 - tick_h), (bar_x1, bar_y1)], fill=(17, 17, 17, 255))

        lbl_x = bx0 + (box_w - lbl_w) / 2.0 - lbl_bbox[0]
        lbl_y = by0 + pad_y - lbl_bbox[1]
        draw.text((lbl_x, lbl_y), scale_label, fill=(17, 17, 17, 255), font=scale_font)

    # 2. Orientation badge (top-left)
    if orientation_label:
        badge_font = _load_font(max(_px(20, ps), int(round(width * 0.025))))
        b_bbox = draw.textbbox((0, 0), orientation_label, font=badge_font)
        b_w = b_bbox[2] - b_bbox[0]
        b_h = b_bbox[3] - b_bbox[1]
        b_pad_x = _px(18, ps)
        b_pad_y = _px(10, ps)
        ox0 = margin
        oy0 = margin
        ox1 = ox0 + b_w + 2 * b_pad_x
        oy1 = oy0 + b_h + 2 * b_pad_y

        draw.rounded_rectangle(
            [(ox0, oy0), (ox1, oy1)],
            radius=_px(12, ps),
            fill=(255, 255, 255, 238),
            outline=(34, 34, 34, 255),
            width=_px(2, ps),
        )
        draw.text(
            (ox0 + b_pad_x - b_bbox[0], oy0 + b_pad_y - b_bbox[1]),
            orientation_label,
            fill=(17, 17, 17, 255),
            font=badge_font,
        )

    # 3. Element legend (top-right): font size follows the image width only, so it looks
    # the same for 1 or 10 elements and at any dpi (width already scales with dpi).
    if legend_elements:
        _draw_element_legend(draw, legend_elements, width, height, margin, ps)

    composited_rgba = Image.alpha_composite(base, overlay)
    composited = composited_rgba if transparent_background else composited_rgba.convert("RGB")

    # 4. Caption footer (bottom strip)
    if caption_text:
        if transparent_background:
            final_img = Image.new("RGBA", (width, height + footer_h), (255, 255, 255, 0))
            final_img.paste(composited, (0, 0), composited)
        else:
            final_img = Image.new("RGB", (width, height + footer_h), (255, 255, 255))
            final_img.paste(composited, (0, 0))
        _draw_centered_footer(
            final_img, caption_text, footer_top=height, footer_height=footer_h, px_scale=ps
        )
    else:
        final_img = composited

    final_img.save(path, format="PNG", dpi=_dpi_pair())
    return path.resolve()


def formula_parts(structure: Structure) -> list[tuple[str, str]]:
    """Reduced formula as [(symbol, count_text)], count_text '' for 1.

    Full-Heusler 2:1:1 compositions use the X2YZ convention (Ni2MnGa, not pymatgen's
    electronegativity-sorted MnGaNi2). Y (transition metal) comes before Z (main-group
    element): the two 1-count elements are ordered by periodic-table group, then by
    electronegativity, so the result does not depend on the site order (a slab of
    Ni2MnGa used to give Ni2GaMn). Everything else uses pymatgen's standard
    electronegativity order (SrTiO3, not O3SrTi).
    """
    reduced, _ = structure.composition.get_reduced_composition_and_factor()
    amounts = sorted(float(v) for v in reduced.values())
    if len(amounts) == 3 and all(abs(x - y) < 1e-6 for x, y in zip(amounts, (1.0, 1.0, 2.0))):

        def _x2yz_key(el: Any) -> tuple[float, float, float]:
            group = getattr(el, "group", None) or 99
            x = getattr(el, "X", None)
            x = float(x) if x is not None and x == x else 99.0
            return (-float(reduced[el]), float(group), x)

        elements = sorted(reduced.elements, key=_x2yz_key)
    else:
        elements = list(reduced.elements)  # pymatgen orders by electronegativity
    parts: list[tuple[str, str]] = []
    for el in elements:
        amt = float(reduced[el])
        if abs(amt - 1.0) < 1e-6:
            parts.append((el.symbol, ""))
        else:
            txt = str(int(round(amt))) if abs(amt - round(amt)) < 1e-6 else f"{amt:g}"
            parts.append((el.symbol, txt))
    return parts


def pretty_formula(structure: Structure) -> str:
    """Plain-text reduced formula, e.g. 'Ni2MnGa'."""
    return "".join(sym + cnt for sym, cnt in formula_parts(structure))


def _format_bulk_caption(structure: Structure, summary: dict[str, Any] | None = None) -> str:
    """Build a publication-ready lattice caption string including c/a ratio for bulk structures."""
    formula = ""
    sg = ""
    if summary:
        formula = str(summary.get("formula") or summary.get("reduced_formula") or "").strip()
        sg = str(
            summary.get("spacegroup_symbol")
            or summary.get("space_group_symbol")
            or summary.get("space_group")
            or ""
        ).strip()
    if not formula or structure.is_ordered:
        formula = pretty_formula(structure)

    lat = structure.lattice
    c_over_a = (lat.c / lat.a) if lat.a > 1e-12 else 0.0
    header = f"{formula} ({sg})" if sg else formula
    return (
        f"{header} | "
        f"a={lat.a:.3f}, b={lat.b:.3f}, c={lat.c:.3f} Å | "
        f"c/a={c_over_a:.3f} | "
        f"α={lat.alpha:.1f}°, β={lat.beta:.1f}°, γ={lat.gamma:.1f}°"
    )


def _format_miller_index(miller_index: tuple[int, int, int]) -> str:
    """Format a Miller index tuple compactly as '(111)' or '(-1 1 0)'."""
    h, k, l = miller_index
    if all(0 <= idx <= 9 for idx in (h, k, l)):
        return f"({h}{k}{l})"
    return f"({h} {k} {l})"


def _format_slab_caption(
    slab: Structure,
    miller_index: tuple[int, int, int] | None = None,
) -> str:
    """Build a default publication caption string for slab views."""
    formula = pretty_formula(slab)
    lat = slab.lattice
    prefix = f"{formula} {_format_miller_index(miller_index)} slab" if miller_index else f"{formula} slab"
    return f"{prefix} | a={lat.a:.3f}, b={lat.b:.3f}, c={lat.c:.3f} Å | γ={lat.gamma:.1f}°"


def _stick_radius() -> float:
    """Stick-style radius (A): render.bond_radius x _STICK_RADIUS_FACTOR."""
    bond_radius = max(_float_setting("render.bond_radius", _BOND_RADIUS), 1e-4)
    return float(bond_radius * _STICK_RADIUS_FACTOR)


def _split_bond_segments(
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]],
    max_bond_length: float | None = None,
) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
    """Split each bond at its midpoint; return {element: (starts, ends)} of the half bonds."""
    if max_bond_length is None:
        max_bond_length = _float_setting("bonds.max_bond_length", 0.0)
    max_len = float(max_bond_length) if max_bond_length and max_bond_length > 0.0 else math.inf
    halves: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for p1_raw, p2_raw, elem1, elem2 in limit_render_bonds(render_bonds):
        p1 = np.asarray(p1_raw, dtype=float)
        p2 = np.asarray(p2_raw, dtype=float)
        length = float(np.linalg.norm(p2 - p1))
        if length < 1e-6 or length > max_len + 1e-6:
            continue
        mid = 0.5 * (p1 + p2)
        for elem, a, b in ((str(elem1), p1, mid), (str(elem2), mid, p2)):
            starts, ends = halves.setdefault(elem, ([], []))
            starts.append(a)
            ends.append(b)
    return halves


def _add_bicolor_bonds(
    plotter: pv.Plotter,
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]],
    as_lines: bool = False,
    radius: float | None = None,
    line_width: float | None = None,
) -> None:
    """Draw bonds coloured half-and-half by element (stick: cylinders, wireframe: lines)."""
    if not render_bonds:
        logger.info("No bonds to draw (isolated atoms or large lattice); bond mesh skipped.")
        return
    halves = _split_bond_segments(render_bonds)
    for elem, (starts, ends) in halves.items():
        if not starts:
            continue
        color = _element_color(elem)
        s_arr = np.asarray(starts, dtype=float)
        e_arr = np.asarray(ends, dtype=float)
        if as_lines:
            n = len(s_arr)
            pts = np.empty((2 * n, 3), dtype=float)
            pts[0::2] = s_arr
            pts[1::2] = e_arr
            lines = np.column_stack(
                [np.full(n, 2, dtype=np.int64), np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)]
            ).ravel()
            mesh = pv.PolyData(pts, lines=lines)
            width = _WIREFRAME_LINE_WIDTH * _px_scale() if line_width is None else float(line_width)
            _safe_add_mesh(plotter, mesh, color=color, line_width=width, lighting=False)
        else:
            r = _stick_radius() if radius is None else max(float(radius), 1e-4)
            mesh = _merge_cylinders(s_arr, e_arr, r)
            _safe_add_mesh(
                plotter,
                mesh,
                color=color,
                smooth_shading=True,
                ambient=0.26,
                diffuse=0.74,
                specular=0.45,
                specular_power=40,
            )


def _add_atom_mesh_with_silhouette(
    plotter: pv.Plotter,
    mesh: pv.PolyData,
    px_scale: float | None = None,
    **kwargs: Any,
) -> None:
    """Add an atom sphere mesh with crisp dark silhouette outline, falling back gracefully.

    The 1.4 px outline is a 300-dpi width, scaled by ``px_scale`` (default render.dpi / 300).
    """
    s = _px_scale() if px_scale is None else float(px_scale)
    sil_color = str(get_color_manager().scene_colors.get("silhouette", "#1F2421"))
    silhouette_cfg = {"color": sil_color, "line_width": 1.4 * s}
    if _mesh_is_empty(mesh):
        return
    try:
        plotter.add_mesh(mesh, silhouette=silhouette_cfg, **kwargs)
    except Exception:
        plotter.add_mesh(mesh, **kwargs)


def _verify_plane_polygon(
    lattice: Any,
    miller_index: tuple[int, int, int],
    d_offset: float,
    pts_cart: np.ndarray,
    tol: float = 1e-6,
) -> None:
    """Check a cut polygon exactly; valid for any (triclinic) metric.

    * every vertex satisfies ``G . r = d_offset`` with ``G = (h, k, l) @ B`` where ``B``
      is the crystallographic reciprocal basis (no 2*pi), i.e. lies on the (hkl) plane;
    * every vertex lies on the unit-cell boundary (fractional coords in [0, 1], at least
      one coordinate equal to 0 or 1);
    * the vertices are in convex order (no self-intersecting polygon).
    """
    recip = np.asarray(lattice.reciprocal_lattice_crystallographic.matrix, dtype=float)
    g_vec = np.asarray(miller_index, dtype=float) @ recip
    resid = pts_cart @ g_vec - float(d_offset)
    if np.max(np.abs(resid)) > tol * max(1.0, abs(float(d_offset))):
        raise ScientificValidationError(
            f"Plane {miller_index}: polygon vertices are off the plane "
            f"(max residual {np.max(np.abs(resid)):.2e})."
        )
    frac = np.asarray(lattice.get_fractional_coords(pts_cart), dtype=float)
    if np.any(frac < -1e-6) or np.any(frac > 1.0 + 1e-6):
        raise ScientificValidationError(f"Plane {miller_index}: polygon vertex outside the unit cell.")
    on_face = np.any((np.abs(frac) < 1e-6) | (np.abs(frac - 1.0) < 1e-6), axis=1)
    if not np.all(on_face):
        raise ScientificValidationError(f"Plane {miller_index}: polygon vertex not on a cell face.")
    n_unit = g_vec / np.linalg.norm(g_vec)
    n_pts = len(pts_cart)
    signs: list[bool] = []
    for i in range(n_pts):
        e1 = pts_cart[(i + 1) % n_pts] - pts_cart[i]
        e2 = pts_cart[(i + 2) % n_pts] - pts_cart[(i + 1) % n_pts]
        s = float(np.dot(np.cross(e1, e2), n_unit))
        if abs(s) > 1e-9:
            signs.append(s > 0.0)
    if signs and not (all(signs) or not any(signs)):
        raise ScientificValidationError(
            f"Plane {miller_index}: polygon vertices are not in convex order."
        )


def _compute_hkl_plane_mesh(
    lattice: Any,
    miller_index: tuple[int, int, int],
    d_offset: float = 1.0,
) -> tuple[pv.PolyData, pv.PolyData] | None:
    """Compute the intersection polygon (mesh + boundary edges) of a (h,k,l) plane with the unit cell."""
    h, k, l = validate_miller_index(miller_index)
    normal_frac = np.array([float(h), float(k), float(l)], dtype=float)
    # Collect intersections with the 12 unit-cell edges: h*u + k*v + l*w = d_offset
    pts_frac: list[np.ndarray] = []
    for f0, f1 in _UNIT_CELL_EDGES:
        p0 = np.asarray(f0, dtype=float)
        p1 = np.asarray(f1, dtype=float)
        v0 = float(np.dot(normal_frac, p0)) - d_offset
        v1 = float(np.dot(normal_frac, p1)) - d_offset
        if abs(v0) < 1e-9 and abs(v1) < 1e-9:
            pts_frac.extend([p0, p1])
        elif abs(v0) < 1e-9:
            pts_frac.append(p0)
        elif abs(v1) < 1e-9:
            pts_frac.append(p1)
        elif (v0 < 0.0 < v1) or (v1 < 0.0 < v0):
            t = -v0 / (v1 - v0)
            pts_frac.append(p0 + t * (p1 - p0))

    # Deduplicate points
    unique_frac: list[np.ndarray] = []
    for p in pts_frac:
        if not any(float(np.linalg.norm(p - q)) < 1e-5 for q in unique_frac):
            unique_frac.append(p)
    if len(unique_frac) < 3:
        return None

    pts_cart = np.array([lattice.get_cartesian_coords(p) for p in unique_frac], dtype=float)
    center = pts_cart.mean(axis=0)
    # Order polygon vertices counter-clockwise in the plane
    v_a = pts_cart[0] - center
    v_a = v_a / (np.linalg.norm(v_a) + 1e-12)
    n_cart = np.cross(v_a, pts_cart[1] - center)
    if np.linalg.norm(n_cart) < 1e-9 and len(pts_cart) > 2:
        n_cart = np.cross(v_a, pts_cart[2] - center)
    n_cart = n_cart / (np.linalg.norm(n_cart) + 1e-12)
    v_b = np.cross(n_cart, v_a)

    angles = [
        math.atan2(float(np.dot(p - center, v_b)), float(np.dot(p - center, v_a)))
        for p in pts_cart
    ]
    order = np.argsort(angles)
    ordered_pts = pts_cart[order]
    n_pts = len(ordered_pts)

    _verify_plane_polygon(lattice, (h, k, l), d_offset, ordered_pts)

    face = np.hstack([[n_pts], np.arange(n_pts, dtype=np.int64)])
    plane_mesh = pv.PolyData(ordered_pts, faces=face)

    line_cells: list[int] = []
    for i in range(n_pts):
        line_cells.extend([2, i, (i + 1) % n_pts])
    plane_edges = pv.PolyData(ordered_pts, lines=np.array(line_cells, dtype=np.int64))
    return plane_mesh, plane_edges


def _build_default_plane(
    lattice: Any, hkl: tuple[int, int, int]
) -> tuple[pv.PolyData, pv.PolyData] | None:
    """Polygon of the (hkl) plane at its default position inside the unit cell."""
    # h*u + k*v + l*w over the cell corners spans [d_lo, d_hi]. Use the first integer
    # plane above d_lo (keeps (111) at d=1, puts (1-10) through the origin, (-1-1-1)
    # at d=-2) and fall back to the mid-cell plane when that one only touches an edge.
    d_lo = float(sum(min(x, 0) for x in hkl))
    d_hi = float(sum(max(x, 0) for x in hkl))
    for d_off in (d_lo + 1.0, 0.5 * (d_lo + d_hi)):
        if d_lo < d_off <= d_hi:
            built = _compute_hkl_plane_mesh(lattice, hkl, d_offset=d_off)
            if built is not None:
                return built
    return None


def add_crystal_planes(
    plotter: pv.Plotter,
    structure: Structure,
    miller_indices: list[tuple[int, int, int]],
    color_manager: ColorManager | None = None,
    custom_planes: list[dict[str, Any]] | None = None,
    offsets: list[float | None] | None = None,
) -> None:
    """Render semi-transparent (hkl) crystal planes with emphasized edges using the active theme.

    ``offsets[i]`` (optional) puts plane i at ``h*x + k*y + l*z = offsets[i]`` (fractional
    coords, i.e. that many d_hkl from the origin); None keeps the default position. An
    explicit offset whose plane does not cut the unit cell raises ValueError.
    """
    if not miller_indices:
        return
    cm = color_manager or get_color_manager()
    plane_pairs: list[tuple[pv.PolyData, pv.PolyData]] = []
    kept_custom: list[dict[str, Any]] = []
    for idx, hkl_raw in enumerate(miller_indices):
        hkl = validate_miller_index(hkl_raw)
        d_off = offsets[idx] if offsets is not None and idx < len(offsets) else None
        if d_off is None:
            built = _build_default_plane(structure.lattice, hkl)
        else:
            built = _compute_hkl_plane_mesh(structure.lattice, hkl, d_offset=float(d_off))
            if built is None:
                d_lo = sum(min(x, 0) for x in hkl)
                d_hi = sum(max(x, 0) for x in hkl)
                raise ValueError(
                    f"Plane {_format_miller_index(hkl)} at d={d_off:g} does not cut the unit cell "
                    f"(use {d_lo} < d < {d_hi})."
                )
        if built is not None:
            plane_pairs.append(built)
            if custom_planes is not None:
                kept_custom.append(custom_planes[idx] if idx < len(custom_planes) else {})
    if custom_planes is not None:
        custom_planes = kept_custom

    styles = cm.resolve_plane_styles(num_planes=len(plane_pairs), custom_planes=custom_planes)
    px_scale = _px_scale()
    for (mesh, edges), st in zip(plane_pairs, styles):
        _safe_add_mesh(
            plotter,
            mesh,
            color=st.fill_color,
            opacity=st.fill_opacity,
            specular=0.05,
            smooth_shading=True,
        )
        _safe_add_mesh(
            plotter,
            edges,
            color=st.edge_color,
            opacity=st.edge_opacity,
            line_width=float(st.line_width) * px_scale,
        )


def _parse_plane_spec(spec: Any) -> dict[str, Any]:
    """One plane: 'h k l[@d][:color][:opacity]' or {'hkl': [h, k, l], 'd', 'color', 'opacity'}."""
    if isinstance(spec, dict):
        hkl_raw = spec.get("hkl", spec.get("miller_index"))
        if isinstance(hkl_raw, str):
            parsed = _parse_plane_spec(hkl_raw)
            hkl_raw = parsed["hkl"]
        d_raw, color, opacity = spec.get("d"), spec.get("color"), spec.get("opacity")
    elif isinstance(spec, (list, tuple)) and len(spec) == 3:
        hkl_raw, d_raw, color, opacity = spec, None, None, None
    elif isinstance(spec, str):
        parts = spec.strip().split(":")
        if len(parts) > 3:
            raise ValueError(f"Plane '{spec}': use 'h k l[@d][:color][:opacity]'.")
        head = parts[0]
        d_raw = None
        if "@" in head:
            head, d_raw = head.split("@", 1)
        head = head.strip().strip("()")
        if re.search(r"[\s,]", head):
            tokens = [t for t in re.split(r"[\s,]+", head) if t]
        else:  # compact '111' / '1-10'
            tokens = re.findall(r"-?\d", head)
        hkl_raw = tokens
        color = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        opacity = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    else:
        raise ValueError(f"Plane {spec!r}: use 'h k l[@d][:color][:opacity]'.")
    try:
        hkl = validate_miller_index(hkl_raw)
    except ValueError as exc:
        raise ValueError(f"Plane {spec!r}: {exc}") from exc
    d_val = None if d_raw in (None, "") else float(d_raw)
    if d_val is not None and not math.isfinite(d_val):
        raise ValueError(f"Plane {spec!r}: d must be a finite number.")
    if color is not None:
        try:
            pv.Color(color)
        except ValueError as exc:
            raise ValueError(f"Plane {spec!r}: unknown colour {color!r}.") from exc
    op_val = None if opacity in (None, "") else float(opacity)
    if op_val is not None and not 0.0 <= op_val <= 1.0:
        raise ValueError(f"Plane {spec!r}: opacity must be between 0 and 1.")
    return {"hkl": hkl, "d": d_val, "color": color, "opacity": op_val}


def normalize_planes(planes: Any) -> list[dict[str, Any]] | None:
    """Planes to draw on bulk views (VESTA "Lattice planes"); None / [] -> None.

    Each item is 'h k l[@d][:color][:opacity]' (e.g. "1 1 1@0.5:#ff0000:0.4", "111",
    "-1 1 0") or a dict {'hkl': [1, 1, 1], 'd': 0.5, 'color': 'red', 'opacity': 0.4}.
    ``d`` is the plane h*x + k*y + l*z = d in fractional coordinates, i.e. d * d_hkl
    from the origin (VESTA's convention); omitted = the default position (d = 1 for
    (111)). Colour / opacity default to the theme's plane palette.
    """
    if planes is None:
        return None
    if isinstance(planes, (str, dict)):
        planes = [planes]
    elif isinstance(planes, (list, tuple)) and len(planes) == 3 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in planes
    ):
        planes = [planes]
    out = [_parse_plane_spec(p) for p in planes]
    return out or None


def _darker_hex(color: str, factor: float = 0.6) -> str:
    r, g, b = pv.Color(color).float_rgb
    return "#{:02X}{:02X}{:02X}".format(*(int(round(255 * c * factor)) for c in (r, g, b)))


def add_bulk_planes(plotter: pv.Plotter, structure: Structure, planes: Any) -> None:
    """Draw `normalize_planes` specs in the unit cell (polygons checked by `_verify_plane_polygon`)."""
    specs = normalize_planes(planes)
    if not specs:
        return
    custom: list[dict[str, Any]] = []
    for sp in specs:
        override: dict[str, Any] = {}
        if sp["color"] is not None:
            override["fill"] = sp["color"]
            override["edge"] = _darker_hex(sp["color"])
        if sp["opacity"] is not None:
            override["opacity"] = sp["opacity"]
        custom.append(override)
    add_crystal_planes(
        plotter,
        structure,
        [sp["hkl"] for sp in specs],
        custom_planes=custom,
        offsets=[sp["d"] for sp in specs],
    )


# ---- atom vectors (magnetic moments) ------------------------------------------------

_VECTOR_SCALE: float = 1.0  # Å of arrow per mu_B
_VECTOR_RADIUS: float = 0.08  # shaft radius, Å
_VECTOR_COLOR: str = "#E4572E"


def normalize_vector_style(
    scale: float | None = None, color: str | None = None, radius: float | None = None
) -> tuple[float, str, float]:
    """(Å per unit, colour hex, shaft radius Å); None -> settings vectors.* or the defaults."""
    scale = _float_setting("vectors.scale", _VECTOR_SCALE) if scale is None else float(scale)
    radius = _float_setting("vectors.radius", _VECTOR_RADIUS) if radius is None else float(radius)
    color = str(_setting("vectors.color", _VECTOR_COLOR) if color is None else color)
    if not (scale > 0.0 and math.isfinite(scale)):
        raise ValueError(f"vector_scale must be > 0 (got {scale}).")
    if not (radius > 0.0 and math.isfinite(radius)):
        raise ValueError(f"vector_radius must be > 0 (got {radius}).")
    try:
        color = pv.Color(color).hex_rgb
    except Exception as exc:
        raise ValueError(f"vector_color {color!r} is not a colour.") from exc
    return scale, color, radius


def vector_arrows(
    structure: Structure,
    vectors: Any,
    basis: str | None = None,
    scale: float | None = None,
    include_boundary_atoms: bool = True,
    frac_range: Any = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(start, end) in Å of one arrow per drawn atom that has a vector (boundary / range
    images included), centred on the atom; length = scale * |vector| (see core.vectors)."""
    from shido.core.vectors import resolve_site_vectors

    site_vecs = resolve_site_vectors(structure, vectors, basis)
    if not site_vecs:
        return []
    scale, _, _ = normalize_vector_style(scale=scale)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, cart in _drawn_site_images(structure, include_boundary_atoms, frac_range=frac_range):
        vec = site_vecs.get(idx)
        if vec is None:
            continue
        half = 0.5 * scale * np.asarray(vec, dtype=float)
        out.append((cart - half, cart + half))
    return out


def build_arrow_mesh(
    arrows: list[tuple[np.ndarray, np.ndarray]], radius: float, resolution: int = 20
) -> pv.PolyData | None:
    """Merged cylinder shafts + cone heads (head length <= 40 % of the arrow, radius 2.2x)."""
    parts: list[pv.PolyData] = []
    for start, end in arrows:
        start = np.asarray(start, dtype=float)
        d = np.asarray(end, dtype=float) - start
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        u = d / length
        head = min(0.4 * length, 4.0 * radius)
        shaft = length - head
        if shaft > 1e-6:
            parts.append(
                pv.Cylinder(center=start + u * (0.5 * shaft), direction=u, radius=radius,
                            height=shaft, resolution=resolution, capping=True).triangulate()
            )
        parts.append(
            pv.Cone(center=start + u * (shaft + 0.5 * head), direction=u, height=head,
                    radius=2.2 * radius, resolution=resolution, capping=True).triangulate()
        )
    if not parts:
        return None
    return pv.merge(parts) if len(parts) > 1 else parts[0]


def add_vector_arrows(
    plotter: pv.Plotter,
    arrows: list[tuple[np.ndarray, np.ndarray]],
    color: str,
    radius: float,
) -> None:
    mesh = build_arrow_mesh(arrows, radius)
    if mesh is None:
        return
    plotter.add_mesh(
        mesh, color=color, smooth_shading=True, ambient=0.3, diffuse=0.7, specular=0.4,
        specular_power=30,
    )


def apply_studio_lighting(
    plotter: pv.Plotter,
    parallel_scale: float = 5.0,
    enable_ssao: bool = True,
) -> None:
    """Apply upper-left studio 3-point lighting, SSAO depth shading, and AA."""
    try:
        plotter.remove_all_lights()
        # Key light: upper-left studio angle (creates crisp 3D gradients on spheres & polyhedra)
        key_light = pv.Light(
            light_type="camera light",
            position=(-0.55, 0.65, 1.0),
            intensity=0.85,
            color="white",
        )
        # Fill light: soft lower-right bounce (prevents shaded facets from turning black)
        fill_light = pv.Light(
            light_type="camera light",
            position=(0.70, -0.45, 0.50),
            intensity=0.34,
            color="white",
        )
        # Rim accent: subtle back-edge separation for overlapping atoms
        rim_light = pv.Light(
            light_type="camera light",
            position=(0.0, 0.20, -0.85),
            intensity=0.18,
            color="white",
        )
        plotter.add_light(key_light)
        plotter.add_light(fill_light)
        plotter.add_light(rim_light)
    except Exception:
        pass

    if enable_ssao:
        try:
            ssao_radius = max(0.8, min(float(parallel_scale) * 0.14, 3.0))
            plotter.enable_ssao(
                radius=ssao_radius,
                bias=0.02,
                kernel_size=128,
                blur=True,
            )
        except Exception:
            pass

    try:
        plotter.enable_anti_aliasing("fxaa")
    except Exception:
        pass


IMAGE_FORMATS: tuple[str, ...] = ("png", "tiff", "jpeg", "pdf", "eps", "svg")
_IMAGE_FORMAT_ALIASES: dict[str, str] = {"tif": "tiff", "jpg": "jpeg"}
_IMAGE_EXT: dict[str, str] = {
    "png": ".png", "tiff": ".tiff", "jpeg": ".jpg", "pdf": ".pdf", "eps": ".eps", "svg": ".svg"
}


def normalize_image_formats(formats: Any) -> list[str]:
    """Normalize "tiff,pdf" / ["tiff", "pdf"] -> ['tiff', 'pdf'] (extra formats; png dropped since
    it is always written). None / "" / "png" -> []."""
    if formats is None:
        return []
    items = formats.split(",") if isinstance(formats, str) else list(formats)
    out: list[str] = []
    for raw in items:
        for part in str(raw).split(","):
            key = part.strip().lower().lstrip(".")
            if not key:
                continue
            key = _IMAGE_FORMAT_ALIASES.get(key, key)
            if key not in IMAGE_FORMATS:
                raise ValueError(
                    f"image format {part.strip()!r}: choose from {', '.join(IMAGE_FORMATS)}."
                )
            if key != "png" and key not in out:
                out.append(key)
    return out


def _flatten_white(im: Image.Image) -> Image.Image:
    """RGB copy; a transparent background becomes white (JPEG / PDF / EPS have no alpha)."""
    if im.mode in ("RGBA", "LA", "P"):
        rgba = im.convert("RGBA")
        flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(flat, rgba).convert("RGB")
    return im.convert("RGB")


def export_image_formats(png_path: str | Path, formats: Any) -> dict[str, str]:
    """Write the finished PNG again as each requested format, same stem, same folder.

    TIFF keeps the alpha channel (LZW); JPEG / PDF / EPS are flattened onto white.
    PDF, EPS and SVG are NOT vector drawings: the rendered bitmap is embedded at the
    render.dpi print size (the 3D scene is a raster from VTK and the labels / legend /
    scale bar are drawn on it with PIL). Returns {format: path}.
    """
    src = Path(png_path)
    fmts = normalize_image_formats(formats)
    out: dict[str, str] = {}
    if not fmts:
        return out
    dpi = _dpi_pair()
    with Image.open(src) as im:
        im.load()
        for fmt in fmts:
            dst = src.with_suffix(_IMAGE_EXT[fmt])
            if fmt == "tiff":
                im.save(dst, format="TIFF", dpi=dpi, compression="tiff_lzw")
            elif fmt == "jpeg":
                _flatten_white(im).save(dst, format="JPEG", dpi=dpi, quality=95, subsampling=0)
            elif fmt == "pdf":
                _flatten_white(im).save(dst, format="PDF", resolution=float(dpi[0]))
            elif fmt == "eps":
                _flatten_white(im).save(dst, format="EPS")
                # PIL writes 1 px = 1 pt; rescale to the render.dpi print size.
                w, h = im.size
                pw = w * 72.0 / float(dpi[0])
                ph = h * 72.0 / float(dpi[1])
                data = dst.read_bytes()
                data = data.replace(
                    f"%%BoundingBox: 0 0 {w} {h}".encode(),
                    f"%%BoundingBox: 0 0 {math.ceil(pw)} {math.ceil(ph)}".encode(),
                    1,
                ).replace(f"\n{w} {h} scale".encode(), f"\n{pw:.3f} {ph:.3f} scale".encode(), 1)
                dst.write_bytes(data)
            elif fmt == "svg":
                buf = io.BytesIO()
                im.save(buf, format="PNG", dpi=dpi)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                w, h = im.size
                w_mm = w / float(dpi[0]) * 25.4
                h_mm = h / float(dpi[1]) * 25.4
                dst.write_text(
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                    f'width="{w_mm:.2f}mm" height="{h_mm:.2f}mm" viewBox="0 0 {w} {h}">\n'
                    f'  <image width="{w}" height="{h}" '
                    f'xlink:href="data:image/png;base64,{b64}"/>\n'
                    "</svg>\n",
                    encoding="utf-8",
                )
            out[fmt] = str(dst.resolve())
    return out


# ---- volumetric data: isosurfaces ----------------------------------------------------

_ISO_COLOR: str = "#F2C200"  # positive level (VESTA-like yellow)
_ISO_COLOR_NEGATIVE: str = "#2F9BD6"  # negative level
_ISO_OPACITY: float = 0.55


def _parse_colour(value: Any, what: str) -> str:
    try:
        return pv.Color(str(value).strip()).hex_rgb
    except Exception as exc:
        raise ValueError(f"{what} {value!r} is not a colour.") from exc


def normalize_iso_style(
    color: Any = None, color_negative: Any = None, opacity: float | None = None
) -> tuple[list[str], str, float]:
    """(positive colours (one per isovalue, cycled), negative colour, opacity 0-1);
    None -> settings isosurface.* or the defaults. ``color`` may be 'c1,c2'."""
    color = _setting("isosurface.color", _ISO_COLOR) if color is None else color
    color_negative = (
        _setting("isosurface.color_negative", _ISO_COLOR_NEGATIVE)
        if color_negative is None else color_negative
    )
    opacity = _float_setting("isosurface.opacity", _ISO_OPACITY) if opacity is None else float(opacity)
    if not (0.0 < opacity <= 1.0):
        raise ValueError(f"iso_opacity must be in (0, 1] (got {opacity}).")
    items = color if isinstance(color, (list, tuple)) else [t for t in str(color).split(",") if t.strip()]
    colours = [_parse_colour(c, "iso_color") for c in items] or [_ISO_COLOR]
    return colours, _parse_colour(color_negative, "iso_color_negative"), opacity


def load_volume(volume: Any) -> Any:
    """A `core.volumetric.VolumetricGrid` from a path (or the grid itself)."""
    from shido.core.volumetric import VolumetricGrid, read_volume

    if isinstance(volume, VolumetricGrid):
        return volume
    return read_volume(volume)


def isosurface_meshes(
    structure: Structure,
    volume: Any,
    isovalue: Any = None,
    both_signs: bool = False,
    color: Any = None,
    color_negative: Any = None,
) -> list[tuple[Any, str, float]]:
    """[(mesh in Å, colour, level), ...] of the isosurfaces of ``volume`` placed on
    ``structure``'s cell (one per level; ``both_signs`` adds -level for each level)."""
    from shido.core.volumetric import isosurface, lattice_mismatch, normalize_isovalues

    grid = load_volume(volume)
    warn = lattice_mismatch(structure.lattice, grid.lattice)
    if warn:
        logger.warning(warn)
    levels = normalize_isovalues(isovalue, grid)
    colours, neg_colour, _ = normalize_iso_style(color, color_negative)
    out: list[tuple[Any, str, float]] = []
    for i, level in enumerate(levels):
        signed = [level, -level] if both_signs else [level]
        for lv in signed:
            mesh = isosurface(grid, lv, lattice_matrix=structure.lattice.matrix)
            if mesh is None:
                logger.warning(
                    "isovalue %g is outside the data range %.4g .. %.4g; no surface.",
                    lv, float(grid.data.min()), float(grid.data.max()),
                )
                continue
            out.append((mesh, neg_colour if lv < 0 else colours[i % len(colours)], lv))
    return out


def add_isosurfaces(plotter: pv.Plotter, meshes: list[tuple[Any, str, float]], opacity: float) -> None:
    for mesh, colour, _ in meshes:
        plotter.add_mesh(
            mesh, color=colour, opacity=opacity, smooth_shading=True, ambient=0.3,
            diffuse=0.7, specular=0.25, specular_power=20,
        )


# ---------------------------------------------------------------------------
# Batch 16: displacement (thermal) ellipsoids, ORTEP / VESTA style
# ---------------------------------------------------------------------------
ELLIPSOID_NO_ADP_RADIUS: float = 0.15  # Å, small sphere for atoms without ADPs
_ELLIPSE_LINE_RADIUS: float = 0.012  # Å, tube radius of the principal ellipses
_ELLIPSE_LINE_COLOR: str = "#1A1A1A"
_ELLIPSOID_BOND_FACTOR: float = 0.45  # bond radius in the ellipsoid style = render.bond_radius x this


def resolve_adp_tensors(structure: Structure, adp: Any) -> tuple[list[np.ndarray | None], list[str]]:
    """U_cart [Å²] per site of ``structure`` (None = no ADP) and notes.

    ``adp``: a CIF path, a ``core.adp.read_cif_adp`` dict, or an already resolved list
    (one 3×3 array or None per site). None -> every site without ADP (small spheres).
    """
    n = len(structure)
    if adp is None:
        return [None] * n, ["ellipsoid style without a CIF with ADPs: atoms drawn as small spheres."]
    if isinstance(adp, (list, tuple)) and len(adp) == n and all(
        t is None or np.shape(t) == (3, 3) for t in adp
    ):
        return [None if t is None else np.asarray(t, dtype=float) for t in adp], []
    from shido.core.adp import site_displacement_tensors

    if isinstance(adp, dict):
        tensors, _, notes = site_displacement_tensors(structure, adp=adp)
    else:
        tensors, _, notes = site_displacement_tensors(structure, cif_path=adp)
    return tensors, notes


def ellipsoid_shapes(
    structure: Structure,
    tensors: list[np.ndarray | None],
    probability: float | None = None,
    include_boundary_atoms: bool = True,
    frac_range: Any = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """One shape per drawn atom (same order as `_collect_atoms_to_draw`) and warnings.

    Each: ``center``, ``semi`` (3 semi-axes, Å), ``axes`` (columns = directions),
    ``kind`` 'ellipsoid' / 'npd' (non-positive-definite: sphere of c(p)·√U_eq, or the
    small sphere if U_eq <= 0) / 'none' (no ADP: ``ELLIPSOID_NO_ADP_RADIUS`` sphere),
    ``radius`` (largest semi-axis) and ``site``. Semi-axes = c(p)·√(eigenvalue of U_cart),
    c(50 %) = 1.5382 (`core.adp.probability_scale`).
    """
    from shido.core.adp import ellipsoid_axes, probability_scale

    p = 50.0 if probability is None else float(probability)
    scale = probability_scale(p)
    per_site: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    npd_sites: list[int] = []
    for idx, u in enumerate(tensors):
        if u is None:
            per_site[idx] = ("none", np.full(3, ELLIPSOID_NO_ADP_RADIUS), np.eye(3))
            continue
        semi, axes, ok = ellipsoid_axes(u, p)
        if ok:
            per_site[idx] = ("ellipsoid", np.asarray(semi, dtype=float), np.asarray(axes, dtype=float))
        else:
            npd_sites.append(idx)
            u_eq = float(np.trace(u) / 3.0)
            r = scale * math.sqrt(u_eq) if u_eq > 0.0 else ELLIPSOID_NO_ADP_RADIUS
            per_site[idx] = ("npd", np.full(3, r), np.eye(3))
    warns: list[str] = []
    if npd_sites:
        names = sorted({str(getattr(structure[i], "label", None) or _site_symbol(structure[i])) for i in npd_sites})
        warns.append(
            "Non-positive-definite ADPs (no ellipsoid, drawn as spheres of sqrt(U_eq)): " + ", ".join(names)
        )
    n_none = sum(1 for t in tensors if t is None)
    if n_none and n_none < len(tensors):
        warns.append(f"{n_none} sites without ADPs drawn as small spheres ({ELLIPSOID_NO_ADP_RADIUS} Å).")
    for w in warns:
        logger.warning("ellipsoid style: %s", w)
    shapes = []
    for idx, cart in _drawn_site_images(structure, include_boundary_atoms, 1e-3, frac_range):
        kind, semi, axes = per_site[idx]
        shapes.append({
            "site": idx, "center": np.asarray(cart, dtype=float), "kind": kind,
            "semi": semi, "axes": axes, "radius": float(np.max(semi)),
        })
    return shapes, warns


def build_ellipsoid_mesh(center: np.ndarray, semi: np.ndarray, axes: np.ndarray, resolution: int = 36) -> pv.PolyData:
    """Ellipsoid surface: unit sphere scaled by ``semi`` along the columns of ``axes``."""
    sph = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)
    sph.point_data.clear()
    ax = np.array(axes, dtype=float)
    if np.linalg.det(ax) < 0:  # eigh may return a left-handed set: keep the faces facing out
        ax[:, 2] = -ax[:, 2]
    m = ax * np.asarray(semi, dtype=float)  # columns scaled
    sph.points = np.asarray(sph.points) @ m.T + np.asarray(center, dtype=float)
    return sph


def principal_ellipse_points(
    center: np.ndarray, semi: np.ndarray, axes: np.ndarray, n: int = 72
) -> list[np.ndarray]:
    """The 3 principal ellipses (sections through two principal axes), closed (n+1 points each)."""
    t = np.linspace(0.0, 2.0 * math.pi, n + 1)
    out = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pts = (
            np.asarray(center, dtype=float)
            + np.outer(semi[i] * np.cos(t), axes[:, i])
            + np.outer(semi[j] * np.sin(t), axes[:, j])
        )
        out.append(pts)
    return out


def add_ellipsoids(
    plotter: pv.Plotter,
    shapes: list[dict[str, Any]],
    elements: list[str],
    angstrom_per_pixel: float,
    px_scale: float,
    add_atoms: Any,
) -> None:
    """Ellipsoid / sphere meshes (grouped by element) + principal ellipse lines."""
    by_elem: dict[str, list[pv.PolyData]] = {}
    lines: list[pv.PolyData] = []
    for sh, elem in zip(shapes, elements):
        res = _sphere_resolution(sh["radius"], angstrom_per_pixel, px_scale=px_scale)
        by_elem.setdefault(elem, []).append(build_ellipsoid_mesh(sh["center"], sh["semi"], sh["axes"], res))
        if sh["kind"] == "ellipsoid":
            for pts in principal_ellipse_points(sh["center"], sh["semi"], sh["axes"]):
                lines.append(pv.lines_from_points(pts).tube(radius=_ELLIPSE_LINE_RADIUS, n_sides=8))
    for elem, meshes in by_elem.items():
        merged = pv.merge(meshes) if len(meshes) > 1 else meshes[0]
        add_atoms(
            merged, color=_element_color(elem), smooth_shading=True,
            ambient=0.26, diffuse=0.74, specular=0.55, specular_power=45,
        )
    if lines:
        plotter.add_mesh(pv.merge(lines) if len(lines) > 1 else lines[0], color=_ELLIPSE_LINE_COLOR,
                         smooth_shading=True, ambient=0.4, diffuse=0.6)


# ---------------------------------------------------------------------------
# Batch 21: overlay more structures in the same picture (molecule on a surface,
# grain boundary). Each overlay is moved by r' = R (r - c) + c + t.
# ---------------------------------------------------------------------------

_OVERLAY_SHIFT_UNITS = {"angstrom": "angstrom", "a": "angstrom", "ang": "angstrom", "cart": "angstrom",
                        "cartesian": "angstrom", "frac": "frac", "fractional": "frac", "crystal": "frac"}


def _numbers(value: Any, what: str) -> list[float]:
    if isinstance(value, str):
        parts = [p for p in value.replace(",", " ").split() if p]
    else:
        parts = list(np.asarray(value, dtype=float).ravel())
    try:
        return [float(p) for p in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} {value!r}: numbers expected.") from exc


def rotation_matrix(axis: Any, angle_deg: float) -> np.ndarray:
    """Right-handed rotation by ``angle_deg`` about the Cartesian ``axis`` (Rodrigues)."""
    k = np.asarray(axis, dtype=float).ravel()
    n = float(np.linalg.norm(k))
    if k.size != 3 or n < 1e-12:
        raise ValueError(f"rotation axis {axis!r} must be 3 numbers, not all zero.")
    k = k / n
    th = math.radians(float(angle_deg))
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def parse_overlay_rotation(rotate: Any = None, matrix: Any = None) -> np.ndarray:
    """3x3 rotation from ``rotate`` ("ax ay az angle_deg", 4 numbers) and/or ``matrix``
    (9 numbers / 3x3, must be orthogonal; applied first). None -> identity."""
    rot = np.eye(3)
    if matrix is not None:
        vals = _numbers(matrix, "overlay matrix")
        if len(vals) != 9:
            raise ValueError(f"overlay matrix needs 9 numbers, got {len(vals)}.")
        m = np.asarray(vals, dtype=float).reshape(3, 3)
        if not np.allclose(m.T @ m, np.eye(3), atol=1e-3):
            raise ValueError("overlay matrix must be a rotation (orthogonal, M^T M = I).")
        rot = m
    if rotate is not None and not (isinstance(rotate, str) and not rotate.strip()):
        vals = _numbers(rotate, "overlay rotate")
        if len(vals) == 9:
            return parse_overlay_rotation(None, vals) @ rot
        if len(vals) != 4:
            raise ValueError(f"overlay rotate {rotate!r}: 'axis_x axis_y axis_z angle_deg' expected.")
        rot = rotation_matrix(vals[:3], vals[3]) @ rot
    return rot


def normalize_overlays(overlays: Any) -> list[dict[str, Any]]:
    """Overlay specs -> list of dicts with a loaded ``structure``.

    An item is a Structure, a path (CIF / POSCAR / XSF / .vesta, read with
    `analyzer.load_structure_robust`) or a dict: ``structure`` or ``path``, ``shift``
    (3 numbers, default 0), ``shift_unit`` 'angstrom' (default, Cartesian Å) or 'frac'
    (fractions of the BASE cell), ``rotate`` "ax ay az angle_deg" (Cartesian axis,
    degrees), ``matrix`` (3x3 rotation), ``about`` 'origin' (default) or 'center'
    (centre of the overlay cell), ``opacity`` 0-1 (default 1), ``color`` (None = element
    colours; one colour paints the whole overlay), ``bonds`` (bond specs as in
    `analyzer.compute_bonds`), ``render_bonds`` (ready-made Cartesian bonds), ``show_cell``.
    """
    if overlays is None:
        return []
    if isinstance(overlays, (Structure, str, Path, dict)):
        overlays = [overlays]
    out: list[dict[str, Any]] = []
    for item in overlays:
        if isinstance(item, (Structure, str, Path)):
            item = {"structure": item}
        elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
            item = {**item[1], "structure": item[0]}
        elif not isinstance(item, dict):
            raise ValueError(f"overlay {item!r}: a Structure, a path or a dict expected.")
        if item.get("_normalized"):
            out.append(item)
            continue
        src = item.get("structure", item.get("path"))
        if isinstance(src, (str, Path)):
            from shido.analyzer import load_structure_robust

            loaded = load_structure_robust(Path(src))
            struct = loaded[0] if isinstance(loaded, tuple) else loaded
            name = Path(src).name
        elif isinstance(src, Structure):
            struct, name = src, item.get("name") or src.composition.reduced_formula
        else:
            raise ValueError("overlay needs 'structure' (Structure or path) or 'path'.")
        shift = _numbers(item.get("shift") if item.get("shift") is not None else [0, 0, 0], "overlay shift")
        if len(shift) != 3:
            raise ValueError(f"overlay shift needs 3 numbers, got {len(shift)}.")
        unit = _OVERLAY_SHIFT_UNITS.get(str(item.get("shift_unit") or "angstrom").strip().lower())
        if unit is None:
            raise ValueError(f"overlay shift_unit {item.get('shift_unit')!r}: 'angstrom' or 'frac'.")
        about = str(item.get("about") or "origin").strip().lower()
        if about not in ("origin", "center", "centre"):
            raise ValueError(f"overlay about {about!r}: 'origin' or 'center'.")
        opacity = 1.0 if item.get("opacity") is None else float(item["opacity"])
        if not 0.0 < opacity <= 1.0:
            raise ValueError(f"overlay opacity {opacity} must be in (0, 1].")
        color = item.get("color")
        out.append({
            "_normalized": True,
            "structure": struct,
            "name": name,
            "rotation": parse_overlay_rotation(item.get("rotate"), item.get("matrix")),
            "shift": np.asarray(shift, dtype=float),
            "shift_unit": unit,
            "about": "center" if about.startswith("cent") else "origin",
            "opacity": opacity,
            "color": _parse_colour(color, "overlay color") if color else None,
            "bonds": item.get("bonds"),
            "render_bonds": item.get("render_bonds"),
            "show_cell": bool(item.get("show_cell", False)),
        })
    return out


def overlay_transform(spec: dict[str, Any], base_structure: Structure) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, t, c) of a normalized overlay: r' = R (r - c) + c + t (Cartesian Å)."""
    shift = np.asarray(spec["shift"], dtype=float)
    t = base_structure.lattice.get_cartesian_coords(shift) if spec["shift_unit"] == "frac" else shift
    c = (
        spec["structure"].lattice.get_cartesian_coords([0.5, 0.5, 0.5])
        if spec["about"] == "center" else np.zeros(3)
    )
    return np.asarray(spec["rotation"], dtype=float), np.asarray(t, dtype=float), np.asarray(c, dtype=float)


def apply_overlay_transform(points: Any, rotation: np.ndarray, shift: np.ndarray, center: np.ndarray) -> np.ndarray:
    """r' = R (r - c) + c + t for an (N, 3) or (3,) array."""
    p = np.asarray(points, dtype=float)
    return (p - center) @ np.asarray(rotation).T + center + shift


def overlay_geometry(
    base_structure: Structure,
    spec: dict[str, Any],
    include_boundary_atoms: bool = True,
    style: str | None = None,
) -> dict[str, Any]:
    """Moved atoms [(cart, species_occs, radius)], bonds and cell edges of one overlay."""
    style = normalize_style(style)
    s = spec["structure"]
    validate_structure(s)
    rot, t, c = overlay_transform(spec, base_structure)
    atom_style = "ball-stick" if style == "ellipsoid" else style
    raw = _collect_atoms_to_draw(s, include_boundary_atoms=include_boundary_atoms, style=atom_style)
    atoms = [(apply_overlay_transform(p, rot, t, c), occ, r) for p, occ, r in raw]
    bonds: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    if style != "space-filling":
        rb = spec.get("render_bonds")
        if rb is None:
            from shido.analyzer import compute_bonds

            _, rb, _ = compute_bonds(s, spec.get("bonds"))
        bonds = [
            (apply_overlay_transform(p, rot, t, c), apply_overlay_transform(q, rot, t, c), e1, e2)
            for p, q, e1, e2 in (rb or [])
        ]
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    if spec.get("show_cell"):
        corners = list(itertools.product([0.0, 1.0], repeat=3))
        for i, a in enumerate(corners):
            for b in corners[i + 1:]:
                if sum(x != y for x, y in zip(a, b)) == 1:
                    edges.append(tuple(apply_overlay_transform(  # type: ignore[arg-type]
                        s.lattice.get_cartesian_coords([a, b]), rot, t, c)))
    return {"atoms": atoms, "bonds": bonds, "cell_edges": edges}


def add_overlay(plotter: pv.Plotter, spec: dict[str, Any], geom: dict[str, Any], style: str,
                angstrom_per_pixel: float, px_scale: float) -> None:
    """Draw one overlay: spheres by majority element (or ``color``), single-colour bonds."""
    opacity = float(spec["opacity"])
    kw: dict[str, Any] = dict(smooth_shading=True, ambient=0.26, diffuse=0.74, specular=0.45,
                              specular_power=40)
    if opacity < 1.0:
        kw["opacity"] = opacity
    if style != "wireframe":
        groups: dict[str, tuple[list[np.ndarray], list[float]]] = {}
        for cart, occs, r in geom["atoms"]:
            sym = max(occs, key=lambda so: so[1])[0] if occs else "X"
            cs, rs = groups.setdefault(sym, ([], []))
            cs.append(cart)
            rs.append(r)
        for sym, (cs, rs) in groups.items():
            mesh = _merge_spheres(cs, rs, _sphere_resolution(max(rs), angstrom_per_pixel, px_scale=px_scale))
            _safe_add_mesh(plotter, mesh, color=spec["color"] or _element_color(sym), **kw)
    if geom["bonds"]:
        radius = {"stick": _stick_radius(), "wireframe": 0.04}.get(
            style, _float_setting("render.bond_radius", _BOND_RADIUS)
            * (_ELLIPSOID_BOND_FACTOR if style == "ellipsoid" else 1.0))
        pairs = [(np.asarray(p), np.asarray(q)) for p, q, _, _ in limit_render_bonds(geom["bonds"])
                 if float(np.linalg.norm(np.asarray(q) - np.asarray(p))) > 1e-6]
        if pairs:
            mesh = _merge_cylinders(np.array([p for p, _ in pairs]), np.array([q for _, q in pairs]), radius)
            _safe_add_mesh(plotter, mesh, color=spec["color"] or str(get_theme().get("bond", "#B8C0CC")), **kw)
    if geom["cell_edges"]:
        lines = [pv.Line(p, q) for p, q in geom["cell_edges"]]
        _safe_add_mesh(plotter, pv.merge(lines).tube(radius=0.03), color="#7A7A7A",
                       **({"opacity": opacity} if opacity < 1.0 else {}))


# ---------------------------------------------------------------------------
# print mode (render_single_view(print_width_mm=...)): finished size, exact pt
# ---------------------------------------------------------------------------

def print_annotation_style(
    width_mm: float,
    height_mm: float | None = None,
    dpi: float | None = None,
    fonts: Any = None,
    text_pt: float | None = None,
    line_pt: float | None = None,
) -> dict[str, Any]:
    """Pixel sizes for a view printed ``width_mm`` x ``height_mm`` at ``dpi``.

    Text and line pixels are rounded up, so the drawn size is never below the asked pt.
    """
    from shido.core.figure_layout import find_font

    d = float(dpi) if dpi else float(_render_dpi())
    if d <= 0 or float(width_mm) <= 0:
        raise ValueError("print_width_mm and print_dpi must be positive.")
    h_mm = float(height_mm) if height_mm else float(width_mm)
    tpt = float(text_pt) if text_pt else 7.0
    lpt = float(line_pt) if line_pt else 0.5
    if isinstance(fonts, str):
        fonts = [fonts]
    font_path, font_name = find_font(list(fonts or ["Arial", "Helvetica"]))
    text_px = max(4, math.ceil(tpt / 72.0 * d - 1e-6))
    line_px = max(1, math.ceil(lpt / 72.0 * d - 1e-6))
    return {
        "dpi": d,
        "w_px": max(16, int(round(float(width_mm) / 25.4 * d))),
        "h_px": max(16, int(round(h_mm / 25.4 * d))),
        "font_path": font_path,
        "font_name": font_name,
        "text_px": text_px,
        "line_px": line_px,
        "text_pt": text_px / d * 72.0,
        "line_pt": line_px / d * 72.0,
    }


def structure_view_aspect(
    structure: Structure,
    camera_dir: Any,
    view_up: Any,
    include_boundary_atoms: bool = True,
    style: str | None = None,
) -> float:
    """Width / height of what `render_single_view` frames from this direction (atoms with
    radii and the cell outline, same rule as `_compute_parallel_framing`)."""
    cam = np.asarray(camera_dir, dtype=float)
    cam = cam / max(float(np.linalg.norm(cam)), 1e-12)
    up = _orthonormal_up(cam, np.asarray(view_up, dtype=float))
    atoms = _collect_atoms_to_draw(structure, include_boundary_atoms=include_boundary_atoms,
                                   style=normalize_style(style))
    _, scale_sq, _ = _compute_parallel_framing(structure, atoms, cam, up, (1000, 1000))
    _, scale_wide, _ = _compute_parallel_framing(structure, atoms, cam, up, (100000, 1000))
    # scale = max(half_h, half_w / aspect) * 1.18; a very wide window leaves half_h
    half_h = scale_wide / 1.18
    half_w = scale_sq / 1.18 if scale_sq / 1.18 > half_h + 1e-9 else None
    if half_w is None:  # taller than wide: find half_w from a tall window
        _, scale_tall, _ = _compute_parallel_framing(structure, atoms, cam, up, (1000, 100000))
        half_w = scale_tall / 1.18 * 0.01
    return float(half_w / max(half_h, 1e-9))


def _px_to_pt(px: float, dpi: float) -> float:
    return round(float(px) / float(dpi) * 72.0, 2)


def _print_triad(structure: Structure, cam_unit: np.ndarray, up_unit: np.ndarray) -> list[dict[str, Any]]:
    """a, b, c directions on the screen: dx (right), dy (down), toward (to the viewer)."""
    right = np.cross(up_unit, cam_unit)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    axes_cfg = get_color_manager().scene_colors.get("axes", {})
    defaults = {"a": "#E63946", "b": "#2A9D8F", "c": "#264653"}
    out = []
    for lab, vec in zip("abc", structure.lattice.matrix):
        u = np.asarray(vec, dtype=float)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        out.append({
            "label": lab,
            "dx": float(np.dot(u, right)),
            "dy": float(-np.dot(u, up_unit)),
            "toward": float(np.dot(u, cam_unit)),
            "color": str(axes_cfg.get(lab, defaults[lab])),
        })
    return out


def _print_strip_layout(pstyle: dict[str, Any], triad: list[dict[str, Any]]) -> dict[str, Any]:
    """Geometry of the strip under the view: triad (left) and scale bar (right)."""
    t = pstyle["text_px"]
    font = ImageFont.truetype(pstyle["font_path"], size=t)
    arrow = max(1.1 * t, min(1.7 * t, 0.12 * pstyle["w_px"]))  # shorter in narrow panels
    end_on = 0.22  # an axis this short on screen points at / away from the viewer
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    items = []
    for ax in triad:
        lb = font.getbbox(ax["label"])
        lw, lh = lb[2] - lb[0], lb[3] - lb[1]
        proj = math.hypot(ax["dx"], ax["dy"])
        if proj < end_on:
            tip = (0.0, 0.0)
            cx, cy = -0.55 * t - lw / 2.0, 0.55 * t + lh / 2.0
            pts += [(-0.3 * t, -0.3 * t), (0.3 * t, 0.3 * t)]
        else:
            ux, uy = ax["dx"] / proj, ax["dy"] / proj
            tip = (ax["dx"] * arrow, ax["dy"] * arrow)
            off = 0.3 * t + 0.5 * max(lw * abs(ux) + lh * abs(uy), 0.6 * t)
            cx, cy = tip[0] + ux * off, tip[1] + uy * off
        items.append({**ax, "tip": tip, "end_on": proj < end_on, "label_c": (cx, cy)})
        pts += [tip, (cx - lw / 2, cy - lh / 2), (cx + lw / 2, cy + lh / 2)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    triad_box = (min(xs), min(ys), max(xs), max(ys))
    triad_h = triad_box[3] - triad_box[1]
    num = font.getbbox("0123456789 Å")
    bar_px = max(2 * pstyle["line_px"], math.ceil(1.0 / 72.0 * pstyle["dpi"]))
    gap = int(round(0.3 * t))
    scale_h = (num[3] - num[1]) + gap + bar_px
    pad = int(round(0.35 * t))
    strip = int(math.ceil(max(triad_h, scale_h) + 2 * pad))
    return {
        "items": items, "triad_box": triad_box, "strip_px": strip, "pad": pad,
        "bar_px": bar_px, "gap": gap, "num_box": num,
        "footer_px": int(round(1.9 * t)),
        "triad_lw": max(pstyle["line_px"], math.ceil(0.75 / 72.0 * pstyle["dpi"])),
    }


def _annotate_print_image(
    img_path: str | Path,
    pstyle: dict[str, Any],
    layout: dict[str, Any],
    angstrom_per_pixel: float | None,
    orientation_label: str | None = None,
    caption_text: str | None = None,
    transparent_background: bool = False,
    legend_elements: list[str] | None = None,
    atom_labels: list[tuple[float, float, str]] | None = None,
    report: dict[str, Any] | None = None,
    elements_present: list[str] | None = None,
    cell_radius: float | None = None,
) -> Path:
    """Print-mode annotation: the 3D picture on top, the triad and scale bar in a strip
    under it, optional caption footer; every text ``pstyle['text_px']`` (exact pt)."""
    from shido.core.figure_layout import draw_legend, legend_size

    path = Path(img_path)
    with Image.open(path) as src:
        base = src.convert("RGBA")
    dpi = pstyle["dpi"]
    t = pstyle["text_px"]
    lpx = pstyle["line_px"]
    font = ImageFont.truetype(pstyle["font_path"], size=t)
    W, H = pstyle["w_px"], pstyle["h_px"]
    bg = (255, 255, 255, 0) if transparent_background else (255, 255, 255, 255)
    canvas = Image.new("RGBA", (W, H), bg)
    canvas.paste(base.crop((0, 0, min(W, base.width), min(H, base.height))), (0, 0))
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ink = (17, 17, 17, 255)
    texts: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    tpt = _px_to_pt(t, dpi)
    margin = int(round(0.5 * t))

    if atom_labels:
        n = _draw_atom_labels(draw, atom_labels, W, dpi / float(_BASE_DPI), font=font,
                              stroke=max(1, lpx))
        if n:
            texts.append({"what": "原子のラベル", "pt": tpt})
    if orientation_label:
        draw.text((margin, margin), orientation_label, fill=ink, font=font, anchor="lt")
        texts.append({"what": "方向の名前", "pt": tpt})
    legend_items = [(e, _element_color(e)) for e in (legend_elements or [])]
    if legend_items:
        lw, _ = legend_size(legend_items, font, t, int(W * 0.45))
        draw_legend(draw, legend_items, W - margin - lw, margin, font, t, lpx, lw, align="right")
        texts.append({"what": "元素の凡例", "pt": tpt})

    # strip under the view: triad (left), scale bar (right)
    y_top = base.height
    pad = layout["pad"]
    x0b, y0b, _, _ = layout["triad_box"]
    ox = margin - x0b
    oy = y_top + pad - y0b
    lw_tri = layout["triad_lw"]
    head = max(3 * lw_tri, 0.4 * t)
    for it in layout["items"]:
        col = it["color"]
        if it["end_on"]:
            r = 0.3 * t
            draw.ellipse([(ox - r, oy - r), (ox + r, oy + r)], outline=col, width=lw_tri)
            if it["toward"] > 0:
                d = 0.3 * r
                draw.ellipse([(ox - d, oy - d), (ox + d, oy + d)], fill=col)
            else:
                d = 0.6 * r
                draw.line([(ox - d, oy - d), (ox + d, oy + d)], fill=col, width=lw_tri)
                draw.line([(ox - d, oy + d), (ox + d, oy - d)], fill=col, width=lw_tri)
        else:
            tx, ty = ox + it["tip"][0], oy + it["tip"][1]
            ln = math.hypot(it["tip"][0], it["tip"][1])
            ux, uy = it["tip"][0] / ln, it["tip"][1] / ln
            bx, by = tx - ux * head, ty - uy * head
            draw.line([(ox, oy), (bx, by)], fill=col, width=lw_tri)
            px_, py_ = -uy * head * 0.45, ux * head * 0.45
            draw.polygon([(tx, ty), (bx + px_, by + py_), (bx - px_, by - py_)], fill=col)
        cx, cy = it["label_c"]
        draw.text((ox + cx, oy + cy), it["label"], fill=col, font=font, anchor="mm")
    texts.append({"what": "方位の a・b・c", "pt": tpt})
    lines.append({"what": "方位の矢印", "pt": _px_to_pt(lw_tri, dpi)})

    if angstrom_per_pixel is not None and angstrom_per_pixel > 0:
        bar_a, _ = _select_scale_bar_length(W, angstrom_per_pixel, px_scale=dpi / float(_BASE_DPI))
        label = _format_scale_label(bar_a)
        for longer in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):  # bar at least as long as its label
            lb = font.getbbox(label)
            if bar_a / angstrom_per_pixel >= (lb[2] - lb[0]) or longer / angstrom_per_pixel > 0.45 * W:
                break
            if longer > bar_a:
                bar_a, label = longer, _format_scale_label(longer)
        bar_len = int(round(bar_a / angstrom_per_pixel))
        bar_h = layout["bar_px"]
        x1 = W - margin
        x0 = x1 - bar_len
        y1 = y_top + layout["strip_px"] - pad
        draw.rectangle([(x0, y1 - bar_h), (x1, y1)], fill=ink)
        draw.text(((x0 + x1) / 2.0, y1 - bar_h - layout["gap"]), label, fill=ink, font=font, anchor="ms")
        texts.append({"what": "縮尺の数字", "pt": tpt})
        lines.append({"what": "縮尺の棒", "pt": _px_to_pt(bar_h, dpi)})
    if cell_radius is not None and angstrom_per_pixel:
        lines.append({"what": "単位胞の線", "pt": _px_to_pt(2.0 * cell_radius / angstrom_per_pixel, dpi)})

    out = Image.alpha_composite(canvas, overlay)
    if caption_text:
        fy = y_top + layout["strip_px"]
        d2 = ImageDraw.Draw(out)
        d2.line([(0, fy), (W, fy)], fill=(205, 205, 205, 255), width=lpx)
        d2.text((W / 2.0, fy + layout["footer_px"] / 2.0), caption_text, fill=ink, font=font, anchor="mm")
        texts.append({"what": "説明の帯", "pt": tpt})
    final = out if transparent_background else out.convert("RGB")
    final.save(path, format="PNG", dpi=(dpi, dpi))
    if report is not None:
        report.update(
            texts=texts, lines=lines, fonts=[pstyle["font_name"]],
            elements=[[e, _element_color(e)] for e in (elements_present or [])],
            size_px=[W, H], dpi=dpi, view_px=[base.width, base.height],
        )
    return path.resolve()


def render_single_view(
    structure: Structure | list[Any],
    output_path: str | Path,
    camera_dir: np.ndarray | list[float],
    view_up: np.ndarray | list[float],
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]] | None = None,
    polyhedra_envs: list[tuple[int, list[np.ndarray], str]] | None = None,
    show_polyhedra: bool = False,
    include_boundary_atoms: bool = True,
    window_size: tuple[int, int] | None = None,
    orientation_label: str | None = None,
    caption_text: str | None = None,
    transparent_background: bool | None = None,
    show_legend: bool = True,
    style: str | None = None,
    perspective: bool = False,
    labels: str | None = None,
    polyhedra_center: Any = None,
    bond_color: str | None = None,
    frac_range: Any = None,
    image_formats: Any = None,
    planes: Any = None,
    vectors: Any = None,
    vector_basis: str | None = None,
    vector_scale: float | None = None,
    vector_color: str | None = None,
    vector_radius: float | None = None,
    volume: Any = None,
    isovalue: Any = None,
    iso_color: Any = None,
    iso_color_negative: Any = None,
    iso_opacity: float | None = None,
    iso_both_signs: bool = False,
    adp: Any = None,
    probability: float | None = None,
    frame_hook: Any = None,
    cutoff_planes: Any = None,
    whole_molecules: bool = False,
    hbonds: Any = None,
    overlays: Any = None,
    print_width_mm: float | None = None,
    print_height_mm: float | None = None,
    print_dpi: float | None = None,
    print_fonts: Any = None,
    print_text_pt: float | None = None,
    print_line_pt: float | None = None,
    print_report: dict[str, Any] | None = None,
) -> Path:
    """Render a single orthographic 3D crystal view to a publication-annotated PNG (render.dpi).

    Print mode (``print_width_mm`` given; all off by default = the old output): the PNG is
    drawn at its finished size, ``print_width_mm`` x ``print_height_mm`` (default square)
    at ``print_dpi`` (pixels = mm / 25.4 x dpi). Every text (scale bar, orientation triad
    a/b/c, legend, badge, caption, atom labels) is ``print_text_pt`` (default 7) in the
    first of ``print_fonts`` found, and the drawn lines (unit cell, scale bar, triad) are
    at least ``print_line_pt`` (default 0.5). The scale bar and the triad sit in a strip
    under the structure (no overlap with atoms), drawn with PIL instead of VTK's axes.
    ``print_report`` (a dict) receives what was drawn: "texts" / "lines" ({"what", "pt"}),
    "fonts", "elements" ([symbol, colour] of the legend), "size_px".

    ``structure`` may also be a list: the first is the base, the others are overlays
    (same as ``overlays``). ``overlays`` (see `normalize_overlays`) draws more structures
    in the same picture, each moved by a rotation (axis + angle or matrix) and a shift
    (Å or fractions of the base cell), optionally translucent / one colour (VESTA
    "multiple phases": a molecule on a surface, a grain boundary). The picture is framed
    on everything; bonds of an overlay come from its own cell (`analyzer.compute_bonds`).

    ``cutoff_planes`` ("1 1 1@0.5", ...) keeps only atoms with h*x+k*y+l*z <= D (bonds /
    polyhedra follow); ``whole_molecules`` draws every molecule (covalent connected
    component) whole, centre of mass in the cell / ``frac_range``; ``hbonds``
    ("O-H...O:2.5:120", ...) draws those hydrogen bonds as dashed lines (see core.molecules).

    ``frame_hook(plotter, info)`` (e.g. `core.animation.RotationHook`) is called after the
    still is taken, with the scene still open, to grab more frames (rotation animation).

    ``style='ellipsoid'`` draws displacement ellipsoids (ORTEP / VESTA) of ``probability``
    percent (default 50: semi-axes 1.5382·√eigenvalue of U_cart) with their 3 principal
    ellipses; ``adp`` is the CIF path (or `resolve_adp_tensors` list). Non-positive-definite
    ADPs are drawn as spheres of √U_eq (warning), atoms without ADPs as small spheres.

    ``volume`` (CHGCAR / LOCPOT / ELFCAR / *.cube / *.xsf / *.grd path, or a
    `core.volumetric.VolumetricGrid`) overlays the isosurfaces at ``isovalue``
    ("0.05" / "0.05,0.1" / list; default 10 % of max |value|) in the unit cell
    (VESTA "Isosurfaces"): ``iso_color`` (one per level, cycled; default #F2C200),
    ``iso_color_negative`` (#2F9BD6, for negative levels), ``iso_opacity`` (0.55),
    ``iso_both_signs`` also draws -level for each level (e.g. charge differences).

    ``frac_range`` [x0, x1, y0, y1, z0, z1] (VESTA "Boundary", e.g. -0.5 1.5 ...) draws
    every atom image in that fractional box; unit-cell bonds / polyhedra are copied onto
    it. ``image_formats`` (e.g. "tiff,pdf") also writes the finished PNG in those
    formats next to it (see `export_image_formats`); the PNG is always written.
    ``planes`` ("1 1 1@0.5:#ff0000:0.4", ... see `normalize_planes`) draws those
    translucent (hkl) planes in the unit cell (VESTA "Lattice planes").
    ``vectors`` ("Mn:0 0 4", "magmom", ... see `core.vectors`) draws an arrow through
    every drawn atom that has a vector (VESTA "Vectors"): length ``vector_scale`` Å per
    unit (default 1.0), ``vector_color``, shaft ``vector_radius`` Å; ``vector_basis``
    'crystal' (default, components along a/|a|, b/|b|, c/|c| as in mCIF) or 'cartesian'.

    ``polyhedra_center`` ('Mn' or ['Mn', 'Ga']) chooses the polyhedron centre elements
    (default: the least frequent element). ``bond_color`` = 'split' colours ball-stick
    bonds half-and-half by element; 'single' / None keeps the theme bond colour
    (stick / wireframe are always split).

    ``perspective=True`` uses a perspective camera (30 deg view angle) instead of the
    parallel projection; the scale bar is then omitted (no single A-per-pixel scale).
    ``labels`` = 'site' (CIF site label, e.g. Ni1) or 'element' writes a text label on
    each visible atom (hidden atoms and labels that would overlap a nearer one are skipped).

    ``style`` is one of RENDER_STYLES (None = 'ball-stick', the original look):
    'space-filling' (full-size atoms, no bonds), 'stick' (thin atoms + element-coloured
    sticks) or 'wireframe' (element-coloured bond lines only, no atom spheres).

    ``show_legend=False`` omits the in-scene element legend (used for multi-panel
    figures, which draw one shared legend below the panels instead).
    ``window_size=None`` is 1200 x 1200 px at 300 dpi, scaled with render.dpi so the
    printed size stays the same.
    """
    if isinstance(structure, (list, tuple)):
        if not structure:
            raise ValueError("render_single_view: empty structure list.")
        extra = list(structure[1:])
        structure = structure[0]
        overlays = extra + (list(overlays) if isinstance(overlays, (list, tuple)) else
                            ([overlays] if overlays is not None else []))
    transparent_background = _resolve_transparent(transparent_background)
    style = normalize_style(style)
    labels = normalize_labels(labels)
    perspective = bool(perspective)
    polyhedra_center = normalize_polyhedra_center(polyhedra_center)
    bond_color = normalize_bond_color(bond_color)
    px_scale = _px_scale()
    pstyle: dict[str, Any] | None = None
    if print_width_mm is not None:
        pstyle = print_annotation_style(
            print_width_mm, print_height_mm, print_dpi, print_fonts, print_text_pt, print_line_pt
        )
        px_scale = pstyle["dpi"] / float(_BASE_DPI)
    if window_size is None:
        side = _px(1200, px_scale)
        window_size = (side, side)
    out_path = Path(output_path)
    # Guardrail: refuse broken cells / overlapping atoms before any drawing.
    validate_structure(structure)
    warn_partial_occupancy(structure, context=f"render {out_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _CURRENT_COLOR_MAP.clear()
    _CURRENT_COLOR_MAP.update(resolve_structure_colors(structure))

    frac_range = normalize_frac_range(frac_range)
    image_formats = normalize_image_formats(image_formats)
    planes = normalize_planes(planes)
    atoms = _collect_atoms_to_draw(
        structure, include_boundary_atoms=include_boundary_atoms, style=style, frac_range=frac_range
    )
    ellipsoids: list[dict[str, Any]] = []
    if style == "ellipsoid":
        tensors, adp_notes = resolve_adp_tensors(structure, adp)
        for note in adp_notes:
            logger.warning("ellipsoid style: %s", note)
        ellipsoids, _ = ellipsoid_shapes(
            structure, tensors, probability, include_boundary_atoms=include_boundary_atoms,
            frac_range=frac_range,
        )
        atoms = [(a[0], a[1], sh["radius"]) for a, sh in zip(atoms, ellipsoids)]
    arrows: list[tuple[np.ndarray, np.ndarray]] = []
    if vectors is not None:
        vector_scale, vector_color, vector_radius = normalize_vector_style(
            vector_scale, vector_color, vector_radius
        )
        arrows = vector_arrows(
            structure, vectors, basis=vector_basis, scale=vector_scale,
            include_boundary_atoms=include_boundary_atoms, frac_range=frac_range,
        )
    iso_meshes: list[tuple[Any, str, float]] = []
    if volume is not None:
        _, _, iso_opacity = normalize_iso_style(iso_color, iso_color_negative, iso_opacity)
        iso_meshes = isosurface_meshes(
            structure, volume, isovalue, both_signs=bool(iso_both_signs),
            color=iso_color, color_negative=iso_color_negative,
        )
    if frac_range is not None:
        render_bonds, polyhedra_envs = _replicate_in_range(
            structure, [a[0] for a in atoms], frac_range, render_bonds, polyhedra_envs
        )
        render_bonds = limit_render_bonds(render_bonds)
    hbond_segments: list[tuple[np.ndarray, np.ndarray]] = []
    keep_on_atoms = None
    if cutoff_planes or whole_molecules or hbonds:
        from shido.core.molecules import keep_near_atoms, restrict_drawing

        if whole_molecules and style == "ellipsoid":
            raise ValueError("whole_molecules cannot be combined with style='ellipsoid'.")
        old_atoms = atoms
        atoms, render_bonds, polyhedra_envs, hbond_segments = restrict_drawing(
            structure, atoms, render_bonds, polyhedra_envs, cutoff_planes=cutoff_planes,
            whole_molecules=bool(whole_molecules), hbonds=hbonds, frac_range=frac_range,
            site_info={
                idx: (_site_species_occupancies(s), r)
                for idx, (s, (_, _, r)) in enumerate(zip(structure, _collect_atoms_to_draw(
                    structure, include_boundary_atoms=False, style=style)))
            } if whole_molecules else None,
        )
        if ellipsoids:
            kept = keep_near_atoms([a[0] for a in old_atoms], atoms)
            ellipsoids = [e for e, k in zip(ellipsoids, kept) if k]
        if arrows:
            arrows = [pq for pq, k in zip(arrows, keep_near_atoms(
                [(np.asarray(p) + np.asarray(q)) / 2 for p, q in arrows], atoms)) if k]
        keep_on_atoms = atoms
        if not atoms:
            raise ValueError("cutoff planes / molecules: no atom is left to draw.")

    cam_vec = np.asarray(camera_dir, dtype=float)
    if float(np.linalg.norm(cam_vec)) < 1e-8:
        cam_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    cam_unit = cam_vec / np.linalg.norm(cam_vec)
    up_unit = _orthonormal_up(cam_unit, np.asarray(view_up, dtype=float))
    print_layout: dict[str, Any] | None = None
    if pstyle is not None:  # 3D part = finished size minus the strip (and caption footer)
        print_layout = _print_strip_layout(pstyle, _print_triad(structure, cam_unit, up_unit))
        reserve = print_layout["strip_px"] + (print_layout["footer_px"] if caption_text else 0)
        window_size = (pstyle["w_px"], max(16, pstyle["h_px"] - reserve))

    overlay_specs = normalize_overlays(overlays)
    overlay_geoms = [
        overlay_geometry(structure, spec, include_boundary_atoms=include_boundary_atoms, style=style)
        for spec in overlay_specs
    ]
    if overlay_specs:  # one palette over every structure, so no two elements share a colour
        from types import SimpleNamespace

        total = structure.composition
        for spec in overlay_specs:
            total = total + spec["structure"].composition
        _CURRENT_COLOR_MAP.clear()
        _CURRENT_COLOR_MAP.update(resolve_structure_colors(SimpleNamespace(composition=total)))

    frame_atoms = atoms
    if arrows:  # keep the arrow heads in the picture
        tip_r = 2.2 * float(vector_radius)
        frame_atoms = atoms + [(p, [], tip_r) for pair in arrows for p in pair]
    if overlay_geoms:
        frame_atoms = frame_atoms + [a for g in overlay_geoms for a in g["atoms"]] + [
            (p, [], 0.05) for g in overlay_geoms for pq in g["cell_edges"] for p in pq
        ]
    cell_center, parallel_scale, angstrom_per_pixel = _compute_parallel_framing(
        structure=structure,
        atoms=frame_atoms,
        cam_unit=cam_unit,
        up_unit=up_unit,
        window_size=window_size,
    )

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    background_color = str(get_color_manager().scene_colors.get("background", "#FFFFFF"))
    try:
        plotter.set_background(background_color)
        if not perspective:
            plotter.enable_parallel_projection()
        apply_studio_lighting(plotter, parallel_scale=parallel_scale)

        # 1. Unit cell wireframe tubes (print mode: at least the journal's line width)
        cell_radius = None
        if pstyle is not None:
            cell_radius = max(
                _float_setting("render.unit_cell_line_radius", _UNIT_CELL_LINE_RADIUS),
                pstyle["line_px"] * angstrom_per_pixel / 2.0,
            )
        _add_unit_cell(plotter, structure, tube_radius=cell_radius)

        # 2. Atom spheres (ordered grouped by element; disordered as latitude-banded multi-color spheres)
        # Mesh resolution follows the on-screen atom size (sub-pixel outline error), so
        # big supercells / small atoms are cheaper while close-ups keep the old 36.
        spheres_by_elem: dict[str, tuple[list[np.ndarray], list[float]]] = {}
        disordered_spheres: list[pv.PolyData] = []
        elements_present_set: set[str] = set()

        for cart, species_occs, radius in atoms:
            for sym, _ in species_occs:
                elements_present_set.add(sym)
            if style in ("wireframe", "ellipsoid"):
                continue  # wireframe: atoms only frame the view; ellipsoids are drawn below

            is_disordered = len(species_occs) > 1 or (
                len(species_occs) == 1 and species_occs[0][1] < 0.995
            )
            if is_disordered:
                # Keep >= 20 so the occupancy bands keep smooth borders.
                res = max(
                    _sphere_resolution(radius, angstrom_per_pixel, max_res=32, px_scale=px_scale), 20
                )
                disordered_spheres.append(
                    _build_disordered_sphere(cart, radius, species_occs, resolution=res)
                )
            else:
                symbol = species_occs[0][0]
                centers, radii = spheres_by_elem.setdefault(symbol, ([], []))
                centers.append(cart)
                radii.append(radius)

        # Stick joints are plain spheres: a silhouette around them would outline every joint.
        add_atoms = plotter.add_mesh if style == "stick" else (
            lambda mesh, **kw: _add_atom_mesh_with_silhouette(plotter, mesh, px_scale=px_scale, **kw)
        )
        for elem, (centers, radii) in spheres_by_elem.items():
            res = _sphere_resolution(max(radii), angstrom_per_pixel, px_scale=px_scale)
            merged = _merge_spheres(centers, radii, res)
            if _mesh_is_empty(merged):
                continue
            add_atoms(
                merged,
                color=_element_color(elem),
                smooth_shading=True,
                ambient=0.26,
                diffuse=0.74,
                specular=0.55,
                specular_power=45,
            )

        if disordered_spheres:
            merged_disordered = (
                pv.merge(disordered_spheres)
                if len(disordered_spheres) > 1
                else disordered_spheres[0]
            )
            add_atoms(
                merged_disordered,
                scalars="rgb",
                rgb=True,
                smooth_shading=True,
                ambient=0.26,
                diffuse=0.74,
                specular=0.55,
                specular_power=45,
            )

        if ellipsoids:  # majority species colours the ellipsoid
            add_ellipsoids(
                plotter, ellipsoids,
                [max(a[1], key=lambda so: so[1])[0] if a[1] else "X" for a in atoms],
                angstrom_per_pixel, px_scale, add_atoms,
            )

        # 3. Bonds as split half-cylinders
        if style in ("ball-stick", "ellipsoid"):
            # ellipsoids are small (~0.15-0.3 Å): thinner bonds so they do not hide them (ORTEP)
            b_radius = _float_setting("render.bond_radius", _BOND_RADIUS) * (
                _ELLIPSOID_BOND_FACTOR if style == "ellipsoid" else 1.0
            )
            if render_bonds and bond_color == "split":
                _add_bicolor_bonds(plotter, render_bonds, as_lines=False, radius=b_radius)
            elif render_bonds:
                _add_bonds(plotter, render_bonds, bond_radius=None if style == "ball-stick" else b_radius)
        elif style == "stick":
            _add_bicolor_bonds(plotter, render_bonds or [], as_lines=False)
        elif style == "wireframe":
            if not render_bonds:
                logger.warning("wireframe style: no bonds found, only the unit cell is drawn.")
            _add_bicolor_bonds(plotter, render_bonds or [], as_lines=True)
        # space-filling: no bonds

        # 4. Coordination polyhedra (least frequent element)
        # (ball-stick only: like VESTA, the other styles are their own representation and
        # translucent polyhedra would hide the full-size atoms / thin sticks / lines)
        if show_polyhedra and polyhedra_envs and style == "ball-stick":
            _add_polyhedra(plotter, structure, polyhedra_envs, center_elements=polyhedra_center)

        # 4a. Overlaid structures (none by default)
        if overlay_geoms:
            if any(s["opacity"] < 1.0 for s in overlay_specs):
                plotter.enable_depth_peeling()
            for spec, geom in zip(overlay_specs, overlay_geoms):
                add_overlay(plotter, spec, geom, style, angstrom_per_pixel, px_scale)
                for _, occs, _ in geom["atoms"]:
                    for sym, _ in occs:
                        elements_present_set.add(sym)

        # 4b. User (hkl) lattice planes (none by default)
        if planes:
            add_bulk_planes(plotter, structure, planes)

        # 4c. Atom vectors / magnetic moments (none by default)
        if arrows:
            add_vector_arrows(plotter, arrows, vector_color, vector_radius)

        # 4c'. Hydrogen bonds as dashed tubes (none by default)
        if hbond_segments:
            from shido.core.molecules import dashed_tubes

            dashes = dashed_tubes(hbond_segments)
            if dashes is not None:
                plotter.add_mesh(dashes, color="#3A86FF", smooth_shading=True)

        # 4d. Isosurfaces of volumetric data (none by default)
        if iso_meshes:
            plotter.enable_depth_peeling()
            add_isosurfaces(plotter, iso_meshes, iso_opacity)

        # 5. Legend for all elements present (including disordered sites)
        for site in structure:
            if hasattr(site, "species") and hasattr(site.species, "elements"):
                for el in site.species.elements:
                    elements_present_set.add(str(el.symbol))

        # Drawn after the screenshot by _annotate_publication_image (fixed font size;
        # VTK's legend box scales the text to the box and cut "Cu" off at the edge).
        elements_present = sorted(elements_present_set)

        # 6. Crystallographic orientation axes (a, b, c with theme axis colors)
        axes_cfg = get_color_manager().scene_colors.get("axes", {})
        if pstyle is None:  # print mode draws the triad itself (exact pt) under the view
            add_orientation_axes(
                plotter,
                xlabel="a",
                ylabel="b",
                zlabel="c",
                line_width=2.5,
                x_color=str(axes_cfg.get("a", "#E63946")),
                y_color=str(axes_cfg.get("b", "#2A9D8F")),
                z_color=str(axes_cfg.get("c", "#264653")),
                color="black",
            )

        # 7. Exact orthographic camera framing (do not call reset_camera after parallel_scale)
        diag_len = float(
            np.linalg.norm(structure.lattice.get_cartesian_coords([1.0, 1.0, 1.0]))
        )
        cam_dist = max(diag_len * 3.0, parallel_scale * 4.0, 20.0)
        if perspective:
            # Fit the framed box at the focal plane; the front atoms (closer by up to
            # the depth half-extent) still fit, since they only grow by d / (d - depth).
            depth_half = max(
                (abs(float(np.dot(a[0] - cell_center, cam_unit))) + a[2] for a in atoms),
                default=0.0,
            )
            cam_dist = (
                parallel_scale / math.tan(math.radians(_PERSPECTIVE_VIEW_ANGLE / 2.0)) + depth_half
            )
        cam_pos = cell_center + cam_unit * cam_dist

        plotter.camera_position = [
            cam_pos.tolist(),
            cell_center.tolist(),
            up_unit.tolist(),
        ]
        if perspective:
            plotter.camera.view_angle = _PERSPECTIVE_VIEW_ANGLE
        else:
            plotter.camera.parallel_scale = parallel_scale
        plotter.renderer.ResetCameraClippingRange()

        img = plotter.screenshot(
            str(out_path), transparent_background=transparent_background, return_img=True
        )
        atom_labels = (
            _project_visible_labels(
                plotter,
                _labels_on_atoms(
                    atom_label_candidates(structure, include_boundary_atoms, labels, frac_range),
                    keep_on_atoms, structure, labels,
                ),
                atoms,
                cam_unit,
                up_unit,
                window_size,
                (int(img.shape[1]), int(img.shape[0])),
            )
            if labels
            else None
        )
        if frame_hook is not None:
            frame_hook(plotter, {
                "structure": structure, "atoms": frame_atoms, "atoms_drawn": atoms,
                "cell_center": cell_center, "cam_unit": cam_unit, "up_unit": up_unit,
                "cam_dist": cam_dist, "parallel_scale": parallel_scale,
                "angstrom_per_pixel": angstrom_per_pixel, "perspective": perspective,
                "window_size": window_size, "transparent_background": transparent_background,
                "labels": labels, "include_boundary_atoms": include_boundary_atoms,
                "frac_range": frac_range, "caption_text": caption_text,
                "legend_elements": elements_present if show_legend else None,
            })
    finally:
        plotter.close()

    # Guardrail: a (near) empty frame means the camera missed the atoms or no mesh was built.
    try:
        validate_blank_image(
            img,
            background=background_color,
            transparent=transparent_background,
            context=f"render {out_path.name}",
        )
    except ScientificValidationError:
        out_path.unlink(missing_ok=True)
        raise

    if pstyle is not None and print_layout is not None:
        final_path = _annotate_print_image(
            out_path,
            pstyle,
            print_layout,
            angstrom_per_pixel=None if perspective else angstrom_per_pixel,
            orientation_label=orientation_label,
            caption_text=caption_text,
            transparent_background=transparent_background,
            legend_elements=elements_present if show_legend else None,
            atom_labels=atom_labels,
            report=print_report,
            elements_present=elements_present,
            cell_radius=cell_radius,
        )
        if image_formats:
            export_image_formats(final_path, image_formats)
        return final_path
    final_path = _annotate_publication_image(
        img_path=out_path,
        angstrom_per_pixel=None if perspective else angstrom_per_pixel,
        orientation_label=orientation_label,
        caption_text=caption_text,
        transparent_background=transparent_background,
        legend_elements=elements_present if show_legend else None,
        atom_labels=atom_labels,
    )
    if image_formats:
        export_image_formats(final_path, image_formats)
    return final_path


_PERSPECTIVE_VIEW_ANGLE: float = 30.0


def _project_visible_labels(
    plotter: Any,
    candidates: list[tuple[np.ndarray, str]],
    atoms: list[tuple[np.ndarray, list[tuple[str, float]], float]],
    cam_unit: np.ndarray,
    up_unit: np.ndarray,
    window_size: tuple[int, int],
    image_size: tuple[int, int],
) -> list[tuple[float, float, str]]:
    """Image-pixel (x, y, text) of labels on atoms not hidden behind a nearer atom,
    ordered front to back (the annotator drops later labels that would overlap)."""
    if not candidates:
        return []
    ren = plotter.renderer
    sx = image_size[0] / max(float(window_size[0]), 1.0)
    sy = image_size[1] / max(float(window_size[1]), 1.0)

    def to_px(p: np.ndarray) -> tuple[float, float]:
        ren.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        return x * sx, image_size[1] - y * sy

    right = np.cross(up_unit, cam_unit)
    right = right / max(float(np.linalg.norm(right)), 1e-12)
    proj = []
    for (cart, text), atom in zip(candidates, atoms):
        x, y = to_px(cart)
        ex, ey = to_px(cart + right * atom[2])
        proj.append((x, y, math.hypot(ex - x, ey - y), float(np.dot(cart, cam_unit)), text))

    out: list[tuple[float, float, str]] = []
    for i in sorted(range(len(proj)), key=lambda k: -proj[k][3]):
        x, y, _, depth, text = proj[i]
        if not (0.0 <= x < image_size[0] and 0.0 <= y < image_size[1]):
            continue
        hidden = any(
            pj[3] > depth + 1e-3 and math.hypot(pj[0] - x, pj[1] - y) < 0.8 * pj[2]
            for j, pj in enumerate(proj)
            if j != i
        )
        if not hidden:
            out.append((x, y, text))
    return out


def _draw_atom_labels(
    draw: Any, labels: list[tuple[float, float, str]], width: int, ps: float,
    font: Any = None, stroke: int | None = None,
) -> int:
    """Draw centred dark labels with a white outline; skip ones overlapping an earlier
    (nearer) label. Returns the number drawn. ``font`` / ``stroke`` override the
    width-based defaults (print mode)."""
    if font is None:
        font = _load_font(max(_px(16, ps), int(round(width * 0.017))))
    if stroke is None:
        stroke = max(_px(3, ps), 2)
    pad = _px(2, ps)
    taken: list[tuple[float, float, float, float]] = []
    for x, y, text in labels:
        bb = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        box = (x - w / 2 - pad, y - h / 2 - pad, x + w / 2 + pad, y + h / 2 + pad)
        if any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
            continue
        taken.append(box)
        draw.text(
            (x - w / 2 - bb[0], y - h / 2 - bb[1]),
            text,
            font=font,
            fill=(17, 17, 17, 255),
            stroke_width=stroke,
            stroke_fill=(255, 255, 255, 255),
        )
    return len(taken)


def screenshot_checked(
    plotter: pv.Plotter,
    output_path: str | Path | None = None,
    transparent_background: bool = False,
    context: str = "render",
) -> np.ndarray:
    """``plotter.screenshot`` + blank-frame guardrail (> 99.5 % background raises).

    Call before ``plotter.close()``. A rejected PNG is deleted so no blank figure is left.
    """
    try:
        bg = tuple(int(v) for v in plotter.background_color.int_rgb)
    except Exception:
        bg = (255, 255, 255)
    img = plotter.screenshot(
        str(output_path) if output_path is not None else None,
        transparent_background=transparent_background,
        return_img=True,
    )
    try:
        validate_blank_image(
            img, background=bg, transparent=transparent_background, context=context
        )
    except ScientificValidationError:
        if output_path is not None:
            Path(output_path).unlink(missing_ok=True)
        raise
    return np.asarray(img)


def _structure_elements_sorted(structure: Structure) -> list[str]:
    """Collect sorted unique element symbols across ordered and disordered sites."""
    elems: set[str] = set()
    for site in structure:
        if hasattr(site, "species") and hasattr(site.species, "elements"):
            for el in site.species.elements:
                elems.add(str(el.symbol))
        else:
            for sym, _ in _site_species_occupancies(site):
                elems.add(sym)
    return sorted(elems)


def _compose_overview_figure(
    panel_paths: list[Path],
    output_path: Path,
    caption_text: str | None = None,
    panel_size: int | None = None,
    strip_panel_footer: bool = True,
    elements_present: list[str] | None = None,
    transparent_background: bool | None = None,
) -> Path:
    """Compose a 3-panel side-by-side publication overview figure (render.dpi PNG).

    When ``elements_present`` is given, one shared element legend is drawn below the panels.
    With a transparent background the canvas is RGBA and the panels keep their alpha.
    ``panel_size=None`` is 900 px at 300 dpi; all sizes scale with render.dpi / 300.
    """
    ps = _px_scale()
    if panel_size is None:
        panel_size = _px(900, ps)
    panel_footer = _px(_FOOTER_HEIGHT, ps)
    transparent = _resolve_transparent(transparent_background)
    mode = "RGBA" if transparent else "RGB"
    panels: list[Image.Image] = []
    for p in panel_paths:
        with Image.open(p) as im:
            rgb = im.convert(mode)
            if strip_panel_footer and rgb.height > panel_footer + _px(100, ps):
                rgb = rgb.crop((0, 0, rgb.width, rgb.height - panel_footer))
            resized = rgb.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            panels.append(resized)

    gap = _px(12, ps)
    pad = _px(16, ps)
    total_w = pad * 2 + panel_size * len(panels) + gap * max(0, len(panels) - 1)
    legend_h = _px(64, ps) if elements_present else 0
    content_h = pad * 2 + panel_size + legend_h
    footer_h = panel_footer if caption_text else 0

    if transparent:
        canvas = Image.new("RGBA", (total_w, content_h + footer_h), (255, 255, 255, 0))
    else:
        canvas = Image.new("RGB", (total_w, content_h + footer_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, panel in enumerate(panels):
        x0 = pad + idx * (panel_size + gap)
        y0 = pad
        canvas.paste(panel, (x0, y0))  # RGBA paste copies alpha as-is (keeps transparency)
        draw.rectangle(
            [(x0, y0), (x0 + panel_size - 1, y0 + panel_size - 1)],
            outline=(215, 215, 215),
            width=_px(2, ps),
        )

    if elements_present:
        font = _load_font(_px(30, ps))
        r = _px(13, ps)
        label_gap = _px(12, ps)
        items = []
        for elem in elements_present:
            bbox = draw.textbbox((0, 0), elem, font=font)
            items.append((elem, bbox[2] - bbox[0]))
        item_gap = _px(48, ps)
        row_w = sum(2 * r + label_gap + w for _, w in items) + item_gap * (len(items) - 1)
        x = (total_w - row_w) / 2.0
        cy = pad + panel_size + legend_h / 2.0
        for elem, w in items:
            fill = _hex_to_rgb(_element_color(elem))
            draw.ellipse(
                [(x, cy - r), (x + 2 * r, cy + r)], fill=fill, outline=(26, 26, 26), width=_px(2, ps)
            )
            draw.text((x + 2 * r + label_gap, cy), elem, fill=(24, 24, 24), font=font, anchor="lm")
            x += 2 * r + label_gap + w + item_gap

    if caption_text:
        _draw_centered_footer(
            canvas, caption_text, footer_top=content_h, footer_height=panel_footer, px_scale=ps
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=_dpi_pair())
    return output_path.resolve()


def _compose_overview_svg(
    panel_specs: list[tuple[str, Path]],
    elements_present: list[str],
    output_path: Path,
    caption_text: str | None = None,
    panel_size: int = 900,
    strip_panel_footer: bool = True,
    transparent_background: bool | None = None,
) -> Path:
    """Compose an editable vector SVG multi-panel publication figure with embedded PNG panels.

    The SVG layout stays in 300-dpi units (same size at any dpi); the embedded panel
    bitmaps get render.dpi / 300 times more pixels.
    """
    ps = _px_scale()
    panel_px = _px(panel_size, ps)
    panel_footer = _px(_FOOTER_HEIGHT, ps)
    transparent = _resolve_transparent(transparent_background)
    gap = 18
    pad = 24
    header_h = 44
    legend_h = 46 if elements_present else 0
    footer_h = 64 if caption_text else 0

    n_panels = max(1, len(panel_specs))
    total_w = pad * 2 + panel_size * n_panels + gap * max(0, n_panels - 1)
    panel_top = pad + header_h
    legend_top = panel_top + panel_size + 14
    footer_top = legend_top + legend_h + (8 if legend_h else 0)
    total_h = footer_top + footer_h + pad

    font_family = "Arial, Helvetica, sans-serif"
    svg_lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{total_w}" height="{total_h}" '
            f'viewBox="0 0 {total_w} {total_h}">'
        ),
        "  <style>",
        f"    text {{ font-family: {font_family}; fill: #181818; }}",
        "    .panel-title { font-size: 26px; font-weight: 700; }",
        "    .legend-label { font-size: 22px; font-weight: 600; }",
        "    .caption-text { font-size: 24px; font-weight: 500; }",
        "  </style>",
    ]
    if not transparent:
        svg_lines.append(f'  <rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#FFFFFF"/>')

    # Embedded PNG panels + editable vector panel labels
    for idx, (label, panel_path) in enumerate(panel_specs):
        x0 = pad + idx * (panel_size + gap)
        with Image.open(panel_path) as im:
            rgb = im.convert("RGBA") if im.mode == "RGBA" else im.convert("RGB")
            if strip_panel_footer and rgb.height > panel_footer + _px(100, ps):
                rgb = rgb.crop((0, 0, rgb.width, rgb.height - panel_footer))
            resized = rgb.resize((panel_px, panel_px), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="PNG", dpi=_dpi_pair())
            b64_png = base64.b64encode(buf.getvalue()).decode("ascii")

        svg_lines.append(f'  <g id="panel_{idx + 1}">')
        svg_lines.append(
            f'    <text x="{x0 + 4}" y="{pad + 30}" class="panel-title" '
            f'font-family="{font_family}">{xml_escape(label)}</text>'
        )
        svg_lines.append(
            f'    <image x="{x0}" y="{panel_top}" width="{panel_size}" height="{panel_size}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'href="data:image/png;base64,{b64_png}" '
            f'xlink:href="data:image/png;base64,{b64_png}"/>'
        )
        svg_lines.append(
            f'    <rect x="{x0}" y="{panel_top}" width="{panel_size}" height="{panel_size}" '
            f'fill="none" stroke="#CCCCCC" stroke-width="2"/>'
        )
        svg_lines.append("  </g>")

    # Editable vector element legend
    if elements_present:
        item_step = 110
        legend_total_w = len(elements_present) * item_step
        start_x = (total_w - legend_total_w) / 2.0 + 20.0
        cy = legend_top + legend_h / 2.0
        svg_lines.append('  <g id="element_legend">')
        for idx, elem in enumerate(elements_present):
            cx = start_x + idx * item_step
            color = _element_color(elem)
            svg_lines.append(
                f'    <circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" '
                f'fill="{color}" stroke="#1A1A1A" stroke-width="1.8"/>'
            )
            svg_lines.append(
                f'    <text x="{cx + 20:.1f}" y="{cy + 7:.1f}" class="legend-label" '
                f'font-family="{font_family}">{xml_escape(elem)}</text>'
            )
        svg_lines.append("  </g>")

    # Editable vector caption footer
    if caption_text:
        svg_lines.append('  <g id="caption_footer">')
        svg_lines.append(
            f'    <line x1="{pad}" y1="{footer_top}" x2="{total_w - pad}" y2="{footer_top}" '
            f'stroke="#CCCCCC" stroke-width="2"/>'
        )
        svg_lines.append(
            f'    <text x="{total_w / 2.0:.1f}" y="{footer_top + 42}" '
            f'text-anchor="middle" class="caption-text" '
            f'font-family="{font_family}">{xml_escape(caption_text)}</text>'
        )
        svg_lines.append("  </g>")

    svg_lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")
    return output_path.resolve()


def render_bulk_views(
    structure: Structure,
    output_dir: str | Path,
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]] | None = None,
    polyhedra_envs: list[tuple[int, list[np.ndarray], str]] | None = None,
    show_polyhedra: bool | None = None,
    summary: dict[str, Any] | None = None,
    transparent_background: bool | None = None,
    style: str | None = None,
    view_dir: list[float] | tuple[float, ...] | None = None,
    view_normal: list[float] | tuple[float, ...] | None = None,
    perspective: bool = False,
    labels: str | None = None,
    polyhedra_center: Any = None,
    bond_color: str | None = None,
    frac_range: Any = None,
    image_formats: Any = None,
    planes: Any = None,
    vectors: Any = None,
    vector_basis: str | None = None,
    vector_scale: float | None = None,
    vector_color: str | None = None,
    vector_radius: float | None = None,
    volume: Any = None,
    isovalue: Any = None,
    iso_color: Any = None,
    iso_color_negative: Any = None,
    iso_opacity: float | None = None,
    iso_both_signs: bool = False,
    adp: Any = None,
    probability: float | None = None,
    cutoff_planes: Any = None,
    whole_molecules: bool = False,
    hbonds: Any = None,
    overlays: Any = None,
) -> dict[str, str]:
    """Render 3 orthographic bulk crystal views + 3-panel PNG (render.dpi) and editable SVG overview.

    ``overlays`` (see `normalize_overlays`) draws those structures on every view (loaded once).

    ``cutoff_planes`` / ``whole_molecules`` / ``hbonds``: see `render_single_view`.

    ``style='ellipsoid'`` + ``adp`` (CIF path with U/B/beta) + ``probability`` (percent,
    default 50) draws displacement ellipsoids on every view (see `render_single_view`).

    ``volume`` / ``isovalue`` / ``iso_*`` (see `render_single_view`) overlay the
    isosurfaces of a CHGCAR / Cube / XSF / grd grid on every view (read once).

    ``transparent_background=None`` uses ``render.transparent_background`` from settings.
    ``style`` is one of RENDER_STYLES (None = 'ball-stick').
    ``view_dir`` [u v w] or ``view_normal`` (h k l) adds one more view
    ``images/view_custom.png`` from that direction (the overview figure is unchanged).
    ``perspective`` / ``labels`` ('site' / 'element') apply to every view, as do
    ``polyhedra_center`` / ``bond_color`` and ``frac_range`` (see `render_single_view`).
    ``image_formats`` ("tiff,jpeg,pdf,eps,svg") also writes every view and the overview
    in those formats (keys ``<name>_<format>``); the overview's ``svg`` stays the
    editable ``figure_overview.svg``.
    ``show_polyhedra=None`` (auto) draws polyhedra only when the structure has an
    anion-forming element or ``polyhedra_center`` is given (see `resolve_show_polyhedra`).
    ``planes`` (see `normalize_planes`) draws those (hkl) planes on every view, and
    ``vectors`` / ``vector_*`` (see `render_single_view`) the atom arrows.
    """
    style = normalize_style(style)
    labels = normalize_labels(labels)
    polyhedra_center = normalize_polyhedra_center(polyhedra_center)
    show_polyhedra = resolve_show_polyhedra(show_polyhedra, structure, polyhedra_center)
    bond_color = normalize_bond_color(bond_color)
    frac_range = normalize_frac_range(frac_range)
    image_formats = normalize_image_formats(image_formats)
    planes = normalize_planes(planes)
    vec_kw: dict[str, Any] = {}
    if vectors is not None:
        from shido.core.vectors import normalize_vector_basis, normalize_vectors

        vector_scale, vector_color, vector_radius = normalize_vector_style(
            vector_scale, vector_color, vector_radius
        )
        vec_kw = {
            "vectors": normalize_vectors(vectors),
            "vector_basis": normalize_vector_basis(vector_basis),
            "vector_scale": vector_scale,
            "vector_color": vector_color,
            "vector_radius": vector_radius,
        }
        if vec_kw["vectors"] is None:
            vec_kw = {}
    if volume is not None:  # the isosurface options ride along with the arrow ones
        normalize_iso_style(iso_color, iso_color_negative, iso_opacity)
        vec_kw.update(
            volume=load_volume(volume), isovalue=isovalue, iso_color=iso_color,
            iso_color_negative=iso_color_negative, iso_opacity=iso_opacity,
            iso_both_signs=bool(iso_both_signs),
        )
    if style == "ellipsoid":  # read the CIF once for all views
        tensors, adp_notes = resolve_adp_tensors(structure, adp)
        for note in adp_notes:
            logger.warning("ellipsoid style: %s", note)
        vec_kw.update(adp=tensors, probability=probability)
    if cutoff_planes or whole_molecules or hbonds:  # batch 23 (none by default)
        vec_kw.update(cutoff_planes=cutoff_planes, whole_molecules=bool(whole_molecules), hbonds=hbonds)
    if overlays:  # batch 21 (none by default); bonds of each overlay computed once
        from shido.analyzer import compute_bonds

        ov_specs = normalize_overlays(overlays)
        for spec in ov_specs:
            if spec["render_bonds"] is None and normalize_style(style) != "space-filling":
                spec["render_bonds"] = compute_bonds(spec["structure"], spec["bonds"])[1]
        vec_kw["overlays"] = ov_specs
    custom_view = (
        resolve_view_direction(structure, view_dir=view_dir, view_normal=view_normal)
        if view_dir is not None or view_normal is not None
        else None
    )
    transparent_background = _resolve_transparent(transparent_background)
    images_dir = Path(output_dir) / FIG_DIR
    images_dir.mkdir(parents=True, exist_ok=True)

    a_vec, b_vec, c_vec = structure.lattice.matrix
    # Standard asymmetric oblique orientation (prevents front/back atomic column overlap)
    iso_dir = 1.0 * a_vec + 0.36 * b_vec + 0.24 * c_vec
    caption = _format_bulk_caption(structure, summary=summary)

    view_a = render_single_view(
        structure=structure,
        output_path=images_dir / "view_a_axis.png",
        perspective=perspective,
        labels=labels,
        camera_dir=a_vec,
        view_up=c_vec,
        render_bonds=render_bonds,
        polyhedra_envs=polyhedra_envs,
        show_polyhedra=show_polyhedra,
        style=style,
        polyhedra_center=polyhedra_center,
        bond_color=bond_color,
        frac_range=frac_range,
        image_formats=image_formats,
        planes=planes,
        **vec_kw,
        orientation_label="[100]",
        caption_text=caption,
        transparent_background=transparent_background,
        show_legend=False,
    )
    view_c = render_single_view(
        structure=structure,
        output_path=images_dir / "view_c_axis.png",
        perspective=perspective,
        labels=labels,
        camera_dir=c_vec,
        view_up=b_vec,
        render_bonds=render_bonds,
        polyhedra_envs=polyhedra_envs,
        show_polyhedra=show_polyhedra,
        style=style,
        polyhedra_center=polyhedra_center,
        bond_color=bond_color,
        frac_range=frac_range,
        image_formats=image_formats,
        planes=planes,
        **vec_kw,
        orientation_label="[001]",
        caption_text=caption,
        transparent_background=transparent_background,
        show_legend=False,
    )
    view_iso = render_single_view(
        structure=structure,
        output_path=images_dir / "view_isometric.png",
        perspective=perspective,
        labels=labels,
        camera_dir=iso_dir,
        view_up=c_vec,
        render_bonds=render_bonds,
        polyhedra_envs=polyhedra_envs,
        show_polyhedra=show_polyhedra,
        style=style,
        polyhedra_center=polyhedra_center,
        bond_color=bond_color,
        frac_range=frac_range,
        image_formats=image_formats,
        planes=planes,
        **vec_kw,
        orientation_label="Isometric",
        caption_text=caption,
        transparent_background=transparent_background,
        show_legend=False,
    )

    view_custom = None
    if custom_view is not None:
        cam_dir, up_dir, view_label = custom_view
        view_custom = render_single_view(
            structure=structure,
            output_path=images_dir / "view_custom.png",
            perspective=perspective,
            labels=labels,
            camera_dir=cam_dir,
            view_up=up_dir,
            render_bonds=render_bonds,
            polyhedra_envs=polyhedra_envs,
            show_polyhedra=show_polyhedra,
            style=style,
            polyhedra_center=polyhedra_center,
            bond_color=bond_color,
            frac_range=frac_range,
            image_formats=image_formats,
            planes=planes,
            **vec_kw,
            orientation_label=view_label,
            caption_text=caption,
            transparent_background=transparent_background,
            show_legend=True,
        )

    elements_present = _structure_elements_sorted(structure)
    for spec in vec_kw.get("overlays") or []:
        elements_present = sorted(set(elements_present) | set(_structure_elements_sorted(spec["structure"])))
    overview_path = _compose_overview_figure(
        panel_paths=[view_a, view_c, view_iso],
        output_path=images_dir / "figure_overview.png",
        caption_text=caption,
        panel_size=None,
        strip_panel_footer=True,
        elements_present=elements_present,
        transparent_background=transparent_background,
    )

    svg_path = _compose_overview_svg(
        panel_specs=[
            ("(a) [100]", view_a),
            ("(b) [001]", view_c),
            ("(c) Isometric", view_iso),
        ],
        elements_present=elements_present,
        output_path=images_dir / "figure_overview.svg",
        caption_text=caption,
        panel_size=900,
        strip_panel_footer=True,
        transparent_background=transparent_background,
    )

    result = {
        "view_a_axis": str(view_a),
        "view_c_axis": str(view_c),
        "view_isometric": str(view_iso),
        "figure_overview": str(overview_path),
        "figure_overview_svg": str(svg_path),
    }
    if view_custom is not None:
        result["view_custom"] = str(view_custom)
    if image_formats:
        overview_extra = export_image_formats(
            overview_path, [f for f in image_formats if f != "svg"]
        )
        for fmt, path in overview_extra.items():
            result[f"figure_overview_{fmt}"] = path
        for key in ("view_a_axis", "view_c_axis", "view_isometric", "view_custom"):
            if key not in result:
                continue
            for fmt in image_formats:
                extra = Path(result[key]).with_suffix(_IMAGE_EXT[fmt])
                if extra.is_file():
                    result[f"{key}_{fmt}"] = str(extra)
    return result


def render_slab_views(
    slab: Structure,
    output_dir: str | Path,
    file_prefix: str,
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]] | None = None,
    miller_index: tuple[int, int, int] | None = None,
    slab_caption: str | None = None,
    transparent_background: bool | None = None,
    style: str | None = None,
) -> dict[str, str]:
    """Render 2 orthographic slab views (top along +c, side along +a) with annotations.

    ``transparent_background=None`` (the default, also what slab_cutter passes) uses
    ``render.transparent_background`` from settings.
    """
    transparent_background = _resolve_transparent(transparent_background)
    images_dir = Path(output_dir) / FIG_DIR
    images_dir.mkdir(parents=True, exist_ok=True)

    a_vec, b_vec, c_vec = slab.lattice.matrix

    if miller_index is not None:
        hkl = _format_miller_index(miller_index)
        top_label = f"{hkl} Top"
        side_label = f"{hkl} Side"
    else:
        top_label = "Top [001]"
        side_label = "Side [100]"

    caption = slab_caption if slab_caption is not None else _format_slab_caption(slab, miller_index)

    top_path = render_single_view(
        structure=slab,
        output_path=images_dir / f"{file_prefix}_top.png",
        camera_dir=c_vec,
        view_up=b_vec,
        render_bonds=render_bonds,
        polyhedra_envs=None,
        show_polyhedra=False,
        style=style,
        orientation_label=top_label,
        caption_text=caption,
        transparent_background=transparent_background,
    )
    side_path = render_single_view(
        structure=slab,
        output_path=images_dir / f"{file_prefix}_side.png",
        camera_dir=a_vec,
        view_up=c_vec,
        render_bonds=render_bonds,
        polyhedra_envs=None,
        show_polyhedra=False,
        style=style,
        orientation_label=side_label,
        caption_text=caption,
        transparent_background=transparent_background,
    )

    return {
        "top": str(top_path),
        "side": str(side_path),
    }


def render_batch_gallery(
    image_entries: list[tuple[str, str | Path]],
    output_path: str | Path,
    cols: int = 4,
    thumb_size: tuple[int, int] | None = None,
) -> Path | None:
    """Create a publication contact-sheet PNG (render.dpi) arranging thumbnails with titles in a grid.

    ``thumb_size=None`` is 480 x 480 px at 300 dpi; sizes scale with render.dpi / 300.
    """
    ps = _px_scale()
    if thumb_size is None:
        thumb_size = (_px(480, ps), _px(480, ps))
    valid_entries: list[tuple[str, Path]] = []
    for title, raw_path in image_entries:
        p = Path(raw_path)
        if p.is_file():
            valid_entries.append((str(title), p))

    if not valid_entries:
        return None

    tw, th = max(_px(64, ps), int(thumb_size[0])), max(_px(64, ps), int(thumb_size[1]))
    n_items = len(valid_entries)
    n_cols = min(max(1, int(cols)), n_items)
    n_rows = math.ceil(n_items / n_cols)

    title_h = _px(46, ps)
    pad = _px(24, ps)
    gap = _px(18, ps)
    cell_w = tw
    cell_h = th + title_h

    canvas_w = pad * 2 + n_cols * cell_w + (n_cols - 1) * gap
    canvas_h = pad * 2 + n_rows * cell_h + (n_rows - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(_px(15, ps), min(_px(20, ps), int(round(tw * 0.04)))))

    for idx, (title, img_path) in enumerate(valid_entries):
        row = idx // n_cols
        col = idx % n_cols
        x0 = pad + col * (cell_w + gap)
        y0 = pad + row * (cell_h + gap)

        # Card background & crisp border
        draw.rectangle(
            [(x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1)],
            fill=(255, 255, 255),
            outline=(195, 195, 195),
            width=_px(2, ps),
        )

        # Thumbnail (aspect-ratio preserved, centered)
        with Image.open(img_path) as im:
            if im.mode in ("RGBA", "LA", "P"):
                # Transparent renders: flatten onto the white card instead of dropping alpha.
                rgba = im.convert("RGBA")
                flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                src = Image.alpha_composite(flat, rgba).convert("RGB")
            else:
                src = im.convert("RGB")
            inset = _px(8, ps)
            thumb = ImageOps.contain(src, (tw - inset, th - inset), Image.Resampling.LANCZOS)
        paste_x = x0 + (tw - thumb.width) // 2
        paste_y = y0 + (th - thumb.height) // 2
        canvas.paste(thumb, (paste_x, paste_y))

        # Title footer divider & centered label
        title_top = y0 + th
        draw.line([(x0, title_top), (x0 + cell_w - 1, title_top)], fill=(220, 220, 220), width=_px(1, ps))

        display_title = title
        max_w = cell_w - _px(20, ps)
        while len(display_title) > 3:
            bbox = draw.textbbox((0, 0), display_title, font=font)
            if (bbox[2] - bbox[0]) <= max_w:
                break
            display_title = display_title[:-2].rstrip() + "…"

        bbox = draw.textbbox((0, 0), display_title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = x0 + (cell_w - text_w) / 2.0 - bbox[0]
        ty = title_top + (title_h - text_h) / 2.0 - bbox[1]
        draw.text((tx, ty), display_title, fill=(24, 24, 24), font=font)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="PNG", dpi=_dpi_pair())
    return out.resolve()


def render_morphology(
    morph: Any,
    output_path: str | Path,
    view_dir: Any = None,
    window_size: tuple[int, int] = (1200, 1000),
    show_legend: bool = True,
    show_axes: bool = True,
) -> Path:
    """Draw a `core.morphology.Morphology` (batch 20): faces coloured per {hkl}, dark edges.

    ``view_dir``: Cartesian viewing direction (from the crystal towards the camera);
    default a general oblique direction. The a, b, c directions are drawn as short
    arrows beside the crystal when ``show_axes``.
    """
    verts = np.asarray(morph.vertices, dtype=float)
    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    radius = float(np.linalg.norm(verts - center, axis=1).max())
    d = np.asarray(view_dir if view_dir is not None else (1.0, 0.55, 0.4), dtype=float)
    if np.linalg.norm(d) < 1e-12:
        raise ValueError("view_dir must not be zero")
    d = d / np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.95 else np.array([0.0, 1.0, 0.0])

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        plotter.set_background(str(get_color_manager().scene_colors.get("background", "#FFFFFF")))
        plotter.enable_parallel_projection()
        legend = []
        for i, fam in enumerate(morph.families):
            if not fam.get("n_on_crystal"):
                continue
            mesh = morph.to_polydata(family=i)
            plotter.add_mesh(mesh, color=fam["color"], show_edges=True, edge_color="#222222",
                             line_width=2.0, smooth_shading=False, ambient=0.25, diffuse=0.75)
            legend.append(fam)
        if show_axes:
            lat = np.asarray(morph.lattice_matrix, dtype=float)
            up_o = up - d * (up @ d)
            up_o /= np.linalg.norm(up_o)
            right = np.cross(-d, up_o)
            base = center - 1.55 * radius * right - 1.2 * radius * up_o
            for vec, name, col in zip(lat, ("a", "b", "c"), ("#D62728", "#2CA02C", "#1F77B4")):
                u = vec / np.linalg.norm(vec)
                plotter.add_mesh(pv.Arrow(start=base, direction=u, scale=0.45 * radius), color=col)
                plotter.add_point_labels([base + u * 0.55 * radius], [name], font_size=22, text_color=col,
                                         shape=None, show_points=False, always_visible=True)
        if show_legend and legend:
            total = sum(f["area"] for f in morph.families) or 1.0
            fs = max(10, int(window_size[1] / 50))
            for k, f in enumerate(legend):
                plotter.add_text(f"{f['family']}  d={f['distance']:g}  {100.0 * f['area'] / total:.1f}%",
                                 position=(int(0.03 * window_size[0]),
                                           int(window_size[1] - (k + 1) * 2.0 * fs - 0.02 * window_size[1])),
                                 font_size=fs, color=f["color"])
        plotter.camera.focal_point = tuple(center)
        plotter.camera.position = tuple(center + d * radius * 10.0)
        plotter.camera.up = tuple(up)
        plotter.reset_camera()
        plotter.camera.parallel_scale = radius * (1.75 if show_axes else 1.15)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        screenshot_checked(plotter, out, context="morphology")
    finally:
        plotter.close()
    return out.resolve()

"""Two-phase side-by-side lattice correspondence comparison and Bain distortion visualizer."""

from __future__ import annotations

import base64
import io
import os
import warnings
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from shido.analyzer import build_ni2mnga_structure, compute_bonds, to_ordered_proxy
from shido.core.output_paths import DATA_DIR, FIG_DIR, STRUCT_DIR  # noqa: E402
from shido.core.validator import validate_composition_conserved
from shido.settings import get as _setting
from shido.renderer import (
    screenshot_checked,
    _add_atom_mesh_with_silhouette,
    _add_bonds,
    _add_unit_cell,
    _build_disordered_sphere,
    _collect_atoms_to_draw,
    _compute_parallel_framing,
    _element_color,
    _format_scale_label,
    _hex_to_rgb,
    _load_font,
    _dpi_pair,
    _merge_spheres,
    _orthonormal_up,
    _px_scale,
    _res_scale,
    _select_scale_bar_length,
    _structure_elements_sorted,
    add_orientation_axes,
    apply_studio_lighting,
    line_radius_ratio,
)

# Suppress noisy pymatgen UserWarnings (e.g., CIF rounding / site tolerance)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen(\..*)?")

# Bain-correspondent 45°-rotated tetragonal subcell inside the cubic parent cell
_BAIN_BASE_FRAC: tuple[tuple[float, float], ...] = (
    (0.5, 0.0),
    (1.0, 0.5),
    (0.5, 1.0),
    (0.0, 0.5),
)

_BAIN_SUBCELL_EDGES: tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]], ...
] = (
    # Bottom face (z = 0)
    ((0.5, 0.0, 0.0), (1.0, 0.5, 0.0)),
    ((1.0, 0.5, 0.0), (0.5, 1.0, 0.0)),
    ((0.5, 1.0, 0.0), (0.0, 0.5, 0.0)),
    ((0.0, 0.5, 0.0), (0.5, 0.0, 0.0)),
    # Top face (z = 1)
    ((0.5, 0.0, 1.0), (1.0, 0.5, 1.0)),
    ((1.0, 0.5, 1.0), (0.5, 1.0, 1.0)),
    ((0.5, 1.0, 1.0), (0.0, 0.5, 1.0)),
    ((0.0, 0.5, 1.0), (0.5, 0.0, 1.0)),
    # Vertical edges
    ((0.5, 0.0, 0.0), (0.5, 0.0, 1.0)),
    ((1.0, 0.5, 0.0), (1.0, 0.5, 1.0)),
    ((0.5, 1.0, 0.0), (0.5, 1.0, 1.0)),
    ((0.0, 0.5, 0.0), (0.0, 0.5, 1.0)),
)

_BAIN_COLOR_HEX: str = "#D62728"
_DEFAULT_CAMERA_DIR: tuple[float, float, float] = (0.35, 1.0, 0.25)
_DEFAULT_VIEW_UP: tuple[float, float, float] = (0.0, 0.0, 1.0)
_PANEL_SIZE: int = 960  # px at 300 dpi; multiplied by render.dpi / 300 at render time


def _k(value: float, px_scale: float) -> int:
    """A 300-dpi pixel length at the current dpi (0 stays 0)."""
    return int(round(float(value) * px_scale))


def build_ni2mnga_martensite_l10(
    a: float | None = None,
    c_over_a: float | None = None,
) -> Structure:
    """Build a constant-volume Bain-distorted tetragonal L1_0-like Ni2MnGa martensite structure.

    Starts from the cubic L2_1 Heusler parent (`build_ni2mnga_structure(a)`) and applies
    a volume-conserving tetragonal Bain distortion:
        scale_ab = (c_over_a) ** (-1/3)
        scale_c  = (c_over_a) ** (2/3)
    while preserving all fractional atomic coordinates and site species.

    ``a`` / ``c_over_a`` left as ``None`` are read from the settings
    (``lattice.a_austenite`` / ``lattice.martensite_c_over_a``) at call time.
    """
    if a is None:
        a = _setting("lattice.a_austenite", 5.825)
    if c_over_a is None:
        c_over_a = _setting("lattice.martensite_c_over_a", 1.25)
    a_val = float(a)
    ratio = float(c_over_a)
    if a_val <= 0.0:
        raise ValueError(f"Lattice parameter 'a' must be positive, got {a_val}.")
    if ratio <= 0.0:
        raise ValueError(f"Axial ratio 'c_over_a' must be positive, got {ratio}.")

    parent = build_ni2mnga_structure(a=a_val)
    scale_ab = ratio ** (-1.0 / 3.0)
    scale_c = ratio ** (2.0 / 3.0)
    tet_lattice = Lattice.tetragonal(a_val * scale_ab, a_val * scale_c)

    species = [site.species for site in parent]
    frac_coords = [np.asarray(site.frac_coords, dtype=float) for site in parent]

    return Structure(
        lattice=tet_lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


def _supercell_diag(mult: int) -> list[list[int]]:
    """Small diagonal supercell matrix with determinant ``mult`` (first candidate)."""
    d = _diag_candidates(mult)[0]
    return [[d[0], 0, 0], [0, d[1], 0], [0, 0, d[2]]]


def _diag_candidates(mult: int) -> list[tuple[int, int, int]]:
    """All diagonal (n1, n2, n3) with n1 * n2 * n3 == mult (every axis order)."""
    out = [
        (i, j, mult // (i * j))
        for i in range(1, mult + 1)
        for j in range(1, mult + 1)
        if mult % (i * j) == 0
    ]
    return sorted(set(out), reverse=True)


def _shape_mismatch(ref_abc: np.ndarray, target_abc: np.ndarray) -> float:
    """0 when the two cells have proportional a:b:c (spread of log length ratios)."""
    r = np.log(np.asarray(target_abc, dtype=float) / np.asarray(ref_abc, dtype=float))
    return float(np.ptp(r))


def _conventional_cell(struct: Structure) -> Structure:
    """Conventional standard cell (falls back to the input if symmetry analysis fails)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SpacegroupAnalyzer(struct, symprec=0.1).get_conventional_standard_structure()
    except Exception:
        return struct


def _lattice_correspondence(
    parent_struct: Structure,
    martensite_struct: Structure,
) -> dict[str, Any] | None:
    """Find the parent <-> martensite lattice correspondence and the shuffle displacements.

    The parent is first converted to its conventional standard cell, so the strains are
    along the parent conventional [100] / [010] / [001] whatever cell the file uses
    (e.g. a 4-atom primitive fcc cell). When the martensite cell is larger, the parent
    supercell (diagonal, every axis order) whose a:b:c is closest to the martensite's is
    tried first, so a 1x1x2 martensite keeps its long axis along the parent c.

    ``StructureMatcher`` (volume-normalised, supercells allowed, same element only)
    rewrites the martensite in a basis like the parent cell and orders its sites
    like the parent's. In that basis the homogeneous lattice strain leaves the
    fractional coordinates unchanged, so ``frac_m - frac_p`` is the internal
    displacement only ("shuffle"). The mean (rigid) shift is subtracted.

    Returns ``None`` when no correspondence is found (different compositions or
    too large a distortion).
    """
    try:
        p_ord = _conventional_cell(to_ordered_proxy(parent_struct))
        m_ord = to_ordered_proxy(martensite_struct)
    except Exception:
        return None
    if p_ord.composition.reduced_composition != m_ord.composition.reduced_composition:
        return None

    matcher = StructureMatcher(
        ltol=float(_setting("phase_compare.match_ltol", 0.35)),
        stol=float(_setting("phase_compare.match_stol", 0.6)),
        angle_tol=float(_setting("phase_compare.match_angle_tol", 10.0)),
        primitive_cell=False,  # get_transformation does not allow True
        scale=True,
        attempt_supercell=True,
    )

    # get_s2_like_s1 needs struct1 to be at least as large: grow the parent if needed.
    m_like = None
    ref = p_ord
    p_abc = np.array(p_ord.lattice.abc, dtype=float)
    m_abc = np.array(m_ord.lattice.abc, dtype=float)
    for mult in (1, 2, 3, 4, 6, 8):
        if len(p_ord) * mult < len(m_ord):
            continue
        cands = sorted(
            _diag_candidates(mult),
            key=lambda d: _shape_mismatch(p_abc * np.asarray(d, dtype=float), m_abc),
        )
        for d in cands:
            ref = p_ord.copy()
            if mult > 1:
                ref.make_supercell([[d[0], 0, 0], [0, d[1], 0], [0, 0, d[2]]])
                validate_composition_conserved(p_ord, ref, context=f"supercell {d}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m_like = matcher.get_s2_like_s1(ref, m_ord)
            except Exception:
                m_like = None
            if m_like is not None and len(m_like) == len(ref):
                break
            m_like = None
        if m_like is not None:
            break
    if m_like is None:
        return None
    if not all(sp.species == sm.species for sp, sm in zip(ref, m_like)):
        return None

    f_p = np.array([site.frac_coords for site in ref], dtype=float)
    f_m = np.array([site.frac_coords for site in m_like], dtype=float)
    df = np.mod(f_m - f_p + 0.5, 1.0) - 0.5
    cart = df @ m_like.lattice.matrix
    cart -= cart.mean(axis=0)
    norms = np.linalg.norm(cart, axis=1)

    p_len = np.array(ref.lattice.abc, dtype=float)
    m_len = np.array(m_like.lattice.abc, dtype=float)

    # Principal stretches of the homogeneous deformation (Bain strains)
    f_grad = np.linalg.solve(ref.lattice.matrix, m_like.lattice.matrix).T
    principal = np.sqrt(np.clip(np.linalg.eigvalsh(f_grad.T @ f_grad), 0.0, None))

    return {
        "stretch_abc": m_len / p_len,  # along the parent a, b, c in the matched basis
        "principal_stretches": np.sort(principal),
        "max_shuffle": float(norms.max()) if len(norms) else 0.0,
        "rms_shuffle": float(np.sqrt(np.mean(norms**2))) if len(norms) else 0.0,
        "num_matched_sites": int(len(ref)),
    }


def _add_bain_subcell(
    plotter: pv.Plotter,
    structure: Structure,
    color: str = _BAIN_COLOR_HEX,
    tube_radius: float | None = None,
    dash_length: float = 0.34,
    gap_length: float = 0.22,
) -> None:
    """Overlay the 45°-rotated Bain tetragonal subcell as red dashed tubes.

    ``tube_radius=None``: 0.045 Å times (render.unit_cell_line_radius / its default).
    """
    if tube_radius is None:
        tube_radius = 0.045 * line_radius_ratio("cell")
    lattice = structure.lattice
    dash_tubes: list[pv.PolyData] = []
    period = dash_length + gap_length

    for f0, f1 in _BAIN_SUBCELL_EDGES:
        p0 = np.asarray(lattice.get_cartesian_coords(f0), dtype=float)
        p1 = np.asarray(lattice.get_cartesian_coords(f1), dtype=float)
        vec = p1 - p0
        total_len = float(np.linalg.norm(vec))
        if total_len < 1e-6:
            continue
        unit = vec / total_len

        dist = 0.0
        while dist < total_len:
            seg_end = min(dist + dash_length, total_len)
            if seg_end - dist > 1e-4:
                a_pt = p0 + unit * dist
                b_pt = p0 + unit * seg_end
                dash_tubes.append(pv.Line(a_pt, b_pt).tube(radius=tube_radius, n_sides=12))
            dist += period

    if dash_tubes:
        merged = pv.merge(dash_tubes) if len(dash_tubes) > 1 else dash_tubes[0]
        plotter.add_mesh(merged, color=color, smooth_shading=True)


def _populate_phase_scene(
    plotter: pv.Plotter,
    structure: Structure,
    atoms: list[tuple[np.ndarray, list[tuple[str, float]], float]],
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]],
    overlay_bain_subcell: bool = False,
) -> None:
    """Populate a PyVista plotter with unit cell, atom spheres, bonds, and optional Bain subcell.

    Sphere radii come from ``atoms`` (render.ball_radius_* / atomic_radii), the cell and
    bonds from render.unit_cell_line_radius / render.bond_radius.
    """
    px_scale = _px_scale()
    _add_unit_cell(plotter, structure)

    spheres_by_elem: dict[str, tuple[list[np.ndarray], list[float]]] = {}
    disordered_spheres: list[pv.PolyData] = []
    res_ordered = _res_scale(24, px_scale)
    res_disordered = _res_scale(32, px_scale)

    for cart, species_occs, radius in atoms:
        is_disordered = len(species_occs) > 1 or (
            len(species_occs) == 1 and species_occs[0][1] < 0.995
        )
        if is_disordered:
            disordered_spheres.append(
                _build_disordered_sphere(cart, radius, species_occs, resolution=res_disordered)
            )
        else:
            centers, radii = spheres_by_elem.setdefault(species_occs[0][0], ([], []))
            centers.append(cart)
            radii.append(radius)

    for elem, (centers, radii) in spheres_by_elem.items():
        merged = _merge_spheres(centers, radii, res_ordered)
        _add_atom_mesh_with_silhouette(
            plotter,
            merged,
            px_scale=px_scale,
            color=_element_color(elem),
            smooth_shading=True,
            specular=0.35,
            specular_power=25,
        )

    if disordered_spheres:
        merged_disordered = (
            pv.merge(disordered_spheres)
            if len(disordered_spheres) > 1
            else disordered_spheres[0]
        )
        _add_atom_mesh_with_silhouette(
            plotter,
            merged_disordered,
            px_scale=px_scale,
            scalars="rgb",
            rgb=True,
            smooth_shading=True,
            specular=0.35,
            specular_power=25,
        )

    if render_bonds:
        _add_bonds(plotter, render_bonds)

    if overlay_bain_subcell:
        _add_bain_subcell(plotter, structure)


def _render_phase_panel(
    structure: Structure,
    atoms: list[tuple[np.ndarray, list[tuple[str, float]], float]],
    render_bonds: list[tuple[np.ndarray, np.ndarray, str, str]],
    cell_center: np.ndarray,
    cam_unit: np.ndarray,
    up_unit: np.ndarray,
    parallel_scale: float,
    output_path: Path,
    window_size: tuple[int, int] = (_PANEL_SIZE, _PANEL_SIZE),
    overlay_bain_subcell: bool = False,
    transparent_background: bool = False,
) -> Path:
    """Render one phase panel with exact shared orthographic scale and [100]/[010]/[001] triad."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        plotter.set_background("white")
        plotter.enable_parallel_projection()
        apply_studio_lighting(plotter, parallel_scale=parallel_scale)

        _populate_phase_scene(
            plotter=plotter,
            structure=structure,
            atoms=atoms,
            render_bonds=render_bonds,
            overlay_bain_subcell=overlay_bain_subcell,
        )

        # Clean 3-axis crystallographic triad in bottom-left
        add_orientation_axes(
            plotter, xlabel="[100]", ylabel="[010]", zlabel="[001]", line_width=2, color="black"
        )

        diag_len = float(
            np.linalg.norm(structure.lattice.get_cartesian_coords([1.0, 1.0, 1.0]))
        )
        cam_dist = max(diag_len * 3.0, parallel_scale * 4.0, 20.0)
        cam_pos = cell_center + cam_unit * cam_dist

        plotter.camera_position = [
            cam_pos.tolist(),
            cell_center.tolist(),
            up_unit.tolist(),
        ]
        plotter.camera.parallel_scale = float(parallel_scale)
        plotter.renderer.ResetCameraClippingRange()

        screenshot_checked(
            plotter,
            output_path,
            transparent_background=transparent_background,
            context=f"phase panel {Path(output_path).name}",
        )
    finally:
        plotter.close()

    return output_path.resolve()


def _format_signed_pct(val: float) -> str:
    """Format a percentage delta with explicit sign."""
    return f"{val:+.2f}%"


def _fmt_pct_or_na(val: float) -> str:
    return "n/a" if val is None or not np.isfinite(val) else _format_signed_pct(val)


def _build_metrics_lines(metrics: dict[str, Any]) -> tuple[str, str]:
    """Format two concise publication lines for the bottom metrics strip.

    Δa/a, Δc/c: strain along the parent a / c axes (matched lattice correspondence).
    ΔV/V: volume per atom. Shuffle: atom displacement left after removing the
    homogeneous lattice strain and the rigid shift.
    """
    line1 = (
        f"Parent: a = {metrics['parent_a']:.3f} Å, c = {metrics['parent_c']:.3f} Å, "
        f"c/a = {metrics['parent_c_over_a']:.3f}   →   "
        f"Martensite: a = {metrics['mart_a']:.3f} Å, c = {metrics['mart_c']:.3f} Å, "
        f"c/a = {metrics['mart_c_over_a']:.3f}"
    )
    line2 = (
        f"Δa/a = {_fmt_pct_or_na(metrics['delta_a_pct'])}, "
        f"Δc/c = {_fmt_pct_or_na(metrics['delta_c_pct'])} (parent axes)   |   "
        f"ΔV/V = {_fmt_pct_or_na(metrics['delta_v_pct'])} (per atom)   |   "
        + (
            "Max shuffle = n/a"
            if not np.isfinite(metrics["max_shuffle_angstrom"])
            else f"Max shuffle (strain removed) = {metrics['max_shuffle_angstrom']:.3f} Å"
        )
    )
    return line1, line2


def _compose_phase_compare_png(
    parent_panel_path: Path,
    mart_panel_path: Path,
    output_path: Path,
    elements_present: list[str],
    angstrom_per_pixel: float,
    metrics: dict[str, float],
    panel_size: int | None = None,
    transparent_background: bool = False,
    px_scale: float | None = None,
) -> Path:
    """Compose the side-by-side 2-phase comparison figure as a render.dpi PNG.

    Layout sizes are 300-dpi pixels times ``px_scale`` (default render.dpi / 300), so the
    printed size is the same at any dpi. ``angstrom_per_pixel`` is per panel pixel.
    """
    ps = _px_scale() if px_scale is None else float(px_scale)
    if panel_size is None:
        panel_size = _k(_PANEL_SIZE, ps)
    pad = _k(32, ps)
    header_h = _k(54, ps)
    arrow_w = _k(136, ps)
    legend_h = _k(62, ps)
    metrics_h = _k(94, ps)

    total_w = pad * 2 + panel_size * 2 + arrow_w
    panel_y0 = pad + header_h
    legend_y0 = panel_y0 + panel_size + _k(14, ps)
    metrics_y0 = legend_y0 + legend_h + _k(10, ps)
    total_h = metrics_y0 + metrics_h + pad

    mode = "RGBA" if transparent_background else "RGB"
    bg = (255, 255, 255, 0) if transparent_background else (255, 255, 255)
    text_dark = (24, 24, 24, 255) if transparent_background else (24, 24, 24)
    text_muted = (72, 72, 72, 255) if transparent_background else (72, 72, 72)
    border_col = (205, 205, 205, 255) if transparent_background else (205, 205, 205)
    arrow_col = (40, 40, 40, 255) if transparent_background else (40, 40, 40)
    bain_rgb = _hex_to_rgb(_BAIN_COLOR_HEX)
    bain_col = (*bain_rgb, 255) if transparent_background else bain_rgb

    canvas = Image.new(mode, (total_w, total_h), bg)
    draw = ImageDraw.Draw(canvas)

    left_x0 = pad
    right_x0 = pad + panel_size + arrow_w

    with Image.open(parent_panel_path) as im_p:
        p_img = im_p.convert(mode)
        if p_img.size != (panel_size, panel_size):
            p_img = p_img.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
    with Image.open(mart_panel_path) as im_m:
        m_img = im_m.convert(mode)
        if m_img.size != (panel_size, panel_size):
            m_img = m_img.resize((panel_size, panel_size), Image.Resampling.LANCZOS)

    if transparent_background:
        canvas.paste(p_img, (left_x0, panel_y0), p_img)
        canvas.paste(m_img, (right_x0, panel_y0), m_img)
    else:
        canvas.paste(p_img, (left_x0, panel_y0))
        canvas.paste(m_img, (right_x0, panel_y0))

    # Crisp panel borders
    for x0 in (left_x0, right_x0):
        draw.rectangle(
            [(x0, panel_y0), (x0 + panel_size - 1, panel_y0 + panel_size - 1)],
            outline=border_col,
            width=_k(2, ps),
        )

    # Panel titles
    title_font = _load_font(_k(30, ps))
    draw.text((left_x0 + _k(4, ps), pad + _k(10, ps)), "(a) Parent Phase (Austenite)", fill=text_dark, font=title_font)
    draw.text((right_x0 + _k(4, ps), pad + _k(10, ps)), "(b) Martensite Phase", fill=text_dark, font=title_font)

    # Transition arrow between panels
    arrow_cx = left_x0 + panel_size + arrow_w // 2
    arrow_cy = panel_y0 + panel_size // 2
    shaft_x0 = left_x0 + panel_size + _k(22, ps)
    shaft_x1 = right_x0 - _k(38, ps)
    draw.line([(shaft_x0, arrow_cy), (shaft_x1, arrow_cy)], fill=arrow_col, width=_k(7, ps))
    head_tip = (right_x0 - _k(18, ps), arrow_cy)
    head_top = (right_x0 - _k(44, ps), arrow_cy - _k(18, ps))
    head_bot = (right_x0 - _k(44, ps), arrow_cy + _k(18, ps))
    draw.polygon([head_tip, head_top, head_bot], fill=arrow_col)

    arrow_font = _load_font(_k(22, ps))
    lbl_top = "Bain"
    bbox_t = draw.textbbox((0, 0), lbl_top, font=arrow_font)
    draw.text(
        (arrow_cx - (bbox_t[2] - bbox_t[0]) / 2.0 - bbox_t[0], arrow_cy - _k(54, ps)),
        lbl_top,
        fill=text_muted,
        font=arrow_font,
    )
    lbl_bot = "distortion"
    bbox_b = draw.textbbox((0, 0), lbl_bot, font=arrow_font)
    draw.text(
        (arrow_cx - (bbox_b[2] - bbox_b[0]) / 2.0 - bbox_b[0], arrow_cy + _k(28, ps)),
        lbl_bot,
        fill=text_muted,
        font=arrow_font,
    )

    # Legend strip (left/middle) + shared scale bar (right)
    legend_cy = legend_y0 + legend_h // 2
    legend_font = _load_font(_k(24, ps))
    cursor_x = left_x0 + _k(18, ps)
    for elem in elements_present:
        rgb = _hex_to_rgb(_element_color(elem))
        fill_c = (*rgb, 255) if transparent_background else rgb
        r = _k(12, ps)
        draw.ellipse(
            [(cursor_x - r, legend_cy - r), (cursor_x + r, legend_cy + r)],
            fill=fill_c,
            outline=text_dark,
            width=_k(2, ps),
        )
        draw.text((cursor_x + _k(20, ps), legend_cy - _k(14, ps)), elem, fill=text_dark, font=legend_font)
        cursor_x += _k(110, ps)

    # Red dashed Bain subcell swatch in legend
    cursor_x += _k(18, ps)
    for dx in (0, 18, 36):
        draw.line(
            [(cursor_x + _k(dx, ps), legend_cy), (cursor_x + _k(dx + 11, ps), legend_cy)],
            fill=bain_col,
            width=_k(4, ps),
        )
    draw.text((cursor_x + _k(58, ps), legend_cy - _k(14, ps)), "Bain subcell", fill=bain_col, font=legend_font)

    # Shared scale bar (bottom-right of legend row, exact Å/px for both panels)
    bar_angstrom, bar_px = _select_scale_bar_length(panel_size, angstrom_per_pixel, px_scale=ps)
    scale_label = f"Scale (identical): {_format_scale_label(bar_angstrom)}"
    scale_font = _load_font(_k(22, ps))
    sb_x1 = right_x0 + panel_size - _k(16, ps)
    sb_x0 = sb_x1 - bar_px
    sb_y = legend_cy + _k(10, ps)
    t2, t4, t10 = _k(2, ps), _k(4, ps), _k(10, ps)
    draw.rectangle([(sb_x0, sb_y - t4), (sb_x1, sb_y + t2)], fill=text_dark)
    draw.rectangle([(sb_x0, sb_y - t10), (sb_x0 + t2, sb_y + t2)], fill=text_dark)
    draw.rectangle([(sb_x1 - t2, sb_y - t10), (sb_x1, sb_y + t2)], fill=text_dark)

    s_bbox = draw.textbbox((0, 0), scale_label, font=scale_font)
    s_w = s_bbox[2] - s_bbox[0]
    draw.text(
        (sb_x0 + (bar_px - s_w) / 2.0 - s_bbox[0], legend_cy - _k(26, ps)),
        scale_label,
        fill=text_dark,
        font=scale_font,
    )

    # Bottom metrics strip
    draw.line([(pad, metrics_y0), (total_w - pad, metrics_y0)], fill=border_col, width=_k(2, ps))
    line1, line2 = _build_metrics_lines(metrics)
    font_l1 = _load_font(_k(23, ps))
    font_l2 = _load_font(_k(26, ps))

    for text_str, font, y_pos, fill in (
        (line1, font_l1, metrics_y0 + _k(14, ps), text_muted),
        (line2, font_l2, metrics_y0 + _k(50, ps), text_dark),
    ):
        bbox = draw.textbbox((0, 0), text_str, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) / 2.0 - bbox[0], y_pos), text_str, fill=fill, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", dpi=_dpi_pair())
    return output_path.resolve()


def _encode_panel_png_b64(panel_path: Path, panel_size: int) -> str:
    """Encode a panel PNG resized to (panel_size, panel_size) pixels as base64."""
    with Image.open(panel_path) as im:
        img = im.convert("RGBA") if im.mode == "RGBA" else im.convert("RGB")
        if img.size != (panel_size, panel_size):
            img = img.resize((panel_size, panel_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", dpi=_dpi_pair())
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _compose_phase_compare_svg(
    parent_panel_path: Path,
    mart_panel_path: Path,
    output_path: Path,
    elements_present: list[str],
    angstrom_per_pixel: float,
    metrics: dict[str, float],
    panel_size: int = _PANEL_SIZE,
    transparent_background: bool = False,
    embed_px: int | None = None,
) -> Path:
    """Compose an editable vector SVG 2-phase comparison figure with embedded base64 panels.

    ``panel_size`` and ``angstrom_per_pixel`` are in SVG units (300-dpi pixels); the panel
    bitmaps are embedded at ``embed_px`` pixels (default panel_size x render.dpi / 300).
    """
    if embed_px is None:
        embed_px = _k(panel_size, _px_scale())
    pad = 32
    header_h = 54
    arrow_w = 136
    legend_h = 62
    metrics_h = 94

    total_w = pad * 2 + panel_size * 2 + arrow_w
    panel_y0 = pad + header_h
    legend_y0 = panel_y0 + panel_size + 14
    metrics_y0 = legend_y0 + legend_h + 10
    total_h = metrics_y0 + metrics_h + pad

    left_x0 = pad
    right_x0 = pad + panel_size + arrow_w

    b64_parent = _encode_panel_png_b64(parent_panel_path, embed_px)
    b64_mart = _encode_panel_png_b64(mart_panel_path, embed_px)

    bar_angstrom, bar_px = _select_scale_bar_length(panel_size, angstrom_per_pixel)
    scale_label = f"Scale (identical): {_format_scale_label(bar_angstrom)}"
    line1, line2 = _build_metrics_lines(metrics)

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
        "    .panel-title { font-size: 30px; font-weight: 700; }",
        "    .arrow-label { font-size: 21px; font-weight: 500; fill: #484848; }",
        "    .legend-label { font-size: 24px; font-weight: 600; }",
        "    .scale-label { font-size: 22px; font-weight: 600; }",
        "    .metrics-sub { font-size: 23px; font-weight: 500; fill: #484848; }",
        "    .metrics-main { font-size: 26px; font-weight: 700; fill: #181818; }",
        "  </style>",
    ]

    if not transparent_background:
        svg_lines.append(
            f'  <rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#FFFFFF"/>'
        )

    # Panel (a): Parent Phase
    svg_lines.extend(
        [
            '  <g id="panel_parent">',
            (
                f'    <text x="{left_x0 + 4}" y="{pad + 36}" class="panel-title" '
                f'font-family="{font_family}">{xml_escape("(a) Parent Phase (Austenite)")}</text>'
            ),
            (
                f'    <image x="{left_x0}" y="{panel_y0}" width="{panel_size}" height="{panel_size}" '
                f'preserveAspectRatio="xMidYMid meet" '
                f'href="data:image/png;base64,{b64_parent}" '
                f'xlink:href="data:image/png;base64,{b64_parent}"/>'
            ),
            (
                f'    <rect x="{left_x0}" y="{panel_y0}" width="{panel_size}" height="{panel_size}" '
                f'fill="none" stroke="#CDCDCD" stroke-width="2"/>'
            ),
            "  </g>",
        ]
    )

    # Center transformation arrow
    arrow_cx = left_x0 + panel_size + arrow_w / 2.0
    arrow_cy = panel_y0 + panel_size / 2.0
    shaft_x0 = left_x0 + panel_size + 22
    shaft_x1 = right_x0 - 38
    svg_lines.extend(
        [
            '  <g id="transformation_arrow">',
            (
                f'    <line x1="{shaft_x0}" y1="{arrow_cy:.1f}" '
                f'x2="{shaft_x1}" y2="{arrow_cy:.1f}" '
                f'stroke="#282828" stroke-width="7" stroke-linecap="round"/>'
            ),
            (
                f'    <polygon points="{right_x0 - 18},{arrow_cy:.1f} '
                f'{right_x0 - 44},{arrow_cy - 18:.1f} '
                f'{right_x0 - 44},{arrow_cy + 18:.1f}" fill="#282828"/>'
            ),
            (
                f'    <text x="{arrow_cx:.1f}" y="{arrow_cy - 34:.1f}" text-anchor="middle" '
                f'class="arrow-label" font-family="{font_family}">Bain</text>'
            ),
            (
                f'    <text x="{arrow_cx:.1f}" y="{arrow_cy + 48:.1f}" text-anchor="middle" '
                f'class="arrow-label" font-family="{font_family}">distortion</text>'
            ),
            "  </g>",
        ]
    )

    # Panel (b): Martensite Phase
    svg_lines.extend(
        [
            '  <g id="panel_martensite">',
            (
                f'    <text x="{right_x0 + 4}" y="{pad + 36}" class="panel-title" '
                f'font-family="{font_family}">{xml_escape("(b) Martensite Phase")}</text>'
            ),
            (
                f'    <image x="{right_x0}" y="{panel_y0}" width="{panel_size}" height="{panel_size}" '
                f'preserveAspectRatio="xMidYMid meet" '
                f'href="data:image/png;base64,{b64_mart}" '
                f'xlink:href="data:image/png;base64,{b64_mart}"/>'
            ),
            (
                f'    <rect x="{right_x0}" y="{panel_y0}" width="{panel_size}" height="{panel_size}" '
                f'fill="none" stroke="#CDCDCD" stroke-width="2"/>'
            ),
            "  </g>",
        ]
    )

    # Legend + shared scale bar
    legend_cy = legend_y0 + legend_h / 2.0
    svg_lines.append('  <g id="legend_and_scale">')
    cursor_x = float(left_x0 + 18)
    for elem in elements_present:
        color = _element_color(elem)
        svg_lines.append(
            f'    <circle cx="{cursor_x:.1f}" cy="{legend_cy:.1f}" r="12" '
            f'fill="{color}" stroke="#181818" stroke-width="2"/>'
        )
        svg_lines.append(
            f'    <text x="{cursor_x + 20:.1f}" y="{legend_cy + 8:.1f}" '
            f'class="legend-label" font-family="{font_family}">{xml_escape(elem)}</text>'
        )
        cursor_x += 110.0

    cursor_x += 18.0
    svg_lines.append(
        f'    <line x1="{cursor_x:.1f}" y1="{legend_cy:.1f}" '
        f'x2="{cursor_x + 48:.1f}" y2="{legend_cy:.1f}" '
        f'stroke="{_BAIN_COLOR_HEX}" stroke-width="4" stroke-dasharray="11,7"/>'
    )
    svg_lines.append(
        f'    <text x="{cursor_x + 58:.1f}" y="{legend_cy + 8:.1f}" '
        f'class="legend-label" style="fill: {_BAIN_COLOR_HEX};" '
        f'font-family="{font_family}">Bain subcell</text>'
    )

    sb_x1 = right_x0 + panel_size - 16
    sb_x0 = sb_x1 - bar_px
    sb_y = legend_cy + 10.0
    svg_lines.extend(
        [
            (
                f'    <text x="{(sb_x0 + sb_x1) / 2.0:.1f}" y="{legend_cy - 8:.1f}" '
                f'text-anchor="middle" class="scale-label" '
                f'font-family="{font_family}">{xml_escape(scale_label)}</text>'
            ),
            (
                f'    <rect x="{sb_x0}" y="{sb_y - 4:.1f}" width="{bar_px}" height="6" '
                f'fill="#181818"/>'
            ),
            (
                f'    <rect x="{sb_x0}" y="{sb_y - 10:.1f}" width="2" height="12" '
                f'fill="#181818"/>'
            ),
            (
                f'    <rect x="{sb_x1 - 2}" y="{sb_y - 10:.1f}" width="2" height="12" '
                f'fill="#181818"/>'
            ),
            "  </g>",
        ]
    )

    # Bottom metrics strip
    svg_lines.extend(
        [
            '  <g id="metrics_strip">',
            (
                f'    <line x1="{pad}" y1="{metrics_y0}" x2="{total_w - pad}" y2="{metrics_y0}" '
                f'stroke="#CDCDCD" stroke-width="2"/>'
            ),
            (
                f'    <text x="{total_w / 2.0:.1f}" y="{metrics_y0 + 36}" text-anchor="middle" '
                f'class="metrics-sub" font-family="{font_family}">{xml_escape(line1)}</text>'
            ),
            (
                f'    <text x="{total_w / 2.0:.1f}" y="{metrics_y0 + 74}" text-anchor="middle" '
                f'class="metrics-main" font-family="{font_family}">{xml_escape(line2)}</text>'
            ),
            "  </g>",
            "</svg>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")
    return output_path.resolve()


def compare_phases(
    parent_struct: Structure,
    martensite_struct: Structure,
    output_dir: str | Path,
    transparent_background: bool = False,
) -> dict[str, Any]:
    """Compare parent and martensite phases, export structures/CSV, and render identical-scale figures."""
    out_dir = Path(output_dir)
    data_dir = out_dir / DATA_DIR
    structures_dir = out_dir / STRUCT_DIR
    images_dir = out_dir / FIG_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export parent and martensite CIF + POSCAR
    parent_cif = structures_dir / "parent.cif"
    parent_poscar = structures_dir / "POSCAR_parent"
    mart_cif = structures_dir / "martensite.cif"
    mart_poscar = structures_dir / "POSCAR_martensite"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        CifWriter(parent_struct).write_file(str(parent_cif))
        Poscar(to_ordered_proxy(parent_struct)).write_file(str(parent_poscar))
        CifWriter(martensite_struct).write_file(str(mart_cif))
        Poscar(to_ordered_proxy(martensite_struct)).write_file(str(mart_poscar))

    # 2. Compute lattice & atomic displacement metrics
    p_lat = parent_struct.lattice
    m_lat = martensite_struct.lattice

    parent_a = float(p_lat.a)
    parent_b = float(p_lat.b)
    parent_c = float(p_lat.c)
    parent_c_over_a = parent_c / parent_a if parent_a > 1e-12 else 0.0
    parent_volume = float(parent_struct.volume)
    # Per atom = per occupied site weight (composition.num_atoms counts partial
    # occupancies / vacancies correctly, len() would count sites).
    parent_atoms = float(parent_struct.composition.num_atoms) or float(len(parent_struct))
    parent_vpa = parent_volume / max(parent_atoms, 1e-12)

    mart_a = float(m_lat.a)
    mart_b = float(m_lat.b)
    mart_c = float(m_lat.c)
    mart_c_over_a = mart_c / mart_a if mart_a > 1e-12 else 0.0
    mart_volume = float(martensite_struct.volume)
    mart_atoms = float(martensite_struct.composition.num_atoms) or float(len(martensite_struct))
    mart_vpa = mart_volume / max(mart_atoms, 1e-12)

    # Volume change per atom, so cells with different atom counts compare correctly.
    delta_v_pct = (mart_vpa - parent_vpa) / parent_vpa * 100.0 if parent_vpa > 1e-12 else 0.0

    # Lattice strains along the parent a, b, c axes from the matched correspondence
    # (independent of which cell setting the martensite file uses).
    nan = float("nan")
    corr = _lattice_correspondence(parent_struct, martensite_struct)
    if corr is not None:
        sx, sy, sz = (float(v) for v in corr["stretch_abc"])
        delta_a_pct, delta_b_pct, delta_c_pct = (sx - 1) * 100, (sy - 1) * 100, (sz - 1) * 100
        principal = [round((float(v) - 1) * 100, 6) for v in corr["principal_stretches"]]
        max_shuffle = corr["max_shuffle"]
        rms_shuffle = corr["rms_shuffle"]
    else:
        delta_a_pct = delta_b_pct = delta_c_pct = nan
        principal = [nan, nan, nan]
        max_shuffle = rms_shuffle = nan

    metrics: dict[str, Any] = {
        "parent_a": round(parent_a, 6),
        "parent_b": round(parent_b, 6),
        "parent_c": round(parent_c, 6),
        "parent_c_over_a": round(parent_c_over_a, 6),
        "parent_volume": round(parent_volume, 6),
        "parent_num_sites": len(parent_struct),
        "parent_volume_per_atom": round(parent_vpa, 6),
        "mart_a": round(mart_a, 6),
        "mart_b": round(mart_b, 6),
        "mart_c": round(mart_c, 6),
        "mart_c_over_a": round(mart_c_over_a, 6),
        "mart_volume": round(mart_volume, 6),
        "mart_num_sites": len(martensite_struct),
        "mart_volume_per_atom": round(mart_vpa, 6),
        "lattice_correspondence_found": corr is not None,
        "delta_a_pct": round(delta_a_pct, 6),
        "delta_b_pct": round(delta_b_pct, 6),
        "delta_c_pct": round(delta_c_pct, 6),
        "principal_strain_1_pct": principal[0],
        "principal_strain_2_pct": principal[1],
        "principal_strain_3_pct": principal[2],
        "delta_v_pct": round(delta_v_pct, 6),
        "max_shuffle_angstrom": round(max_shuffle, 6),
        "rms_shuffle_angstrom": round(rms_shuffle, 6),
    }

    csv_path = data_dir / "phase_comparison.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

    # 3. Compute identical orthographic framing across both structures
    px_scale = _px_scale()
    panel_px = _k(_PANEL_SIZE, px_scale)  # 960 px at 300 dpi
    window_size = (panel_px, panel_px)
    cam_vec = np.asarray(_DEFAULT_CAMERA_DIR, dtype=float)
    cam_unit = cam_vec / np.linalg.norm(cam_vec)
    up_unit = _orthonormal_up(cam_unit, np.asarray(_DEFAULT_VIEW_UP, dtype=float))

    parent_atoms = _collect_atoms_to_draw(parent_struct, include_boundary_atoms=True)
    mart_atoms = _collect_atoms_to_draw(martensite_struct, include_boundary_atoms=True)

    parent_center, parent_scale, _ = _compute_parallel_framing(
        structure=parent_struct,
        atoms=parent_atoms,
        cam_unit=cam_unit,
        up_unit=up_unit,
        window_size=window_size,
    )
    mart_center, mart_scale, _ = _compute_parallel_framing(
        structure=martensite_struct,
        atoms=mart_atoms,
        cam_unit=cam_unit,
        up_unit=up_unit,
        window_size=window_size,
    )

    shared_parallel_scale = float(max(parent_scale, mart_scale))
    shared_angstrom_per_pixel = (2.0 * shared_parallel_scale) / float(window_size[1])

    # 4. Render parent (with red dashed Bain subcell) and martensite panels at identical scale
    _, parent_bonds, _ = compute_bonds(parent_struct)
    _, mart_bonds, _ = compute_bonds(martensite_struct)

    panel_parent_path = _render_phase_panel(
        structure=parent_struct,
        atoms=parent_atoms,
        render_bonds=parent_bonds,
        cell_center=parent_center,
        cam_unit=cam_unit,
        up_unit=up_unit,
        parallel_scale=shared_parallel_scale,
        output_path=images_dir / "panel_parent.png",
        window_size=window_size,
        overlay_bain_subcell=True,
        transparent_background=transparent_background,
    )
    panel_mart_path = _render_phase_panel(
        structure=martensite_struct,
        atoms=mart_atoms,
        render_bonds=mart_bonds,
        cell_center=mart_center,
        cam_unit=cam_unit,
        up_unit=up_unit,
        parallel_scale=shared_parallel_scale,
        output_path=images_dir / "panel_martensite.png",
        window_size=window_size,
        overlay_bain_subcell=False,
        transparent_background=transparent_background,
    )

    # 5. Compose publication 300 dpi PNG and editable vector SVG
    elements_present = sorted(
        set(_structure_elements_sorted(parent_struct))
        | set(_structure_elements_sorted(martensite_struct))
    )

    fig_png_path = _compose_phase_compare_png(
        parent_panel_path=panel_parent_path,
        mart_panel_path=panel_mart_path,
        output_path=images_dir / "figure_phase_compare.png",
        elements_present=elements_present,
        angstrom_per_pixel=shared_angstrom_per_pixel,
        metrics=metrics,
        panel_size=panel_px,
        transparent_background=transparent_background,
        px_scale=px_scale,
    )
    fig_svg_path = _compose_phase_compare_svg(
        parent_panel_path=panel_parent_path,
        mart_panel_path=panel_mart_path,
        output_path=images_dir / "figure_phase_compare.svg",
        elements_present=elements_present,
        # SVG units are 300-dpi pixels: convert the per-pixel scale back to them
        angstrom_per_pixel=shared_angstrom_per_pixel * px_scale,
        metrics=metrics,
        panel_size=_PANEL_SIZE,
        transparent_background=transparent_background,
        embed_px=panel_px,
    )

    return {
        "output_dir": str(out_dir.resolve()),
        "parent_cif": str(parent_cif.resolve()),
        "parent_poscar": str(parent_poscar.resolve()),
        "martensite_cif": str(mart_cif.resolve()),
        "martensite_poscar": str(mart_poscar.resolve()),
        "phase_comparison_csv": str(csv_path.resolve()),
        "panel_parent_png": str(panel_parent_path),
        "panel_martensite_png": str(panel_mart_path),
        "figure_phase_compare_png": str(fig_png_path),
        "figure_phase_compare_svg": str(fig_svg_path),
        "shared_parallel_scale": round(shared_parallel_scale, 6),
        "scale_angstrom_per_pixel": round(shared_angstrom_per_pixel, 6),
        "metrics": metrics,
    }

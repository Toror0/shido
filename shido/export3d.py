"""3D export: printable STL (mm) and a self-contained, offline, rotatable HTML viewer.

- ``export_stl``: atoms (spheres) + bonds (capped cylinders) + optional unit-cell frame
  as one closed triangle mesh in millimetres (default 1 Å = 10 mm; optional
  ``max_size_mm`` shrinks it to fit a given printer). Overlapping closed shells are written together; slicers
  (Bambu Studio, PrusaSlicer, Cura) union them when slicing.
- ``export_html``: a single HTML file with an embedded WebGL2 viewer (no CDN, no
  network access; a Content-Security-Policy blocks every external request).

renderer.py is only imported, never modified.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from pymatgen.core import Structure
from scipy.spatial import cKDTree

from shido.core.species import site_occupancies
from shido.core.validator import ScientificValidationError, validate_structure
from shido.renderer import (
    _UNIT_CELL_EDGES,
    _element_color,
    _element_radius,
    _float_setting,
    _merge_spheres,
    _radius_params,
    limit_render_bonds,
    pretty_formula,
)
from shido.theme_manager import get_theme, resolve_structure_colors

logger = logging.getLogger(__name__)

RenderBonds = list[tuple[np.ndarray, np.ndarray, str, str]]

DEFAULT_SCALE_MM_PER_ANGSTROM: float = 10.0
DEFAULT_MAX_SIZE_MM: float | None = None  # no limit unless the caller gives one (e.g. 180)
DEFAULT_MIN_BOND_RADIUS_MM: float = 1.2  # radius -> bonds >= 2.4 mm thick
DEFAULT_MIN_FRAME_RADIUS_MM: float = 1.0
VACUUM_GAP_ANGSTROM: float = 5.0  # an empty band thicker than this along an axis = vacuum
_BOND_MATCH_TOL: float = 0.05  # Å, bond end <-> atom centre
_BOUNDARY_TOL: float = 1e-3


# ---------------------------------------------------------------------------
# Scene in Angstrom (shared by STL and HTML)
# ---------------------------------------------------------------------------


@dataclass
class _Scene:
    centers: np.ndarray  # (N, 3) Å
    radii: np.ndarray  # (N,) Å (display radius, same as the PNG figures)
    elements: list[str]  # dominant element per drawn atom
    bond_pairs: list[tuple[int, int]]  # indices into centers
    cell_edges: list[tuple[np.ndarray, np.ndarray]]  # empty if no frame
    vacuum_axes: list[int]
    structure: Structure  # (possibly shifted) structure actually drawn
    warnings: list[str] = field(default_factory=list)


def _largest_gap(fracs: np.ndarray) -> tuple[float, float]:
    """Largest empty band of fractional coords on a circle: (gap width, start of atoms after it)."""
    f = np.sort(np.mod(fracs, 1.0))
    if len(f) == 1:
        return 1.0, float(f[0])
    gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))
    k = int(np.argmax(gaps))
    return float(gaps[k]), float(f[(k + 1) % len(f)])


def _prepare_structure(structure: Structure, notes: list[str]) -> tuple[Structure, list[int]]:
    """Detect vacuum axes (slabs) and shift so the atoms are contiguous along them."""
    s = structure.copy()
    frac = np.asarray(s.frac_coords, dtype=float)
    lengths = np.asarray(s.lattice.abc, dtype=float)
    # Thickness of a fractional band along axis i = frac width * interplanar spacing.
    recip = np.asarray(s.lattice.reciprocal_lattice_crystallographic.abc, dtype=float)
    spacing = 1.0 / np.where(recip > 0, recip, np.inf)
    vacuum_axes: list[int] = []
    shift = np.zeros(3)
    for ax in range(3):
        gap, start = _largest_gap(frac[:, ax])
        if gap * spacing[ax] >= VACUUM_GAP_ANGSTROM and gap < 1.0 - 1e-9:
            vacuum_axes.append(ax)
            shift[ax] = 0.02 - start
    if vacuum_axes:
        s.translate_sites(list(range(len(s))), shift, frac_coords=True, to_unit_cell=True)
        notes.append(
            "Vacuum detected along axis "
            + ",".join("abc"[a] for a in vacuum_axes)
            + f" (cell {np.round(lengths, 2).tolist()} Å); size follows the atoms, not the cell."
        )
    return s, vacuum_axes


def _collect_atoms(
    s: Structure, vacuum_axes: list[int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Atoms in the cell + boundary images (not along vacuum axes)."""
    lattice = s.lattice
    params = _radius_params()
    centers: list[np.ndarray] = []
    radii: list[float] = []
    elems: list[str] = []
    seen: set[tuple[int, int, int]] = set()
    for site in s:
        occs = site_occupancies(site)
        elem = occs[0][0] if occs else str(site.species)
        r = _element_radius(site, params=params)
        f = np.mod(np.asarray(site.frac_coords, dtype=float), 1.0)
        f = np.where(np.isclose(f, 1.0, atol=_BOUNDARY_TOL), 0.0, f)
        per_axis = [
            [0.0, 1.0] if (ax not in vacuum_axes and f[ax] <= _BOUNDARY_TOL) else [0.0]
            for ax in range(3)
        ]
        for sx in per_axis[0]:
            for sy in per_axis[1]:
                for sz in per_axis[2]:
                    cart = np.asarray(lattice.get_cartesian_coords(f + [sx, sy, sz]), dtype=float)
                    key = tuple(np.round(cart * 1000.0).astype(int))
                    if key in seen:
                        continue
                    seen.add(key)
                    centers.append(cart)
                    radii.append(float(r))
                    elems.append(elem)
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float), elems


def _build_scene(
    structure: Structure,
    render_bonds: RenderBonds | None,
    include_unit_cell: bool,
) -> _Scene:
    if not isinstance(structure, Structure):
        raise ScientificValidationError(f"Expected a pymatgen Structure, got {type(structure)!r}.")
    validate_structure(structure)
    notes: list[str] = []
    s, vacuum_axes = _prepare_structure(structure, notes)
    if vacuum_axes and render_bonds is not None:
        notes.append("Structure was shifted for the vacuum axis; bonds were recomputed.")
        render_bonds = None
    centers, radii, elems = _collect_atoms(s, vacuum_axes)
    if len(centers) == 0:
        raise ScientificValidationError("No atoms to export.")

    if render_bonds is None:
        from shido.analyzer import compute_bonds  # local import: analyzer is heavy

        _, render_bonds, _ = compute_bonds(s)

    max_len = _float_setting("bonds.max_bond_length", 0.0)
    max_len = max_len if max_len > 0.0 else math.inf
    bonds = limit_render_bonds(list(render_bonds or []))
    tree = cKDTree(centers)
    pairs: set[tuple[int, int]] = set()
    dropped = 0
    for p1, p2, *_ in bonds:
        a = np.asarray(p1, dtype=float)
        b = np.asarray(p2, dtype=float)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6 or length > max_len + 1e-6:
            continue
        da, ia = tree.query(a)
        db, ib = tree.query(b)
        if da > _BOND_MATCH_TOL or db > _BOND_MATCH_TOL or ia == ib:
            dropped += 1
            continue
        pairs.add((min(int(ia), int(ib)), max(int(ia), int(ib))))
    if dropped:
        notes.append(f"{dropped} bond(s) did not end on a drawn atom and were skipped.")

    edges: list[tuple[np.ndarray, np.ndarray]] = []
    if include_unit_cell:
        if vacuum_axes:
            notes.append("Unit-cell frame skipped: it would enclose the vacuum and float.")
        else:
            lat = s.lattice
            edges = [
                (np.asarray(lat.get_cartesian_coords(f0), dtype=float),
                 np.asarray(lat.get_cartesian_coords(f1), dtype=float))
                for f0, f1 in _UNIT_CELL_EDGES
            ]

    for n in notes:
        logger.info(n)
    return _Scene(
        centers=centers,
        radii=radii,
        elements=elems,
        bond_pairs=sorted(pairs),
        cell_edges=edges,
        vacuum_axes=vacuum_axes,
        structure=s,
        warnings=notes,
    )


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------


@dataclass
class PrintModel:
    """Meshes in mm, already placed on the bed (min corner at 0,0,0)."""

    atoms_by_element: dict[str, pv.PolyData]
    bonds: pv.PolyData | None
    frame: pv.PolyData | None
    scale_mm_per_angstrom: float
    size_mm: tuple[float, float, float]
    bond_radius_mm: float
    n_atoms: int
    n_bonds: int
    n_removed_floating: int
    warnings: list[str]

    def combined(self) -> pv.PolyData:
        parts = [m for m in [*self.atoms_by_element.values(), self.bonds, self.frame] if m is not None]
        return _finish_mesh(_merge_parts(parts))


def _finish_mesh(mesh: pv.PolyData) -> pv.PolyData:
    # No global clean(): merging coincident points of different parts (e.g. two
    # collinear bonds meeting in one atom) would stitch their caps together.
    mesh = mesh.triangulate()
    for key in list(mesh.point_data.keys()):
        del mesh.point_data[key]
    return mesh


def _merge_parts(parts: list[pv.PolyData]) -> pv.PolyData:
    return pv.merge(parts, merge_points=False) if len(parts) > 1 else parts[0]


def _closed_cylinders(
    starts: np.ndarray, ends: np.ndarray, radius: float, resolution: int = 24
) -> pv.PolyData:
    """Capped cylinders, each a closed shell (template welded before copying)."""
    tmpl = pv.Cylinder(
        center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), radius=radius,
        height=1.0, resolution=resolution, capping=True,
    ).clean().triangulate()
    t_pts = np.asarray(tmpl.points, dtype=float)
    t_tri = np.asarray(tmpl.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    vec = ends - starts
    lengths = np.linalg.norm(vec, axis=1)
    d = vec / lengths[:, None]
    helper = np.where(np.abs(d[:, [0]]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    u = np.cross(d, helper)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    centers = 0.5 * (starts + ends)
    pts = (
        centers[:, None, :]
        + t_pts[None, :, 0:1] * u[:, None, :]
        + t_pts[None, :, 1:2] * v[:, None, :]
        + (t_pts[None, :, 2:3] * lengths[:, None, None]) * d[:, None, :]
    ).reshape(-1, 3)
    n_b, n_p = len(starts), len(t_pts)
    tri = (t_tri[None, :, :] + (np.arange(n_b) * n_p)[:, None, None]).reshape(-1, 3)
    faces = np.column_stack([np.full(len(tri), 3, dtype=np.int64), tri]).ravel()
    return pv.PolyData(pts, faces=faces)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab[None, :]), axis=1)


def _connected_keep_mask(
    n_atoms: int,
    pairs: list[tuple[int, int]],
    centers_mm: np.ndarray,
    radii_mm: np.ndarray,
    edges_mm: list[tuple[np.ndarray, np.ndarray]],
    frame_r: float,
) -> np.ndarray:
    """Atoms in the largest connected piece (bonds, and touching the frame)."""
    parent = list(range(n_atoms + 1))  # last node = frame

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in pairs:
        union(i, j)
    if edges_mm:
        for a, b in edges_mm:
            d = _point_segment_distance(centers_mm, a, b)
            for i in np.nonzero(d < radii_mm + frame_r - 1e-6)[0]:
                union(int(i), n_atoms)
    nodes = list(range(n_atoms)) + ([n_atoms] if edges_mm else [])
    roots = [find(i) for i in nodes]
    weights: dict[int, int] = {}
    for i, r in zip(nodes, roots):
        weights[r] = weights.get(r, 0) + (1 if i < n_atoms else 0)
    best = max(weights, key=lambda r: weights[r])
    return np.array([find(i) == best for i in range(n_atoms)], dtype=bool)


def build_print_model(
    structure: Structure,
    render_bonds: RenderBonds | None = None,
    include_unit_cell: bool = True,
    scale: float = DEFAULT_SCALE_MM_PER_ANGSTROM,
    max_size_mm: float | None = DEFAULT_MAX_SIZE_MM,
    min_bond_radius_mm: float = DEFAULT_MIN_BOND_RADIUS_MM,
    remove_floating: bool = True,
    sphere_resolution: int = 32,
) -> PrintModel:
    """Build the printable meshes (mm). ``scale`` = mm per Å; shrunk to fit ``max_size_mm`` if given."""
    if not (scale > 0 and math.isfinite(scale)):
        raise ValueError(f"scale must be a positive number (mm per Å), got {scale!r}")
    if max_size_mm is not None and not (max_size_mm > 0 and math.isfinite(max_size_mm)):
        raise ValueError(f"max_size_mm must be positive, got {max_size_mm!r}")
    scene = _build_scene(structure, render_bonds, include_unit_cell)
    notes = list(scene.warnings)

    bond_r_ang = _float_setting("render.bond_radius", 0.075)
    frame_r_ang = _float_setting("render.unit_cell_line_radius", 0.018)
    requested = float(scale)
    sc = requested
    centers_mm = radii_mm = None
    bond_r = frame_r = 0.0
    lo = hi = np.zeros(3)
    for _ in range(6):
        bond_r = max(bond_r_ang * sc, float(min_bond_radius_mm))
        frame_r = max(frame_r_ang * sc, DEFAULT_MIN_FRAME_RADIUS_MM)
        centers_mm = scene.centers * sc
        # Atoms must stay fatter than bonds so the bond ends are buried inside them.
        radii_mm = np.maximum(scene.radii * sc, bond_r * 1.35)
        lo = (centers_mm - radii_mm[:, None]).min(axis=0)
        hi = (centers_mm + radii_mm[:, None]).max(axis=0)
        if scene.cell_edges:
            corners = np.array([p for e in scene.cell_edges for p in e]) * sc
            lo = np.minimum(lo, corners.min(axis=0) - frame_r)
            hi = np.maximum(hi, corners.max(axis=0) + frame_r)
        extent = float((hi - lo).max())
        if max_size_mm is None or extent <= max_size_mm + 1e-6:
            break
        sc *= (max_size_mm / extent) * 0.995
    else:
        raise ScientificValidationError(
            f"Model cannot fit {max_size_mm} mm (minimum bond/atom sizes too large for this cell)."
        )
    if sc < requested - 1e-9:
        msg = (
            f"Model larger than {max_size_mm:g} mm at {requested:g} mm/Å; "
            f"shrunk to {sc:.3f} mm/Å."
        )
        logger.warning(msg)
        notes.append(msg)
    if sc < 2.0:
        msg = f"Scale {sc:.2f} mm/Å is small: atoms will be tiny and hard to print."
        logger.warning(msg)
        notes.append(msg)

    offset = -lo  # place min corner at the origin (sits on the bed)
    centers_mm = centers_mm + offset
    edges_mm = [(a * sc + offset, b * sc + offset) for a, b in scene.cell_edges]

    n = len(centers_mm)
    keep = np.ones(n, dtype=bool)
    if remove_floating:
        keep = _connected_keep_mask(n, scene.bond_pairs, centers_mm, radii_mm, edges_mm, frame_r)
    n_removed = int((~keep).sum())
    if n_removed:
        msg = f"Removed {n_removed} floating atom(s) not connected to the main model."
        logger.warning(msg)
        notes.append(msg)

    atoms_by_element: dict[str, pv.PolyData] = {}
    for elem in sorted(set(scene.elements)):
        idx = [i for i in range(n) if keep[i] and scene.elements[i] == elem]
        if not idx:
            continue
        mesh = _merge_spheres([centers_mm[i] for i in idx], [float(radii_mm[i]) for i in idx],
                              sphere_resolution)
        atoms_by_element[elem] = _finish_mesh(mesh)
    if not atoms_by_element:
        raise ScientificValidationError("No atoms left to export (empty mesh).")

    pairs = [(i, j) for i, j in scene.bond_pairs if keep[i] and keep[j]]
    bonds_mesh = None
    if pairs:
        # End each bond half-way inside its atoms (not at the centre): still buried in
        # the sphere (rim at <= 0.89 r), and collinear bonds no longer share cap vertices,
        # which would leave open edges once a slicer welds the STL by position.
        a = np.array([centers_mm[i] for i, _ in pairs])
        b = np.array([centers_mm[j] for _, j in pairs])
        ra = np.array([radii_mm[i] for i, _ in pairs])[:, None]
        rb = np.array([radii_mm[j] for _, j in pairs])[:, None]
        d = (b - a) / np.linalg.norm(b - a, axis=1)[:, None]
        starts = a + d * (0.5 * ra)
        ends = b - d * (0.5 * rb)
        bonds_mesh = _finish_mesh(_closed_cylinders(starts, ends, bond_r, resolution=24))

    frame_mesh = None
    if edges_mm:
        starts = np.array([a for a, _ in edges_mm])
        ends = np.array([b for _, b in edges_mm])
        tubes = _closed_cylinders(starts, ends, frame_r, resolution=16)
        corners = np.unique(np.round(np.vstack([starts, ends]), 6), axis=0)
        joints = _merge_spheres(list(corners), [frame_r * 1.2] * len(corners), 16)
        frame_mesh = _finish_mesh(_merge_parts([tubes, joints]))

    size = tuple(float(v) for v in (hi - lo))
    return PrintModel(
        atoms_by_element=atoms_by_element,
        bonds=bonds_mesh,
        frame=frame_mesh,
        scale_mm_per_angstrom=float(sc),
        size_mm=size,  # type: ignore[arg-type]
        bond_radius_mm=float(bond_r),
        n_atoms=int(keep.sum()),
        n_bonds=len(pairs),
        n_removed_floating=n_removed,
        warnings=notes,
    )


def _write_stl(mesh: pv.PolyData, path: Path) -> Path:
    if mesh is None or mesh.n_points == 0 or mesh.n_cells == 0:
        raise ScientificValidationError(f"Refusing to write an empty mesh: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path), binary=True)
    return path


def export_stl(
    structure: Structure,
    output_path: str | Path,
    render_bonds: RenderBonds | None = None,
    include_unit_cell: bool = True,
    scale: float = DEFAULT_SCALE_MM_PER_ANGSTROM,
    split_by_element: bool = False,
    max_size_mm: float | None = DEFAULT_MAX_SIZE_MM,
    min_bond_radius_mm: float = DEFAULT_MIN_BOND_RADIUS_MM,
    remove_floating: bool = True,
) -> Path:
    """Write a printable STL (units: mm) and return its path.

    ``scale`` is mm per Å (default 10). ``max_size_mm`` (default None = no limit): if
    given and the model exceeds it on any axis, it is shrunk to fit and a warning is logged. ``split_by_element=True`` also writes
    ``<stem>_<El>.stl`` per element and ``<stem>_bonds.stl`` (bonds + unit-cell frame),
    all in the same coordinates, for multi-colour printing.
    """
    out = Path(output_path)
    if out.suffix.lower() != ".stl":
        out = out.with_suffix(".stl")
    model = build_print_model(
        structure,
        render_bonds=render_bonds,
        include_unit_cell=include_unit_cell,
        scale=scale,
        max_size_mm=max_size_mm,
        min_bond_radius_mm=min_bond_radius_mm,
        remove_floating=remove_floating,
    )
    _write_stl(model.combined(), out)
    if split_by_element:
        for elem, mesh in model.atoms_by_element.items():
            _write_stl(mesh, out.with_name(f"{out.stem}_{elem}.stl"))
        rest = [m for m in (model.bonds, model.frame) if m is not None]
        if rest:
            _write_stl(_finish_mesh(_merge_parts(rest)),
                       out.with_name(f"{out.stem}_bonds.stl"))
    logger.info(
        "STL %s: %.1f x %.1f x %.1f mm at %.3f mm/Å (%d atoms, %d bonds)",
        out.name, *model.size_mm, model.scale_mm_per_angstrom, model.n_atoms, model.n_bonds,
    )
    return out


def export_vrml(
    structure: Structure,
    output_path: str | Path,
    render_bonds: RenderBonds | None = None,
    include_unit_cell: bool = True,
    scale: float = DEFAULT_SCALE_MM_PER_ANGSTROM,
    max_size_mm: float | None = DEFAULT_MAX_SIZE_MM,
    remove_floating: bool = True,
) -> Path:
    """Write a colour VRML 2.0 (*.wrl, units: mm) of the same model as `export_stl`.

    Atoms are coloured per element (theme colours), bonds and frame use the theme's
    bond / unit-cell colours, so a colour 3D-print service can use the file as-is.
    """
    out = Path(output_path)
    if out.suffix.lower() != ".wrl":
        out = out.with_suffix(".wrl")
    model = build_print_model(
        structure,
        render_bonds=render_bonds,
        include_unit_cell=include_unit_cell,
        scale=scale,
        max_size_mm=max_size_mm,
        remove_floating=remove_floating,
    )
    cmap = resolve_structure_colors(structure)
    theme = get_theme()
    plotter = pv.Plotter(off_screen=True)
    try:
        for elem, mesh in model.atoms_by_element.items():
            plotter.add_mesh(mesh, color=_element_color(elem, cmap), smooth_shading=True)
        if model.bonds is not None:
            plotter.add_mesh(model.bonds, color=str(theme.get("bond", "#B8C0CC")))
        if model.frame is not None:
            plotter.add_mesh(model.frame, color=str(theme.get("unit_cell", "#2B2D42")))
        out.parent.mkdir(parents=True, exist_ok=True)
        plotter.export_vrml(str(out))
    finally:
        plotter.close()
    return out


def split_stl_paths(output_path: str | Path, structure: Structure) -> list[Path]:
    """Paths ``export_stl(..., split_by_element=True)`` writes besides the main file."""
    out = Path(output_path).with_suffix(".stl")
    elems = sorted({site_occupancies(s)[0][0] for s in structure})
    return [out.with_name(f"{out.stem}_{e}.stl") for e in elems] + [
        out.with_name(f"{out.stem}_bonds.stl")
    ]


# ---------------------------------------------------------------------------
# HTML (self-contained WebGL2 viewer, no network)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:;">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:#fff;font-family:"Segoe UI","Yu Gothic UI",Meiryo,sans-serif;color:#222;overflow:hidden}
#c{position:fixed;inset:0;width:100%;height:100%;display:block;touch-action:none;cursor:grab}
#hud{position:fixed;left:12px;top:10px;background:rgba(255,255,255,.88);border:1px solid #ddd;border-radius:8px;padding:8px 12px;font-size:14px;pointer-events:none}
#hud h1{font-size:17px;margin:0 0 6px}
.lg{display:flex;align-items:center;gap:8px;margin:3px 0}
.sw{width:16px;height:16px;border-radius:50%;border:1px solid rgba(0,0,0,.35);flex:none}
#help{position:fixed;left:12px;bottom:10px;font-size:12px;color:#555;background:rgba(255,255,255,.8);padding:4px 8px;border-radius:6px}
#err{display:none;position:fixed;inset:0;align-items:center;justify-content:center;font-size:16px}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud"><h1>__TITLE__</h1><div id="legend"></div></div>
<div id="help">ドラッグ：回転　ホイール／ピンチ：拡大縮小　Shift＋ドラッグ：移動　ダブルクリック：元に戻す</div>
<div id="err">このブラウザでは WebGL2 が使えないため表示できません。</div>
<script type="application/json" id="scene">__DATA__</script>
<script>
"use strict";
(function(){
const D=JSON.parse(document.getElementById("scene").textContent);
const lg=document.getElementById("legend");
D.legend.forEach(function(e){const r=document.createElement("div");r.className="lg";
 const s=document.createElement("span");s.className="sw";s.style.background=e.color;
 const t=document.createElement("span");t.textContent=e.element;r.appendChild(s);r.appendChild(t);lg.appendChild(r);});
const cv=document.getElementById("c");
const gl=cv.getContext("webgl2",{antialias:true,preserveDrawingBuffer:true});
if(!gl){document.getElementById("err").style.display="flex";return;}
function sh(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
 if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
function prog(vs,fs){const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));
 gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
 if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;}
const FS=`#version 300 es
precision highp float;in vec3 vN;in vec3 vC;in vec3 vP;out vec4 o;uniform float uLit;
void main(){if(uLit<0.5){o=vec4(vC,1.0);return;}
 vec3 n=normalize(vN);vec3 L=normalize(vec3(0.45,0.6,1.0));vec3 V=vec3(0.0,0.0,1.0);
 float d=max(dot(n,L),0.0);vec3 H=normalize(L+V);float s=pow(max(dot(n,H),0.0),48.0);
 float fill=max(dot(n,normalize(vec3(-0.6,-0.3,0.6))),0.0)*0.18;
 o=vec4(vC*(0.30+0.62*d+fill)+vec3(0.38*s),1.0);}`;
const VS_SPH=`#version 300 es
in vec3 aP;in vec4 aI;in vec3 aCol;uniform mat4 uMV;uniform mat4 uPr;
out vec3 vN;out vec3 vC;out vec3 vP;
void main(){vec4 p=uMV*vec4(aI.xyz+aP*aI.w,1.0);vN=mat3(uMV)*aP;vC=aCol;vP=p.xyz;gl_Position=uPr*p;}`;
const VS_CYL=`#version 300 es
in vec3 aP;in vec3 aA;in vec3 aB;uniform mat4 uMV;uniform mat4 uPr;uniform float uR;uniform vec3 uCol;
out vec3 vN;out vec3 vC;out vec3 vP;
void main(){vec3 d=aB-aA;vec3 dn=normalize(d);vec3 h=abs(dn.x)<0.9?vec3(1,0,0):vec3(0,1,0);
 vec3 u=normalize(cross(dn,h));vec3 v=cross(dn,u);vec3 w=aA+d*aP.z+(u*aP.x+v*aP.y)*uR;
 vec4 p=uMV*vec4(w,1.0);vN=mat3(uMV)*(u*aP.x+v*aP.y);vC=uCol;vP=p.xyz;gl_Position=uPr*p;}`;
const VS_CONE=`#version 300 es
in vec3 aP;in vec3 aA;in vec3 aB;uniform mat4 uMV;uniform mat4 uPr;uniform float uR;uniform vec3 uCol;
out vec3 vN;out vec3 vC;out vec3 vP;
void main(){vec3 d=aB-aA;float L=length(d);vec3 dn=d/L;vec3 h=abs(dn.x)<0.9?vec3(1,0,0):vec3(0,1,0);
 vec3 u=normalize(cross(dn,h));vec3 v=cross(dn,u);vec3 w=aA+d*aP.z+(u*aP.x+v*aP.y)*uR*(1.0-aP.z);
 vec4 p=uMV*vec4(w,1.0);vN=mat3(uMV)*normalize((u*aP.x+v*aP.y)*L+dn*uR);vC=uCol;vP=p.xyz;gl_Position=uPr*p;}`;
const pS=prog(VS_SPH,FS),pC=prog(VS_CYL,FS),pK=prog(VS_CONE,FS);
function sphereMesh(n){const v=[];const m=2*n;
 function pt(i,j){const t=Math.PI*i/n,f=2*Math.PI*j/m;return [Math.sin(t)*Math.cos(f),Math.sin(t)*Math.sin(f),Math.cos(t)];}
 for(let i=0;i<n;i++)for(let j=0;j<m;j++){const a=pt(i,j),b=pt(i+1,j),c=pt(i+1,j+1),d=pt(i,j+1);v.push(...a,...b,...c,...a,...c,...d);}
 return new Float32Array(v);}
function cylMesh(m){const v=[];for(let j=0;j<m;j++){const f0=2*Math.PI*j/m,f1=2*Math.PI*(j+1)/m;
 const a=[Math.cos(f0),Math.sin(f0)],b=[Math.cos(f1),Math.sin(f1)];
 v.push(a[0],a[1],0,b[0],b[1],0,b[0],b[1],1,a[0],a[1],0,b[0],b[1],1,a[0],a[1],1);}return new Float32Array(v);}
const nA=D.atoms.length/4;const res=nA>4000?10:(nA>800?14:20);
const sphV=sphereMesh(res),cylV=cylMesh(16);
function buf(data){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);return b;}
function attr(p,name,b,size,div){const l=gl.getAttribLocation(p,name);if(l<0)return;gl.bindBuffer(gl.ARRAY_BUFFER,b);
 gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,size,gl.FLOAT,false,0,0);gl.vertexAttribDivisor(l,div);}
const vaoS=gl.createVertexArray();gl.bindVertexArray(vaoS);
attr(pS,"aP",buf(sphV),3,0);attr(pS,"aI",buf(new Float32Array(D.atoms)),4,1);attr(pS,"aCol",buf(new Float32Array(D.atom_colors)),3,1);
const nB=D.bonds.length/6;let vaoB=null;
if(nB>0){const A=new Float32Array(nB*3),B=new Float32Array(nB*3);
 for(let i=0;i<nB;i++){for(let k=0;k<3;k++){A[3*i+k]=D.bonds[6*i+k];B[3*i+k]=D.bonds[6*i+3+k];}}
 vaoB=gl.createVertexArray();gl.bindVertexArray(vaoB);attr(pC,"aP",buf(cylV),3,0);attr(pC,"aA",buf(A),3,1);attr(pC,"aB",buf(B),3,1);}
const nE=D.cell.length/6;let vaoE=null;
if(nE>0){const A=new Float32Array(nE*3),B=new Float32Array(nE*3);
 for(let i=0;i<nE;i++){for(let k=0;k<3;k++){A[3*i+k]=D.cell[6*i+k];B[3*i+k]=D.cell[6*i+3+k];}}
 vaoE=gl.createVertexArray();gl.bindVertexArray(vaoE);attr(pC,"aP",buf(cylV),3,0);attr(pC,"aA",buf(A),3,1);attr(pC,"aB",buf(B),3,1);}
function segVao(p,seg){const n=seg.length/6;if(n===0)return null;const A=new Float32Array(n*3),B=new Float32Array(n*3);
 for(let i=0;i<n;i++){for(let k=0;k<3;k++){A[3*i+k]=seg[6*i+k];B[3*i+k]=seg[6*i+3+k];}}
 const o=gl.createVertexArray();gl.bindVertexArray(o);attr(p,"aP",buf(cylV),3,0);attr(p,"aA",buf(A),3,1);attr(p,"aB",buf(B),3,1);return o;}
const AR=D.arrows||null;const nAr=AR?AR.shafts.length/6:0,nAh=AR?AR.heads.length/6:0;
const vaoAs=AR?segVao(pC,AR.shafts):null,vaoAh=AR?segVao(pK,AR.heads):null;
gl.bindVertexArray(null);
function hex(c){return [parseInt(c.substr(1,2),16)/255,parseInt(c.substr(3,2),16)/255,parseInt(c.substr(5,2),16)/255];}
const bondCol=hex(D.bond_color),cellCol=hex(D.cell_color);
const C=D.center,R0=D.radius;
let rot=[1,0,0,0,1,0,0,0,1],zoom=1,pan=[0,0];
function mul3(a,b){const r=new Array(9);for(let i=0;i<3;i++)for(let j=0;j<3;j++){r[3*i+j]=a[3*i]*b[j]+a[3*i+1]*b[3+j]+a[3*i+2]*b[6+j];}return r;}
function axisRot(x,y,z,t){const c=Math.cos(t),s=Math.sin(t),k=1-c;
 return [c+x*x*k,x*y*k-z*s,x*z*k+y*s, y*x*k+z*s,c+y*y*k,y*z*k-x*s, z*x*k-y*s,z*y*k+x*s,c+z*z*k];}
function reset(){rot=mul3(axisRot(1,0,0,-1.15),axisRot(0,0,1,-0.55));zoom=1;pan=[0,0];draw();}
function draw(){const dpr=window.devicePixelRatio||1;const w=Math.max(1,Math.round(cv.clientWidth*dpr)),h=Math.max(1,Math.round(cv.clientHeight*dpr));
 if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
 gl.viewport(0,0,w,h);gl.clearColor(1,1,1,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);
 const asp=w/h,half=R0*1.08/zoom,hx=asp>=1?half*asp:half,hy=asp>=1?half:half/asp,far=R0*8;
 const pr=[1/hx,0,0,0, 0,1/hy,0,0, 0,0,-2/(2*far),0, -pan[0]/hx,-pan[1]/hy,0,1];
 const r=rot;const t=[-(r[0]*C[0]+r[1]*C[1]+r[2]*C[2]),-(r[3]*C[0]+r[4]*C[1]+r[5]*C[2]),-(r[6]*C[0]+r[7]*C[1]+r[8]*C[2])];
 const mv=[r[0],r[3],r[6],0, r[1],r[4],r[7],0, r[2],r[5],r[8],0, t[0],t[1],t[2],1];
 function common(p){gl.useProgram(p);gl.uniformMatrix4fv(gl.getUniformLocation(p,"uMV"),false,mv);
  gl.uniformMatrix4fv(gl.getUniformLocation(p,"uPr"),false,pr);gl.uniform1f(gl.getUniformLocation(p,"uLit"),1);}
 common(pS);gl.bindVertexArray(vaoS);gl.drawArraysInstanced(gl.TRIANGLES,0,sphV.length/3,nA);
 if(vaoB||vaoE){common(pC);}
 if(vaoB){gl.uniform1f(gl.getUniformLocation(pC,"uR"),D.bond_radius);gl.uniform3fv(gl.getUniformLocation(pC,"uCol"),bondCol);
  gl.bindVertexArray(vaoB);gl.drawArraysInstanced(gl.TRIANGLES,0,cylV.length/3,nB);}
 if(vaoE){gl.uniform1f(gl.getUniformLocation(pC,"uR"),D.cell_radius);gl.uniform3fv(gl.getUniformLocation(pC,"uCol"),cellCol);
  gl.uniform1f(gl.getUniformLocation(pC,"uLit"),0);gl.bindVertexArray(vaoE);gl.drawArraysInstanced(gl.TRIANGLES,0,cylV.length/3,nE);}
 if(AR){const ac=hex(AR.color);
  if(vaoAs){common(pC);gl.uniform1f(gl.getUniformLocation(pC,"uR"),AR.radius);gl.uniform3fv(gl.getUniformLocation(pC,"uCol"),ac);
   gl.bindVertexArray(vaoAs);gl.drawArraysInstanced(gl.TRIANGLES,0,cylV.length/3,nAr);}
  if(vaoAh){common(pK);gl.uniform1f(gl.getUniformLocation(pK,"uR"),AR.head_radius);gl.uniform3fv(gl.getUniformLocation(pK,"uCol"),ac);
   gl.bindVertexArray(vaoAh);gl.drawArraysInstanced(gl.TRIANGLES,0,cylV.length/3,nAh);}}
 gl.bindVertexArray(null);}
let drag=null;const pts=new Map();
cv.addEventListener("pointerdown",function(e){cv.setPointerCapture(e.pointerId);pts.set(e.pointerId,[e.clientX,e.clientY]);
 drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};cv.style.cursor="grabbing";});
cv.addEventListener("pointermove",function(e){if(!pts.has(e.pointerId))return;
 if(pts.size===2){const o=[...pts.values()];const d0=Math.hypot(o[0][0]-o[1][0],o[0][1]-o[1][1]);pts.set(e.pointerId,[e.clientX,e.clientY]);
  const n=[...pts.values()];const d1=Math.hypot(n[0][0]-n[1][0],n[0][1]-n[1][1]);if(d0>0){zoom=Math.min(20,Math.max(0.2,zoom*d1/d0));draw();}return;}
 pts.set(e.pointerId,[e.clientX,e.clientY]);if(!drag)return;
 const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
 if(drag.pan){const s=2/Math.min(cv.clientWidth,cv.clientHeight);pan[0]-=dx*s;pan[1]+=dy*s;}
 else{const a=Math.hypot(dx,dy);if(a>0){rot=mul3(axisRot(dy/a,dx/a,0,a*0.008),rot);}}draw();});
function up(e){pts.delete(e.pointerId);if(pts.size===0){drag=null;cv.style.cursor="grab";}}
cv.addEventListener("pointerup",up);cv.addEventListener("pointercancel",up);
cv.addEventListener("contextmenu",function(e){e.preventDefault();});
cv.addEventListener("wheel",function(e){e.preventDefault();zoom=Math.min(20,Math.max(0.2,zoom*Math.exp(-e.deltaY*0.0015)));draw();},{passive:false});
cv.addEventListener("dblclick",reset);window.addEventListener("resize",draw);
reset();
})();
</script>
</body>
</html>
"""


def _round_list(arr: Any, nd: int = 4) -> list[float]:
    return [round(float(x), nd) for x in np.asarray(arr, dtype=float).ravel()]


def _scene_site_indices(scene: _Scene) -> list[int]:
    """Site index (in scene.structure) of every drawn centre (boundary images included)."""
    s = scene.structure
    site_frac = np.mod(np.asarray(s.frac_coords, dtype=float), 1.0)
    out: list[int] = []
    for c in scene.centers:
        f = np.asarray(s.lattice.get_fractional_coords(c), dtype=float)
        d = np.mod(f - site_frac + 0.5, 1.0) - 0.5
        dist = np.linalg.norm(d @ np.asarray(s.lattice.matrix, dtype=float), axis=1)
        out.append(int(np.argmin(dist)))
    return out


def html_vector_arrows(
    scene: _Scene,
    vectors: Any,
    basis: str | None = None,
    scale: float | None = None,
    color: str | None = None,
    radius: float | None = None,
) -> dict[str, Any] | None:
    """Arrow shafts / heads (start, end pairs in Å) for the HTML viewer; None if no arrow."""
    from shido.core.vectors import resolve_site_vectors
    from shido.renderer import normalize_vector_style

    site_vecs = resolve_site_vectors(scene.structure, vectors, basis)
    if not site_vecs:
        return None
    scale, color, radius = normalize_vector_style(scale, color, radius)
    shafts: list[float] = []
    heads: list[float] = []
    for c, idx in zip(scene.centers, _scene_site_indices(scene)):
        vec = site_vecs.get(idx)
        if vec is None:
            continue
        d = scale * np.asarray(vec, dtype=float)
        length = float(np.linalg.norm(d))
        u = d / length
        start = c - 0.5 * d
        head = min(0.4 * length, 4.0 * radius)
        mid = start + u * (length - head)
        shafts.extend(_round_list(np.concatenate([start, mid])))
        heads.extend(_round_list(np.concatenate([mid, start + d])))
    if not heads:
        return None
    return {"shafts": shafts, "heads": heads, "radius": radius, "head_radius": 2.2 * radius,
            "color": color}


def export_html(
    structure: Structure,
    output_path: str | Path,
    render_bonds: RenderBonds | None = None,
    include_unit_cell: bool = True,
    title: str | None = None,
    vectors: Any = None,
    vector_basis: str | None = None,
    vector_scale: float | None = None,
    vector_color: str | None = None,
    vector_radius: float | None = None,
) -> Path:
    """Write a single offline HTML file with a mouse-rotatable 3D view and element legend.

    Everything (data + WebGL2 viewer) is embedded; the page loads nothing from the
    network and its Content-Security-Policy forbids any external request.
    ``vectors`` / ``vector_*`` add the atom arrows as in `renderer.render_single_view`.
    """
    out = Path(output_path)
    if out.suffix.lower() not in (".html", ".htm"):
        out = out.with_suffix(".html")
    scene = _build_scene(structure, render_bonds, include_unit_cell=False)
    s = scene.structure

    cmap = resolve_structure_colors(structure)
    colors = {e: _element_color(e, cmap) for e in set(scene.elements)}
    atom_rgb = []
    for e in scene.elements:
        h = colors[e].lstrip("#")
        atom_rgb.extend(round(int(h[i:i + 2], 16) / 255.0, 4) for i in (0, 2, 4))
    atoms = np.column_stack([scene.centers, scene.radii])
    bonds = [np.concatenate([scene.centers[i], scene.centers[j]]) for i, j in scene.bond_pairs]

    # The frame is drawn in the viewer even for slabs (it does not float there),
    # but the view is framed on the atoms.
    cell: list[float] = []
    if include_unit_cell:
        lat = s.lattice
        for f0, f1 in _UNIT_CELL_EDGES:
            cell.extend(_round_list(lat.get_cartesian_coords(f0)))
            cell.extend(_round_list(lat.get_cartesian_coords(f1)))
    pts = [scene.centers - scene.radii[:, None], scene.centers + scene.radii[:, None]]
    if include_unit_cell and not scene.vacuum_axes:
        pts.append(np.asarray(cell, dtype=float).reshape(-1, 3))
    allp = np.vstack(pts)
    center = 0.5 * (allp.min(axis=0) + allp.max(axis=0))
    radius = float(np.linalg.norm(allp - center, axis=1).max())

    theme = get_theme()
    present = sorted(set(scene.elements))
    data = {
        "atoms": _round_list(atoms),
        "atom_colors": atom_rgb,
        "bonds": _round_list(np.asarray(bonds)) if bonds else [],
        "cell": cell,
        "center": _round_list(center),
        "radius": round(max(radius, 1e-3), 4),
        "bond_radius": _float_setting("render.bond_radius", 0.075),
        "cell_radius": max(_float_setting("render.unit_cell_line_radius", 0.018), 0.012),
        "bond_color": str(theme.get("bond", "#B8C0CC")),
        "cell_color": str(theme.get("unit_cell", "#2B2D42")),
        "legend": [{"element": e, "color": colors[e]} for e in present],
    }
    if vectors is not None:
        arrows = html_vector_arrows(scene, vectors, vector_basis, vector_scale, vector_color, vector_radius)
        if arrows is not None:
            data["arrows"] = arrows
    if not data["atoms"]:
        raise ScientificValidationError("No atoms to export (empty scene).")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page_title = title or f"{pretty_formula(structure)} 3D"
    html_title = (
        page_title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    html = _HTML_TEMPLATE.replace("__TITLE__", html_title).replace("__DATA__", payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Crystal shape (morphology, batch 20)
# ---------------------------------------------------------------------------

def export_morphology_stl(morph: Any, output_path: str | Path, size_mm: float = 30.0) -> Path:
    """Write the crystal shape as a closed STL (mm); the longest bounding-box side = ``size_mm``."""
    out = Path(output_path)
    if out.suffix.lower() != ".stl":
        out = out.with_suffix(".stl")
    verts = np.asarray(morph.vertices, dtype=float)
    span = float((verts.max(axis=0) - verts.min(axis=0)).max())
    if span <= 0 or size_mm <= 0:
        raise ScientificValidationError("Crystal shape has no size.")
    mesh = morph.to_polydata(scale=size_mm / span).triangulate()
    mesh.points -= mesh.points.min(axis=0)  # sits on z = 0 at the origin corner
    return _write_stl(mesh, out)


_MORPH_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline';">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:#fff;font-family:"Segoe UI","Yu Gothic UI",Meiryo,sans-serif;color:#222;overflow:hidden}
#c{position:fixed;inset:0;width:100%;height:100%;touch-action:none;cursor:grab}
#hud{position:fixed;left:12px;top:10px;background:rgba(255,255,255,.9);border:1px solid #ddd;border-radius:8px;padding:8px 12px;font-size:14px}
.sw{display:inline-block;width:12px;height:12px;border:1px solid #555;margin-right:6px;vertical-align:-1px}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud"><b>__TITLE__</b><div id="lg"></div><div style="color:#777;font-size:12px">ドラッグで回転・ホイールで拡大</div></div>
<script>
const D=__DATA__;
const cv=document.getElementById('c'),cx=cv.getContext('2d');
document.getElementById('lg').innerHTML=D.legend.map(l=>'<div><span class="sw" style="background:'+l.color+'"></span>'+l.label+'</div>').join('');
let R=[[1,0,0],[0,1,0],[0,0,1]],zoom=1,drag=null;
function mul(A,B){return A.map(r=>[0,1,2].map(j=>r[0]*B[0][j]+r[1]*B[1][j]+r[2]*B[2][j]));}
function rot(ax,t){const c=Math.cos(t),s=Math.sin(t);return ax==='x'?[[1,0,0],[0,c,-s],[0,s,c]]:[[c,0,s],[0,1,0],[-s,0,c]];}
function ap(v){return [0,1,2].map(i=>R[i][0]*v[0]+R[i][1]*v[1]+R[i][2]*v[2]);}
R=mul(rot('x',-1.1),mul([[0,1,0],[0,0,1],[1,0,0]],[[1,0,0],[0,1,0],[0,0,1]]));
R=mul(rot('y',0.5),R);
function hex(h,k){const n=parseInt(h.slice(1),16);return 'rgb('+[(n>>16)&255,(n>>8)&255,n&255].map(x=>Math.round(x*k)).join(',')+')';}
function draw(){
 const w=cv.width=innerWidth*devicePixelRatio,h=cv.height=innerHeight*devicePixelRatio;
 cx.fillStyle='#fff';cx.fillRect(0,0,w,h);
 const s=zoom*0.42*Math.min(w,h)/D.radius,L=[0.35,0.45,0.82];
 const fs=D.faces.map(f=>{const n=ap(f.n);const p=f.v.map(i=>ap(D.verts[i].map((x,j)=>x-D.center[j])));return {n,p,c:f.c,z:p.reduce((a,q)=>a+q[2],0)/p.length};}).sort((a,b)=>a.z-b.z);
 for(const f of fs){ if(f.n[2]<=0)continue;
  const k=0.45+0.55*Math.max(0,f.n[0]*L[0]+f.n[1]*L[1]+f.n[2]*L[2]);
  cx.beginPath();f.p.forEach((q,i)=>{const X=w/2+q[0]*s,Y=h/2-q[1]*s;i?cx.lineTo(X,Y):cx.moveTo(X,Y);});cx.closePath();
  cx.fillStyle=hex(f.c,k);cx.fill();cx.lineWidth=1.5*devicePixelRatio;cx.strokeStyle='#222';cx.stroke();}
}
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointerup',()=>drag=null);
cv.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];drag=[e.clientX,e.clientY];R=mul(rot('y',dx*0.01),mul(rot('x',dy*0.01),R));draw();});
cv.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*0.001);draw();},{passive:false});
addEventListener('resize',draw);draw();
</script>
</body>
</html>
"""


def export_morphology_html(morph: Any, output_path: str | Path, title: str | None = None) -> Path:
    """Offline HTML (2D canvas, no network) with the crystal shape, rotatable with the mouse."""
    out = Path(output_path)
    if out.suffix.lower() not in (".html", ".htm"):
        out = out.with_suffix(".html")
    verts = np.asarray(morph.vertices, dtype=float)
    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    total = sum(f["area"] for f in morph.families) or 1.0
    data = {
        "verts": _round_list(verts, 6),
        "center": _round_list(center, 6),
        "radius": round(float(np.linalg.norm(verts - center, axis=1).max()) or 1.0, 6),
        "faces": [
            {"v": list(f["polygon"]), "n": _round_list(f["normal"], 6),
             "c": morph.families[f["family_index"]]["color"]}
            for f in morph.visible_faces
        ],
        "legend": [
            {"label": f"{f['family']} d={f['distance']:g} ({100.0 * f['area'] / total:.1f}%)", "color": f["color"]}
            for f in morph.families if f.get("n_on_crystal")
        ],
    }
    if not data["faces"]:
        raise ScientificValidationError("Crystal shape has no faces.")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    t = (title or "Crystal shape").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_MORPH_HTML.replace("__TITLE__", t).replace("__DATA__", payload), encoding="utf-8")
    return out


__all__ = [
    "PrintModel",
    "build_print_model",
    "export_html",
    "export_morphology_html",
    "export_morphology_stl",
    "export_stl",
    "export_vrml",
    "split_stl_paths",
]

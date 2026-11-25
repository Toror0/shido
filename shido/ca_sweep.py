"""DFT c/a sweep batch generator and publication-grade energy curve auto-plotter."""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
import pyvista as pv
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar

from shido.analyzer import to_ordered_proxy
from shido.core.output_paths import DATA_DIR, FIG_DIR, STRUCT_DIR  # noqa: E402
from shido.core.validator import validate_composition_conserved, validate_structure
from shido.renderer import (
    screenshot_checked,
    _UNIT_CELL_EDGES,
    _collect_atoms_to_draw,
    _element_color,
    _merge_spheres,
    _px_scale,
    _render_dpi,
    _res_scale,
    line_radius_ratio,
)

_THUMB_PX: int = 360  # thumbnail size at 300 dpi (scaled with render.dpi / 300)
_THUMB_ZOOM: float = 0.25  # OffsetImage zoom at 300 dpi

# Suppress noisy pymatgen UserWarnings during CIF/POSCAR I/O
warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen(\..*)?")

SWEEP_TABLE_COLUMNS: list[str] = [
    "step_index",
    "c_over_a",
    "a_angstrom",
    "b_angstrom",
    "c_angstrom",
    "volume_angstrom3",
    "cif_file",
    "poscar_file",
]

MINIMA_COLUMNS: list[str] = [
    "c_over_a",
    "delta_E_meV_per_atom",
    "phase_label",
]


def apply_constant_volume_ca(structure: Structure, target_ca: float) -> Structure:
    """Apply a volume-conserving tetragonal (Bain-path) distortion to achieve ``c'/a' = target_ca``.

    Preserves total cell volume ``V = structure.volume`` and all fractional
    atomic coordinates while scaling lattice vectors so that ``a' = b'`` and
    ``c' / a' = target_ca``.

    For cubic/tetragonal cells, ``a' = b' = (V / target_ca) ** (1/3)`` and
    ``c' = target_ca * a'``. For general lattices, vector lengths are scaled by
    ``s_a, s_b, s_c`` with ``s_a * s_b * s_c == 1.0`` (preserving interaxial
    angles and exact volume).
    """
    ca = float(target_ca)
    if not np.isfinite(ca) or ca <= 0.0:
        raise ValueError(f"target_ca must be a positive finite number, got {target_ca!r}.")

    mat = np.asarray(structure.lattice.matrix, dtype=float)
    a_len = float(np.linalg.norm(mat[0]))
    b_len = float(np.linalg.norm(mat[1]))
    c_len = float(np.linalg.norm(mat[2]))
    if a_len <= 0.0 or b_len <= 0.0 or c_len <= 0.0:
        raise ValueError("Input structure has a degenerate zero-length lattice vector.")

    # Enforce a' = b' = target_ab, c' = ca * target_ab, and s_a * s_b * s_c == 1.0
    target_ab = ((a_len * b_len * c_len) / ca) ** (1.0 / 3.0)
    s_a = target_ab / a_len
    s_b = target_ab / b_len
    s_c = (ca * target_ab) / c_len

    new_matrix = np.vstack([mat[0] * s_a, mat[1] * s_b, mat[2] * s_c])
    new_lattice = Lattice(new_matrix)

    return Structure(
        lattice=new_lattice,
        species=[site.species for site in structure],
        coords=[np.asarray(site.frac_coords, dtype=float) for site in structure],
        coords_are_cartesian=False,
        site_properties=dict(structure.site_properties) if structure.site_properties else None,
    )


def generate_ca_sweep(
    structure: Structure,
    output_dir: str | Path,
    ca_min: float | None = None,
    ca_max: float | None = None,
    ca_step: float | None = None,
) -> dict[str, Any]:
    """Generate a batch of volume-conserving c/a-distorted CIF and VASP POSCAR files.

    ``ca_min`` / ``ca_max`` / ``ca_step`` left as ``None`` are read from the settings
    (``ca_sweep.*``) at call time.

    Creates ``<output_dir>/structures/`` and ``<output_dir>/data/``, converts
    disordered structures to an ordered proxy via ``to_ordered_proxy``, writes
    ``structures/ca_{ca:.2f}.cif`` and ``structures/POSCAR_ca_{ca:.2f}`` for
    each step, and exports ``data/ca_sweep_table.csv``.
    """
    from shido.settings import get

    ca_min_f = float(ca_min) if ca_min is not None else float(get("ca_sweep.ca_min", 0.85))
    ca_max_f = float(ca_max) if ca_max is not None else float(get("ca_sweep.ca_max", 1.35))
    ca_step_f = float(ca_step) if ca_step is not None else float(get("ca_sweep.ca_step", 0.02))

    if ca_min_f <= 0.0 or ca_max_f <= 0.0:
        raise ValueError("ca_min and ca_max must be positive.")
    if ca_max_f < ca_min_f - 1e-9:
        raise ValueError(f"ca_max ({ca_max_f}) must be >= ca_min ({ca_min_f}).")
    if ca_step_f <= 0.0:
        raise ValueError(f"ca_step ({ca_step_f}) must be positive.")

    out_root = Path(output_dir)
    structures_dir = out_root / STRUCT_DIR
    data_dir = out_root / DATA_DIR
    structures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    work_struct = to_ordered_proxy(structure)
    validate_structure(work_struct)

    # Build the sweep grid without floating-point accumulation drift. When the range
    # contains the cubic reference c/a = 1, anchor the grid on 1.0 so the reference
    # point is always computed (e.g. 0.85..1.35 step 0.02 -> 0.86, ..., 1.00, ..., 1.34);
    # otherwise start from ca_min.
    anchor = 1.0 if ca_min_f - 1e-6 <= 1.0 <= ca_max_f + 1e-6 else ca_min_f
    k_lo = math.ceil((ca_min_f - anchor) / ca_step_f - 1e-6)
    k_hi = math.floor((ca_max_f - anchor) / ca_step_f + 1e-6)
    ca_values: list[float] = [
        float(round(anchor + k * ca_step_f, 10)) for k in range(k_lo, k_hi + 1)
    ]

    if not ca_values:
        ca_values = [ca_min_f]

    # Use .2f per specification; widen precision only if step < 0.01 to avoid filename collisions
    fmt = ".2f" if ca_step_f >= 0.01 - 1e-9 else ".3f"

    rows: list[dict[str, Any]] = []
    cif_files: list[str] = []
    poscar_files: list[str] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for step_index, ca in enumerate(ca_values):
            distorted = apply_constant_volume_ca(work_struct, ca)
            # Guardrail: the Bain distortion must keep composition and not squeeze atoms.
            validate_composition_conserved(work_struct, distorted, context=f"c/a = {ca:.4f}")
            validate_structure(distorted)

            tag = format(ca, fmt)
            cif_path = structures_dir / f"ca_{tag}.cif"
            poscar_path = structures_dir / f"POSCAR_ca_{tag}"

            CifWriter(distorted).write_file(str(cif_path))
            Poscar(
                distorted,
                comment=f"{distorted.composition.reduced_formula} c/a={ca:.4f}",
            ).write_file(str(poscar_path))

            cif_resolved = str(cif_path.resolve())
            poscar_resolved = str(poscar_path.resolve())
            cif_files.append(cif_resolved)
            poscar_files.append(poscar_resolved)

            rows.append(
                {
                    "step_index": int(step_index),
                    "c_over_a": round(float(ca), 6),
                    "a_angstrom": round(float(distorted.lattice.a), 6),
                    "b_angstrom": round(float(distorted.lattice.b), 6),
                    "c_angstrom": round(float(distorted.lattice.c), 6),
                    "volume_angstrom3": round(float(distorted.volume), 6),
                    "cif_file": cif_resolved,
                    "poscar_file": poscar_resolved,
                }
            )

    sweep_table_path = data_dir / "ca_sweep_table.csv"
    df = pd.DataFrame(rows, columns=SWEEP_TABLE_COLUMNS)
    df.to_csv(sweep_table_path, index=False)

    return {
        "output_dir": str(out_root.resolve()),
        "num_steps": int(len(rows)),
        "ca_min": float(ca_values[0]),
        "ca_max": float(ca_values[-1]),
        "ca_step": float(ca_step_f),
        "sweep_table_csv": str(sweep_table_path.resolve()),
        "cif_files": cif_files,
        "poscar_files": poscar_files,
    }


def _synthetic_ni2mnga_bain_energy(ca: np.ndarray) -> np.ndarray:
    """Evaluate a realistic synthetic Ni2MnGa Bain-path DFT curve (meV/atom).

    Produces a classic double-well landscape:
      - Metastable cubic austenitic minimum at c/a = 1.00 (0.0 meV/atom)
      - Energy barrier near c/a ~ 1.09 (~ +5.7 meV/atom)
      - Global tetragonal martensite minimum at c/a = 1.25 (~ -28.0 meV/atom)
    """
    x = np.asarray(ca, dtype=float)

    def _raw(val: np.ndarray | float) -> np.ndarray | float:
        harmonic = 320.0 * (val - 1.125) ** 2 + 12.0
        well_austenite = -17.05 * np.exp(-(((val - 0.986) / 0.075) ** 2))
        well_martensite = -45.55 * np.exp(-(((val - 1.258) / 0.095) ** 2))
        return harmonic + well_austenite + well_martensite

    ref_at_1 = float(_raw(1.00))
    return np.asarray(_raw(x) - ref_at_1, dtype=float)


def generate_demo_ca_energy_csv(
    sweep_table_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    """Generate a realistic synthetic Ni2MnGa Bain-path energy CSV from a sweep table.

    Reads ``c_over_a`` values from ``sweep_table_csv``, computes
    ``delta_E_meV_per_atom`` with a metastable cubic minimum at ``c/a = 1.00``
    (``0.0 meV/atom``), barrier near ``c/a ~ 1.09``, and global tetragonal
    martensite minimum at ``c/a = 1.25`` (``~ -28.0 meV/atom``), writes
    ``output_csv``, and returns the resolved ``Path``.
    """
    sweep_path = Path(sweep_table_csv)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep table CSV not found: {sweep_path}")

    df_sweep = pd.read_csv(sweep_path)
    ca_col = _find_column(df_sweep, ("c_over_a", "c/a", "ca"))
    if ca_col is None:
        raise ValueError(
            f"Could not locate a 'c_over_a' column in {sweep_path}. "
            f"Found columns: {list(df_sweep.columns)}"
        )

    ca_vals = pd.to_numeric(df_sweep[ca_col], errors="coerce").dropna().to_numpy(dtype=float)
    if ca_vals.size == 0:
        raise ValueError(f"No valid numeric c/a values found in {sweep_path}.")

    delta_e = _synthetic_ni2mnga_bain_energy(ca_vals)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "c_over_a": np.round(ca_vals, 6),
            "delta_E_meV_per_atom": np.round(delta_e, 4),
        }
    )
    out_df.to_csv(out_path, index=False)
    return out_path.resolve()


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name in ``df`` (exact or case-insensitive)."""
    lookup = {str(col).strip().lower(): str(col) for col in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _detect_minima(
    ca: np.ndarray,
    energy: np.ndarray,
    max_minima: int = 3,
    min_ca_separation: float = 0.04,
) -> list[dict[str, Any]]:
    """Detect up to ``max_minima`` prominent interior local minima.

    The first and last sweep points are never reported as minima: a lowest value at
    a range edge only means the energy is still falling there (see ``_range_edge_lowest``).
    Each item has ``index``, ``c_over_a``, ``energy`` (plot unit), ``is_global``
    (lowest of all sampled points) and ``prominence``.
    """
    n = len(ca)
    if n < 3:
        return []

    global_e = float(np.min(energy))
    candidate_indices: list[int] = []

    # Interior local minima (collapse flat plateaus to their midpoint)
    i = 1
    while i < n - 1:
        if energy[i] <= energy[i - 1] and energy[i] <= energy[i + 1]:
            j = i
            while j + 1 < n - 1 and np.isclose(energy[j + 1], energy[i], atol=1e-12):
                j += 1
            if energy[i] < energy[i - 1] and energy[j] < energy[j + 1]:
                candidate_indices.append((i + j) // 2)
            i = j + 1
        else:
            i += 1

    # Score prominence for each candidate (both sides exist for interior points)
    scored: list[dict[str, Any]] = []
    for idx in sorted(set(candidate_indices)):
        left_peak = float(np.max(energy[: idx + 1]))
        right_peak = float(np.max(energy[idx:]))
        prom = min(left_peak - float(energy[idx]), right_peak - float(energy[idx]))
        scored.append(
            {
                "index": int(idx),
                "c_over_a": float(ca[idx]),
                "energy": float(energy[idx]),
                "is_global": bool(np.isclose(float(energy[idx]), global_e, atol=1e-12)),
                "prominence": float(max(prom, 0.0)),
            }
        )

    # Lowest minimum first, then highest prominence, then lowest energy
    scored.sort(key=lambda d: (not d["is_global"], -d["prominence"], d["energy"]))

    selected: list[dict[str, Any]] = []
    for item in scored:
        if len(selected) >= max_minima:
            break
        if any(abs(item["c_over_a"] - s["c_over_a"]) < min_ca_separation for s in selected):
            continue
        selected.append(item)

    # Order left-to-right by c/a
    selected.sort(key=lambda d: d["c_over_a"])

    for item in selected:
        val = item["c_over_a"]
        if abs(val - 1.0) <= 0.03:
            item["phase_label"] = "Austenite (cubic)"
        elif val > 1.03:
            item["phase_label"] = "Martensite (tetragonal)"
        else:
            item["phase_label"] = "Tetragonal (c/a < 1)"

    return selected


def _range_edge_lowest(ca: np.ndarray, energy: np.ndarray) -> dict[str, Any] | None:
    """If the lowest sampled energy sits on the first/last sweep point, describe it (not a minimum).

    An edge counts only when it equals the global minimum AND is strictly lower than its
    neighbour (energy still falling outward). Needs >= 2 points, so a flat curve or a
    single point gives no warning, and a tie such as [5, -10, 0, -10] still flags the
    upper edge. If both edges qualify, the lower energy (then the lower side) is returned.
    """
    n = len(ca)
    if n < 2:
        return None
    e = np.asarray(energy, dtype=float)
    g_min = float(np.min(e))
    found: list[tuple[float, int, str]] = []
    if np.isclose(e[0], g_min, atol=1e-12) and e[0] < e[1]:
        found.append((float(e[0]), 0, "lower"))
    if np.isclose(e[-1], g_min, atol=1e-12) and e[-1] < e[-2]:
        found.append((float(e[-1]), n - 1, "upper"))
    if not found:
        return None
    _, idx, side = min(found)
    return {
        "c_over_a": float(ca[idx]),
        "energy": float(e[idx]),
        "side": side,
    }


def _catmull_rom_smooth(x: np.ndarray, y: np.ndarray, samples_per_seg: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Compute a smooth cubic Hermite (PCHIP-like monotone) curve without requiring SciPy."""
    n = len(x)
    if n < 3:
        return x.copy(), y.copy()

    h = np.diff(x)
    if np.any(h <= 0):
        return x.copy(), y.copy()

    delta = np.diff(y) / h
    m = np.zeros(n, dtype=float)
    # Interior tangents via weighted harmonic mean (Fritsch-Carlson monotone cubic)
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            m[k] = 0.0
    m[0] = delta[0]
    m[-1] = delta[-1]

    xs_out: list[np.ndarray] = []
    ys_out: list[np.ndarray] = []
    for k in range(n - 1):
        t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=(k == n - 2))
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        xs_out.append(x[k] + t * h[k])
        ys_out.append(h00 * y[k] + h10 * h[k] * m[k] + h01 * y[k + 1] + h11 * h[k] * m[k + 1])

    return np.concatenate(xs_out), np.concatenate(ys_out)


def _compute_thumbnail_parallel_scale(structures: list[Structure]) -> float:
    """Compute a shared PyVista parallel_scale so all phase thumbnails share exact physical scale."""
    max_half_extent = 1.0
    for struct in structures:
        corners = np.array(
            [
                struct.lattice.get_cartesian_coords([i, j, k])
                for i in (0, 1)
                for j in (0, 1)
                for k in (0, 1)
            ],
            dtype=float,
        )
        center = corners.mean(axis=0)
        radii = np.linalg.norm(corners - center, axis=1)
        max_half_extent = max(max_half_extent, float(np.max(radii)) + 0.65)
    return max_half_extent * 1.08


def _render_crystal_thumbnail(
    structure: Structure,
    size: int | None = None,
    parallel_scale: float | None = None,
) -> np.ndarray:
    """Render a crisp off-screen parallel-projection PyVista thumbnail (RGB uint8 array).

    ``size=None`` is 360 px at 300 dpi, scaled with render.dpi / 300. Atom sizes follow
    render.ball_radius_* / atomic_radii, the cell tube follows render.unit_cell_line_radius.
    """
    px_scale = _px_scale()
    if size is None:
        size = max(1, int(round(_THUMB_PX * px_scale)))
    work = to_ordered_proxy(structure)
    lattice = work.lattice
    corners = np.array(
        [
            lattice.get_cartesian_coords([i, j, k])
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ],
        dtype=float,
    )
    center = corners.mean(axis=0)

    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    try:
        plotter.set_background("white")
        plotter.enable_parallel_projection()
        try:
            plotter.enable_anti_aliasing("msaa")
        except Exception:
            pass

        # Crisp unit-cell wireframe (0.055 Å at the default render.unit_cell_line_radius)
        cell_tube_radius = 0.055 * line_radius_ratio("cell")
        for start_f, end_f in _UNIT_CELL_EDGES:
            p0 = lattice.get_cartesian_coords(start_f)
            p1 = lattice.get_cartesian_coords(end_f)
            line = pv.Line(p0, p1)
            tube = line.tube(radius=cell_tube_radius, n_sides=12)
            plotter.add_mesh(
                tube,
                color="#2C3E50",
                smooth_shading=True,
                ambient=0.35,
                diffuse=0.65,
                specular=0.1,
            )

        # Atoms (with boundary images so the unit cell looks complete and symmetric)
        atoms = _collect_atoms_to_draw(work, include_boundary_atoms=True)
        by_elem: dict[str, tuple[list[np.ndarray], list[float]]] = {}
        for cart, species_occs, radius in atoms:
            sym = species_occs[0][0] if species_occs else "X"
            centers, radii = by_elem.setdefault(sym, ([], []))
            centers.append(np.asarray(cart, dtype=float))
            radii.append(float(radius) * 0.95)
        res = _res_scale(28, px_scale)
        for sym, (centers, radii) in by_elem.items():
            plotter.add_mesh(
                _merge_spheres(centers, radii, res),
                color=_element_color(sym),
                smooth_shading=True,
                ambient=0.25,
                diffuse=0.72,
                specular=0.28,
                specular_power=25,
            )

        # Consistent camera orientation with c-axis strictly vertical so c/a elongation is obvious
        view_dir = np.array([1.75, 1.15, 0.55], dtype=float)
        view_dir /= np.linalg.norm(view_dir)
        scale = parallel_scale if parallel_scale is not None else _compute_thumbnail_parallel_scale([work])
        cam_dist = scale * 4.0

        plotter.camera_position = [
            tuple(center + view_dir * cam_dist),
            tuple(center),
            (0.0, 0.0, 1.0),
        ]
        plotter.camera.parallel_scale = float(scale)

        return screenshot_checked(plotter, None, context="c/a thumbnail")
    finally:
        plotter.close()


def _formula_mathtext(structure: Structure) -> str:
    """Reduced formula with mathtext subscripts (Ni$_2$MnGa); ordering rules in renderer.formula_parts."""
    from shido.renderer import formula_parts

    return "".join(rf"{sym}$_{{{cnt}}}$" if cnt else sym for sym, cnt in formula_parts(structure))


def plot_ca_energy_curve(
    energy_csv: str | Path,
    base_structure: Structure | None,
    output_dir: str | Path,
    points_label: str = "DFT sweep points",
    n_atoms: int | None = None,
) -> dict[str, Any]:
    """Plot a publication-grade ΔE (meV/atom) vs c/a curve with embedded crystal thumbnails.

    Reads ``energy_csv`` (auto-detecting ``c_over_a`` and ``delta_E_meV_per_atom``,
    ``energy_meV_per_atom``, or raw ``energy_eV``), detects up to 3 prominent
    local minima, exports ``data/energy_minima.csv``, and writes both
    ``images/figure_ca_energy.png`` (render.dpi) and ``images/figure_ca_energy.svg``.

    Units: per-atom columns are plotted in meV/atom. For raw ``energy_eV`` (energy of
    the whole sweep cell) the atom count comes from ``n_atoms``, an ``n_atoms`` /
    ``num_atoms`` / ``num_sites`` column, or ``base_structure`` (in that order); if none
    is known the curve is plotted and labelled in meV/cell. ``ca_sweep.energy_unit``
    (``meV/atom`` or ``meV/cell``) chooses the axis when the atom count is known.
    The first/last sweep points are never reported as minima.
    """
    csv_path = Path(energy_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Energy CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    ca_col = _find_column(df, ("c_over_a", "c/a", "ca"))
    if ca_col is None:
        raise ValueError(
            f"Could not locate a 'c_over_a' column in {csv_path}. "
            f"Found columns: {list(df.columns)}"
        )

    e_col = _find_column(
        df,
        ("delta_E_meV_per_atom", "energy_meV_per_atom", "delta_E_meV_per_cell", "energy_eV"),
    )
    if e_col is None:
        raise ValueError(
            f"Could not locate an energy column ('delta_E_meV_per_atom', "
            f"'energy_meV_per_atom', or 'energy_eV') in {csv_path}. "
            f"Found columns: {list(df.columns)}"
        )

    sub = df[[ca_col, e_col]].copy()
    sub[ca_col] = pd.to_numeric(sub[ca_col], errors="coerce")
    sub[e_col] = pd.to_numeric(sub[e_col], errors="coerce")
    sub = sub.dropna().sort_values(by=ca_col, ascending=True).reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No valid numeric rows found in {csv_path}.")

    ca_arr = sub[ca_col].to_numpy(dtype=float)
    raw_e = sub[e_col].to_numpy(dtype=float)

    # Atom count of the computed cell: argument > CSV column > base structure
    n_cell: int | None = int(n_atoms) if n_atoms is not None and int(n_atoms) > 0 else None
    if n_cell is None:
        n_atoms_col = _find_column(df, ("n_atoms", "num_atoms", "num_sites"))
        if n_atoms_col is not None:
            parsed_n = pd.to_numeric(df[n_atoms_col], errors="coerce").dropna()
            if not parsed_n.empty and float(parsed_n.iloc[0]) > 0:
                n_cell = int(round(float(parsed_n.iloc[0])))
    if n_cell is None and base_structure is not None and len(base_structure) > 0:
        n_cell = len(to_ordered_proxy(base_structure))

    # Energies relative to the point closest to c/a = 1 (cubic reference)
    ref_idx = int(np.argmin(np.abs(ca_arr - 1.0)))
    e_key = e_col.strip().lower()
    e_per_atom: np.ndarray | None
    e_per_cell: np.ndarray | None
    if e_key == "delta_e_mev_per_atom":
        e_per_atom = raw_e
        e_per_cell = raw_e * n_cell if n_cell else None
    elif e_key == "energy_mev_per_atom":
        e_per_atom = raw_e - raw_e[ref_idx]
        e_per_cell = e_per_atom * n_cell if n_cell else None
    elif e_key == "delta_e_mev_per_cell":
        # Already relative (like delta_E_meV_per_atom): do not re-reference, otherwise a
        # partial range without c/a = 1 would shift the per-cell curve but not the per-atom one.
        e_per_cell = raw_e
        e_per_atom = e_per_cell / float(n_cell) if n_cell else None
    else:  # energy_eV (whole cell, absolute) -> relative to the point closest to c/a = 1
        e_per_cell = (raw_e - raw_e[ref_idx]) * 1000.0
        e_per_atom = e_per_cell / float(n_cell) if n_cell else None

    try:
        from shido.settings import get

        unit_pref = str(get("ca_sweep.energy_unit", "meV/atom")).strip().lower()
    except Exception:  # noqa: BLE001 - settings are optional for plotting
        unit_pref = "meV/atom"
    if e_per_atom is not None and (unit_pref != "mev/cell" or e_per_cell is None):
        delta_e, energy_unit = e_per_atom, "meV/atom"
    else:
        delta_e, energy_unit = e_per_cell, "meV/cell"
    assert delta_e is not None
    energy_col_name = "delta_E_meV_per_atom" if energy_unit == "meV/atom" else "delta_E_meV_per_cell"

    out_root = Path(output_dir)
    data_dir = out_root / DATA_DIR
    images_dir = out_root / FIG_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    detected = _detect_minima(ca_arr, delta_e, max_minima=3)
    edge = _range_edge_lowest(ca_arr, delta_e)
    warnings_out: list[str] = []
    if edge is not None:
        warnings_out.append(
            f"Lowest energy is at the {edge['side']} end of the sweep (c/a = {edge['c_over_a']:.2f}); "
            "it is not a minimum. Extend the c/a range to find it."
        )
    if energy_unit == "meV/cell" and n_cell is None:
        warnings_out.append(
            "Atom count unknown (no n_atoms argument/column, no base structure): energies are per cell (meV/cell)."
        )
    minima_records: list[dict[str, Any]] = [
        {
            "c_over_a": round(float(m["c_over_a"]), 6),
            energy_col_name: round(float(m["energy"]), 4),
            "phase_label": str(m["phase_label"]),
        }
        for m in detected
    ]

    minima_csv_path = data_dir / "energy_minima.csv"
    minima_cols = [energy_col_name if c == "delta_E_meV_per_atom" else c for c in MINIMA_COLUMNS]
    pd.DataFrame(minima_records, columns=minima_cols).to_csv(minima_csv_path, index=False)

    # Prepare crystal thumbnails for each detected minimum if base_structure is given
    thumbnails: list[np.ndarray] = []
    if base_structure is not None and detected:
        work_base = to_ordered_proxy(base_structure)
        distorted_min_structs = [
            apply_constant_volume_ca(work_base, float(m["c_over_a"])) for m in detected
        ]
        shared_scale = _compute_thumbnail_parallel_scale(distorted_min_structs)
        for struct in distorted_min_structs:
            thumbnails.append(_render_crystal_thumbnail(struct, parallel_scale=shared_scale))

    # Publication-grade Matplotlib styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = True  # true minus sign (−20), not a hyphen
    plt.rcParams["svg.fonttype"] = "none"

    # render.dpi: same figure size in inches, more pixels (fonts/lines are in points)
    fig_dpi = _render_dpi()
    px_scale = fig_dpi / 300.0
    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=fig_dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Smooth curve + data markers
    xs_smooth, ys_smooth = _catmull_rom_smooth(ca_arr, delta_e, samples_per_seg=14)
    ax.axhline(0.0, color="#7F8C8D", linestyle="--", linewidth=1.0, zorder=1)
    ax.plot(
        xs_smooth,
        ys_smooth,
        color="#1F4E79",
        linewidth=2.2,
        zorder=2,
        label="Bain-path energy curve",
    )
    ax.scatter(
        ca_arr,
        delta_e,
        s=34,
        facecolor="white",
        edgecolor="#1F4E79",
        linewidths=1.4,
        zorder=3,
        label=points_label,
    )

    x_min, x_max = float(np.min(ca_arr)), float(np.max(ca_arr))
    x_span = max(x_max - x_min, 0.1)
    y_min, y_max = float(np.min(delta_e)), float(np.max(delta_e))
    y_span = max(y_max - y_min, 10.0)

    # Highlight minima and place thumbnails directly above each minimum
    box_x_positions: list[float] = [float(m["c_over_a"]) for m in detected]
    min_sep = 0.22 * x_span
    for k in range(1, len(box_x_positions)):
        if box_x_positions[k] - box_x_positions[k - 1] < min_sep:
            mid = 0.5 * (box_x_positions[k] + box_x_positions[k - 1])
            box_x_positions[k - 1] = max(x_min + 0.04 * x_span, mid - 0.5 * min_sep)
            box_x_positions[k] = min(x_max - 0.04 * x_span, mid + 0.5 * min_sep)

    has_thumbs = len(thumbnails) == len(detected) and len(thumbnails) > 0
    y_top_needed = y_max + 0.15 * y_span

    for idx_m, m in enumerate(detected):
        mx = float(m["c_over_a"])
        my = float(m["energy"])
        is_global = bool(m.get("is_global", False))
        accent = "#C0392B" if is_global else "#D97706"
        role_tag = "Lowest min" if is_global else "Local min"
        marker_shape = "o" if is_global else "D"

        ax.scatter(
            [mx],
            [my],
            s=88,
            marker=marker_shape,
            facecolor=accent,
            edgecolor="black",
            linewidths=0.9,
            zorder=5,
            label=rf"{m['phase_label']}: $c/a$ = {mx:.2f}",
        )

        # Compute clean vertical slot directly above this minimum
        local_mask = np.abs(ca_arr - box_x_positions[idx_m]) <= 0.10 * x_span
        local_ceiling = float(np.max(delta_e[local_mask])) if np.any(local_mask) else my

        if has_thumbs:
            y_box = max(local_ceiling + 0.42 * y_span, my + 0.52 * y_span)
            y_top_needed = max(y_top_needed, y_box + 0.44 * y_span)

            # thumbnails have px_scale x more pixels; shrink zoom so the printed size stays
            imagebox = OffsetImage(thumbnails[idx_m], zoom=_THUMB_ZOOM / px_scale)
            ab = AnnotationBbox(
                imagebox,
                xy=(mx, my),
                xybox=(box_x_positions[idx_m], y_box),
                xycoords="data",
                boxcoords="data",
                pad=0.28,
                frameon=True,
                bboxprops=dict(
                    edgecolor=accent,
                    facecolor="white",
                    linewidth=1.3,
                    boxstyle="round,pad=0.25",
                ),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=accent,
                    linewidth=1.4,
                    shrinkA=6,
                    shrinkB=7,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=6,
            )
            ax.add_artist(ab)

            my_txt = f"{my:+.1f}".replace("-", "−")  # true minus sign
            caption = rf"{role_tag}" + "\n" + rf"$c/a$ = {mx:.2f} | {my_txt} {energy_unit}"
            ax.text(
                box_x_positions[idx_m],
                y_box + 0.29 * y_span,
                caption,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=accent,
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.88,
                ),
                zorder=7,
            )
        else:
            y_text = max(local_ceiling + 0.18 * y_span, my + 0.22 * y_span)
            y_top_needed = max(y_top_needed, y_text + 0.18 * y_span)
            ax.annotate(
                f"{m['phase_label']}\nc/a = {mx:.2f}, " + f"{my:+.1f}".replace("-", "−") + f" {energy_unit}",
                xy=(mx, my),
                xytext=(box_x_positions[idx_m], y_text),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=accent,
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="white",
                    edgecolor=accent,
                    linewidth=1.0,
                ),
                arrowprops=dict(arrowstyle="-|>", color=accent, linewidth=1.3, shrinkB=6),
                zorder=6,
            )

    if edge is not None:
        ex, ey = float(edge["c_over_a"]), float(edge["energy"])
        ax.scatter(
            [ex], [ey], s=80, marker="s", facecolor="#7F8C8D", edgecolor="black",
            linewidths=0.9, zorder=5, label=rf"Lowest at range edge ($c/a$ = {ex:.2f}), not a minimum",
        )
        y_note = ey + 0.20 * y_span
        y_top_needed = max(y_top_needed, y_note + 0.15 * y_span)
        ax.annotate(
            "Still decreasing at range edge\n(extend the sweep)",
            xy=(ex, ey),
            xytext=(ex - 0.14 * x_span if edge["side"] == "upper" else ex + 0.14 * x_span, y_note),
            ha="center", va="bottom", fontsize=8.5, color="#4D5656",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#7F8C8D", linewidth=0.9),
            arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", linewidth=1.1, shrinkB=6),
            zorder=6,
        )

    ax.set_xlim(x_min - 0.04 * x_span, x_max + 0.04 * x_span)
    ax.set_ylim(y_min - 0.14 * y_span, y_top_needed)

    # Academic graph rules (from jikken-report/CLAUDE.md):
    # - Axis labels: "Quantity name, symbol [unit]" with italic variables and upright Delta/units
    # - No background grid lines
    # - Full 4-sided box frame, inward ticks on all 4 sides, 1/5 minor ticks without numbers
    ax.set_xlabel(r"Axial ratio, $c/a$", fontsize=11.5, labelpad=8)
    ax.set_ylabel(rf"Relative energy, $\mathrm{{\Delta}}E$ [{energy_unit}]", fontsize=11.5, labelpad=8)

    formula_prefix = (
        f"{_formula_mathtext(base_structure)}: "
        if base_structure is not None
        else ""
    )
    ax.set_title(
        rf"{formula_prefix}Constant-volume Bain path ($\mathrm{{\Delta}}E$ vs. $c/a$)",
        fontsize=12.0,
        pad=10,
    )

    ax.grid(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=6.0,
        width=1.1,
        labelsize=10,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=3.0,
        width=0.8,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#111111")

    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#333333",
        framealpha=1.0,
        fontsize=8.5,
    )

    fig.tight_layout()

    png_path = images_dir / "figure_ca_energy.png"
    svg_path = images_dir / "figure_ca_energy.svg"
    fig.savefig(png_path, dpi=fig_dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "output_dir": str(out_root.resolve()),
        "energy_minima_csv": str(minima_csv_path.resolve()),
        "minima": minima_records,
        "energy_unit": energy_unit,
        "n_atoms": n_cell,
        "range_edge_lowest": (
            {"c_over_a": round(edge["c_over_a"], 6), energy_col_name: round(edge["energy"], 4), "side": edge["side"]}
            if edge is not None
            else None
        ),
        "warnings": warnings_out,
        "figure_ca_energy_png": str(png_path.resolve()),
        "figure_ca_energy_svg": str(svg_path.resolve()),
    }

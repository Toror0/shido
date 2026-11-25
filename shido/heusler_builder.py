"""Heusler alloy sublattice & substitution model generator (Design 2).

Generates L2_1 and B2 ordered/disordered Heusler supercells (and ordered
inverse-Heusler XA, F-43m) from an alloy
composition (e.g., Ni50Mn35In15 or Ni2MnGa), computes nearest-neighbor
magnetic-exchange pair distances, and renders publication-ready 300 dpi PNG
and editable SVG figures using headless PyVista (parallel projection).
"""

from __future__ import annotations

import base64
import itertools
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar

from shido.renderer import (
    screenshot_checked,
    _compute_parallel_framing as _renderer_parallel_framing,
    _dpi_pair,
    _merge_spheres,
    _px_scale,
    _radius_params,
    _res_scale,
    _resolve_transparent,
    add_orientation_axes,
    apply_studio_lighting,
    line_radius_ratio,
    relative_ball_radius,
)
from shido.core.output_paths import DATA_DIR, FIG_DIR, STRUCT_DIR  # noqa: E402
from shido.core.species import dominant_element
from shido.theme_manager import get_color_manager


def _scene_colors() -> dict[str, Any]:
    """Background / silhouette / unit-cell colors of the active theme (render.theme).

    Sublattice and highlight colors stay fixed: they encode Wyckoff sites, not elements.
    """
    sc = get_color_manager().scene_colors
    return {
        "background": str(sc.get("background", "#FFFFFF")),
        "silhouette": str(sc.get("silhouette", "#1A1A1A")),
        "unit_cell": str(sc.get("unit_cell", "#222222")),
    }

# Pixel sizes below (window 1200 px, fonts, paddings, line widths) are 300-dpi values;
# they are multiplied by render.dpi / 300 so the printed size stays the same.
_WINDOW_PX: int = 1200


def _k(value: float, px_scale: float) -> int:
    """A 300-dpi pixel length at the current dpi (0 stays 0)."""
    return int(round(float(value) * px_scale))


# Conventional 16-atom Fm-3m (No. 225) Heusler Wyckoff positions
WYCKOFF_4A: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
)
WYCKOFF_4B: tuple[tuple[float, float, float], ...] = (
    (0.5, 0.5, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)
WYCKOFF_8C: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
    (0.75, 0.75, 0.75),
    (0.75, 0.25, 0.25),
    (0.25, 0.75, 0.25),
    (0.25, 0.25, 0.75),
)

# Conventional 16-atom F-43m (No. 216) inverse-Heusler XA positions (Hg2CuTi type).
# Same 16 sites as above: 8c splits into 4c (1/4,1/4,1/4) and 4d (3/4,3/4,3/4).
# Coordinates follow text2cif's heusler_xa prototype: X -> 4a + 4d, Y -> 4c, Z -> 4b.
WYCKOFF_4C: tuple[tuple[float, float, float], ...] = (
    (0.25, 0.25, 0.25),
    (0.25, 0.75, 0.75),
    (0.75, 0.25, 0.75),
    (0.75, 0.75, 0.25),
)
WYCKOFF_4D: tuple[tuple[float, float, float], ...] = (
    (0.75, 0.75, 0.75),
    (0.75, 0.25, 0.25),
    (0.25, 0.75, 0.25),
    (0.25, 0.25, 0.75),
)

SUPPORTED_ORDERINGS: tuple[str, ...] = ("L21", "B2", "XA")

SUBLATTICE_COLORS: dict[str, str] = {
    "8c": "#4E79A7",  # steel blue
    "4a": "#9C7AC7",  # purple
    "4b": "#F28E2B",  # amber
    "4c": "#59A14F",  # green (XA only)
    "4d": "#76B7B2",  # teal (XA only)
}


def heusler_sublattice_keys(ordering: str) -> tuple[str, ...]:
    """Wyckoff sublattice keys used by an ordering.

    L21 / B2 (Fm-3m): ("8c", "4a", "4b"). XA (F-43m): ("4a", "4d", "4c", "4b").
    """
    o = str(ordering).strip().upper()
    if o in ("L21", "B2"):
        return ("8c", "4a", "4b")
    if o == "XA":
        return ("4a", "4d", "4c", "4b")
    raise ValueError(f"Unsupported ordering {ordering!r}. Expected 'L21', 'B2' or 'XA'.")


def _sublattice_basis(ordering: str) -> tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...]:
    """(key, fractional positions in the 16-atom cubic cell) per sublattice of an ordering."""
    coords = {
        "8c": WYCKOFF_8C, "4a": WYCKOFF_4A, "4b": WYCKOFF_4B,
        "4c": WYCKOFF_4C, "4d": WYCKOFF_4D,
    }
    return tuple((key, coords[key]) for key in heusler_sublattice_keys(ordering))

HIGHLIGHT_BOND_COLOR: str = "#D62728"  # vivid red
MUTED_ATOM_COLOR: str = "#C8CDD3"

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

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9.]*)")


def parse_alloy_composition(formula: str) -> dict[str, float]:
    """Parse an alloy formula (e.g. 'Ni50Mn35In15' or 'Ni2MnGa') into atomic fractions summing to 1.0."""
    cleaned = formula.strip().replace(" ", "")
    if not cleaned:
        raise ValueError("Alloy formula must not be empty.")

    matches = _FORMULA_TOKEN_RE.findall(cleaned)
    if not matches:
        raise ValueError(f"Could not parse any element tokens from formula: {formula!r}")

    reconstructed = "".join(f"{el}{num}" for el, num in matches)
    if reconstructed != cleaned:
        raise ValueError(
            f"Formula {formula!r} contains unrecognized characters (parsed {reconstructed!r})."
        )

    raw_counts: dict[str, float] = {}
    for elem, num_str in matches:
        if num_str == "":
            val = 1.0
        else:
            try:
                val = float(num_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid stoichiometric number {num_str!r} for element {elem!r}."
                ) from exc
        if val < 0.0:
            raise ValueError(f"Negative stoichiometry for element {elem!r}: {val}")
        raw_counts[elem] = raw_counts.get(elem, 0.0) + val

    total = sum(raw_counts.values())
    if total <= 0.0:
        raise ValueError(f"Total composition weight must be positive for formula: {formula!r}")

    return {elem: count / total for elem, count in raw_counts.items()}


def _allocate_integer_counts(fractions: dict[str, float], total_sites: int) -> dict[str, int]:
    """Allocate integer site counts summing to total_sites using the largest-remainder method."""
    exact = {el: frac * total_sites for el, frac in fractions.items()}
    floors = {el: int(math.floor(val)) for el, val in exact.items()}
    remainder = total_sites - sum(floors.values())

    if remainder > 0:
        order = sorted(
            exact.keys(),
            key=lambda el: (exact[el] - floors[el], fractions[el]),
            reverse=True,
        )
        for el in order[:remainder]:
            floors[el] += 1

    return floors


def _build_supercell_template(
    supercell: tuple[int, int, int],
    a: float,
    ordering: str = "L21",
) -> tuple[Lattice, list[np.ndarray], list[str]]:
    """Build fractional coordinates and Wyckoff labels for an (nx, ny, nz) Heusler supercell."""
    nx, ny, nz = (int(supercell[0]), int(supercell[1]), int(supercell[2]))
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError(f"Supercell dimensions must all be >= 1, got {supercell!r}")
    if a <= 0.0:
        raise ValueError(f"Lattice parameter 'a' must be positive, got {a!r}")

    lattice = Lattice.orthorhombic(a * nx, a * ny, a * nz)
    frac_coords: list[np.ndarray] = []
    wyckoff_labels: list[str] = []

    sublattice_defs: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...] = (
        _sublattice_basis(ordering)
    )

    for ix, iy, iz in itertools.product(range(nx), range(ny), range(nz)):
        for label, basis in sublattice_defs:
            for fx, fy, fz in basis:
                frac_coords.append(
                    np.array(
                        [(fx + ix) / nx, (fy + iy) / ny, (fz + iz) / nz],
                        dtype=float,
                    )
                )
                wyckoff_labels.append(label)

    return lattice, frac_coords, wyckoff_labels


def _heusler_roles(
    element_order: list[str],
    fractions: dict[str, float] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Group elements into X2YZ roles (X -> 8c, Y -> 4a, Z -> 4b) from the written formula order.

    The first written element is X. Following elements join the X group while the
    group stays within its 50 at.% (e.g. Co in Ni45Co5Mn37In13 substitutes Ni). The
    next element starts Y, and further elements join Y while it stays within 25 at.%;
    everything after is Z. The written order (not the abundance) decides the roles,
    so Mn-rich Ni40Mn45In15 still has X=Ni, Y=Mn, Z=In.
    Without ``fractions`` the plain order X, Y, Z, Z... is used.
    """
    if not element_order:
        raise ValueError("Formula has no elements.")
    tol = 0.005  # 0.5 at.% rounding tolerance
    x_els = [element_order[0]]
    rest = list(element_order[1:])
    if fractions is not None:
        cum = fractions.get(x_els[0], 0.0)
        while (
            len(rest) > 2  # keep at least one element each for Y and Z
            and cum + fractions.get(rest[0], 0.0) <= 0.5 + tol
            and cum < 0.5 - tol
        ):
            cum += fractions.get(rest[0], 0.0)
            x_els.append(rest.pop(0))
    y_els: list[str] = []
    if rest:
        y_els.append(rest.pop(0))
        if fractions is not None:
            cum = fractions.get(y_els[0], 0.0)
            while (
                len(rest) > 1  # keep at least one element for Z
                and cum + fractions.get(rest[0], 0.0) <= 0.25 + tol
                and cum < 0.25 - tol
            ):
                cum += fractions.get(rest[0], 0.0)
                y_els.append(rest.pop(0))
    return x_els, y_els, rest


def _split_even(counts: dict[str, int], order: list[str], n_a: int, n_b: int) -> tuple[list[str], list[str]]:
    """Split each element's count as evenly as possible between two equal sublattices."""
    pool_a: list[str] = []
    pool_b: list[str] = []
    odd: list[str] = []
    for el in order:
        c = int(counts.get(el, 0))
        half = c // 2
        pool_a.extend([el] * half)
        pool_b.extend([el] * half)
        if c % 2:
            odd.append(el)
    for el in odd:  # alternate odd leftovers so neither side overflows
        if len(pool_a) < n_a and (len(pool_a) <= len(pool_b) or len(pool_b) >= n_b):
            pool_a.append(el)
        else:
            pool_b.append(el)
    if len(pool_a) != n_a or len(pool_b) != n_b:
        raise RuntimeError(
            f"Internal site-allocation mismatch on 4a/4b: {len(pool_a)}/{n_a}, {len(pool_b)}/{n_b}"
        )
    return pool_a, pool_b


def _assign_species_for_pattern(
    wyckoff_labels: list[str],
    target_counts: dict[str, int],
    element_order: list[str],
    ordering: str,
    rng: np.random.Generator,
) -> list[str]:
    """Assign element symbols to sites according to L21 or B2 Heusler ordering (X2YZ).

    Roles come from ``_heusler_roles`` (written formula order):

    * L21: X -> 8c, Y -> 4a, Z -> 4b. Atoms that do not fit on their own
      sublattice (off-stoichiometry) go to vacant sites, preferring
      Y excess -> 4b (Z site; e.g. excess Mn on In sites in Ni-Mn-In) then 8c,
      X excess -> 4a then 4b, Z excess -> 4a then 8c.
    * B2: X -> 8c; Y and Z share 4a+4b at random, with each element split
      50:50 between 4a and 4b (same composition on both). An 8c deficit is
      filled by the Y excess over 25 % first, then the Z excess over 25 %.
    * XA (inverse Heusler, F-43m; ordered only): X -> 4a + 4d, Y -> 4c, Z -> 4b.
      Off-stoichiometric excess simply fills vacant sites (Y -> 4b first,
      X and Z -> 4c first). No XA disorder is modelled.
    """
    ordering_upper = ordering.upper()
    if ordering_upper not in set(SUPPORTED_ORDERINGS):
        raise ValueError(f"Unsupported ordering {ordering!r}. Expected 'L21', 'B2' or 'XA'.")
    sub_keys = heusler_sublattice_keys(ordering_upper)

    idx = {sub: [i for i, w in enumerate(wyckoff_labels) if w == sub] for sub in sub_keys}
    cap = {sub: len(v) for sub, v in idx.items()}
    pools: dict[str, list[str]] = {sub: [] for sub in sub_keys}
    remaining = {el: int(target_counts.get(el, 0)) for el in element_order}
    n_total = max(sum(remaining.values()), 1)

    x_els, y_els, z_els = _heusler_roles(
        element_order, {el: remaining[el] / n_total for el in element_order}
    )

    def free(sub: str) -> int:
        return cap[sub] - len(pools[sub])

    def put_group(els: list[str], sub: str, limit: int | None = None) -> None:
        """Put as many atoms of the group on ``sub`` as fit (random pick if they don't all fit).

        ``limit`` caps the number of atoms moved.
        """
        pool: list[str] = []
        for el in els:
            pool.extend([el] * remaining[el])
        rng.shuffle(pool)
        n_put = free(sub) if limit is None else max(0, min(free(sub), int(limit)))
        for el in pool[:n_put]:
            pools[sub].append(el)
            remaining[el] -= 1

    # Step 1: each group on its own (home) sublattice
    if ordering_upper != "XA":
        put_group(x_els, "8c")
    if ordering_upper == "XA":
        # Ordered inverse Heusler: X on 4a + 4d, Y on 4c, Z on 4b
        put_group(x_els, "4a")
        put_group(x_els, "4d")
        put_group(y_els, "4c")
        put_group(z_els, "4b")
        # Minimal off-stoichiometry handling: excess onto vacant sublattices
        for els, order in (
            (y_els, ("4b", "4a", "4d", "4c")),
            (x_els, ("4c", "4b")),
            (z_els, ("4c", "4a", "4d")),
        ):
            for sub in order:
                put_group(els, sub)
        if any(remaining.values()):
            raise RuntimeError(f"Internal site-allocation mismatch: left over {remaining}")
    elif ordering_upper == "L21":
        put_group(y_els, "4a")
        put_group(z_els, "4b")
        # Step 2: off-stoichiometric excess onto vacant sublattices
        for els, order in ((y_els, ("4b", "8c", "4a")), (x_els, ("4a", "4b")), (z_els, ("4a", "8c"))):
            for sub in order:
                put_group(els, sub)
        if any(remaining.values()):
            raise RuntimeError(f"Internal site-allocation mismatch: left over {remaining}")
    else:  # B2
        # X-poor alloys: the 8c deficit is filled by the Y / Z excess over their own
        # 25 % share (4a resp. 4b capacity) first, Y excess before Z excess; only then
        # (if 8c is still not full) by further Y, then Z atoms.
        def group_excess(els: list[str], share: int) -> int:
            return max(0, sum(remaining[el] for el in els) - share)

        y_excess = group_excess(y_els, cap["4a"])
        z_excess = group_excess(z_els, cap["4b"])
        put_group(y_els, "8c", limit=y_excess)
        put_group(z_els, "8c", limit=z_excess)
        put_group(y_els, "8c")
        put_group(z_els, "8c")
        # Everything left (Y, Z and any X excess) goes to 4a+4b, split 50:50 per element
        pools["4a"], pools["4b"] = _split_even(remaining, element_order, cap["4a"], cap["4b"])
        for el in element_order:
            remaining[el] = 0

    species: list[str] = [""] * len(wyckoff_labels)
    for sub in sub_keys:
        if len(pools[sub]) != cap[sub]:
            raise RuntimeError(f"Internal site-allocation mismatch on {sub}: {len(pools[sub])} != {cap[sub]}")
        rng.shuffle(pools[sub])
        for site_idx, el in zip(idx[sub], pools[sub]):
            species[site_idx] = el
    return species


def _compute_sublattice_summary(
    structure: Structure,
    wyckoff_labels: list[str],
    element_order: list[str],
    sublattices: tuple[str, ...] = ("8c", "4a", "4b"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute site counts and element fractions per sublattice (8c/4a/4b, or 4a/4d/4c/4b for XA)."""
    rows: list[dict[str, Any]] = []
    summary_dict: dict[str, Any] = {}

    for sub in sublattices:
        indices = [i for i, w in enumerate(wyckoff_labels) if w == sub]
        total = len(indices)
        counts = Counter(dominant_element(structure[i]) for i in indices)

        row: dict[str, Any] = {"sublattice": sub, "total_sites": total}
        sub_entry: dict[str, Any] = {
            "total_sites": total,
            "counts": {},
            "fractions": {},
        }
        parts: list[str] = []
        for el in element_order:
            c = int(counts.get(el, 0))
            f = (c / total) if total > 0 else 0.0
            row[f"count_{el}"] = c
            row[f"fraction_{el}"] = round(f, 6)
            sub_entry["counts"][el] = c
            sub_entry["fractions"][el] = round(f, 6)
            if c > 0:
                parts.append(f"{el}: {100.0 * f:.1f}%")

        row["composition"] = ", ".join(parts)
        sub_entry["composition"] = row["composition"]
        rows.append(row)
        summary_dict[sub] = sub_entry

    return pd.DataFrame(rows), summary_dict


def _compute_pair_distances(
    structure: Structure,
    wyckoff_labels: list[str],
    highlight_pair: tuple[str, str],
    cutoff: float,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray, float, str]]]:
    """Compute unique undirected highlight_pair distances within cutoff (with PBC)."""
    el_a, el_b = str(highlight_pair[0]), str(highlight_pair[1])
    same_elem = el_a == el_b

    all_neighbors = structure.get_all_neighbors(r=cutoff)
    seen_keys: set[tuple[int, int, tuple[int, int, int]]] = set()
    rows: list[dict[str, Any]] = []
    render_segments: list[tuple[np.ndarray, np.ndarray, float, str]] = []

    for i, nbrs in enumerate(all_neighbors):
        sym_i = dominant_element(structure[i])
        if sym_i not in (el_a, el_b):
            continue
        sub_i = wyckoff_labels[i]
        pos_i = np.asarray(structure[i].coords, dtype=float)

        for nbr in nbrs:
            j = int(nbr.index)
            sym_j = dominant_element(structure[j])
            if same_elem:
                if sym_i != el_a or sym_j != el_a:
                    continue
            else:
                if {sym_i, sym_j} != {el_a, el_b}:
                    continue

            img = tuple(int(round(x)) for x in nbr.image)
            if i == j and img == (0, 0, 0):
                continue

            if i < j:
                key = (i, j, img)
            elif i > j:
                key = (j, i, (-img[0], -img[1], -img[2]))
            else:
                # Self-image across PBC: canonicalize by first non-zero image component > 0
                first_nz = next((v for v in img if v != 0), 1)
                canon_img = img if first_nz > 0 else (-img[0], -img[1], -img[2])
                key = (i, j, canon_img)

            if key in seen_keys:
                continue
            seen_keys.add(key)

            sub_j = wyckoff_labels[j]
            dist = float(nbr.nn_distance)

            if same_elem:
                s1, s2 = sorted((sub_i, sub_j))
                pair_type = f"{el_a}({s1})-{el_a}({s2})"
            else:
                if sym_i == el_a:
                    pair_type = f"{sym_i}({sub_i})-{sym_j}({sub_j})"
                else:
                    pair_type = f"{sym_j}({sub_j})-{sym_i}({sub_i})"

            rows.append(
                {
                    "site_i": i,
                    "sublattice_i": sub_i,
                    "site_j": j,
                    "sublattice_j": sub_j,
                    "pair_type": pair_type,
                    "distance_angstrom": round(dist, 6),
                }
            )

            # For 3D visualization, keep segments strictly inside the supercell box (image == (0,0,0))
            if img == (0, 0, 0):
                pos_j = np.asarray(structure[j].coords, dtype=float)
                render_segments.append((pos_i, pos_j, dist, pair_type))

    columns = [
        "site_i",
        "sublattice_i",
        "site_j",
        "sublattice_j",
        "pair_type",
        "distance_angstrom",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(
            by=["distance_angstrom", "site_i", "site_j"], kind="mergesort"
        ).reset_index(drop=True)

    return df, render_segments


def _orthonormal_up(cam_dir: np.ndarray, view_up: np.ndarray) -> np.ndarray:
    """Return a unit view_up vector orthogonal to cam_dir."""
    c = cam_dir / np.linalg.norm(cam_dir)
    u = np.asarray(view_up, dtype=float)
    u_norm = float(np.linalg.norm(u))
    u = np.array([0.0, 0.0, 1.0], dtype=float) if u_norm < 1e-8 else (u / u_norm)
    u_orth = u - np.dot(u, c) * c
    if float(np.linalg.norm(u_orth)) < 1e-6:
        fallback = (
            np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(c[1]) < 0.9
            else np.array([1.0, 0.0, 0.0], dtype=float)
        )
        u_orth = fallback - np.dot(fallback, c) * c
    return u_orth / np.linalg.norm(u_orth)


_FRAMING_MARGIN = 1.28  # padding factor used by the Heusler figures
_RENDERER_FRAMING_MARGIN = 1.18  # padding factor applied inside renderer._compute_parallel_framing


def _compute_parallel_framing(
    lattice: Lattice,
    atom_points: list[tuple[np.ndarray, float]],
    cam_unit: np.ndarray,
    up_unit: np.ndarray,
    window_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Compute (focal_point, parallel_scale) for orthographic framing.

    Thin adapter over ``renderer._compute_parallel_framing`` (atom bounding box with
    radii; cell corners kept only for bulk-like cells), so slabs with a long vacuum gap
    are framed the same way here as in the main renderer. The Heusler figures keep
    their slightly wider margin (1.28 instead of the renderer's 1.18).
    """
    atoms = [(np.asarray(cart, dtype=float), [], float(radius)) for cart, radius in atom_points]
    focal_point, parallel_scale, _ = _renderer_parallel_framing(
        Structure(lattice, [], []), atoms, cam_unit, up_unit, window_size
    )
    return focal_point, float(parallel_scale) * (_FRAMING_MARGIN / _RENDERER_FRAMING_MARGIN)


def _add_unit_cell_wireframe(
    plotter: pv.Plotter,
    lattice: Lattice,
    tube_radius: float = 0.032,
    color: str = "#222222",
) -> None:
    """Draw the 12 edges of a lattice box as crisp dark tubes."""
    tubes: list[pv.PolyData] = []
    for f0, f1 in _UNIT_CELL_EDGES:
        p0 = lattice.get_cartesian_coords(f0)
        p1 = lattice.get_cartesian_coords(f1)
        tubes.append(pv.Line(p0, p1).tube(radius=tube_radius, n_sides=12))
    plotter.add_mesh(pv.merge(tubes), color=color, smooth_shading=True)


def _add_mesh_with_silhouette(
    plotter: pv.Plotter,
    mesh: pv.PolyData,
    line_width: float = 2.0,
    **kwargs: Any,
) -> None:
    """Add a PyVista mesh with crisp dark silhouette outline, falling back gracefully."""
    silhouette_cfg = {"color": _scene_colors()["silhouette"], "line_width": line_width}
    try:
        plotter.add_mesh(mesh, silhouette=silhouette_cfg, **kwargs)
    except Exception:
        plotter.add_mesh(mesh, **kwargs)


def _render_sublattice_diagram(
    output_path: Path,
    a: float,
    sublattice_summary: dict[str, Any],
    window_size: tuple[int, int] | None = None,
    transparent_background: bool = False,
    ordering: str = "L21",
) -> Path:
    """Render the 1x1x1 Heusler unit cell colored distinctly by Wyckoff sublattice (8c / 4a / 4b).

    For XA the sublattices are 4a / 4d / 4c / 4b (the X sites 4a, 4d use the 8c size).

    Sphere sizes: 0.44 Å (8c) / 0.50 Å (4a, 4b) with default settings, scaled by each
    sublattice's composition-weighted display radius relative to the defaults
    (render.ball_radius_*, atomic_radii). Cell / strut tubes follow
    render.unit_cell_line_radius / render.bond_radius the same way.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    px_scale = _px_scale()
    if window_size is None:
        side = _k(_WINDOW_PX, px_scale)
        window_size = (side, side)
    lattice = Lattice.cubic(a)

    # Collect unit-cell sites with periodic boundary mirroring for complete corners/faces
    tol = 1e-3
    sub_keys = heusler_sublattice_keys(ordering)
    is_xa = "8c" not in sub_keys
    atoms_by_sub: dict[str, list[np.ndarray]] = {sub: [] for sub in sub_keys}
    seen: set[tuple[int, int, int]] = set()
    all_points: list[tuple[np.ndarray, float]] = []

    if is_xa:
        base_radius = {"4a": 0.44, "4d": 0.44, "4c": 0.50, "4b": 0.50}
    else:
        base_radius = {"8c": 0.44, "4a": 0.50, "4b": 0.50}
    params = _radius_params()
    radius_map: dict[str, float] = {}
    for sub, base in base_radius.items():
        fracs = (sublattice_summary.get(sub, {}) or {}).get("fractions", {}) or {}
        occs = [(str(el), float(f)) for el, f in fracs.items() if float(f) > 0.0]
        radius_map[sub] = relative_ball_radius(base, occs, params) if occs else base
    basis_map = _sublattice_basis(ordering)

    for sub, coords in basis_map:
        rad = radius_map[sub]
        for fx, fy, fz in coords:
            frac = np.array([fx, fy, fz], dtype=float)
            shifts = [([0.0, 1.0] if c <= tol else [0.0]) for c in frac]
            for sh in itertools.product(*shifts):
                img_frac = frac + np.asarray(sh, dtype=float)
                cart = np.asarray(lattice.get_cartesian_coords(img_frac), dtype=float)
                key = tuple(int(round(v * 1000.0)) for v in cart)
                if key not in seen:
                    seen.add(key)
                    atoms_by_sub[sub].append(cart)
                    all_points.append((cart, rad))

    cam_dir = np.array([1.0, 0.42, 0.28], dtype=float)
    cam_unit = cam_dir / np.linalg.norm(cam_dir)
    up_unit = _orthonormal_up(cam_unit, np.array([0.0, 0.0, 1.0], dtype=float))
    cell_center, parallel_scale = _compute_parallel_framing(
        lattice=lattice,
        atom_points=all_points,
        cam_unit=cam_unit,
        up_unit=up_unit,
        window_size=window_size,
    )

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        colors = _scene_colors()
        plotter.set_background(colors["background"])
        plotter.enable_parallel_projection()
        apply_studio_lighting(plotter, parallel_scale=parallel_scale)

        _add_unit_cell_wireframe(
            plotter, lattice, tube_radius=0.032 * line_radius_ratio("cell"), color=colors["unit_cell"]
        )

        # Subtle thin internal 8c-4a / 8c-4b tetrahedral struts for visual clarity
        strut_starts: list[np.ndarray] = []
        strut_ends: list[np.ndarray] = []
        bond_cutoff = 0.46 * a  # sqrt(3)/4 * a ~= 0.433 * a
        if is_xa:  # same tetrahedral network: (4c, 4d) <-> (4a, 4b)
            pts_8c = atoms_by_sub["4c"] + atoms_by_sub["4d"]
        else:
            pts_8c = atoms_by_sub["8c"]
        pts_4ab = atoms_by_sub["4a"] + atoms_by_sub["4b"]
        for p8 in pts_8c:
            for p4 in pts_4ab:
                if float(np.linalg.norm(p8 - p4)) <= bond_cutoff:
                    strut_starts.append(p8)
                    strut_ends.append(p4)
        if strut_starts:
            strut_r = 0.028 * line_radius_ratio("bond")
            plotter.add_mesh(
                pv.merge(
                    [
                        pv.Line(p0, p1).tube(radius=strut_r, n_sides=10)
                        for p0, p1 in zip(strut_starts, strut_ends)
                    ]
                ),
                color="#D0D4D9",
                smooth_shading=True,
                opacity=0.65,
            )

        # Render each Wyckoff sublattice with its dedicated color and crisp silhouette
        res = _res_scale(28, px_scale)
        for sub in sub_keys:
            pts = atoms_by_sub[sub]
            if pts:
                merged = _merge_spheres(pts, [radius_map[sub]] * len(pts), res)
                _add_mesh_with_silhouette(
                    plotter,
                    merged,
                    line_width=2.0 * px_scale,
                    color=SUBLATTICE_COLORS[sub],
                    smooth_shading=True,
                    specular=0.35,
                    specular_power=25,
                )

        add_orientation_axes(
            plotter, xlabel="[100]", ylabel="[010]", zlabel="[001]", line_width=2, color="black"
        )

        diag_len = float(np.linalg.norm(lattice.get_cartesian_coords([1.0, 1.0, 1.0])))
        cam_dist = max(diag_len * 3.0, parallel_scale * 4.0, 20.0)
        plotter.camera_position = [
            (cell_center + cam_unit * cam_dist).tolist(),
            cell_center.tolist(),
            up_unit.tolist(),
        ]
        plotter.camera.parallel_scale = parallel_scale
        plotter.renderer.ResetCameraClippingRange()

        screenshot_checked(
            plotter, output_path, transparent_background=transparent_background,
            context=f"heusler {Path(output_path).name}",
        )
    finally:
        plotter.close()

    return output_path.resolve()


def _render_pair_highlight(
    output_path: Path,
    structure: Structure,
    wyckoff_labels: list[str],
    highlight_pair: tuple[str, str],
    render_segments: list[tuple[np.ndarray, np.ndarray, float, str]],
    a: float,
    window_size: tuple[int, int] | None = None,
    transparent_background: bool = False,
    ordering: str = "L21",
) -> Path:
    """Render pattern 0 supercell highlighting nearest-neighbor target pairs in vivid red.

    Target / other atoms are 0.44 / 0.28 Å with default settings, scaled per element by
    the display radius relative to the defaults (render.ball_radius_*, atomic_radii).
    The red paths and the cell follow render.bond_radius / render.unit_cell_line_radius.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    px_scale = _px_scale()
    if window_size is None:
        side = _k(_WINDOW_PX, px_scale)
        window_size = (side, side)
    lattice = structure.lattice
    target_elems = {str(highlight_pair[0]), str(highlight_pair[1])}
    el1, el2 = str(highlight_pair[0]), str(highlight_pair[1])

    # (centers, radii) per sublattice for target atoms, and for the muted others
    sub_keys = heusler_sublattice_keys(ordering)
    target_by_sub: dict[str, tuple[list[np.ndarray], list[float]]] = {
        sub: ([], []) for sub in sub_keys
    }
    muted_centers: list[np.ndarray] = []
    muted_radii: list[float] = []
    all_points: list[tuple[np.ndarray, float]] = []

    params = _radius_params()
    radius_cache: dict[tuple[str, bool], float] = {}

    def _radius(sym: str, is_target: bool) -> float:
        key = (sym, is_target)
        if key not in radius_cache:
            radius_cache[key] = relative_ball_radius(0.44 if is_target else 0.28, sym, params)
        return radius_cache[key]

    for idx, site in enumerate(structure):
        sym = dominant_element(site)
        sub = wyckoff_labels[idx]
        cart = np.asarray(site.coords, dtype=float)
        if sym in target_elems:
            rad = _radius(sym, True)
            target_by_sub[sub][0].append(cart)
            target_by_sub[sub][1].append(rad)
        else:
            rad = _radius(sym, False)
            muted_centers.append(cart)
            muted_radii.append(rad)
        all_points.append((cart, rad))

    # Select nearest-neighbor shell to highlight prominently:
    # Prefer 1NN (<= 0.58 * a, i.e., a/2) if present; otherwise highlight shortest shell found.
    primary_segments: list[tuple[np.ndarray, np.ndarray]] = []
    if render_segments:
        min_dist = min(seg[2] for seg in render_segments)
        nn1_thresh = 0.58 * a
        if min_dist <= nn1_thresh:
            primary_segments = [
                (p1, p2) for p1, p2, d, _ in render_segments if d <= nn1_thresh
            ]
        else:
            shell_thresh = min_dist * 1.05
            primary_segments = [
                (p1, p2) for p1, p2, d, _ in render_segments if d <= shell_thresh
            ]

    cam_dir = np.array([1.0, 0.42, 0.28], dtype=float)
    cam_unit = cam_dir / np.linalg.norm(cam_dir)
    up_unit = _orthonormal_up(cam_unit, np.array([0.0, 0.0, 1.0], dtype=float))
    cell_center, parallel_scale = _compute_parallel_framing(
        lattice=lattice,
        atom_points=all_points,
        cam_unit=cam_unit,
        up_unit=up_unit,
        window_size=window_size,
    )

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        colors = _scene_colors()
        plotter.set_background(colors["background"])
        plotter.enable_parallel_projection()
        apply_studio_lighting(plotter, parallel_scale=parallel_scale)

        _add_unit_cell_wireframe(
            plotter, lattice, tube_radius=0.045 * line_radius_ratio("cell"), color=colors["unit_cell"]
        )

        # 1. Muted semi-transparent non-target atoms
        if muted_centers:
            merged_muted = _merge_spheres(muted_centers, muted_radii, _res_scale(16, px_scale))
            plotter.add_mesh(
                merged_muted,
                color=MUTED_ATOM_COLOR,
                opacity=0.22,
                smooth_shading=True,
            )

        # 2. Vivid red thick tubes for nearest-neighbor magnetic exchange paths
        if primary_segments:
            bond_r = 0.13 * line_radius_ratio("bond")
            bond_tubes = [
                pv.Line(p1, p2).tube(radius=bond_r, n_sides=16)
                for p1, p2 in primary_segments
            ]
            merged_bonds = pv.merge(bond_tubes) if len(bond_tubes) > 1 else bond_tubes[0]
            plotter.add_mesh(
                merged_bonds,
                color=HIGHLIGHT_BOND_COLOR,
                smooth_shading=True,
                specular=0.30,
                specular_power=20,
            )

        # 3. Opaque target atoms colored by sublattice with crisp silhouettes
        res_target = _res_scale(22, px_scale)
        draw_order = ("4a", "4b", "8c") if "8c" in sub_keys else sub_keys
        for sub in draw_order:
            t_centers, t_radii = target_by_sub[sub]
            if not t_centers:
                continue
            merged_t = _merge_spheres(t_centers, t_radii, res_target)
            _add_mesh_with_silhouette(
                plotter,
                merged_t,
                line_width=1.8 * px_scale,
                color=SUBLATTICE_COLORS[sub],
                smooth_shading=True,
                specular=0.35,
                specular_power=25,
            )

        add_orientation_axes(
            plotter, xlabel="[100]", ylabel="[010]", zlabel="[001]", line_width=2, color="black"
        )

        diag_len = float(np.linalg.norm(lattice.get_cartesian_coords([1.0, 1.0, 1.0])))
        cam_dist = max(diag_len * 3.0, parallel_scale * 4.0, 20.0)
        plotter.camera_position = [
            (cell_center + cam_unit * cam_dist).tolist(),
            cell_center.tolist(),
            up_unit.tolist(),
        ]
        plotter.camera.parallel_scale = parallel_scale
        plotter.renderer.ResetCameraClippingRange()

        screenshot_checked(
            plotter, output_path, transparent_background=transparent_background,
            context=f"heusler {Path(output_path).name}",
        )
    finally:
        plotter.close()

    return output_path.resolve()


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load Arial / Helvetica-compatible TrueType font, falling back to Pillow default."""
    candidates = (
        [
            "arialbd.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        if bold
        else [
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    )
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    return (80, 80, 80)


def _draw_legend_strip_pil(
    draw: ImageDraw.ImageDraw,
    entries: list[tuple[str, str, str]],
    x0: int,
    y0: int,
    width: int,
    height: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    px_scale: float = 1.0,
) -> None:
    """Draw a clean, single- or two-row legend strip inside [x0, x0+width] x [y0, y0+height].

    Each entry is (label, hex_color, kind) where kind is 'circle' or 'line'.
    Swatch sizes are 300-dpi pixels times ``px_scale``.
    """
    if not entries:
        return

    # Split into rows if total width exceeds strip width
    swatch_w = _k(24, px_scale)
    swatch_gap = _k(10, px_scale)
    item_gap = _k(30, px_scale)
    pad_x = _k(18, px_scale)

    def _item_w(label: str) -> float:
        bb = draw.textbbox((0, 0), label, font=font)
        return swatch_w + swatch_gap + (bb[2] - bb[0])

    rows: list[list[tuple[str, str, str]]] = []
    cur_row: list[tuple[str, str, str]] = []
    cur_w = 0.0
    max_row_w = width - 2 * pad_x

    for item in entries:
        iw = _item_w(item[0])
        needed = iw if not cur_row else (cur_w + item_gap + iw)
        if cur_row and needed > max_row_w:
            rows.append(cur_row)
            cur_row = [item]
            cur_w = iw
        else:
            cur_row.append(item)
            cur_w = needed
    if cur_row:
        rows.append(cur_row)

    row_h = height / max(len(rows), 1)
    for r_idx, row in enumerate(rows):
        row_total_w = sum(_item_w(it[0]) for it in row) + item_gap * max(len(row) - 1, 0)
        cursor_x = x0 + max(pad_x, (width - row_total_w) / 2.0)
        center_y = y0 + row_h * (r_idx + 0.5)

        for label, hex_col, kind in row:
            rgb = _hex_to_rgb(hex_col)
            if kind == "line":
                draw.line(
                    [(cursor_x, center_y), (cursor_x + swatch_w, center_y)],
                    fill=rgb,
                    width=_k(6, px_scale),
                )
            else:
                r = _k(9, px_scale)
                cx = cursor_x + swatch_w / 2.0
                draw.ellipse(
                    [(cx - r, center_y - r), (cx + r, center_y + r)],
                    fill=rgb,
                    outline=(35, 35, 35),
                    width=_k(2, px_scale),
                )
            tx = cursor_x + swatch_w + swatch_gap
            draw.text((tx, center_y), label, fill=(25, 25, 25), font=font, anchor="lm")
            cursor_x += _item_w(label) + item_gap


def _build_legend_strip_svg(
    entries: list[tuple[str, str, str]],
    x0: int,
    y0: int,
    width: int,
    height: int,
    font_size: int = 22,
) -> str:
    """Build crisp editable vector SVG markup for a panel legend strip."""
    if not entries:
        return ""

    swatch_w = 24
    swatch_gap = 10
    item_gap = 28
    pad_x = 18

    def _approx_w(label: str) -> float:
        return swatch_w + swatch_gap + len(label) * (font_size * 0.56)

    rows: list[list[tuple[str, str, str]]] = []
    cur_row: list[tuple[str, str, str]] = []
    cur_w = 0.0
    max_row_w = width - 2 * pad_x
    for item in entries:
        iw = _approx_w(item[0])
        needed = iw if not cur_row else (cur_w + item_gap + iw)
        if cur_row and needed > max_row_w:
            rows.append(cur_row)
            cur_row = [item]
            cur_w = iw
        else:
            cur_row.append(item)
            cur_w = needed
    if cur_row:
        rows.append(cur_row)

    parts: list[str] = []
    row_h = height / max(len(rows), 1)
    for r_idx, row in enumerate(rows):
        row_total_w = sum(_approx_w(it[0]) for it in row) + item_gap * max(len(row) - 1, 0)
        cursor_x = x0 + max(pad_x, (width - row_total_w) / 2.0)
        cy = y0 + row_h * (r_idx + 0.5)
        for label, hex_col, kind in row:
            if kind == "line":
                parts.append(
                    f'    <line x1="{cursor_x:.1f}" y1="{cy:.1f}" '
                    f'x2="{cursor_x + swatch_w:.1f}" y2="{cy:.1f}" '
                    f'stroke="{hex_col}" stroke-width="6" stroke-linecap="round"/>'
                )
            else:
                cx = cursor_x + swatch_w / 2.0
                parts.append(
                    f'    <circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" '
                    f'fill="{hex_col}" stroke="#232323" stroke-width="1.8"/>'
                )
            tx = cursor_x + swatch_w + swatch_gap
            parts.append(
                f'    <text x="{tx:.1f}" y="{cy:.1f}" dominant-baseline="central" '
                f'class="legend-label">{xml_escape(label)}</text>'
            )
            cursor_x += _approx_w(label) + item_gap
    return "\n".join(parts)


def _compose_composite_figure(
    left_png: Path,
    right_png: Path,
    out_png: Path,
    out_svg: Path,
    left_title: str,
    right_title: str,
    caption_text: str,
    left_legend: list[tuple[str, str, str]] | None = None,
    right_legend: list[tuple[str, str, str]] | None = None,
    transparent_background: bool = False,
) -> tuple[Path, Path]:
    """Compose a side-by-side render.dpi PNG and an editable Arial/Helvetica SVG figure.

    PNG sizes are 300-dpi pixels times render.dpi / 300 (panels are rendered that large
    too). The SVG keeps its layout in 300-dpi units and embeds the full-resolution panels.
    """
    ps = _px_scale()
    mode = "RGBA" if transparent_background else "RGB"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(left_png) as img_l_raw:
        img_l = img_l_raw.convert(mode)
    with Image.open(right_png) as img_r_raw:
        img_r = img_r_raw.convert(mode)

    panel_w = max(img_l.width, img_r.width)
    panel_h = max(img_l.height, img_r.height)
    if img_l.size != (panel_w, panel_h):
        img_l = img_l.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    if img_r.size != (panel_w, panel_h):
        img_r = img_r.resize((panel_w, panel_h), Image.Resampling.LANCZOS)

    margin_x = _k(48, ps)
    gap_x = _k(40, ps)
    header_h = _k(84, ps)
    legend_h = _k(92, ps) if (left_legend or right_legend) else 0
    footer_h = _k(86, ps)

    total_w = 2 * margin_x + 2 * panel_w + gap_x
    total_h = header_h + panel_h + legend_h + footer_h

    left_x = margin_x
    right_x = margin_x + panel_w + gap_x
    panel_y = header_h
    legend_y = panel_y + panel_h

    # --- 1. render.dpi PNG composite ---
    if transparent_background:
        canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 0))
        canvas.paste(img_l, (left_x, panel_y), img_l)
        canvas.paste(img_r, (right_x, panel_y), img_r)
    else:
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        canvas.paste(img_l, (left_x, panel_y))
        canvas.paste(img_r, (right_x, panel_y))

    draw = ImageDraw.Draw(canvas)
    border_color = (210, 210, 210)
    draw.rectangle(
        [(left_x, panel_y), (left_x + panel_w - 1, panel_y + panel_h - 1)],
        outline=border_color,
        width=_k(2, ps),
    )
    draw.rectangle(
        [(right_x, panel_y), (right_x + panel_w - 1, panel_y + panel_h - 1)],
        outline=border_color,
        width=_k(2, ps),
    )

    title_font = _load_font(_k(34, ps), bold=True)
    for x0, title in ((left_x, left_title), (right_x, right_title)):
        draw.text(
            (x0 + panel_w / 2.0, header_h / 2.0),
            title,
            fill=(20, 20, 20),
            font=title_font,
            anchor="mm",
        )

    if legend_h > 0:
        leg_font = _load_font(_k(22, ps), bold=False)
        if left_legend:
            _draw_legend_strip_pil(
                draw, left_legend, left_x, legend_y, panel_w, legend_h, leg_font, px_scale=ps
            )
        if right_legend:
            _draw_legend_strip_pil(
                draw, right_legend, right_x, legend_y, panel_w, legend_h, leg_font, px_scale=ps
            )

    footer_y = legend_y + legend_h
    rule = _k(8, ps)
    draw.line(
        [(margin_x, footer_y + rule), (total_w - margin_x, footer_y + rule)],
        fill=(200, 200, 200),
        width=_k(2, ps),
    )

    # Caption font: 26 px at 300 dpi, shrunk in 2 px steps (down to 14) until it fits
    cap_size = 26
    cap_font = _load_font(_k(cap_size, ps), bold=False)
    max_cap_w = total_w - 2 * margin_x
    while cap_size > 14:
        cb = draw.textbbox((0, 0), caption_text, font=cap_font)
        if (cb[2] - cb[0]) <= max_cap_w:
            break
        cap_size -= 2
        cap_font = _load_font(_k(cap_size, ps), bold=False)

    draw.text(
        (total_w / 2.0, footer_y + rule + (footer_h - rule) / 2.0),
        caption_text,
        fill=(30, 30, 30),
        font=cap_font,
        anchor="mm",
    )

    canvas.save(out_png, format="PNG", dpi=_dpi_pair())

    # --- 2. Editable vector SVG with embedded high-res panels ---
    # Layout in 300-dpi units so the SVG has the same size at any dpi.
    b64_left = base64.b64encode(left_png.read_bytes()).decode("ascii")
    b64_right = base64.b64encode(right_png.read_bytes()).decode("ascii")

    panel_w = int(round(panel_w / ps))
    panel_h = int(round(panel_h / ps))
    margin_x, gap_x, header_h = 48, 40, 84
    legend_h = 92 if (left_legend or right_legend) else 0
    footer_h = 86
    total_w = 2 * margin_x + 2 * panel_w + gap_x
    total_h = header_h + panel_h + legend_h + footer_h
    left_x = margin_x
    right_x = margin_x + panel_w + gap_x
    panel_y = header_h
    legend_y = panel_y + panel_h
    footer_y = legend_y + legend_h

    left_cx = left_x + panel_w / 2.0
    right_cx = right_x + panel_w / 2.0
    title_y = header_h / 2.0
    cap_y = footer_y + 8 + (footer_h - 8) / 2.0

    left_leg_svg = (
        _build_legend_strip_svg(left_legend, left_x, legend_y, panel_w, legend_h)
        if (legend_h > 0 and left_legend)
        else ""
    )
    right_leg_svg = (
        _build_legend_strip_svg(right_legend, right_x, legend_y, panel_w, legend_h)
        if (legend_h > 0 and right_legend)
        else ""
    )

    bg_rect = (
        ""
        if transparent_background
        else f'  <rect x="0" y="0" width="{total_w}" height="{total_h}" fill="#FFFFFF"/>\n'
    )
    svg_content = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{total_w}" height="{total_h}"
     viewBox="0 0 {total_w} {total_h}">
  <style>
    .panel-title {{
      font-family: Arial, Helvetica, sans-serif;
      font-size: 34px;
      font-weight: 700;
      fill: #141414;
    }}
    .legend-label {{
      font-family: Arial, Helvetica, sans-serif;
      font-size: 22px;
      font-weight: 400;
      fill: #191919;
    }}
    .figure-caption {{
      font-family: Arial, Helvetica, sans-serif;
      font-size: {cap_size}px;
      font-weight: 400;
      fill: #1E1E1E;
    }}
  </style>
{bg_rect}
  <g id="panel-a">
    <text x="{left_cx:.1f}" y="{title_y:.1f}" text-anchor="middle" dominant-baseline="central" class="panel-title">{xml_escape(left_title)}</text>
    <image x="{left_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}"
           preserveAspectRatio="xMidYMid meet"
           href="data:image/png;base64,{b64_left}"/>
    <rect x="{left_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}"
          fill="none" stroke="#D2D2D2" stroke-width="2"/>
{left_leg_svg}
  </g>

  <g id="panel-b">
    <text x="{right_cx:.1f}" y="{title_y:.1f}" text-anchor="middle" dominant-baseline="central" class="panel-title">{xml_escape(right_title)}</text>
    <image x="{right_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}"
           preserveAspectRatio="xMidYMid meet"
           href="data:image/png;base64,{b64_right}"/>
    <rect x="{right_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}"
          fill="none" stroke="#D2D2D2" stroke-width="2"/>
{right_leg_svg}
  </g>

  <g id="caption">
    <line x1="{margin_x}" y1="{footer_y + 8}" x2="{total_w - margin_x}" y2="{footer_y + 8}"
          stroke="#C8C8C8" stroke-width="2"/>
    <text x="{total_w / 2.0:.1f}" y="{cap_y:.1f}" text-anchor="middle" dominant-baseline="central" class="figure-caption">{xml_escape(caption_text)}</text>
  </g>
</svg>
"""
    out_svg.write_text(svg_content, encoding="utf-8")

    return out_png.resolve(), out_svg.resolve()


def generate_heusler_models(
    formula: str | None = None,
    ordering: str | None = None,
    supercell: tuple[int, int, int] | None = None,
    num_patterns: int | None = None,
    a: float | None = None,
    output_dir: str | Path = "output_heusler",
    highlight_pair: tuple[str, str] | None = None,
    seed: int | None = None,
    pair_cutoff_factor: float | None = None,
    transparent_background: bool | None = None,
) -> dict[str, Any]:
    """Generate Heusler L21/B2 (and ordered inverse-Heusler XA) supercell models, pair-distance tables, and figures.

    Any argument left as ``None`` is read from the settings at call time
    (``heusler.*`` and ``lattice.heusler_a``), so GUI changes apply without a restart.
    Site roles follow the X2YZ convention in written formula order (X -> 8c, Y -> 4a, Z -> 4b;
    for XA, F-43m: X -> 4a + 4d, Y -> 4c, Z -> 4b, ordered only).
    ``transparent_background=None`` uses ``render.transparent_background``; the scene
    background / silhouette / cell colors follow the active theme (``render.theme``).
    """
    transparent_background = _resolve_transparent(transparent_background)
    from shido.settings import get

    if formula is None:
        formula = str(get("heusler.formula", "Ni50Mn35In15"))
    if ordering is None:
        ordering = str(get("heusler.ordering", "L21"))
    if supercell is None:
        supercell = tuple(int(v) for v in get("heusler.supercell", [2, 2, 2]))  # type: ignore[assignment]
    if num_patterns is None:
        num_patterns = int(get("heusler.num_patterns", 3))
    if a is None:
        a = float(get("lattice.heusler_a", 6.00))
    if highlight_pair is None:
        hp = get("heusler.highlight_pair", ["Mn", "Mn"])
        highlight_pair = (str(hp[0]), str(hp[1]))
    if seed is None:
        seed = int(get("heusler.seed", 42))
    if pair_cutoff_factor is None:
        pair_cutoff_factor = float(get("heusler.pair_cutoff_factor", 0.75))
    supercell = (int(supercell[0]), int(supercell[1]), int(supercell[2]))

    ordering_norm = ordering.strip().upper()
    if ordering_norm not in set(SUPPORTED_ORDERINGS):
        raise ValueError(f"Unsupported ordering {ordering!r}. Expected 'L21', 'B2' or 'XA'.")
    sub_keys = heusler_sublattice_keys(ordering_norm)
    if num_patterns < 1:
        raise ValueError(f"num_patterns must be >= 1, got {num_patterns!r}")
    if pair_cutoff_factor <= 0.0:
        raise ValueError(f"pair_cutoff_factor must be positive, got {pair_cutoff_factor!r}")

    comp_fractions = parse_alloy_composition(formula)
    element_order = list(comp_fractions.keys())

    out_dir = Path(output_dir)
    structures_dir = out_dir / STRUCT_DIR
    data_dir = out_dir / DATA_DIR
    images_dir = out_dir / FIG_DIR
    for d in (structures_dir, data_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)

    lattice, frac_coords, wyckoff_labels = _build_supercell_template(
        supercell=supercell, a=a, ordering=ordering_norm
    )
    n_total = len(wyckoff_labels)
    target_counts = _allocate_integer_counts(comp_fractions, n_total)
    actual_fractions = {el: round(cnt / n_total, 6) for el, cnt in target_counts.items()}

    tag = re.sub(r"[^A-Za-z0-9_.-]+", "", formula) or "alloy"

    cif_paths: list[Path] = []
    poscar_paths: list[Path] = []
    pattern_structures: list[Structure] = []

    for pat_idx in range(num_patterns):
        rng = np.random.default_rng(int(seed) + pat_idx)
        species = _assign_species_for_pattern(
            wyckoff_labels=wyckoff_labels,
            target_counts=target_counts,
            element_order=element_order,
            ordering=ordering_norm,
            rng=rng,
        )
        struct = Structure(
            lattice=lattice,
            species=species,
            coords=frac_coords,
            coords_are_cartesian=False,
            site_properties={"wyckoff": list(wyckoff_labels)},
        )
        pattern_structures.append(struct)

        cif_path = structures_dir / f"heusler_{tag}_{ordering_norm}_pat{pat_idx}.cif"
        poscar_path = structures_dir / f"POSCAR_heusler_{tag}_{ordering_norm}_pat{pat_idx}"

        CifWriter(struct).write_file(str(cif_path))
        Poscar(struct.get_sorted_structure()).write_file(str(poscar_path))

        cif_paths.append(cif_path.resolve())
        poscar_paths.append(poscar_path.resolve())

    # Analyze representative pattern 0
    rep_struct = pattern_structures[0]
    sublattice_df, sublattice_summary = _compute_sublattice_summary(
        structure=rep_struct,
        wyckoff_labels=wyckoff_labels,
        element_order=element_order,
        sublattices=sub_keys,
    )
    sublattice_csv = (data_dir / "sublattice_summary.csv").resolve()
    sublattice_df.to_csv(sublattice_csv, index=False)

    cutoff = float(pair_cutoff_factor) * float(a)
    pair_df, render_segments = _compute_pair_distances(
        structure=rep_struct,
        wyckoff_labels=wyckoff_labels,
        highlight_pair=highlight_pair,
        cutoff=cutoff,
    )
    # Label coordination shells (1 = nearest) so the CSV separates a/2 and a/sqrt(2) pairs
    if len(pair_df) > 0:
        shell_dists = sorted({round(float(d), 3) for d in pair_df["distance_angstrom"]})
        merged: list[float] = []
        for d in shell_dists:
            if not merged or d > merged[-1] * 1.02:
                merged.append(d)
        pair_df["shell"] = [
            1 + max(k for k, ref in enumerate(merged) if float(d) >= ref * 0.98)
            for d in pair_df["distance_angstrom"]
        ]
    pair_csv = (data_dir / "pair_distances.csv").resolve()
    pair_df.to_csv(pair_csv, index=False)

    # Summarize pair distance statistics
    el1, el2 = str(highlight_pair[0]), str(highlight_pair[1])
    total_pairs = int(len(pair_df))
    if total_pairs > 0:
        min_d = float(pair_df["distance_angstrom"].min())
        mean_d = float(pair_df["distance_angstrom"].mean())
        max_d = float(pair_df["distance_angstrom"].max())
        by_pair_type: dict[str, dict[str, Any]] = {}
        for ptype, grp in pair_df.groupby("pair_type", sort=True):
            by_pair_type[str(ptype)] = {
                "count": int(len(grp)),
                "min_distance_angstrom": round(float(grp["distance_angstrom"].min()), 6),
                "mean_distance_angstrom": round(float(grp["distance_angstrom"].mean()), 6),
                "max_distance_angstrom": round(float(grp["distance_angstrom"].max()), 6),
            }
    else:
        min_d = None
        mean_d = None
        max_d = None
        by_pair_type = {}

    # Render individual PyVista panels
    sublattice_png = _render_sublattice_diagram(
        output_path=images_dir / "sublattice_diagram.png",
        a=float(a),
        sublattice_summary=sublattice_summary,
        transparent_background=transparent_background,
        ordering=ordering_norm,
    )
    pair_png = _render_pair_highlight(
        output_path=images_dir / "pair_highlight.png",
        structure=rep_struct,
        wyckoff_labels=wyckoff_labels,
        highlight_pair=(el1, el2),
        render_segments=render_segments,
        a=float(a),
        transparent_background=transparent_background,
        ordering=ordering_norm,
    )

    # Compose side-by-side 300 dpi PNG + editable vector SVG
    sc_str = f"{supercell[0]}x{supercell[1]}x{supercell[2]}"
    if total_pairs > 0 and min_d is not None and mean_d is not None:
        nn_df = pair_df[pair_df["shell"] == 1]
        dist_caption = (
            f"NN {el1}–{el2}: N={len(nn_df)}, d={min_d:.3f} Å "
            f"(all pairs r ≤ {cutoff:.2f} Å: N={total_pairs}, see CSV)"
        )
    else:
        dist_caption = f"{el1}–{el2} pairs (r ≤ {cutoff:.2f} Å): N=0"

    caption_text = (
        f"{formula} ({ordering_norm}, {sc_str} supercell, {n_total} atoms, a={float(a):.3f} Å) | "
        f"{dist_caption}"
    )

    left_legend: list[tuple[str, str, str]] = []
    for sub in sub_keys:
        comp_str = sublattice_summary.get(sub, {}).get("composition", "")
        lbl = f"{sub} ({comp_str})" if comp_str else sub
        left_legend.append((lbl, SUBLATTICE_COLORS[sub], "circle"))

    target_elems = {el1, el2}
    target_prefix = el1 if el1 == el2 else f"{el1}/{el2}"
    right_legend: list[tuple[str, str, str]] = []
    for sub in (("4a", "4b", "8c") if "8c" in sub_keys else sub_keys):
        has_target = any(
            wyckoff_labels[i] == sub and dominant_element(rep_struct[i]) in target_elems
            for i in range(len(rep_struct))
        )
        if has_target:
            right_legend.append((f"{target_prefix} ({sub})", SUBLATTICE_COLORS[sub], "circle"))
    right_legend.append((f"NN {el1}-{el2} bond", HIGHLIGHT_BOND_COLOR, "line"))
    right_legend.append(("Other atoms", MUTED_ATOM_COLOR, "circle"))

    fig_png, fig_svg = _compose_composite_figure(
        left_png=sublattice_png,
        right_png=pair_png,
        out_png=images_dir / "figure_heusler_model.png",
        out_svg=images_dir / "figure_heusler_model.svg",
        left_title=f"(a) Wyckoff Sublattice ({' / '.join(sub_keys)})",
        right_title=f"(b) Nearest-Neighbor {el1}-{el2} Paths",
        caption_text=caption_text,
        left_legend=left_legend,
        right_legend=right_legend,
        transparent_background=transparent_background,
    )

    x_els, y_els, z_els = _heusler_roles(
        element_order, {el: target_counts[el] / max(n_total, 1) for el in element_order}
    )
    if ordering_norm == "XA":
        site_roles = {"X_4a_4d": x_els, "Y_4c": y_els, "Z_4b": z_els}
    else:
        site_roles = {"X_8c": x_els, "Y_4a": y_els, "Z_4b": z_els}
    return {
        "formula": formula,
        "site_roles": site_roles,
        "normalized_composition": {k: round(float(v), 6) for k, v in comp_fractions.items()},
        "actual_composition": actual_fractions,
        "target_site_counts": {k: int(v) for k, v in target_counts.items()},
        "ordering": ordering_norm,
        "supercell": [int(supercell[0]), int(supercell[1]), int(supercell[2])],
        "num_atoms": n_total,
        "lattice_parameter_a": float(a),
        "num_patterns": int(num_patterns),
        "highlight_pair": [el1, el2],
        "cutoff_angstrom": round(cutoff, 6),
        "pair_cutoff_factor": float(pair_cutoff_factor),
        "output_dir": str(out_dir.resolve()),
        "structure_files": {
            "cif": [str(p) for p in cif_paths],
            "poscar": [str(p) for p in poscar_paths],
        },
        "data_files": {
            "sublattice_summary_csv": str(sublattice_csv),
            "pair_distances_csv": str(pair_csv),
        },
        "image_files": {
            "sublattice_diagram": str(sublattice_png),
            "pair_highlight": str(pair_png),
            "figure_png": str(fig_png),
            "figure_svg": str(fig_svg),
        },
        "sublattice_summary": sublattice_summary,
        "pair_distance_stats": {
            "pair": f"{el1}-{el2}",
            "cutoff_angstrom": round(cutoff, 6),
            "total_pairs": total_pairs,
            "min_distance_angstrom": round(min_d, 6) if min_d is not None else None,
            "mean_distance_angstrom": round(mean_d, 6) if mean_d is not None else None,
            "max_distance_angstrom": round(max_d, 6) if max_d is not None else None,
            "by_pair_type": by_pair_type,
        },
    }

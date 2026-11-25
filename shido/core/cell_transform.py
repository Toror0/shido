"""Cell transformations (VESTA-like): Niggli reduction, standardisation
(conventional / primitive standard cell), a general basis change with a matrix P
and an origin shift p, and supercells.

Convention (ITA, same as VESTA "Transformation matrix P"):
    (a', b', c') = (a, b, c) P      -- the columns of P are the new axes in the old basis
    x' = P^-1 (x - p)               -- p is the new origin in old fractional coordinates

P may contain fractions (e.g. F -> P: [[0,1/2,1/2],[1/2,0,1/2],[1/2,1/2,0]]).
Every new axis must be a lattice translation of the structure; otherwise a
``ValueError`` is raised. The result is checked: composition conserved and
site count == old count * det(P). det(P) < 0 changes the handedness (warning).

Order of operations: ``cell`` (niggli / conventional / primitive) -> ``P`` +
``origin_shift`` -> ``supercell``.
"""

from __future__ import annotations

import itertools
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from shido.core.validator import ScientificValidationError, validate_composition_conserved

CELL_CHOICES = ("input", "niggli", "conventional", "standard", "primitive")
_TOL = 1e-4  # fractional-coordinate tolerance


def _num(v: Any) -> float:
    if isinstance(v, str):
        return float(Fraction(v.strip()))
    return float(v)


def parse_matrix(value: Any) -> np.ndarray:
    """3x3 matrix from 9 numbers (row-major P11 P12 ... P33), 3 numbers (diagonal),
    a nested list, or a string like "0 1/2 1/2  1/2 0 1/2  1/2 1/2 0"."""
    if isinstance(value, str):
        items: list[Any] = value.replace(",", " ").split()
    else:
        arr = np.asarray(value, dtype=object)
        items = list(arr.ravel())
    vals = [_num(v) for v in items]
    if len(vals) == 3:
        return np.diag(vals)
    if len(vals) == 9:
        return np.asarray(vals, dtype=float).reshape(3, 3)
    raise ValueError(f"transform matrix needs 3 or 9 numbers, got {len(vals)}.")


def parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        items: list[Any] = value.replace(",", " ").split()
    else:
        items = list(value)
    vals = [_num(v) for v in items]
    if len(vals) != 3:
        raise ValueError(f"origin shift needs 3 numbers, got {len(vals)}.")
    return np.asarray(vals, dtype=float)


def _species_key(site) -> str:
    return site.species_string


def _is_lattice_translation(structure: Structure, t: np.ndarray, tol: float = 1e-3) -> bool:
    """True if translating every site by fractional vector t lands on a site of the same species."""
    frac = np.asarray(structure.frac_coords, dtype=float)
    keys = [_species_key(s) for s in structure]
    for i in range(len(structure)):
        d = frac - (frac[i] + t)
        d -= np.round(d)
        dist = np.linalg.norm(structure.lattice.get_cartesian_coords(d), axis=1)
        ok = [j for j in np.where(dist < tol * max(structure.lattice.abc))[0] if keys[j] == keys[i]]
        if not ok:
            return False
    return True


def apply_transform(
    structure: Structure,
    P: Any,
    origin_shift: Any = (0.0, 0.0, 0.0),
) -> tuple[Structure, list[str]]:
    """New cell (a',b',c') = (a,b,c) P with origin p. Returns (structure, warnings)."""
    P = parse_matrix(P) if not isinstance(P, np.ndarray) else np.asarray(P, dtype=float)
    p = parse_vector(origin_shift) if not isinstance(origin_shift, np.ndarray) else origin_shift
    det = float(np.linalg.det(P))
    if abs(det) < 1e-8:
        raise ValueError("transform matrix is singular (det = 0).")
    msgs: list[str] = []
    if det < 0:
        msgs.append(
            f"det(P) = {det:.4g} < 0: the new axes are left-handed (handedness / "
            "chirality of the structure is inverted)."
        )
    for k in range(3):
        col = P[:, k]
        if np.allclose(col, np.round(col), atol=1e-8):
            continue
        if not _is_lattice_translation(structure, col):
            raise ValueError(
                f"new axis {k + 1} = {np.round(col, 6).tolist()} (old fractional) is not a "
                "lattice translation of this structure; choose another P."
            )

    old_m = np.asarray(structure.lattice.matrix, dtype=float)  # rows = a, b, c
    new_lat = Lattice(P.T @ old_m)
    Pinv = np.linalg.inv(P)

    # Range of old cells that can overlap the new cell.
    corners = np.array([p + P @ np.array(c, float) for c in itertools.product((0, 1), repeat=3)])
    lo = np.floor(corners.min(axis=0)).astype(int) - 1
    hi = np.ceil(corners.max(axis=0)).astype(int) + 1
    shifts = np.array(
        list(itertools.product(*(range(lo[i], hi[i] + 1) for i in range(3)))), dtype=float
    )

    species, coords, labels, props = [], [], [], []
    seen: list[np.ndarray] = []
    for site in structure:
        f0 = np.asarray(site.frac_coords, dtype=float)
        new_f = (Pinv @ (f0 + shifts - p).T).T
        new_f[np.abs(new_f) < _TOL] = 0.0
        new_f[np.abs(new_f - 1.0) < _TOL] = 1.0
        inside = np.all((new_f >= 0.0) & (new_f < 1.0), axis=1)
        for f in new_f[inside]:
            f = np.where(np.abs(f - 1.0) < _TOL, 0.0, f)
            dup = False
            for g in seen:
                d = f - g
                d -= np.round(d)
                if np.all(np.abs(d) < _TOL):
                    dup = True
                    break
            if dup:
                continue
            seen.append(f)
            species.append(site.species)
            coords.append(f)
            labels.append(getattr(site, "label", None))
            props.append(dict(site.properties))

    prop_keys = sorted({k for d in props for k in d})
    site_props = {k: [d.get(k) for d in props] for k in prop_keys} if prop_keys else None
    new = Structure(new_lat, species, coords, labels=labels, site_properties=site_props)

    expected = len(structure) * abs(det)
    if abs(expected - len(new)) > 0.01 * max(expected, 1.0):
        raise ScientificValidationError(
            f"cell transformation: {len(new)} sites, expected {expected:.3f} = "
            f"{len(structure)} x |det(P)| (boundary atoms lost or duplicated)."
        )
    validate_composition_conserved(structure, new, context="cell transformation")
    return new, msgs


def reduce_cell(structure: Structure, cell: str = "input", symprec: float = 0.01) -> Structure:
    """input / niggli / conventional (= standard) / primitive standard cell."""
    cell = str(cell).lower()
    if cell not in CELL_CHOICES:
        raise ValueError(f"cell must be one of {CELL_CHOICES}, got {cell!r}.")
    if cell == "input":
        return structure.copy()
    if cell == "niggli":
        # Niggli cell of the primitive lattice (a centred input is reduced first).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prim = structure.get_primitive_structure(tolerance=symprec * 10)
        new = prim.get_reduced_structure("niggli")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sga = SpacegroupAnalyzer(structure, symprec=symprec)
            if cell in ("conventional", "standard"):
                new = sga.get_conventional_standard_structure()
            else:
                new = sga.get_primitive_standard_structure()
    validate_composition_conserved(structure, new, context=f"{cell} cell")
    expected = len(structure) * float(new.volume) / float(structure.volume)
    if abs(expected - len(new)) > 0.01 * max(expected, 1.0):
        raise ScientificValidationError(
            f"{cell} cell: {len(new)} sites, expected {expected:.3f} from the volume ratio."
        )
    return new


def _summary(structure: Structure, symprec: float) -> dict[str, Any]:
    lat = structure.lattice
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sga = SpacegroupAnalyzer(structure, symprec=symprec)
            sg = (sga.get_space_group_symbol(), int(sga.get_space_group_number()))
    except Exception:  # noqa: BLE001 - symmetry is informative only
        sg = ("", 0)
    return {
        "formula": structure.composition.reduced_formula,
        "n_sites": len(structure),
        "a": round(lat.a, 6),
        "b": round(lat.b, 6),
        "c": round(lat.c, 6),
        "alpha": round(lat.alpha, 4),
        "beta": round(lat.beta, 4),
        "gamma": round(lat.gamma, 4),
        "volume": round(float(lat.volume), 6),
        "density_g_cm3": round(float(structure.density), 6),
        "spacegroup_symbol": sg[0],
        "spacegroup_number": sg[1],
    }


def transform_cell(
    structure: Structure,
    cell: str = "input",
    transform: Any = None,
    origin_shift: Any = None,
    supercell: Any = None,
    symprec: float = 0.01,
) -> tuple[Structure, dict[str, Any]]:
    """cell -> transform (P, p) -> supercell. Returns (new structure, report dict)."""
    msgs: list[str] = []
    work = reduce_cell(structure, cell, symprec=symprec)
    if transform is not None or origin_shift is not None:
        P = parse_matrix(transform) if transform is not None else np.eye(3)
        p = parse_vector(origin_shift) if origin_shift is not None else np.zeros(3)
        work, w = apply_transform(work, P, p)
        msgs += w
    if supercell is not None:
        S = parse_matrix(supercell)
        if not np.allclose(S, np.round(S), atol=1e-8):
            raise ValueError("supercell matrix must be integer.")
        work, w = apply_transform(work, np.round(S), np.zeros(3))
        msgs += w
    old_m = np.asarray(structure.lattice.matrix, float)
    P_total = np.linalg.solve(old_m.T, np.asarray(work.lattice.matrix, float).T)
    report = {
        "cell": str(cell).lower(),
        "transform": None if transform is None else parse_matrix(transform).round(6).tolist(),
        "origin_shift": None if origin_shift is None else parse_vector(origin_shift).round(6).tolist(),
        "supercell": None if supercell is None else parse_matrix(supercell).round(6).tolist(),
        "P_total": np.round(P_total, 6).tolist(),
        "det_P_total": round(float(np.linalg.det(P_total)), 6),
        "volume_ratio": round(float(work.volume / structure.volume), 6),
        "input": _summary(structure, symprec),
        "output": _summary(work, symprec),
        "warnings": msgs,
    }
    return work, report


def write_transformed(
    structure: Structure,
    output_dir: str | Path,
    stem: str = "transformed",
    cif_symprec: float | None = None,
) -> dict[str, str]:
    """Write <stem>.cif (P1 unless cif_symprec) and <stem>.vasp (POSCAR)."""
    from pymatgen.io.cif import CifWriter
    from pymatgen.io.vasp import Poscar

    from shido.analyzer import to_ordered_proxy  # lazy: analyzer imports core

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cif_path = out / f"{stem}.cif"
    poscar_path = out / f"{stem}.vasp"
    CifWriter(structure, symprec=cif_symprec).write_file(str(cif_path))
    Poscar(to_ordered_proxy(structure)).write_file(str(poscar_path))
    return {"cif": str(cif_path.resolve()), "poscar": str(poscar_path.resolve())}


def atoms_in_range(
    structure: Structure, frac_range: Sequence[float], tol: float = 1e-4
) -> list[dict[str, Any]]:
    """All atom images with fractional coords inside [x0,x1]x[y0,y1]x[z0,z1]
    (VESTA "Boundary"). Helper for drawing beyond 0..1; renderer not wired yet."""
    r = [float(v) for v in frac_range]
    if len(r) != 6:
        raise ValueError("range needs 6 numbers: x0 x1 y0 y1 z0 z1.")
    lo = np.array(r[0::2])
    hi = np.array(r[1::2])
    if np.any(hi < lo):
        raise ValueError("range: each upper bound must be >= the lower bound.")
    out: list[dict[str, Any]] = []
    for i, site in enumerate(structure):
        f0 = np.asarray(site.frac_coords, float) % 1.0
        f0[np.abs(f0 - 1.0) < tol] = 0.0
        ranges = [
            range(int(np.floor(lo[k] - f0[k] - tol)), int(np.ceil(hi[k] - f0[k] + tol)) + 1)
            for k in range(3)
        ]
        for n in itertools.product(*ranges):
            f = f0 + np.array(n, float)
            if np.all(f >= lo - tol) and np.all(f <= hi + tol):
                out.append(
                    {
                        "site": i,
                        "image": [int(v) for v in n],
                        "frac": [round(float(v), 6) for v in f],
                        "cart": [round(float(v), 6) for v in structure.lattice.get_cartesian_coords(f)],
                    }
                )
    return out

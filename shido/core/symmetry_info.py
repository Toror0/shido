"""Symmetry information table (VESTA-like): space group, symmetry operations
(equivalent positions as "x,y,z" strings), Wyckoff letters, multiplicities and
site symmetries.

Uses spglib on the input cell. Symmetry operations are given in the basis of the
input cell (for a conventional cell this is the ITA list incl. centring, e.g. 192
for Fm-3m). Wyckoff letters, multiplicities and site symmetries refer to the
standard (conventional) setting, so a primitive input still reports "4a".
"""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import spglib

from shido.core.species import dominant_element

SITE_COLUMNS: list[str] = [
    "site",
    "label",
    "element",
    "species",
    "wyckoff",
    "multiplicity",
    "site_symmetry",
    "site_symmetry_oriented",
    "orbit",
    "equivalent_to",
    "frac_x",
    "frac_y",
    "frac_z",
    "cart_x",
    "cart_y",
    "cart_z",
]
OPERATION_COLUMNS: list[str] = ["index", "xyz", "rotation", "translation"]


def _frac_str(v: float) -> str:
    fr = Fraction(float(v)).limit_denominator(24)
    return str(fr)


def _axis_terms(row: np.ndarray, t: float) -> str:
    parts = ""
    for coef, name in zip(row, "xyz"):
        c = int(round(coef))
        if c == 0:
            continue
        sign = "-" if c < 0 else ("+" if parts else "")
        mag = "" if abs(c) == 1 else str(abs(c))
        parts += f"{sign}{mag}{name}"
    t = float(t) % 1.0
    if abs(t) > 1e-6 and abs(t - 1.0) > 1e-6:
        parts += "+" + _frac_str(t) if parts else _frac_str(t)
    return parts or "0"


def operation_to_xyz(rotation, translation) -> str:
    """(R, t) -> "x', y', z'" string, e.g. "-y+1/2,x,z"."""
    R = np.asarray(rotation, dtype=float)
    t = np.asarray(translation, dtype=float)
    return ",".join(_axis_terms(R[i], t[i]) for i in range(3))


def _spglib_cell(structure) -> tuple[tuple, list[str]]:
    """Cell for spglib; each distinct species/occupancy string gets its own type."""
    keys = [s.species_string for s in structure]
    kinds = {k: i + 1 for i, k in enumerate(dict.fromkeys(keys))}
    cell = (
        np.asarray(structure.lattice.matrix, dtype=float),
        np.asarray(structure.frac_coords, dtype=float),
        [kinds[k] for k in keys],
    )
    return cell, keys


def compute_symmetry_info(
    structure, symprec: float = 0.01, angle_tolerance: float = -1.0
) -> dict[str, Any]:
    """Space group, symmetry operations and the per-site / per-orbit symmetry table."""
    cell, species = _spglib_cell(structure)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    if ds is None:
        raise ValueError(f"spglib could not find the symmetry (symprec={symprec}).")

    rotations = np.asarray(ds.rotations)
    translations = np.asarray(ds.translations)
    n_input = len(rotations)
    std_ops = spglib.get_symmetry_from_database(int(ds.hall_number))
    n_std = len(std_ops["rotations"]) if std_ops else n_input

    equiv = [int(v) for v in ds.equivalent_atoms]
    counts: dict[int, int] = {}
    for e in equiv:
        counts[e] = counts.get(e, 0) + 1
    orbit_ids = {rep: k + 1 for k, rep in enumerate(dict.fromkeys(equiv))}
    lat = structure.lattice

    sites: list[dict[str, Any]] = []
    for i, s in enumerate(structure):
        f = np.asarray(s.frac_coords, dtype=float)
        c = lat.get_cartesian_coords(f)
        rep = equiv[i]
        mult = counts[rep] * n_std / n_input
        oriented = str(ds.site_symmetry_symbols[i])
        sites.append(
            {
                "site": i,
                "label": str(getattr(s, "label", "") or ""),
                "element": dominant_element(s),
                "species": species[i],
                "wyckoff": str(ds.wyckoffs[i]),
                "multiplicity": int(round(mult)),
                "site_symmetry": oriented.replace(".", ""),
                "site_symmetry_oriented": oriented,
                "orbit": orbit_ids[rep],
                "equivalent_to": rep,
                "frac_x": round(float(f[0]), 6),
                "frac_y": round(float(f[1]), 6),
                "frac_z": round(float(f[2]), 6),
                "cart_x": round(float(c[0]), 6),
                "cart_y": round(float(c[1]), 6),
                "cart_z": round(float(c[2]), 6),
            }
        )

    orbits: list[dict[str, Any]] = []
    for rep, k in orbit_ids.items():
        members = [r for r in sites if r["equivalent_to"] == rep]
        first = members[0]
        orbits.append(
            {
                "orbit": k,
                "element": first["element"],
                "species": first["species"],
                "label": first["label"],
                "wyckoff": f"{first['multiplicity']}{first['wyckoff']}",
                "wyckoff_letter": first["wyckoff"],
                "multiplicity": first["multiplicity"],
                "site_symmetry": first["site_symmetry"],
                "site_symmetry_oriented": first["site_symmetry_oriented"],
                "n_in_cell": len(members),
                "sites": [r["site"] for r in members],
                "equivalent_positions": [
                    [r["frac_x"], r["frac_y"], r["frac_z"]] for r in members
                ],
            }
        )

    operations = [
        {
            "index": j + 1,
            "xyz": operation_to_xyz(rotations[j], translations[j]),
            "rotation": [[int(v) for v in row] for row in rotations[j]],
            "translation": [round(float(v), 6) for v in translations[j]],
        }
        for j in range(n_input)
    ]

    return {
        "symprec": symprec,
        "spacegroup_number": int(ds.number),
        "spacegroup_symbol": str(ds.international),
        "hall_symbol": str(ds.hall),
        "hall_number": int(ds.hall_number),
        "choice": str(ds.choice),
        "point_group": str(ds.pointgroup),
        "n_operations": n_input,
        "n_operations_standard": n_std,
        "n_sites": len(structure),
        "orbits": orbits,
        "sites": sites,
        "operations": operations,
    }


def write_symmetry_report(info: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """symmetry_info.json, symmetry_sites.csv, symmetry_operations.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "symmetry_info.json"
    sites_path = out / "symmetry_sites.csv"
    ops_path = out / "symmetry_operations.csv"
    json_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    with sites_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SITE_COLUMNS)
        w.writeheader()
        for r in info["sites"]:
            w.writerow({k: r[k] for k in SITE_COLUMNS})
    with ops_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OPERATION_COLUMNS)
        w.writeheader()
        for op in info["operations"]:
            w.writerow(
                {
                    "index": op["index"],
                    "xyz": op["xyz"],
                    "rotation": " ".join(str(v) for row in op["rotation"] for v in row),
                    "translation": " ".join(str(v) for v in op["translation"]),
                }
            )
    return {
        "json": str(json_path.resolve()),
        "sites_csv": str(sites_path.resolve()),
        "operations_csv": str(ops_path.resolve()),
    }

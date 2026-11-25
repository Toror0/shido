"""Batch 8: structure data export (VESTA "Export Data").

Formats (``formats=`` names):
  xyz       XMol XYZ, Cartesian Å (extended-XYZ comment line with the lattice)
  p1        CIF in P1 (all atoms, no symmetry) -> <name>_P1.cif
  xtl       XTL (cell + fractional coordinates, P1)
  pdb       PDB (CRYST1 + SCALE + one ATOM line per species and site, with occupancy)
  wrl       colour VRML 2.0 (same mesh as the STL, mm) via export3d.export_vrml
  multicif  all given structures in one CIF (one data_ block each) -> <multi_name>_all.cif
  vesta     VESTA file (cell, space group, sites, occupancies, bonds, colours) via core.vesta_io

CIF/POSCAR of a transformed cell stay in core.cell_transform.write_transformed;
STL/HTML stay in export3d.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from pymatgen.core import Structure

from shido.core.species import site_occupancies

FORMATS: tuple[str, ...] = ("xyz", "p1", "xtl", "pdb", "wrl", "vesta", "multicif")
DEFAULT_FORMATS: tuple[str, ...] = ("xyz", "p1", "xtl", "pdb")
_ALIASES = {
    "vrml": "wrl",
    "p1cif": "p1",
    "cif_p1": "p1",
    "multi": "multicif",
    "multi_cif": "multicif",
    "cifs": "multicif",
    "all": "all",
}


def parse_formats(value: str | Sequence[str] | None) -> list[str]:
    """"xyz,pdb" / ["xyz", "pdb"] / None (defaults) -> validated, de-duplicated list."""
    if value is None:
        return list(DEFAULT_FORMATS)
    items: Iterable[str] = value.replace(";", ",").split(",") if isinstance(value, str) else value
    out: list[str] = []
    for raw in items:
        for part in str(raw).replace(" ", ",").split(","):
            key = part.strip().lower().lstrip(".")
            if not key:
                continue
            key = _ALIASES.get(key, key)
            if key == "all":
                for f in FORMATS:
                    if f not in out:
                        out.append(f)
                continue
            if key not in FORMATS:
                raise ValueError(f"Unknown export format {part!r}. Use: {', '.join(FORMATS)}.")
            if key not in out:
                out.append(key)
    if not out:
        raise ValueError("No export format given.")
    return out


def _safe_name(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_.\-]+", "_", str(name)).strip("._")
    return s or "structure"


def _ordered(structure: Structure, fmt: str, warnings: list[str], name: str) -> Structure:
    if structure.is_ordered:
        return structure
    from shido.analyzer import to_ordered_proxy  # lazy: analyzer imports core

    warnings.append(
        f"{name}: {fmt.upper()} has no occupancy field; partial sites were written as an "
        "ordered proxy that keeps the composition."
    )
    return to_ordered_proxy(structure)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_xyz(structure: Structure, path: str | Path, comment: str = "") -> Path:
    """XYZ (Cartesian Å). Disordered input should be made ordered first."""
    p = Path(path)
    lat = structure.lattice.matrix
    lattice = " ".join(f"{v:.8f}" for v in lat.ravel())
    title = f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"'
    if comment:
        title += f" comment={_safe_name(comment)}"
    lines = [str(len(structure)), title]
    for site in structure:
        el = site_occupancies(site)[0][0]
        x, y, z = (float(v) for v in site.coords)
        lines.append(f"{el:<3s} {x:14.8f} {y:14.8f} {z:14.8f}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_p1_cif(structure: Structure, path: str | Path) -> Path:
    """CIF with symmetry P1 (every atom listed, occupancies kept)."""
    from pymatgen.io.cif import CifWriter

    p = Path(path)
    CifWriter(structure, symprec=None).write_file(str(p))
    return p


def write_xtl(structure: Structure, path: str | Path, title: str = "") -> Path:
    """XTL: CELL a b c alpha beta gamma, SYMMETRY P1, ATOMS NAME X Y Z (fractional)."""
    p = Path(path)
    lat = structure.lattice
    lines = [
        f"TITLE {title or structure.composition.reduced_formula}",
        "CELL",
        "  " + " ".join(f"{v:.6f}" for v in (*lat.abc, *lat.angles)),
        "SYMMETRY NUMBER 1",
        "SYMMETRY LABEL  P1",
        "ATOMS",
        "NAME         X           Y           Z",
    ]
    for site in structure:
        el = site_occupancies(site)[0][0]
        x, y, z = (float(v) for v in site.frac_coords)
        lines.append(f"{el:<4s} {x:11.6f} {y:11.6f} {z:11.6f}")
    lines.append("EOF")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def pdb_orthogonal_matrix(a: float, b: float, c: float,
                          alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Rows = a, b, c vectors in the PDB frame (a along x, b in the xy plane)."""
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([[a, 0.0, 0.0], [b * np.cos(ga), b * np.sin(ga), 0.0], [cx, cy, cz]])


def write_pdb(structure: Structure, path: str | Path, title: str = "") -> Path:
    """PDB: CRYST1 (P 1) + SCALEn + one ATOM line per (site, species) with occupancy.

    Coordinates use the PDB orthogonalisation (a along x, b in the xy plane), so they
    agree with CRYST1 / SCALE; columns follow the PDB v3.3 fixed-width layout.
    """
    p = Path(path)
    lat = structure.lattice
    a, b, c = lat.abc
    al, be, ga = lat.angles
    m = pdb_orthogonal_matrix(a, b, c, al, be, ga)
    scale = np.linalg.inv(m.T)  # frac = scale @ cart
    lines = []
    if title:
        lines.append(f"TITLE     {title[:70]}")
    lines.append(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {'P 1':<11s}{1:4d}")
    for i in range(3):
        r = scale[i]
        lines.append(f"SCALE{i + 1}    {r[0]:10.6f}{r[1]:10.6f}{r[2]:10.6f}     {0.0:10.5f}")
    serial = 0
    for k, site in enumerate(structure):
        xyz = np.asarray(site.frac_coords, float) @ m
        for el, occ in site_occupancies(site):
            serial += 1
            name = f"{el.upper():<3s}"[:4] if len(el) == 2 else f" {el.upper():<3s}"[:4]
            lines.append(
                f"HETATM{serial % 100000:5d} {name:4s} {'UNK':>3s} A{(k + 1) % 10000:4d}    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f}{0.0:6.2f}          "
                f"{el.upper():>2s}"
            )
    lines.append("END")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_multi_cif(entries: Sequence[tuple[str, Structure]], path: str | Path,
                    symprec: float | None = None) -> Path:
    """All structures in one CIF, one ``data_<name>`` block each (VESTA 3.5 style)."""
    from pymatgen.io.cif import CifWriter

    if not entries:
        raise ValueError("No structures to write.")
    p = Path(path)
    used: set[str] = set()
    chunks = ["# generated by CrystalAnalyzerMCP (multi-structure CIF)"]
    for name, s in entries:
        block = _safe_name(name)
        base, n = block, 2
        while block.lower() in used:
            block, n = f"{base}_{n}", n + 1
        used.add(block.lower())
        text = str(CifWriter(s, symprec=symprec))
        text = re.sub(r"^#.*\n", "", text)
        text = re.sub(r"^data_\S*", f"data_{block}", text, count=1, flags=re.M)
        chunks.append(text.strip())
    p.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Front end
# ---------------------------------------------------------------------------


def export_structures(
    entries: Sequence[tuple[str, Structure]],
    formats: str | Sequence[str] | None,
    output_dir: str | Path,
    multi_name: str | None = None,
) -> dict[str, Any]:
    """Write each structure in every requested format; ``multicif`` writes one file for all.

    Returns {"formats", "structures": [{name, formula, n_sites, files: {fmt: path}}],
    "multicif" (path or None), "warnings"}.
    """
    fmts = parse_formats(formats)
    if not entries:
        raise ValueError("No structures to export.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    report: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, s in entries:
        stem = _safe_name(name)
        base, n = stem, 2
        while stem.lower() in seen:
            stem, n = f"{base}_{n}", n + 1
        seen.add(stem.lower())
        files: dict[str, str] = {}
        for f in fmts:
            if f == "xyz":
                path = write_xyz(_ordered(s, f, warnings, stem), out / f"{stem}.xyz", comment=stem)
            elif f == "p1":
                path = write_p1_cif(s, out / f"{stem}_P1.cif")
            elif f == "xtl":
                path = write_xtl(_ordered(s, f, warnings, stem), out / f"{stem}.xtl", title=stem)
            elif f == "pdb":
                path = write_pdb(s, out / f"{stem}.pdb", title=stem)
            elif f == "wrl":
                from shido.export3d import export_vrml  # lazy: pyvista is heavy

                path = export_vrml(s, out / f"{stem}.wrl")
            elif f == "vesta":
                from shido.core.vesta_io import write_vesta

                path = write_vesta(s, out / f"{stem}.vesta", title=stem)
            else:
                continue
            files[f] = str(Path(path).resolve())
        report.append({
            "name": stem,
            "formula": s.composition.reduced_formula,
            "n_sites": len(s),
            "files": files,
        })
    multi = None
    if "multicif" in fmts:
        mname = _safe_name(multi_name or (entries[0][0] if len(entries) == 1 else "structures"))
        multi = str(write_multi_cif(entries, out / f"{mname}_all.cif").resolve())
    return {"formats": fmts, "structures": report, "multicif": multi, "warnings": warnings}


__all__ = [
    "DEFAULT_FORMATS",
    "FORMATS",
    "export_structures",
    "parse_formats",
    "pdb_orthogonal_matrix",
    "write_multi_cif",
    "write_p1_cif",
    "write_pdb",
    "write_xtl",
    "write_xyz",
]

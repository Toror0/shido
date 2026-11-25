"""Batch 24: VESTA format (*.vesta), structure part only.

Write: lattice (CELLP), space group (GROUP + every operation in SYMOP, centring
included), asymmetric unit with occupancies (STRUC; partially occupied sites are
several STRUC lines at the same position, as VESTA writes them), bonds (SBOND),
site / element radius and colour (SITET / ATOMT), plus the default display blocks.
Read: CELLP + SYMOP + STRUC (and SBOND / SITET as extras) -> pymatgen Structure.

There is no published specification; the layout follows files written by VESTA
3.3.0 / 3.5.4 found on GitHub (WMD-group/Phonons 2015_MAPbI3/MAPbI3-Cubic.vesta,
goirijo/multishifter docs/tutorials/i/mg.vesta, vitroid/GenIce
genice2/lattices/preparing/hs6.vesta (P6/mcc, shows SYMOP is t1 t2 t3 then the
rotation row by row), orex/supercell data/examples/synth_complex/sc.vesta
(disordered sites)). Inferred, not seen in a sample: whether VESTA expects the
centring translations in SYMOP (we write them; duplicates are harmless when read
back) and the exact spacing of the Hermann-Mauguin symbol on the GROUP line.

Symmetry is written only when spglib finds the input cell already in its standard
setting (identity transformation, no origin shift); otherwise the file is P 1 with
every atom, so VESTA can never rebuild a different cell from the group number.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from pymatgen.core import Element, Lattice, Structure

from shido.core.species import site_occupancies

VESTA_VERSION = "3.3.0"

# Fixed blocks copied from VESTA-written files (orientation / display defaults).
_HEAD_FIXED = """TRANM 0
 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1
LTRANSL
 -1
 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
LORIENT
 -1   0   0   0   0
 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000
LMATRIX
 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
 0.000000  0.000000  0.000000"""

_MID_FIXED = """VECTR
 0 0 0 0 0
VECTT
 0 0 0 0 0
SPLAN
  0   0   0   0
LBLAT
 -1
LBLSP
 -1
DLATM
 -1
DLBND
 -1
DLPLY
 -1
PLN2D
  0   0   0   0"""

_LIGHT_OFF = """ 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
 0.000000  0.000000 20.000000  0.000000
 0.000000  0.000000 -1.000000"""

_TAIL_FIXED = f"""SCENE
 1.000000  0.000000  0.000000  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.000000  0.000000  0.000000  1.000000
  0.000   0.000
  0.000
  1.000
HBOND 0 2

STYLE
DISPF 54531010
MODEL   0  1  0
SURFS   0  1  1
SECTS  96  1
FORMS   0  1
ATOMS   0  0  1
BONDS   1
POLYS   1
VECTS 1.000000
FORMP
  1  1.0   0   0   0
ATOMP
 24  24   0  50  2.0   0
BONDP
  1  16  0.250  2.000 127 127 127
POLYP
 204 1  1.000 180 180 180
ISURF
  0   0   0   0
TEX3P
  1  0.00000E+00  1.00000E+00
SECTP
  1  5.00000E-01  5.00000E-01  0.00000E+00
HKLPP
 192 1  1.000 255   0 255
UCOLP
   2   1  5.000   0   0   0
COMPS 1
LABEL 1    12  1.000 0
PROJT 0  0.962
BKGRC
 255 255 255
DPTHQ 1 -0.5000  3.5000
LIGHT0 0
{_LIGHT_OFF}
  26  26  26 255
 179 179 179 255
 255 255 255 255
LIGHT1
{_LIGHT_OFF}
   0   0   0   0
   0   0   0   0
   0   0   0   0
LIGHT2
{_LIGHT_OFF}
   0   0   0   0
   0   0   0   0
   0   0   0   0
LIGHT3
{_LIGHT_OFF}
   0   0   0   0
   0   0   0   0
   0   0   0   0
ATOMM
   0   0   0 255
   1.280
BONDM
 255 255 255 255
 128.000
POLYM
 255 255 255 255
 128.000
SURFM
   0   0   0 255
 128.000
FORMM
 255 255 255 255
 128.000
HKLPM
 255 255 255 255
 128.000"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _element_rgb(el: str) -> tuple[int, int, int]:
    try:
        from shido.theme_manager import JMOL_FALLBACK_COLORS

        hexcol = JMOL_FALLBACK_COLORS.get(el, "#808080")
    except Exception:
        hexcol = "#808080"
    h = hexcol.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _element_radius(el: str) -> float:
    try:
        r = Element(el).atomic_radius
        return float(r) if r else 1.0
    except Exception:
        return 1.0


def spaced_hm(symbol: str) -> str:
    """'Fm-3m' -> 'F m -3 m', 'P6_3/mmc' -> 'P 63/m m c' (VESTA GROUP line style)."""
    s = symbol.replace(" ", "").split(":")[0]  # drop a setting suffix ('Fd-3m:2')
    if not s:
        return "P 1"
    head, rest = s[0], s[1:]
    parts = re.findall(r"-?\d(?:_\d)?(?:/[a-z])?|[a-z]", rest)
    return " ".join([head] + [p.replace("_", "") for p in parts])


def _symmetry(structure: Structure, symprec: float) -> dict[str, Any] | None:
    """Space group data when the cell is already spglib's standard setting, else None."""
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        ds = sga.get_symmetry_dataset()
    except Exception:
        return None
    if ds is None:
        return None
    get = (lambda k: getattr(ds, k)) if not isinstance(ds, dict) else ds.__getitem__
    tmat = np.asarray(get("transformation_matrix"), float)
    shift = np.asarray(get("origin_shift"), float)
    shift = shift - np.round(shift)
    if not (np.allclose(tmat, np.eye(3), atol=1e-6) and np.allclose(shift, 0, atol=1e-6)):
        return None
    number = int(get("number"))
    if number == 1:
        return None
    return {
        "number": number,
        "symbol": str(get("international")),
        "rotations": np.asarray(get("rotations"), int),
        "translations": np.asarray(get("translations"), float),
        "equivalent": np.asarray(get("equivalent_atoms"), int),
        "wyckoffs": list(get("wyckoffs")),
        "site_sym": list(get("site_symmetry_symbols")),
        "multiplicity": Counter(np.asarray(get("equivalent_atoms"), int).tolist()),
    }


def default_bonds(structure: Structure, factor: float = 1.25,
                  rmax: float = 4.0) -> list[tuple[str, str, float, float]]:
    """(A, B, 0, 1.1*shortest A-B distance) for element pairs closer than
    factor*(rA+rB) (pymatgen atomic radii) and within 1.15x the nearest contact of
    A or B (first shell only). Mirrors VESTA's 'add bond by elements'."""
    shortest: dict[tuple[str, str], float] = {}
    els = [site_occupancies(s)[0][0] for s in structure]
    for i, neigh in enumerate(structure.get_all_neighbors(rmax)):
        for n in neigh:
            a, b = sorted((els[i], els[n.index]))
            d = float(n.nn_distance)
            if d > 1e-3 and d < shortest.get((a, b), np.inf):
                shortest[(a, b)] = d
    nearest: dict[str, float] = {}
    for (a, b), d in shortest.items():
        nearest[a] = min(nearest.get(a, np.inf), d)
        nearest[b] = min(nearest.get(b, np.inf), d)
    out = []
    for (a, b), d in sorted(shortest.items()):
        # first shell only: skip pairs much longer than either element's nearest contact
        if d > 1.15 * min(nearest[a], nearest[b]):
            continue
        if d <= factor * (_element_radius(a) + _element_radius(b)):
            out.append((a, b, 0.0, round(d * 1.1, 3)))
    return out


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


def write_vesta(structure: Structure, path: str | Path, title: str = "",
                bonds: Sequence[tuple[str, str, float, float]] | None = None,
                symprec: float = 1e-3, use_symmetry: bool = True) -> Path:
    """Write ``structure`` as a VESTA file. ``bonds`` = [(A, B, dmin, dmax), ...]
    (None: `default_bonds`). Returns the path."""
    p = Path(path)
    sym = _symmetry(structure, symprec) if use_symmetry else None
    lat = structure.lattice

    if sym is None:
        group_line = "1 1 P 1"
        ops = [(np.eye(3, dtype=int), np.zeros(3))]
        reps = list(range(len(structure)))
        wy = ["1a"] * len(structure)
        ss = ["1"] * len(structure)
    else:
        group_line = f"{sym['number']} 1 {spaced_hm(sym['symbol'])}"
        ops = list(zip(sym["rotations"], sym["translations"]))
        reps = sorted(set(sym["equivalent"].tolist()))
        wy = [f"{sym['multiplicity'][int(sym['equivalent'][i])]}{sym['wyckoffs'][i]}"
              for i in range(len(structure))]
        ss = [str(sym["site_sym"][i]).replace(".", "") or "1" for i in range(len(structure))]

    lines = [f"#VESTA_FORMAT_VERSION {VESTA_VERSION}", "", "", "CRYSTAL", "", "TITLE",
             (title or structure.composition.reduced_formula), "", "GROUP", group_line, "SYMOP"]
    for rot, tr in ops:
        t = np.asarray(tr, float) % 1.0
        t[np.isclose(t, 1.0)] = 0.0
        r = " ".join(f"{int(v):2d}" for v in rot[0]) + "  " + " ".join(
            f"{int(v):2d}" for v in rot[1]) + "  " + " ".join(f"{int(v):2d}" for v in rot[2])
        lines.append(f" {t[0]:9.6f} {t[1]:9.6f} {t[2]:9.6f} {r}   1")
    lines.append(" -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0")
    lines.append(_HEAD_FIXED)
    lines.append("CELLP")
    lines.append("  " + "  ".join(f"{v:9.6f}" for v in (*lat.abc, *lat.angles)))
    lines.append("  " + "  ".join(f"{0.0:9.6f}" for _ in range(6)))

    # STRUC: one line per (representative site, species); labels unique per element.
    counter: Counter[str] = Counter()
    entries: list[tuple[str, str, float, np.ndarray, str, str]] = []
    for i in reps:
        site = structure[i]
        for el, occ in site_occupancies(site):
            counter[el] += 1
            entries.append((el, f"{el}{counter[el]}", float(occ),
                            np.asarray(site.frac_coords, float), wy[i], ss[i]))
    lines.append("STRUC")
    for k, (el, label, occ, xyz, w, s) in enumerate(entries, 1):
        lines.append(f"{k:3d} {el:>2s} {label:>10s}  {occ:6.4f}  {xyz[0]:9.6f}  {xyz[1]:9.6f}"
                     f"  {xyz[2]:9.6f}  {w:>5s} {s:>7s}")
        lines.append(f"{'':28s}0.000000   0.000000   0.000000  0.00")
    lines.append("  0 0 0 0 0 0 0")
    lines.append("THERI 0")
    for k, (_, label, *_r) in enumerate(entries, 1):
        lines.append(f"{k:3d} {label:>10s}  1.000000")
    lines.append("  0 0 0")
    lines.append("SHAPE")
    lines.append("  0       0       0       0   0.000000  0   192   192   192   192")
    lines.append("BOUND")
    lines.append("       0        1         0        1         0        1")
    lines.append("  0   0   0   0  0")

    lines.append("SBOND")
    blist = default_bonds(structure) if bonds is None else list(bonds)
    for k, (a, b, dmin, dmax) in enumerate(blist, 1):
        lines.append(f"{k:3d} {a:>5s} {b:>5s}  {float(dmin):9.5f}  {float(dmax):9.5f}"
                     "  0  1  0  0  1  0.250  2.000 127 127 127")
    lines.append("  0 0 0 0")

    lines.append("SITET")
    for k, (el, label, *_r) in enumerate(entries, 1):
        r, g, b = _element_rgb(el)
        lines.append(f"{k:3d} {label:>10s}  {_element_radius(el):6.4f} {r:3d} {g:3d} {b:3d}"
                     f" {r:3d} {g:3d} {b:3d} 204  0")
    lines.append("  0 0 0 0 0 0")
    lines.append(_MID_FIXED)
    lines.append("ATOMT")
    for k, el in enumerate(sorted(counter, key=lambda e: Element(e).Z if Element.is_valid_symbol(e) else 999), 1):
        r, g, b = _element_rgb(el)
        lines.append(f"{k:3d} {el:>10s}  {_element_radius(el):6.4f} {r:3d} {g:3d} {b:3d}"
                     f" {r:3d} {g:3d} {b:3d} 204")
    lines.append("  0 0 0 0 0 0")
    lines.append(_TAIL_FIXED)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

_KEYWORD = re.compile(r"^[A-Z][A-Z0-9]{2,}\b")


def _sections(text: str) -> dict[str, list[str]]:
    """First occurrence of each keyword block -> its lines (keyword line first)."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    title_line = False
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if title_line:  # the line after TITLE is the title even if it looks like a keyword (H2O, C60)
            title_line = False
            if current == "TITLE":
                out[current].append(raw)
                continue
        if _KEYWORD.match(raw):
            key = raw.split()[0]
            current = key if key not in out else None
            if current:
                out[current] = [raw]
            title_line = key == "TITLE"
            continue
        if current:
            out[current].append(raw)
    return out


def is_vesta_text(head: str) -> bool:
    return head.lstrip("﻿").lstrip().startswith("#VESTA_FORMAT_VERSION")


def read_vesta(path: str | Path, merge_tol: float = 1e-3) -> tuple[Structure, dict[str, Any]]:
    """Read a VESTA file -> (Structure with all atoms of the cell, info).

    info = {"version", "title", "group", "n_symops", "bonds": [(A, B, dmin, dmax)],
    "site_styles": {label: {"radius", "rgb"}}}. Partially occupied sites (several
    STRUC entries at the same position) become one site with mixed species.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    first = text.lstrip("﻿").split("\n", 1)[0].strip()
    if not is_vesta_text(first):
        raise ValueError(f"'{p}' is not a VESTA file (no #VESTA_FORMAT_VERSION line).")
    sec = _sections(text)
    for key in ("CELLP", "STRUC"):
        if key not in sec:
            raise ValueError(f"'{p}': VESTA block {key} not found.")

    cell = [float(v) for v in sec["CELLP"][1].split()[:6]]
    lattice = Lattice.from_parameters(*cell)

    ops: list[tuple[np.ndarray, np.ndarray]] = []
    for ln in sec.get("SYMOP", [])[1:]:
        t = ln.split()
        if len(t) < 12:
            continue
        vals = [float(v) for v in t[:12]]
        if vals[:3] == [-1.0, -1.0, -1.0]:
            break
        ops.append((np.array(vals[3:12]).reshape(3, 3), np.array(vals[:3])))
    if not ops:
        ops = [(np.eye(3), np.zeros(3))]

    asym: list[tuple[str, str, float, np.ndarray]] = []
    for ln in sec["STRUC"][1:]:
        t = ln.split()
        if len(t) < 7:
            continue
        if t[0] == "0":
            break
        try:
            int(t[0])
        except ValueError:
            continue
        el = t[1]
        if not Element.is_valid_symbol(el):
            m = re.match(r"[A-Z][a-z]?", t[2])
            el = m.group(0) if m and Element.is_valid_symbol(m.group(0)) else el
        asym.append((el, t[2], float(t[3]), np.array([float(v) for v in t[4:7]])))
    if not asym:
        raise ValueError(f"'{p}': no atoms in STRUC.")

    # Expand with every operation; group by position (mixed species on one site).
    tol = max(merge_tol * 10, 1e-3)  # Å

    def _find(x: np.ndarray, pts: list[np.ndarray]) -> int | None:
        for j, q in enumerate(pts):
            d = x - q
            d -= np.round(d)
            if np.linalg.norm(lattice.get_cartesian_coords(d)) < tol:
                return j
        return None

    positions: list[np.ndarray] = []
    species: list[dict[str, float]] = []
    for el, _label, occ, xyz in asym:
        orbit: list[np.ndarray] = []
        for rot, tr in ops:
            x = (rot @ xyz + tr) % 1.0
            x[np.isclose(x, 1.0, atol=1e-6)] = 0.0
            if _find(x, orbit) is None:
                orbit.append(x)
        for x in orbit:
            j = _find(x, positions)
            if j is None:
                positions.append(x)
                species.append({el: occ})
            elif el in species[j]:
                # the same element written twice for one position (equivalent STRUC
                # entries): one site, not occupancy 2
                species[j][el] = max(species[j][el], occ)
            else:
                species[j][el] = occ

    struct = Structure(lattice, [
        (list(sp)[0] if len(sp) == 1 and abs(list(sp.values())[0] - 1.0) < 1e-6 else sp)
        for sp in species
    ], positions)

    bonds = []
    for ln in sec.get("SBOND", [])[1:]:
        t = ln.split()
        if len(t) < 5 or t[0] == "0":
            break
        try:
            bonds.append((t[1], t[2], float(t[3]), float(t[4])))
        except ValueError:
            continue
    styles: dict[str, dict[str, Any]] = {}
    for ln in sec.get("SITET", [])[1:]:
        t = ln.split()
        if len(t) < 6 or t[0] == "0":
            break
        try:
            styles[t[1]] = {"radius": float(t[2]), "rgb": tuple(int(v) for v in t[3:6])}
        except ValueError:
            continue
    group = sec.get("GROUP", ["", ""])
    info = {
        "version": first.split()[-1],
        "title": (sec.get("TITLE", ["", ""]) + [""])[1].strip(),
        "group": group[1].strip() if len(group) > 1 else "",
        "n_symops": len(ops),
        "bonds": bonds,
        "site_styles": styles,
    }
    return struct, info


__all__ = ["default_bonds", "is_vesta_text", "read_vesta", "spaced_hm", "write_vesta"]

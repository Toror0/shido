"""Atomic displacement parameters (ADP, "thermal ellipsoids") read directly from a CIF.

pymatgen's ``CifParser`` drops the displacement parameters, so the CIF is read again here:

* ``_atom_site_aniso_U_ij`` / ``_atom_site_aniso_B_ij`` / ``_atom_site_aniso_beta_ij``
  (matched to ``_atom_site_label`` through ``_atom_site_aniso_label``),
* ``_atom_site_U_iso_or_equiv`` / ``_atom_site_B_iso_or_equiv`` for the isotropic sites
  (anisotropic values win when a label has both).

Conventions (Grosse-Kunstleve & Adams, J. Appl. Cryst. 35 (2002) 477; Trueblood et al.,
Acta Cryst. A52 (1996) 770):

* ``U_cif`` [Å²] is the CIF tensor, ``N = diag(a*, b*, c*)``,
  ``U* = N U_cif Nᵀ`` (dimensionless, = covariance of the fractional displacement),
  ``U_cart = A U* Aᵀ`` with ``A`` = direct-lattice matrix (columns = a, b, c in Å).
* ``B_ij = 8π² U_ij``, ``β = 2π² U*`` (T = exp(−hᵀβh)), ``U_iso``: ``U_cart = U_iso·I``.
* ``U_eq = trace(U_cart) / 3``. Principal axes = eigenvectors of ``U_cart``; the
  root-mean-square displacement along an axis is √(eigenvalue).
* Ellipsoid of probability p: semi-axes = c(p)·√(eigenvalue), c(p) = √(χ²₃⁻¹(p))
  (50 % → 1.5382, as ORTEP / VESTA).
* Symmetry-equivalent sites: ``U*' = R U* Rᵀ`` for x' = R x + t.

Main entry points (used by the ellipsoid drawing of batch 16):
``read_cif_adp`` (asymmetric unit), ``site_displacement_tensors`` (U_cart per site of a
pymatgen Structure), ``ellipsoid_axes`` / ``probability_scale``, ``analyze_displacement``
(table + report) and ``write_displacement_report`` (``displacement.csv`` / ``.json``).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.io.cif import CifFile

from shido.core.validator import ScientificValidationError

_IJ = ("11", "22", "33", "12", "13", "23")
_IDX = {"11": (0, 0), "22": (1, 1), "33": (2, 2), "12": (0, 1), "13": (0, 2), "23": (1, 2)}
_SYMOP_KEYS = (
    "_space_group_symop_operation_xyz",
    "_symmetry_equiv_pos_as_xyz",
    "_space_group_symop.operation_xyz",
)
_MATCH_TOL = 0.05  # Å, structure site <-> CIF image of an asymmetric-unit atom

DISPLACEMENT_COLUMNS: list[str] = [
    "label", "element", "adp_type", "source",
    "U11", "U22", "U33", "U12", "U13", "U23",
    "U_eq", "U_iso_cif",
    "U1", "U2", "U3", "rms1", "rms2", "rms3",
    "semi_axis1", "semi_axis2", "semi_axis3",
    "axis1_x", "axis1_y", "axis1_z", "axis2_x", "axis2_y", "axis2_z", "axis3_x", "axis3_y", "axis3_z",
    "anisotropy", "positive_definite", "n_sites_in_cell",
]


# --- small helpers ------------------------------------------------------------------
def _num(value: Any) -> float | None:
    """CIF number without its esd: '0.0085(8)' -> 0.0085; '.', '?' -> None."""
    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    if text in ("", ".", "?"):
        return None
    text = re.sub(r"\(\d+\)$", "", text)
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _element(label: str, type_symbol: str | None) -> str:
    for text in (type_symbol, label):
        m = re.match(r"([A-Z][a-z]?)", str(text or ""))
        if m:
            return m.group(1)
    return str(label)


def probability_scale(probability: float = 50.0) -> float:
    """Radius factor c(p) = √(χ²₃⁻¹(p)) of the probability ellipsoid (50 -> 1.5382).

    ``probability`` is in percent (0 < p < 100); values in (0, 1) are taken as fractions.
    """
    from scipy.stats import chi2

    p = float(probability)
    if not math.isfinite(p) or p <= 0.0 or p >= 100.0:
        raise ScientificValidationError(f"probability must be in (0, 100) percent, got {probability!r}.")
    if p >= 1.0:
        p /= 100.0
    return float(math.sqrt(chi2.ppf(p, 3)))


def _n_matrix(lattice: Lattice) -> np.ndarray:
    return np.diag(lattice.reciprocal_lattice_crystallographic.abc)


def ucif_to_ustar(u_cif: np.ndarray, lattice: Lattice) -> np.ndarray:
    """U* = N U_cif Nᵀ (dimensionless)."""
    n = _n_matrix(lattice)
    return n @ np.asarray(u_cif, dtype=float) @ n.T


def ustar_to_ucif(u_star: np.ndarray, lattice: Lattice) -> np.ndarray:
    ninv = np.linalg.inv(_n_matrix(lattice))
    return ninv @ np.asarray(u_star, dtype=float) @ ninv.T


def ustar_to_ucart(u_star: np.ndarray, lattice: Lattice) -> np.ndarray:
    """U_cart = A U* Aᵀ [Å²] in the Cartesian frame of ``lattice.matrix``."""
    a = lattice.matrix.T
    return a @ np.asarray(u_star, dtype=float) @ a.T


def ucart_to_ustar(u_cart: np.ndarray, lattice: Lattice) -> np.ndarray:
    ainv = np.linalg.inv(lattice.matrix.T)
    return ainv @ np.asarray(u_cart, dtype=float) @ ainv.T


def ucif_to_ucart(u_cif: np.ndarray, lattice: Lattice) -> np.ndarray:
    """U_cart = A N U_cif Nᵀ Aᵀ (Grosse-Kunstleve & Adams 2002)."""
    return ustar_to_ucart(ucif_to_ustar(u_cif, lattice), lattice)


def u_equivalent(u_cart: np.ndarray) -> float:
    """U_eq = trace(U_cart)/3 [Å²]."""
    return float(np.trace(np.asarray(u_cart, dtype=float)) / 3.0)


def principal_axes(u_cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(eigenvalues ascending [Å²], unit eigenvectors as columns) of the symmetric U_cart."""
    u = np.asarray(u_cart, dtype=float)
    w, v = np.linalg.eigh(0.5 * (u + u.T))
    return w, v


def ellipsoid_axes(u_cart: np.ndarray, probability: float = 50.0) -> tuple[np.ndarray, np.ndarray, bool]:
    """Semi-axes [Å] (ascending), axis directions (columns, Cartesian) and positive-definiteness.

    Semi-axis = c(p)·√(eigenvalue); a non-positive eigenvalue gives NaN for that axis
    (non-positive-definite ADP, the ellipsoid cannot be drawn).
    """
    w, v = principal_axes(u_cart)
    pd_ok = bool(np.all(w > 0.0))
    with np.errstate(invalid="ignore"):
        semi = probability_scale(probability) * np.where(w > 0.0, np.sqrt(np.clip(w, 0.0, None)), np.nan)
    return semi, v, pd_ok


def _sym_u(vals: dict[str, float]) -> np.ndarray:
    u = np.zeros((3, 3))
    for ij, x in vals.items():
        i, j = _IDX[ij]
        u[i, j] = u[j, i] = x
    return u


# --- reading ------------------------------------------------------------------------
def _block(cif_path: str | Path) -> dict[str, Any]:
    text = Path(cif_path).read_text(encoding="utf-8", errors="replace")
    cf = CifFile.from_str(text)
    blocks = list(cf.data.values())
    if not blocks:
        raise ScientificValidationError(f"No data block in {cif_path}.")
    for blk in blocks:
        if "_atom_site_label" in blk.data or "_atom_site_aniso_label" in blk.data:
            return blk.data
    return blocks[0].data


def _col(data: dict[str, Any], key: str) -> list[Any] | None:
    for k, v in data.items():
        if k.lower() == key.lower():
            return v if isinstance(v, list) else [v]
    return None


def read_cif_adp(cif_path: str | Path) -> dict[str, Any]:
    """Displacement parameters of the asymmetric unit, straight from the CIF.

    Returns ``{"lattice": Lattice, "symops": [SymmOp], "atoms": [...], "notes": [...]}``.
    Each atom: ``label``, ``element``, ``frac`` (np.ndarray or None), ``adp_type``
    ('Uani' / 'Uiso' / None), ``source`` ('U', 'B', 'beta', 'Uiso', 'Biso' or None),
    ``u_star`` (3×3, None without ADP), ``u_cif`` [Å²], ``u_cart`` [Å²],
    ``u_iso_cif`` (the U_iso_or_equiv column, B converted, or None).
    """
    data = _block(cif_path)
    notes: list[str] = []
    try:
        lattice = Lattice.from_parameters(*[
            _num(_col(data, f"_cell_{k}")[0]) for k in
            ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")
        ])
    except (TypeError, IndexError) as exc:
        raise ScientificValidationError(f"Cell parameters missing in {cif_path}.") from exc

    ops: list[SymmOp] = []
    for key in _SYMOP_KEYS:
        col = _col(data, key)
        if col:
            for s in col:
                try:
                    ops.append(SymmOp.from_xyz_str(str(s).strip().strip("'\"").replace(" ", "")))
                except Exception:  # noqa: BLE001 - odd symop text: skip, noted below
                    notes.append(f"Symmetry operation {s!r} could not be read.")
            break
    if not ops:
        ops = [SymmOp.from_xyz_str("x,y,z")]
        notes.append("No symmetry operations in the CIF: P1 assumed.")

    labels = [str(x) for x in (_col(data, "_atom_site_label") or [])]
    types = _col(data, "_atom_site_type_symbol") or [None] * len(labels)
    fx, fy, fz = (_col(data, f"_atom_site_fract_{c}") for c in "xyz")
    u_iso = _col(data, "_atom_site_U_iso_or_equiv")
    b_iso = _col(data, "_atom_site_B_iso_or_equiv")
    adp_t = _col(data, "_atom_site_adp_type") or _col(data, "_atom_site_thermal_displace_type")

    atoms: dict[str, dict[str, Any]] = {}
    for i, lab in enumerate(labels):
        frac = None
        if fx and fy and fz:
            xyz = [_num(fx[i]), _num(fy[i]), _num(fz[i])]
            frac = None if None in xyz else np.array(xyz, dtype=float)
        uiso, src = None, None
        if u_iso is not None and _num(u_iso[i]) is not None:
            uiso, src = _num(u_iso[i]), "Uiso"
        elif b_iso is not None and _num(b_iso[i]) is not None:
            uiso, src = _num(b_iso[i]) / (8.0 * math.pi**2), "Biso"
        atype = str(adp_t[i]).strip() if adp_t and str(adp_t[i]) not in (".", "?") else None
        atoms[lab] = {
            "label": lab, "element": _element(lab, types[i] if i < len(types) else None), "frac": frac,
            "adp_type": None, "source": None, "u_star": None, "u_iso_cif": uiso, "_iso_src": src,
            "_declared": atype,
        }

    # anisotropic loop (U preferred over B over beta when several are given)
    aniso_labels = [str(x) for x in (_col(data, "_atom_site_aniso_label") or [])]
    aniso_types = _col(data, "_atom_site_aniso_type_symbol") or [None] * len(aniso_labels)
    found: dict[str, list[list[Any]]] = {}
    for kind in ("U", "B", "beta"):
        cols = [_col(data, f"_atom_site_aniso_{kind}_{ij}") for ij in _IJ]
        if all(c is not None for c in cols):
            found[kind] = cols
    if len(found) > 1:
        notes.append(f"Both {' and '.join(found)} anisotropic tensors are given: {next(iter(found))} is used.")
    for k, lab in enumerate(aniso_labels):
        for kind, cols in found.items():
            vals = {ij: _num(c[k]) for ij, c in zip(_IJ, cols)}
            if any(v is None for v in vals.values()):
                continue
            u = _sym_u(vals)
            if lab not in atoms:
                atoms[lab] = {"label": lab, "element": _element(lab, aniso_types[k]), "frac": None,
                              "u_iso_cif": None, "_iso_src": None, "_declared": None}
                notes.append(f"Aniso label {lab!r} is not in the _atom_site loop (no position).")
            if kind == "U":
                u_star = ucif_to_ustar(u, lattice)
            elif kind == "B":
                u_star = ucif_to_ustar(u / (8.0 * math.pi**2), lattice)
            else:
                u_star = u / (2.0 * math.pi**2)
            atoms[lab].update(adp_type="Uani", source=kind, u_star=u_star)
            break

    out: list[dict[str, Any]] = []
    for a in atoms.values():
        if a.get("u_star") is None and a["u_iso_cif"] is not None:
            a.update(adp_type="Uiso", source=a["_iso_src"],
                     u_star=ucart_to_ustar(a["u_iso_cif"] * np.eye(3), lattice))
            if a["_declared"] and a["_declared"].lower().endswith("ani"):
                notes.append(f"{a['label']}: adp_type {a['_declared']} but no aniso values; U_iso used.")
        if a.get("u_star") is None:
            a.update(adp_type=None, source=None, u_star=None)
        a["u_cif"] = None if a["u_star"] is None else ustar_to_ucif(a["u_star"], lattice)
        a["u_cart"] = None if a["u_star"] is None else ustar_to_ucart(a["u_star"], lattice)
        a.pop("_iso_src", None)
        a.pop("_declared", None)
        out.append(a)
    if not any(a["u_star"] is not None for a in out):
        notes.append("No displacement parameters (U/B/beta, aniso or iso) in the CIF.")
    return {"lattice": lattice, "symops": ops, "atoms": out, "notes": notes}


# --- mapping onto a Structure ---------------------------------------------------------
def _match_sites(structure: Structure, adp: dict[str, Any]) -> tuple[list[tuple[dict, np.ndarray] | None], list[str]]:
    """For each structure site: (asym atom, rotation R of x' = R x + t) or None."""
    notes: list[str] = []
    cif_lat: Lattice = adp["lattice"]
    lat = structure.lattice
    if not (np.allclose(lat.abc, cif_lat.abc, rtol=1e-3) and np.allclose(lat.angles, cif_lat.angles, atol=0.05)):
        notes.append("The structure cell differs from the CIF cell: sites matched in fractional coordinates "
                     "of the structure cell, the U tensors may not apply.")
    images: list[tuple[dict, np.ndarray, np.ndarray]] = []  # (atom, R, frac image)
    for a in adp["atoms"]:
        if a["frac"] is None or a["u_star"] is None:
            continue
        for op in adp["symops"]:
            images.append((a, op.rotation_matrix, op.operate(a["frac"])))
    result: list[tuple[dict, np.ndarray] | None] = []
    n_miss = 0
    for site in structure:
        best, best_d = None, _MATCH_TOL
        label = getattr(site, "label", None)
        for prefer_label in (True, False):
            for a, rot, f in images:
                if prefer_label and label is not None and a["label"] != label:
                    continue
                d = f - site.frac_coords
                d -= np.round(d)
                dist = float(np.linalg.norm(lat.get_cartesian_coords(d)))
                if dist < best_d:
                    best, best_d = (a, rot), dist
            if best is not None:
                break
        if best is None:
            n_miss += 1
        result.append(best)
    if n_miss:
        notes.append(f"{n_miss} structure sites have no displacement parameters in the CIF.")
    return result, notes


def site_displacement_tensors(
    structure: Structure, cif_path: str | Path | None = None, adp: dict[str, Any] | None = None
) -> tuple[list[np.ndarray | None], list[str | None], list[str]]:
    """U_cart [Å²] for every site of ``structure`` (Cartesian frame of ``structure.lattice``).

    Symmetry-equivalent sites get ``R U* Rᵀ`` of their asymmetric-unit atom.
    Returns ``(tensors, labels, notes)``; a site without ADP gets ``None``.
    Give ``cif_path`` or an already read ``adp`` (``read_cif_adp``).
    """
    if adp is None:
        if cif_path is None:
            raise ScientificValidationError("Give cif_path or adp.")
        adp = read_cif_adp(cif_path)
    matches, notes = _match_sites(structure, adp)
    tensors: list[np.ndarray | None] = []
    labels: list[str | None] = []
    for m in matches:
        if m is None:
            tensors.append(None)
            labels.append(None)
            continue
        a, rot = m
        tensors.append(ustar_to_ucart(rot @ a["u_star"] @ rot.T, structure.lattice))
        labels.append(a["label"])
    return tensors, labels, list(adp["notes"]) + notes


def attach_ucif_site_properties(structure: Structure, cif_path: str | Path) -> Structure:
    """Copy of ``structure`` with the site properties U11_cif ... U12_cif [Å²] (pymatgen's names).

    ``core.structure_factors.compute_structure_factors`` (pymatgen ``get_ustar``) then uses
    the anisotropic Debye-Waller factors. Every site must have ADPs.
    """
    tensors, _, _ = site_displacement_tensors(structure, cif_path)
    missing = [i for i, t in enumerate(tensors) if t is None]
    if missing:
        raise ScientificValidationError(f"No displacement parameters for sites {missing}.")
    s = structure.copy()
    ucif = [ustar_to_ucif(ucart_to_ustar(t, s.lattice), s.lattice) for t in tensors]
    for ij in ("11", "22", "33", "23", "13", "12"):
        i, j = _IDX[ij]
        s.add_site_property(f"U{ij}_cif", [float(u[i, j]) for u in ucif])
    return s


# --- table ----------------------------------------------------------------------------
def analyze_displacement(
    cif_path: str | Path, probability: float = 50.0, structure: Structure | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Principal axes / U_eq per asymmetric-unit atom (``DISPLACEMENT_COLUMNS``) and a report.

    Axis vectors are Cartesian (pymatgen ``Lattice.from_parameters`` frame: a along x,
    b in the xy plane). ``structure`` (optional) only fills ``n_sites_in_cell``.
    """
    scale = probability_scale(probability)
    adp = read_cif_adp(cif_path)
    notes = list(adp["notes"])
    counts: dict[str, int] = {}
    if structure is not None:
        _, labels, n = site_displacement_tensors(structure, adp=adp)
        notes += n[len(adp["notes"]):]
        for lab in labels:
            if lab is not None:
                counts[lab] = counts.get(lab, 0) + 1
    rows: list[dict[str, Any]] = []
    npd: list[str] = []
    for a in adp["atoms"]:
        row: dict[str, Any] = {c: np.nan for c in DISPLACEMENT_COLUMNS}
        row.update(label=a["label"], element=a["element"], adp_type=a["adp_type"] or "",
                   source=a["source"] or "", positive_definite=False,
                   U_iso_cif=np.nan if a["u_iso_cif"] is None else a["u_iso_cif"],
                   n_sites_in_cell=counts.get(a["label"], 0) if structure is not None else np.nan)
        if a["u_cart"] is not None:
            for ij in _IJ:
                i, j = _IDX[ij]
                row[f"U{ij}"] = float(a["u_cif"][i, j])
            w, v = principal_axes(a["u_cart"])
            ok = bool(np.all(w > 0.0))
            row["U_eq"] = u_equivalent(a["u_cart"])
            for k in range(3):
                row[f"U{k + 1}"] = float(w[k])
                row[f"rms{k + 1}"] = float(math.sqrt(w[k])) if w[k] > 0 else np.nan
                row[f"semi_axis{k + 1}"] = scale * row[f"rms{k + 1}"]
                for c, x in zip("xyz", v[:, k]):
                    row[f"axis{k + 1}_{c}"] = float(x)
            row["anisotropy"] = float(w[2] / w[0]) if w[0] > 0 else np.nan
            row["positive_definite"] = ok
            if not ok:
                npd.append(a["label"])
        rows.append(row)
    df = pd.DataFrame(rows, columns=DISPLACEMENT_COLUMNS)
    if npd:
        notes.append(f"Non-positive-definite ADPs (no ellipsoid): {', '.join(npd)}.")
    report: dict[str, Any] = {
        "cif": str(cif_path),
        "probability_percent": float(probability) * (100.0 if float(probability) < 1.0 else 1.0),
        "radius_scale": scale,
        "n_atoms": int(len(df)),
        "n_anisotropic": int((df["adp_type"] == "Uani").sum()),
        "n_isotropic": int((df["adp_type"] == "Uiso").sum()),
        "n_without_adp": int((df["adp_type"] == "").sum()),
        "non_positive_definite": npd,
        "conventions": (
            "U11..U23 = U_cif [A^2] (B, beta and B_iso converted); U_cart = A N U_cif N^T A^T "
            "(Grosse-Kunstleve & Adams 2002); U_eq = trace(U_cart)/3; U1<=U2<=U3 = eigenvalues of U_cart, "
            "rms = sqrt(U); semi_axis = radius_scale*rms, radius_scale = sqrt(chi2_3^-1(p)); "
            "axis vectors Cartesian (a along x, b in the xy plane)."
        ),
        "columns": list(df.columns),
        "atoms": df.replace({np.nan: None}).to_dict(orient="records"),
        "warnings": notes,
    }
    return df, report


def write_displacement_report(
    df: pd.DataFrame, report: dict[str, Any], output_dir: str | Path, stem: str = "displacement"
) -> dict[str, str]:
    """Write ``displacement.csv`` / ``displacement.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out / f"{stem}.csv", out / f"{stem}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}

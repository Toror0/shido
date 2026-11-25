"""Structure factor table F(hkl) (VESTA "Structure factors"), X-ray or neutron.

    F(h) = Σ_j occ_j · f_j(s) · T_j(h) · exp(2πi h·r_j),   s = sinθ/λ = 1 / (2d)

* X-ray: f_j(s) = Z − 41.78214 s² Σ_k a_k exp(−b_k s²) with the coefficients pymatgen's
  ``XRDCalculator`` uses (``ATOMIC_SCATTERING_PARAMS``), no anomalous dispersion. F in electrons.
* Neutron: f_j = b_j, the bound coherent scattering length of pymatgen's ``NDCalculator``
  (``ATOMIC_SCATTERING_LEN``). F in fm.
* T_j = exp(−B s²) with B [Å²] per element (``debye_waller_factors``, same meaning as in
  ``xrd.compute_xrd``), or the anisotropic exp(−2π² hᵀU*h) when the structure carries the
  U11_cif … U12_cif site properties (as pymatgen does).
* Partial occupancy: occupancy-weighted sum per site (average crystal), as ``xrd.py``.

The sign convention, the form factors and the Lorentz-polarization factor
(1 + cos²2θ)/(sin²θ cosθ) (neutron: 1/(sin²θ cosθ)) are those of pymatgen, so
``multiplicity · |F|² · Lp`` reproduces the relative intensities of ``xrd.compute_xrd``.

Reflections are grouped into families with the Laue group of the cell as given
(spglib rotations + Friedel pair), so ``multiplicity`` is the true number of equivalent hkl
(for non-cubic cells it can differ from pymatgen's permutation-based count). Zero
reflections are marked in ``extinct``: ``"systematic"`` = forbidden by the space group
(lattice centring, screw axes, glide planes; checked with an atom on a general position),
``"structure"`` = allowed by the space group but zero for this structure (atoms on special
positions, e.g. Si (2 2 2) with spherical atoms).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.core import get_anisotropic_debye_waller_factors, get_ustar
from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS
from pymatgen.core import Structure

from shido.core.validator import ScientificValidationError, validate_structure

SF_COLUMNS: list[str] = [
    "h", "k", "l", "hkl", "d_angstrom", "two_theta_deg", "multiplicity",
    "F_real", "F_imag", "F_abs", "phase_deg", "F_squared", "lp_factor", "intensity", "intensity_rel",
    "extinct",
]

_DEFAULT_DMIN = 0.8  # Å
_DEFAULT_WAVELENGTH = "CuKa"
_ZERO_REL = 1e-6  # |F| below this fraction of Σ occ|f| counts as zero
_GENERIC_POINTS = ((0.1234, 0.2718, 0.3779), (0.4141, 0.0577, 0.1732))


def _normalize_radiation(radiation: str | None) -> str:
    rad = str(radiation or "xray").strip().lower().replace("-", "")
    rad = {"x": "xray", "xrays": "xray", "n": "neutron", "neutrons": "neutron"}.get(rad, rad)
    if rad not in ("xray", "neutron"):
        raise ScientificValidationError(f"radiation must be 'xray' or 'neutron', got {radiation!r}.")
    return rad


def _symmetry(structure: Structure, symprec: float) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], str | None, int | None, list[str]]:
    """Rotations (fractional, for hkl: h' = h R) and full ops (R, t) of the cell as given."""
    notes: list[str] = []
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        ops = sga.get_symmetry_operations(cartesian=False)
        full = [(np.asarray(o.rotation_matrix, dtype=float), np.asarray(o.translation_vector, dtype=float)) for o in ops]
        symbol, number = sga.get_space_group_symbol(), int(sga.get_space_group_number())
    except Exception as exc:  # noqa: BLE001 - fall back to P-1 Laue group (Friedel only)
        notes.append(f"Symmetry search failed ({exc}); families use only the Friedel pair, extinct = 'structure' only.")
        full, symbol, number = [], None, None
    rots = [np.rint(r).astype(int) for r, _ in full] or [np.eye(3, dtype=int)]
    return rots, full, symbol, number, notes


def _scattering_terms(structure: Structure, rad: str, dw: dict[str, float]) -> tuple[list[dict[str, Any]], list[str]]:
    """One entry per element: coefficients / scattering length, B, sites (frac, occ, index)."""
    groups: dict[str, dict[str, Any]] = {}
    for idx, site in enumerate(structure):
        for sp, occ in site.species.items():
            sym = getattr(sp, "symbol", str(sp))
            table = ATOMIC_SCATTERING_LEN if rad == "neutron" else ATOMIC_SCATTERING_PARAMS
            if sym not in table:
                kind = "neutron scattering length" if rad == "neutron" else "X-ray scattering coefficients"
                raise ScientificValidationError(f"No {kind} for {sym!r} (site {idx}).")
            g = groups.setdefault(sym, {
                "element": sym,
                "z": getattr(sp, "Z", 0),
                "coeff": None if rad == "neutron" else np.asarray(ATOMIC_SCATTERING_PARAMS[sym], dtype=float),
                "b_fm": float(ATOMIC_SCATTERING_LEN[sym]) if rad == "neutron" else None,
                "B": float(dw.get(sym, 0.0)),
                "frac": [], "occ": [], "sites": [],
            })
            g["frac"].append(np.asarray(site.frac_coords, dtype=float))
            g["occ"].append(float(occ))
            g["sites"].append(idx)
    unknown = sorted(set(dw) - set(groups))
    notes = [f"debye_waller_factors given for elements not in the structure: {unknown}"] if unknown else []
    return list(groups.values()), notes


def atomic_scattering_factor(element: str, s: Any, radiation: str = "xray", b_iso: float = 0.0) -> np.ndarray:
    """f(s) (electrons) or b (fm) of one element at s = sinθ/λ [1/Å], times exp(−B s²)."""
    rad = _normalize_radiation(radiation)
    s2 = np.asarray(s, dtype=float) ** 2
    if rad == "neutron":
        f = np.full_like(s2, float(ATOMIC_SCATTERING_LEN[element]))
    else:
        from pymatgen.core import Element

        c = np.asarray(ATOMIC_SCATTERING_PARAMS[element], dtype=float)
        f = Element(element).Z - 41.78214 * s2 * np.sum(c[:, 0] * np.exp(-c[:, 1] * s2[..., None]), axis=-1)
    return f * np.exp(-float(b_iso) * s2)


def _hkl_text(hkl: tuple[int, int, int], hexagonal: bool) -> str:
    h, k, l = (int(i) for i in hkl)
    idx = (h, k, -h - k, l) if hexagonal else (h, k, l)
    return "(" + " ".join(str(i) for i in idx) + ")"


def compute_structure_factors(
    structure: Structure,
    dmin: float | None = None,
    radiation: str = "xray",
    wavelength: str | float | None = None,
    debye_waller_factors: dict[str, float] | None = None,
    include_extinct: bool = True,
    symprec: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Table of F(hkl) for all families with d >= dmin.

    Args:
        structure: pymatgen Structure (cell used as given; partial occupancy allowed).
        dmin: smallest d-spacing in Å (default 0.8). Reflections with d < λ/2 are listed
            but have no 2θ / Lp / intensity.
        radiation: 'xray' (form factors, F in e) or 'neutron' (scattering lengths, F in fm).
        wavelength: source key ('CuKa', 'MoKa', ...) or Å; default settings ``xrd.wavelength``
            (CuKa). Only 2θ, Lp and the intensity depend on it.
        debye_waller_factors: {element: B [Å²]}, T = exp(−B s²) on F.
        include_extinct: False drops the zero reflections from the table.
        symprec: spglib tolerance for the Laue group and the space-group absences.

    Returns:
        (DataFrame in ``SF_COLUMNS`` order plus ``f_<El>`` columns, report dict). One row per
        family, representative = fewest negative indices then largest (h, k, l), sorted by d (large first).
        ``intensity`` = multiplicity · |F|² · Lp, ``intensity_rel`` scaled to max 100.
    """
    from shido.xrd import _resolve_wavelength, _setting  # same wavelength keys / settings as xrd.py

    validate_structure(structure)
    rad = _normalize_radiation(radiation)
    if wavelength is None:
        wavelength = _setting("xrd.wavelength", _DEFAULT_WAVELENGTH)
    lam, lam_label = _resolve_wavelength(wavelength)
    dmin = _DEFAULT_DMIN if dmin is None else float(dmin)
    if not math.isfinite(dmin) or dmin <= 0.0:
        raise ScientificValidationError(f"dmin must be a positive d-spacing in Å, got {dmin!r}.")
    dw = {str(k): float(v) for k, v in (debye_waller_factors or {}).items()}
    ustar = get_ustar(structure)
    if ustar is not None and dw:
        raise ScientificValidationError("Give either debye_waller_factors or the U11_cif ... site properties, not both.")

    notes: list[str] = []
    if rad == "xray" and dmin < 0.25:
        notes.append("The form-factor fit of pymatgen is meant for s = 1/(2d) <= 2 1/Å (d >= 0.25 Å).")
    groups, n = _scattering_terms(structure, rad, dw)
    notes += n
    rots, full_ops, sg_symbol, sg_number, n = _symmetry(structure, symprec)
    notes += n

    lattice = structure.lattice
    hexagonal = lattice.is_hexagonal()
    recip = lattice.reciprocal_lattice_crystallographic
    pts = recip.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.0 / dmin + 1e-9)
    hkls = np.rint(np.array([p[0] for p in pts], dtype=float)).astype(int).reshape(-1, 3)
    gs = np.array([p[1] for p in pts], dtype=float)
    keep = gs > 1e-9
    hkls, gs = hkls[keep], gs[keep]

    # --- families (Laue group = rotations ∪ −rotations) -------------------------------
    rot_stack = np.stack(rots)  # (n_ops, 3, 3)
    fam_of: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    count: dict[tuple[int, int, int], int] = {}
    for hkl in hkls:
        t = tuple(int(i) for i in hkl)
        if t in fam_of:
            continue
        orbit = np.einsum("i,nij->nj", hkl, rot_stack)
        orbit_set = {tuple(int(i) for i in o) for o in np.vstack([orbit, -orbit])}
        # fewest negative indices first, then the largest tuple: (1 1 1), (1 1 -2 0) in hexagonal
        rep = max(orbit_set, key=lambda o: (-sum(i < 0 for i in o), o))
        for o in orbit_set:
            fam_of[o] = rep
    for hkl in hkls:
        rep = fam_of[tuple(int(i) for i in hkl)]
        count[rep] = count.get(rep, 0) + 1
    reps = np.array(sorted(count), dtype=int).reshape(-1, 3)
    g_rep = np.array([np.linalg.norm(recip.get_cartesian_coords(r)) for r in reps])
    s = g_rep / 2.0
    s2 = s * s

    # --- F(h) -------------------------------------------------------------------------
    F = np.zeros(len(reps), dtype=np.complex128)
    f_cols: dict[str, np.ndarray] = {}
    scale = np.zeros(len(reps))  # Σ occ |f| for the zero test
    hf = reps.astype(float)
    for g in groups:
        if rad == "neutron":
            f = np.full(len(reps), g["b_fm"])
        else:
            f = g["z"] - 41.78214 * s2 * np.sum(g["coeff"][:, 0] * np.exp(-g["coeff"][:, 1] * s2[:, None]), axis=1)
        if g["B"]:
            f = f * np.exp(-g["B"] * s2)
        f_cols[g["element"]] = f
        phase = np.exp(2j * math.pi * (hf @ np.asarray(g["frac"]).T))  # (M, m)
        if ustar is not None:
            phase = phase * get_anisotropic_debye_waller_factors(hf, ustar[g["sites"]])
        occ = np.asarray(g["occ"])
        F += f * (phase @ occ)
        scale += np.abs(f) * occ.sum()
    zero = np.abs(F) <= _ZERO_REL * np.maximum(scale, 1e-30)

    # space-group absences: F of a general-position orbit (unit atoms)
    sys_abs = np.zeros(len(reps), dtype=bool)
    if full_ops:
        sys_abs[:] = True
        for x0 in _GENERIC_POINTS:
            pos = np.array([(r @ np.asarray(x0) + t) % 1.0 for r, t in full_ops])
            pos = np.unique(np.round(pos, 6) % 1.0, axis=0)
            fg = np.exp(2j * math.pi * (hf @ pos.T)).sum(axis=1)
            sys_abs &= np.abs(fg) <= _ZERO_REL * len(pos)
    extinct = np.where(zero & sys_abs, "systematic", np.where(zero, "structure", ""))

    F = np.where(zero, 0.0, F)
    re = np.where(np.abs(F.real) <= _ZERO_REL * scale, 0.0, F.real)
    im = np.where(np.abs(F.imag) <= _ZERO_REL * scale, 0.0, F.imag)
    F = re + 1j * im
    phase_deg = np.degrees(np.angle(F))
    phase_deg = np.where(phase_deg <= -180.0 + 1e-9, 180.0, phase_deg)
    phase_deg = np.where(zero, 0.0, phase_deg)
    F2 = (F * F.conjugate()).real

    d = 1.0 / g_rep
    sin_t = lam / (2.0 * d)
    reach = sin_t <= 1.0
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        theta = np.where(reach, np.arcsin(np.clip(sin_t, -1.0, 1.0)), np.nan)
        cos2t = np.cos(2 * theta)
        lp = 1.0 / (np.sin(theta) ** 2 * np.cos(theta))
        if rad == "xray":
            lp = lp * (1.0 + cos2t ** 2)
    mult = np.array([count[tuple(int(i) for i in r)] for r in reps])
    inten = mult * F2 * lp
    i_max = float(np.nanmax(inten)) if np.any(np.isfinite(inten)) else 0.0
    inten_rel = inten * (100.0 / i_max) if i_max > 0.0 else inten

    df = pd.DataFrame({
        "h": reps[:, 0], "k": reps[:, 1], "l": reps[:, 2],
        "hkl": [_hkl_text(tuple(r), hexagonal) for r in reps],
        "d_angstrom": np.round(d, 6),
        "two_theta_deg": np.round(np.degrees(2 * theta), 4),
        "multiplicity": mult,
        "F_real": np.round(re, 5), "F_imag": np.round(im, 5), "F_abs": np.round(np.abs(F), 5),
        "phase_deg": np.round(phase_deg, 3), "F_squared": np.round(F2, 4),
        "lp_factor": np.round(lp, 5), "intensity": inten, "intensity_rel": np.round(inten_rel, 4),
        "extinct": extinct,
    })
    for el, f in f_cols.items():
        df[f"f_{el}"] = np.round(f, 5)
    df = df.assign(_neg=[tuple(-int(i) for i in r) for r in reps])
    df = df.sort_values(["d_angstrom", "_neg"], ascending=[False, True], kind="stable").drop(columns="_neg")
    if not include_extinct:
        df = df[df["extinct"] == ""]
    df = df.reset_index(drop=True)

    if not reach.all():
        notes.append(f"{int((~reach).sum())} families have d < λ/2 = {lam / 2:.4f} Å: no 2θ / Lp / intensity.")
    if any(not site.is_ordered for site in structure):
        notes.append("Partial occupancies: occupancy-weighted scattering per site (average crystal).")
    unit = "fm" if rad == "neutron" else "electrons"
    report: dict[str, Any] = {
        "formula": structure.composition.reduced_formula,
        "radiation": rad,
        "F_unit": unit,
        "scattering": (
            {g["element"]: g["b_fm"] for g in groups} if rad == "neutron"
            else "pymatgen XRDCalculator atomic scattering factors (no anomalous dispersion)"
        ),
        "wavelength_angstrom": lam,
        "wavelength_label": lam_label,
        "dmin_angstrom": dmin,
        "debye_waller_factors": dw or None,
        "anisotropic_U": ustar is not None,
        "spacegroup_symbol": sg_symbol,
        "spacegroup_number": sg_number,
        "n_families": int(len(df)),
        "n_extinct_systematic": int((df["extinct"] == "systematic").sum()),
        "n_extinct_structure": int((df["extinct"] == "structure").sum()),
        "conventions": (
            "F = Σ occ·f·T·exp(2πi h·r) for the cell as given; phase in deg (−180, 180]; "
            "intensity = multiplicity·|F|²·Lp, Lp = (1+cos²2θ)/(sin²θ cosθ) (neutron: 1/(sin²θ cosθ)); "
            "extinct: 'systematic' = space-group absence, 'structure' = zero from the atom positions."
        ),
        "columns": list(df.columns),
        "reflections": df.replace({np.nan: None}).to_dict(orient="records"),
        "warnings": notes,
    }
    return df, report


def write_structure_factor_report(
    df: pd.DataFrame, report: dict[str, Any], output_dir: str | Path, stem: str | None = None
) -> dict[str, str]:
    """Write ``structure_factors.csv`` / ``.json`` (neutron: ``structure_factors_neutron.*``)."""
    if stem is None:
        stem = "structure_factors_neutron" if report.get("radiation") == "neutron" else "structure_factors"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out / f"{stem}.csv", out / f"{stem}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}

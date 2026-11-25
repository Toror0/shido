"""Coordination-polyhedron metrics, bond valence sums and Madelung energy
(VESTA-like "polyhedron information", "bond valence sum" and MADEL).

Bonds are exactly those of ``analyzer.compute_bonds`` (CrystalNN, MinimumDistanceNN
fallback, or ``bond_specs`` element-pair ranges), so the numbers match
``bond_lengths.csv`` and the drawn polyhedra.

Per site (centre atom with n bonded neighbours, bond lengths l_i):

- ``volume_A3``: convex-hull volume of the n neighbours (n >= 4, not flat).
- ``mean_bond_length_A``: l_av = mean(l_i).
- ``distortion_index``: Baur (1974) D = (1/n) sum |l_i - l_av| / l_av.
- ``quadratic_elongation``: Robinson et al. (1971) <lambda> = (1/n) sum (l_i/l_0)^2,
  l_0 = centre-to-vertex distance of the regular polyhedron of the same volume.
  Only for tetrahedra (n=4), octahedra (n=6) and cubes (n=8).
- ``bond_angle_variance_deg2``: sigma^2 = sum (theta_i - theta_0)^2 / (m - 1) over the
  m polyhedron-edge angles (the m smallest vertex-centre-vertex angles; m = 6, 12, 12,
  theta_0 = 109.47, 90, 70.53 deg for n = 4, 6, 8).
- ``ECoN``: Hoppe (1979) effective coordination number,
  l_w = sum l_i w_i / sum w_i with w_i = exp(1 - (l_i/l_min)^6), ECoN = sum exp(1 - (l_i/l_w)^6).
- ``bvs``: bond valence sum over the same bonds, s_ij = exp((R0 - l_ij)/b), positive on
  the less electronegative atom, same-element bonds do not count. R0 from the small
  Brese & O'Keeffe (1991) ion-pair table `BV_R0` (b = 0.37) when the pair is in it
  (``bvs_method = "brese_okeeffe"``), else pymatgen's element parameters (R0 from r
  and c, b = 0.31, as ``pymatgen.analysis.bond_valence.calculate_bv_sum``;
  ``bvs_method = "pymatgen"``, rougher: NaCl gives 0.84 instead of 0.98).

Madelung (``madelung=True``): pymatgen ``EwaldSummation`` on the structure with
oxidation states (given, else the composition guess used by ``slab_cutter``, else
``BVAnalyzer``). Total and per-formula-unit energies (eV), site energies and site
potentials (V). For a two-element compound the Madelung constant referred to the
shortest cation-anion distance r_0 is also given:
M = -E_fu * r_0 / (k * |z_cation * z_anion|), k = e^2/(4 pi eps_0) = 14.399645 eV A
(NaCl 1.7476, CsCl 1.7627).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COULOMB_EV_A = 14.399645  # e^2 / (4 pi eps_0) in eV * Angstrom

# n -> (ideal angle in deg, number of edges, l_0 from volume V)
_REGULAR: dict[int, tuple[float, int, Any]] = {
    4: (math.degrees(math.acos(-1.0 / 3.0)), 6, lambda v: (v * 9.0 * math.sqrt(3.0) / 8.0) ** (1.0 / 3.0)),
    6: (90.0, 12, lambda v: (v * 3.0 / 4.0) ** (1.0 / 3.0)),
    8: (math.degrees(math.acos(1.0 / 3.0)), 12, lambda v: (v * 3.0 * math.sqrt(3.0) / 8.0) ** (1.0 / 3.0)),
}

POLYHEDRA_COLUMNS: list[str] = [
    "site", "element", "label", "coordination", "ligands", "volume_A3",
    "mean_bond_length_A", "min_bond_length_A", "max_bond_length_A", "distortion_index",
    "quadratic_elongation", "bond_angle_variance_deg2", "ECoN", "bvs", "bvs_method",
    "oxidation_state", "site_energy_eV", "site_potential_V",
]


def _hull_volume(points: np.ndarray) -> float | None:
    if len(points) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(points).volume)
    except Exception:  # flat / degenerate set of neighbours
        return None


def polyhedron_metrics(center: np.ndarray, neighbors: np.ndarray) -> dict[str, Any]:
    """Metrics of one polyhedron from Cartesian centre and neighbour positions (A)."""
    center = np.asarray(center, dtype=float)
    pts = np.asarray(neighbors, dtype=float).reshape(-1, 3)
    n = len(pts)
    out: dict[str, Any] = {
        "coordination": n, "volume_A3": None, "mean_bond_length_A": None,
        "min_bond_length_A": None, "max_bond_length_A": None, "distortion_index": None,
        "quadratic_elongation": None, "bond_angle_variance_deg2": None, "ECoN": None,
    }
    if n == 0:
        return out
    vecs = pts - center
    lengths = np.linalg.norm(vecs, axis=1)
    l_av = float(lengths.mean())
    l_min = float(lengths.min())
    out["mean_bond_length_A"] = l_av
    out["min_bond_length_A"] = l_min
    out["max_bond_length_A"] = float(lengths.max())
    out["distortion_index"] = float(np.mean(np.abs(lengths - l_av)) / l_av)
    w = np.exp(1.0 - (lengths / l_min) ** 6)
    l_w = float(np.sum(lengths * w) / np.sum(w))
    out["ECoN"] = float(np.sum(np.exp(1.0 - (lengths / l_w) ** 6)))
    vol = _hull_volume(pts)
    out["volume_A3"] = vol
    if n in _REGULAR:
        theta0, m_edges, l0_of_v = _REGULAR[n]
        if vol is not None and vol > 0:
            l0 = l0_of_v(vol)
            out["quadratic_elongation"] = float(np.mean((lengths / l0) ** 2))
        unit = vecs / lengths[:, None]
        angles = []
        for i in range(n):
            for j in range(i + 1, n):
                c = float(np.clip(np.dot(unit[i], unit[j]), -1.0, 1.0))
                angles.append(math.degrees(math.acos(c)))
        edge_angles = np.sort(np.asarray(angles))[:m_edges]
        out["bond_angle_variance_deg2"] = float(np.sum((edge_angles - theta0) ** 2) / (m_edges - 1))
    return out


# Brese & O'Keeffe, Acta Cryst. B47 (1991) 192: R0 (A) for cation-anion pairs, b = 0.37.
# Only cations with one common oxidation state (so no Fe/Ti/Mn ambiguity).
BV_R0: dict[tuple[str, str], float] = {
    ("Li", "O"): 1.466, ("Na", "O"): 1.803, ("K", "O"): 2.132, ("Mg", "O"): 1.693,
    ("Ca", "O"): 1.967, ("Sr", "O"): 2.118, ("Ba", "O"): 2.29, ("Al", "O"): 1.651,
    ("Si", "O"): 1.624, ("Zn", "O"): 1.704, ("Ca", "F"): 1.842, ("Na", "Cl"): 2.15,
    ("K", "Cl"): 2.519, ("Cs", "Cl"): 2.791,
}
BV_B_TABLE = 0.37


def bond_valence_table(elem_a: str, elem_b: str, distance: float) -> float | None:
    """Signed bond valence from `BV_R0` (+ on the cation), or None if the pair is not in it."""
    if (elem_a, elem_b) in BV_R0:
        return math.exp((BV_R0[(elem_a, elem_b)] - distance) / BV_B_TABLE)
    if (elem_b, elem_a) in BV_R0:
        return -math.exp((BV_R0[(elem_b, elem_a)] - distance) / BV_B_TABLE)
    return None


def bond_valence(elem_a: str, elem_b: str, distance: float) -> float:
    """Signed bond valence of a-b at ``distance`` (A), pymatgen ``calculate_bv_sum`` rule.

    Positive when ``elem_a`` is the less electronegative atom, 0 for same element or
    when neither atom is in pymatgen's electronegative list / parameter table.
    """
    from pymatgen.analysis.bond_valence import BV_PARAMS, ELECTRONEG
    from pymatgen.core import Element

    e1, e2 = Element(elem_a), Element(elem_b)
    if e1 == e2 or not (e1 in ELECTRONEG or e2 in ELECTRONEG):
        return 0.0
    if e1 not in BV_PARAMS or e2 not in BV_PARAMS:
        return 0.0
    r1, r2 = BV_PARAMS[e1]["r"], BV_PARAMS[e2]["r"]
    c1, c2 = BV_PARAMS[e1]["c"], BV_PARAMS[e2]["c"]
    r_ij = r1 + r2 - r1 * r2 * (math.sqrt(c1) - math.sqrt(c2)) ** 2 / (c1 * r1 + c2 * r2)
    v = math.exp((r_ij - distance) / 0.31)
    return v if e1.X < e2.X else -v


def _oxidation_states(structure, given: dict[str, float] | None) -> tuple[dict[str, float] | None, str]:
    if given:
        return {str(k).strip().capitalize(): float(v) for k, v in given.items()}, "given"
    from shido.slab_cutter import guess_bulk_oxidation_states

    oxi = guess_bulk_oxidation_states(structure)
    if oxi:
        return oxi, "composition_guess"
    try:
        from pymatgen.analysis.bond_valence import BVAnalyzer

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = BVAnalyzer().get_oxi_state_decorated_structure(structure)
        out: dict[str, float] = {}
        for site in dec:
            out.setdefault(site.specie.symbol, float(site.specie.oxi_state))
        if any(abs(v) > 1e-6 for v in out.values()):
            return out, "bv_analyzer"
    except Exception:
        pass
    return None, "none"


def compute_madelung(structure, oxidation_states: dict[str, float] | None = None) -> dict[str, Any]:
    """Ewald (Madelung) energy of an ordered structure with oxidation states."""
    from pymatgen.analysis.ewald import EwaldSummation

    from shido.core.species import dominant_element

    oxi, source = _oxidation_states(structure, oxidation_states)
    res: dict[str, Any] = {"oxidation_states": oxi, "oxidation_source": source}
    if not oxi:
        res["error"] = "No oxidation states (give oxidation_states, e.g. {'Na': 1, 'Cl': -1})."
        return res
    s = structure.copy()
    elems = [dominant_element(site) for site in s]
    missing = sorted(set(elems) - set(oxi))
    if missing:
        res["error"] = f"No oxidation state for {missing}."
        return res
    s = s.__class__(s.lattice, elems, s.frac_coords)
    s.add_oxidation_state_by_element(oxi)
    charges = np.array([oxi[e] for e in elems], dtype=float)
    res["total_charge"] = float(charges.sum())
    if abs(res["total_charge"]) > 1e-6:
        res["warning"] = f"Cell is not neutral (total charge {res['total_charge']:+g}); a background charge is used."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ew = EwaldSummation(s)
        e_tot = float(ew.total_energy)
        site_e = [float(ew.get_site_energy(i)) for i in range(len(s))]
    comp = s.composition.remove_charges()
    _, n_fu = comp.get_reduced_composition_and_factor()
    res.update({
        "total_energy_eV": e_tot,
        "n_formula_units": float(n_fu),
        "energy_per_formula_unit_eV": e_tot / n_fu,
        "energy_per_atom_eV": e_tot / len(s),
        "site_energies_eV": site_e,
        "site_potentials_V": [2.0 * e / q if abs(q) > 1e-9 else None for e, q in zip(site_e, charges)],
        "madelung_constant": None, "r0_A": None,
    })
    cats = [e for e in oxi if oxi[e] > 0 and e in elems]
    ans = [e for e in oxi if oxi[e] < 0 and e in elems]
    if len(set(elems)) == 2 and len(cats) == 1 and len(ans) == 1:
        ci = [i for i, e in enumerate(elems) if e == cats[0]]
        ai = [i for i, e in enumerate(elems) if e == ans[0]]
        dm = s.distance_matrix  # minimum-image (periodic) distances
        r0 = float(dm[np.ix_(ci, ai)].min())
        zz = abs(oxi[cats[0]] * oxi[ans[0]])
        res["r0_A"] = r0
        res["madelung_constant"] = -res["energy_per_formula_unit_eV"] * r0 / (COULOMB_EV_A * zz)
    return res


def analyze_polyhedra(
    structure,
    bond_specs: Any = None,
    madelung: bool = False,
    oxidation_states: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Per-site polyhedron metrics + BVS (+ Madelung) using the `compute_bonds` bonds.

    Returns ``(df, report)``; ``df`` has `POLYHEDRA_COLUMNS` (one row per site of the
    ordered proxy, the same site numbering as ``bond_lengths.csv``).
    """
    from shido.analyzer import _compute_bonds_full, to_ordered_proxy
    from shido.core.species import dominant_element

    work = to_ordered_proxy(structure)
    _, _, _, nmap = _compute_bonds_full(structure, bond_specs)
    elems = [dominant_element(site) for site in work]
    mad = compute_madelung(work, oxidation_states) if madelung else None
    rows: list[dict[str, Any]] = []
    for i, site in enumerate(work):
        c = np.asarray(site.coords, dtype=float)
        uniq: dict[tuple[int, tuple[int, int, int]], np.ndarray] = {}
        for j, img, p_j in nmap.get(i, []):
            uniq.setdefault((int(j), tuple(int(x) for x in img)), np.asarray(p_j, dtype=float))
        keys = sorted(uniq, key=lambda k: (float(np.linalg.norm(uniq[k] - c)), k))
        pts = np.array([uniq[k] for k in keys]).reshape(-1, 3)
        row: dict[str, Any] = {"site": i, "element": elems[i], "label": getattr(site, "label", None) or ""}
        row.update(polyhedron_metrics(c, pts))
        lig: dict[str, int] = {}
        bvs = 0.0
        methods: set[str] = set()
        for k in keys:
            ej = elems[k[0]]
            lig[ej] = lig.get(ej, 0) + 1
            d = float(np.linalg.norm(uniq[k] - c))
            v = bond_valence_table(elems[i], ej, d)
            if v is None:
                v = bond_valence(elems[i], ej, d)
                if v != 0.0:
                    methods.add("pymatgen")
            else:
                methods.add("brese_okeeffe")
            bvs += v
        row["ligands"] = " ".join(f"{e}{n}" for e, n in sorted(lig.items()))
        row["bvs"] = bvs
        row["bvs_method"] = "+".join(sorted(methods))
        row["oxidation_state"] = None
        row["site_energy_eV"] = None
        row["site_potential_V"] = None
        if mad and "error" not in mad:
            row["oxidation_state"] = mad["oxidation_states"].get(elems[i])
            row["site_energy_eV"] = mad["site_energies_eV"][i]
            row["site_potential_V"] = mad["site_potentials_V"][i]
        rows.append(row)
    df = pd.DataFrame(rows, columns=POLYHEDRA_COLUMNS)
    report: dict[str, Any] = {
        "formula": work.composition.reduced_formula,
        "n_sites": len(work),
        "bond_specs": bond_specs,
        "sites": df.to_dict(orient="records"),
    }
    if mad is not None:
        mad = dict(mad)
        mad.pop("site_energies_eV", None)
        mad.pop("site_potentials_V", None)
        report["madelung"] = mad
    return df, report


def write_polyhedra_report(df: pd.DataFrame, report: dict[str, Any], output_dir: str | Path,
                           stem: str = "polyhedra") -> dict[str, str]:
    """Write ``<stem>.csv`` and ``<stem>.json``; returns their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out / f"{stem}.csv", out / f"{stem}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}

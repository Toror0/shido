"""Cut-off planes, molecules (connected components) and hydrogen bonds (VESTA batch 23).

* ``cutoff_planes`` "h k l@D": keep only atoms with h*x + k*y + l*z <= D (fractional
  coords, i.e. the side of the plane at D * d_hkl from the origin that holds the origin;
  "-1 -1 -1@-0.5" keeps the other side). Several planes = all must hold.
* `find_molecules`: covalent bonds (d <= tolerance * (r_cov,i + r_cov,j)) joined across
  periodic images into molecules; each molecule is unwrapped (contiguous) with its centre
  of mass in [0, 1). A component that bonds to its own image is "extended" (network solid).
* `find_hbonds` "D-H...A:dmax:angle_min": H covalently bonded to D, H...A <= dmax Å and
  angle D-H...A >= angle_min degrees.
"""

from __future__ import annotations

import csv
import math
import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CUTOFF_D = 0.5
DEFAULT_BOND_TOLERANCE = 1.15
DEFAULT_HBOND_DMAX = 2.5
DEFAULT_HBOND_ANGLE = 120.0
_PLANE_TOL = 1e-3


# ---------------------------------------------------------------- cut-off planes
def parse_cutoff_plane(spec: Any) -> tuple[np.ndarray, float]:
    """'1 1 1@0.5' / '1,1,1@0.5' / [1, 1, 1, 0.5] -> (hkl, D). D defaults to 0.5."""
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], np.ndarray):
        return np.asarray(spec[0], dtype=float), float(spec[1])  # already parsed
    if isinstance(spec, (list, tuple)) and not isinstance(spec, str):
        vals = [float(v) for v in spec]
        if len(vals) not in (3, 4):
            raise ValueError(f"cutoff plane {spec!r}: need h k l [D].")
        hkl, d = vals[:3], (vals[3] if len(vals) == 4 else DEFAULT_CUTOFF_D)
    else:
        text = str(spec).strip()
        head, _, tail = text.partition("@")
        parts = [p for p in re.split(r"[\s,]+", head.strip("() ")) if p]
        if len(parts) != 3:
            raise ValueError(f"cutoff plane {spec!r}: write 'h k l@D', e.g. '1 1 1@0.5'.")
        hkl = [float(p) for p in parts]
        d = float(tail) if tail.strip() else DEFAULT_CUTOFF_D
    h = np.asarray(hkl, dtype=float)
    if float(np.abs(h).sum()) < 1e-9:
        raise ValueError("cutoff plane: h k l must not be 0 0 0.")
    return h, float(d)


def normalize_cutoff_planes(planes: Any) -> list[tuple[np.ndarray, float]] | None:
    if planes is None:
        return None
    items = [planes] if isinstance(planes, str) else list(planes)
    out = [parse_cutoff_plane(p) for p in items if str(p).strip()]
    return out or None


def plane_keep_mask(structure: Any, carts: Any, planes: Any) -> np.ndarray:
    """True for every cartesian point on the kept side of all planes."""
    planes = normalize_cutoff_planes(planes)
    pts = np.asarray(carts, dtype=float).reshape(-1, 3)
    if not planes or len(pts) == 0:
        return np.ones(len(pts), dtype=bool)
    frac = np.asarray(structure.lattice.get_fractional_coords(pts), dtype=float).reshape(-1, 3)
    keep = np.ones(len(pts), dtype=bool)
    for hkl, d in planes:
        keep &= frac @ hkl <= d + _PLANE_TOL
    return keep


# ---------------------------------------------------------------- molecules
def _covalent_radius(symbol: str) -> float:
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

    return float(CovalentRadius.radius.get(symbol, 1.5))


def _site_symbol(site: Any) -> str:
    sp = max(site.species.items(), key=lambda kv: kv[1])[0]
    return str(getattr(sp, "symbol", None) or getattr(sp, "element", sp)).rstrip("+-0123456789")


def _site_label(structure: Any, idx: int) -> str:
    label = getattr(structure[idx], "label", None)
    sym = _site_symbol(structure[idx])
    if label and str(label).strip() and str(label).strip() != sym:
        return str(label).strip()
    return f"{sym}{idx + 1}"


def covalent_edges(structure: Any, tolerance: float = DEFAULT_BOND_TOLERANCE) -> list[tuple[int, int, np.ndarray, float]]:
    """(i, j, image, d): site j at frac_j + image is covalently bonded to site i (both directions)."""
    radii = [_covalent_radius(_site_symbol(s)) for s in structure]
    if not radii:
        return []
    cutoff = 2.0 * max(radii) * float(tolerance) + 0.01
    ci, pi, imgs, dists = structure.get_neighbor_list(r=cutoff)
    out = []
    for i, j, img, d in zip(ci, pi, imgs, dists):
        if d < 0.1 or d > float(tolerance) * (radii[int(i)] + radii[int(j)]):
            continue
        out.append((int(i), int(j), np.asarray(np.round(img), dtype=int), float(d)))
    return out


def _hill_formula(symbols: list[str]) -> str:
    """Hill order (C, H, then alphabetical; alphabetical without C), e.g. H2O, CO2, CH4."""
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    order = (["C", "H"] if "C" in counts else []) + sorted(k for k in counts if "C" not in counts or k not in ("C", "H"))
    return "".join(f"{el}{counts[el] if counts[el] != 1 else ''}" for el in order if el in counts)


def find_molecules(structure: Any, tolerance: float = DEFAULT_BOND_TOLERANCE) -> list[dict[str, Any]]:
    """Connected components of the covalent-bond graph, unwrapped across periodic images.

    Each molecule: ``sites`` (indices), ``offsets`` (integer image of each site so that the
    molecule is contiguous and its centre of mass lies in [0, 1)), ``formula``,
    ``centroid_frac`` / ``centroid_cart`` (mass-weighted), ``extended`` (bonded to its own
    periodic image: a 1D/2D/3D network, not a finite molecule), ``bonds`` (i, j, image).
    """

    n = len(structure)
    adj: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(n)}
    edges = covalent_edges(structure, tolerance)
    for i, j, img, _ in edges:
        adj[i].append((j, img))
    frac = np.asarray(structure.frac_coords, dtype=float)
    offset: dict[int, np.ndarray] = {}
    molecules: list[dict[str, Any]] = []
    for start in range(n):
        if start in offset:
            continue
        offset[start] = np.zeros(3, dtype=int)
        comp = [start]
        extended = False
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j, img in adj[i]:
                want = offset[i] + img
                if j not in offset:
                    offset[j] = want
                    comp.append(j)
                    queue.append(j)
                elif not np.array_equal(offset[j], want):
                    extended = True
        comp.sort()
        masses = np.array([float(structure[k].species.weight) for k in comp])
        pos = np.array([frac[k] + offset[k] for k in comp])
        centroid = (masses[:, None] * pos).sum(axis=0) / masses.sum()
        shift = np.floor(centroid + 1e-9).astype(int)
        offs = [offset[k] - shift for k in comp]
        centroid = centroid - shift
        symbols = [_site_symbol(structure[k]) for k in comp]
        in_comp = set(comp)
        # each bond once: i < j, or for a bond to the own image (i == j) only the image whose
        # first non-zero component is positive (+a, not also -a)
        bonds = [(i, j, img) for i, j, img, _ in edges if i in in_comp and (
            i < j or (i == j and next((v for v in img if v != 0), 0) > 0))]
        molecules.append({
            "sites": comp,
            "offsets": offs,
            "symbols": symbols,
            "formula": _hill_formula(symbols),
            "n_atoms": len(comp),
            "centroid_frac": centroid,
            "centroid_cart": np.asarray(structure.lattice.get_cartesian_coords(centroid), dtype=float),
            "extended": bool(extended),
            "bonds": bonds,
        })
    for k, mol in enumerate(molecules, start=1):
        mol["id"] = k
    return molecules


def write_molecules_csv(structure: Any, molecules: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow([
            "molecule", "formula", "n_atoms", "extended", "centroid_x_frac", "centroid_y_frac",
            "centroid_z_frac", "centroid_x_A", "centroid_y_A", "centroid_z_A", "sites",
        ])
        for m in molecules:
            cf, cc = m["centroid_frac"], m["centroid_cart"]
            w.writerow([
                m["id"], m["formula"], m["n_atoms"], "yes" if m["extended"] else "no",
                *(f"{v:.5f}" for v in cf), *(f"{v:.4f}" for v in cc),
                " ".join(_site_label(structure, k) for k in m["sites"]),
            ])
    return path


def molecule_atoms(structure: Any, molecules: list[dict[str, Any]], frac_range: Any = None) -> list[tuple[int, np.ndarray]]:
    """(site index, cart) of whole molecules: every image whose centre of mass lies in
    ``frac_range`` [x0 x1 y0 y1 z0 z1] (default: the unit cell [0, 1)). Extended networks
    are drawn once (one unit cell's worth of atoms, unwrapped)."""
    return [(k, c) for _mi, _cen, items in _molecule_images(structure, molecules, frac_range) for k, c in items]


def _molecule_images(structure: Any, molecules: list[dict[str, Any]], frac_range: Any = None,
                     ) -> list[tuple[int, np.ndarray, list[tuple[int, np.ndarray]]]]:
    """Like `molecule_atoms`, grouped per drawn image: (molecule index, centre of mass of
    that image (cart), [(site index, cart), ...])."""
    frac = np.asarray(structure.frac_coords, dtype=float)
    lat = structure.lattice
    out: list[tuple[int, np.ndarray, list[tuple[int, np.ndarray]]]] = []
    for mi, m in enumerate(molecules):
        c = np.asarray(m["centroid_frac"], dtype=float)
        if frac_range is None or m["extended"]:
            shifts = [np.zeros(3, dtype=int)]
        else:
            rng = [float(v) for v in frac_range]
            axes = []
            for k in range(3):
                lo, hi = rng[2 * k], rng[2 * k + 1]
                axes.append([t for t in range(math.floor(lo - c[k]) - 1, math.ceil(hi - c[k]) + 2)
                             if lo - 1e-6 <= c[k] + t <= hi + 1e-6])
            shifts = [np.array([a, b, cc]) for a in axes[0] for b in axes[1] for cc in axes[2]]
        for t in shifts:
            items = [(int(k), np.asarray(lat.get_cartesian_coords(frac[k] + off + t), dtype=float))
                     for k, off in zip(m["sites"], m["offsets"])]
            cen = np.asarray(lat.get_cartesian_coords(c + t), dtype=float)
            out.append((mi, cen, items))
    return out


def bonds_between(structure: Any, carts: list[np.ndarray], symbols: list[str],
                  tolerance: float = DEFAULT_BOND_TOLERANCE) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """Covalent bonds among the given drawn points (render_bonds tuples)."""
    from scipy.spatial import cKDTree

    if len(carts) < 2:
        return []
    pts = np.asarray(carts, dtype=float)
    radii = np.array([_covalent_radius(s) for s in symbols])
    tree = cKDTree(pts)
    out = []
    for i, j in sorted(tree.query_pairs(2.0 * radii.max() * tolerance + 0.01)):
        d = float(np.linalg.norm(pts[i] - pts[j]))
        if 0.1 < d <= tolerance * (radii[i] + radii[j]):
            out.append((pts[i].copy(), pts[j].copy(), symbols[i], symbols[j]))
    return out


# ---------------------------------------------------------------- hydrogen bonds
def parse_hbond_spec(spec: Any) -> dict[str, Any]:
    """'O-H...O:2.5:120' -> donor O, hydrogen H, acceptor O, H...A <= 2.5 Å, angle >= 120°.
    The donor / acceptor may be lists ('N,O-H...O'); dmax / angle are optional."""
    text = str(spec).strip()
    parts = text.split(":")
    m = re.fullmatch(r"\s*([A-Za-z,*]+)\s*-\s*([A-Za-z]+)\s*\.{2,3}\s*([A-Za-z,*]+)\s*", parts[0])
    if not m or len(parts) > 3:
        raise ValueError(f"hbond {spec!r}: write 'D-H...A:dmax:angle', e.g. 'O-H...O:2.5:120'.")

    def els(s: str) -> set[str]:
        return {p.strip()[:1].upper() + p.strip()[1:].lower() for p in s.split(",") if p.strip()}

    dmax = float(parts[1]) if len(parts) > 1 and parts[1].strip() else DEFAULT_HBOND_DMAX
    ang = float(parts[2]) if len(parts) > 2 and parts[2].strip() else DEFAULT_HBOND_ANGLE
    if dmax <= 0:
        raise ValueError("hbond: dmax must be > 0.")
    return {"donor": els(m.group(1)), "hydrogen": els(m.group(2)).pop(), "acceptor": els(m.group(3)),
            "dmax": dmax, "angle_min": ang, "spec": text}


def normalize_hbond_specs(specs: Any) -> list[dict[str, Any]] | None:
    if specs is None:
        return None
    items = [specs] if isinstance(specs, str) else list(specs)
    out = [parse_hbond_spec(s) for s in items if str(s).strip()]
    return out or None


def find_hbonds(structure: Any, specs: Any, tolerance: float = DEFAULT_BOND_TOLERANCE) -> list[dict[str, Any]]:
    """Every D-H...A hydrogen bond (one row per H...A contact, acceptor image included)."""
    specs = normalize_hbond_specs(specs)
    if not specs:
        return []
    lat = structure.lattice
    frac = np.asarray(structure.frac_coords, dtype=float)
    syms = [_site_symbol(s) for s in structure]
    edges = covalent_edges(structure, tolerance)
    donors_of: dict[int, list[tuple[int, np.ndarray]]] = {}
    for i, j, img, _ in edges:
        donors_of.setdefault(i, []).append((j, img))
    dmax_all = max(s["dmax"] for s in specs)
    ci, pi, imgs, dists = structure.get_neighbor_list(r=dmax_all + 1e-6)
    near: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for i, j, img, d in zip(ci, pi, imgs, dists):
        near.setdefault(int(i), []).append((int(j), np.asarray(np.round(img), dtype=int), float(d)))
    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for spec in specs:
        for h in range(len(structure)):
            if syms[h] != spec["hydrogen"]:
                continue
            h_cart = np.asarray(lat.get_cartesian_coords(frac[h]), dtype=float)
            for dn, dimg in donors_of.get(h, []):
                if "*" not in spec["donor"] and syms[dn] not in spec["donor"]:
                    continue
                d_cart = np.asarray(lat.get_cartesian_coords(frac[dn] + dimg), dtype=float)
                for a, aimg, dha in near.get(h, []):
                    if dha > spec["dmax"] or ("*" not in spec["acceptor"] and syms[a] not in spec["acceptor"]):
                        continue
                    if a == dn and np.array_equal(aimg, dimg):
                        continue
                    if any(a == j and np.array_equal(aimg, img) for j, img in donors_of.get(h, [])):
                        continue  # covalently bonded to H: not an H-bond acceptor
                    a_cart = np.asarray(lat.get_cartesian_coords(frac[a] + aimg), dtype=float)
                    v1, v2 = d_cart - h_cart, a_cart - h_cart
                    cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                    if angle < spec["angle_min"] - 1e-6:
                        continue
                    key = (dn, tuple(dimg), h, a, tuple(aimg))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        "spec": spec["spec"], "donor": dn, "hydrogen": h, "acceptor": a,
                        "donor_image": dimg, "acceptor_image": aimg,
                        "d_DH": float(np.linalg.norm(v1)), "d_HA": dha,
                        "d_DA": float(np.linalg.norm(a_cart - d_cart)), "angle_DHA": angle,
                        "h_cart": h_cart, "a_cart": a_cart,
                    })
    return out


def write_hbonds_csv(structure: Any, hbonds: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["donor", "hydrogen", "acceptor", "acceptor_image", "d_DH_A", "d_HA_A", "d_DA_A",
                    "angle_DHA_deg", "spec"])
        for hb in hbonds:
            w.writerow([
                _site_label(structure, hb["donor"]), _site_label(structure, hb["hydrogen"]),
                _site_label(structure, hb["acceptor"]), " ".join(str(int(v)) for v in hb["acceptor_image"]),
                f"{hb['d_DH']:.4f}", f"{hb['d_HA']:.4f}", f"{hb['d_DA']:.4f}", f"{hb['angle_DHA']:.2f}",
                hb["spec"],
            ])
    return path


def hbond_segments_on_atoms(structure: Any, hbonds: list[dict[str, Any]], atom_carts: list[np.ndarray],
                            tol: float = 0.02) -> list[tuple[np.ndarray, np.ndarray]]:
    """H...A segments moved by lattice translations onto drawn atoms (both ends drawn)."""
    from scipy.spatial import cKDTree

    if not hbonds or not atom_carts:
        return []
    pts = np.asarray(atom_carts, dtype=float)
    tree = cKDTree(pts)
    mat = np.asarray(structure.lattice.matrix, dtype=float)
    fr = np.asarray(structure.lattice.get_fractional_coords(pts), dtype=float)
    lo, hi = np.floor(fr.min(axis=0)).astype(int) - 1, np.ceil(fr.max(axis=0)).astype(int) + 1
    shifts = [np.array([a, b, c]) @ mat for a in range(lo[0], hi[0] + 1)
              for b in range(lo[1], hi[1] + 1) for c in range(lo[2], hi[2] + 1)]
    out, seen = [], set()
    for hb in hbonds:
        for t in shifts:
            p, q = hb["h_cart"] + t, hb["a_cart"] + t
            if tree.query(p, distance_upper_bound=tol)[0] > tol or tree.query(q, distance_upper_bound=tol)[0] > tol:
                continue
            key = (tuple(np.round(p * 100).astype(int)), tuple(np.round(q * 100).astype(int)))
            if key not in seen:
                seen.add(key)
                out.append((p, q))
    return out


def dashed_tubes(segments: list[tuple[np.ndarray, np.ndarray]], radius: float = 0.035,
                 dash: float = 0.16, gap: float = 0.11) -> Any:
    """One PolyData of short tube pieces along every segment (dashed line), or None."""
    import pyvista as pv

    pieces = []
    for p, q in segments:
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        length = float(np.linalg.norm(q - p))
        if length < 1e-6:
            continue
        u = (q - p) / length
        s = 0.0
        while s < length:
            e = min(s + dash, length)
            pieces.append(pv.Cylinder(center=p + u * (s + e) / 2, direction=u, radius=radius,
                                      height=e - s, resolution=12))
            s = e + gap
    if not pieces:
        return None
    return pv.merge(pieces) if len(pieces) > 1 else pieces[0]


# ---------------------------------------------------------------- renderer hook
def restrict_drawing(
    structure: Any,
    atoms: list[tuple[np.ndarray, list, float]],
    render_bonds: list | None,
    polyhedra_envs: list | None,
    cutoff_planes: Any = None,
    whole_molecules: bool = False,
    hbonds: Any = None,
    frac_range: Any = None,
    site_info: dict[int, tuple[list, float]] | None = None,
) -> tuple[list, list | None, list | None, list[tuple[np.ndarray, np.ndarray]]]:
    """Apply whole-molecule drawing, cut-off planes and H-bond segments to a view.

    Returns (atoms, render_bonds, polyhedra_envs, hbond_segments). ``site_info`` maps the
    site index to (species_occs, radius) (needed for ``whole_molecules``).
    """
    planes = normalize_cutoff_planes(cutoff_planes)
    if whole_molecules:
        mols = find_molecules(structure)
        drawn = []
        for mi, cen, items in _molecule_images(structure, mols, frac_range):
            if not planes:
                drawn.extend(items)
            elif mols[mi]["extended"]:  # a network has no molecule to keep: cut atom by atom
                mask = plane_keep_mask(structure, [c for _, c in items], planes)
                drawn.extend(it for it, k in zip(items, mask) if k)
            elif plane_keep_mask(structure, [cen], planes)[0]:
                # keep or drop each drawn image of a molecule by its own centre of mass
                drawn.extend(items)
        info = site_info or {}
        atoms = [(c, info[k][0], info[k][1]) for k, c in drawn]
        render_bonds = bonds_between(structure, [c for _, c in drawn], [_site_symbol(structure[k]) for k, _ in drawn])
        polyhedra_envs = None
    elif planes:
        mask = plane_keep_mask(structure, [a[0] for a in atoms], planes)
        atoms = [a for a, k in zip(atoms, mask) if k]
        if render_bonds:
            render_bonds = [b for b in render_bonds if plane_keep_mask(structure, [b[0], b[1]], planes).all()]
        if polyhedra_envs:
            polyhedra_envs = [
                p for p in polyhedra_envs
                if len(p[1]) and plane_keep_mask(structure, list(p[1]), planes).all()
            ]
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    if hbonds:
        hb = hbonds if isinstance(hbonds, list) and hbonds and isinstance(hbonds[0], dict) and "h_cart" in hbonds[0] \
            else find_hbonds(structure, hbonds)
        segments = hbond_segments_on_atoms(structure, hb, [a[0] for a in atoms])
    return atoms, render_bonds, polyhedra_envs, segments


def keep_near_atoms(points: list[np.ndarray], atoms: list[tuple[np.ndarray, Any, Any]], tol: float = 0.02) -> list[bool]:
    """For labels / arrows: True where the point sits on a drawn atom."""
    from scipy.spatial import cKDTree

    if not atoms:
        return [False] * len(points)
    tree = cKDTree(np.asarray([a[0] for a in atoms], dtype=float))
    return [bool(tree.query(np.asarray(p, dtype=float), distance_upper_bound=tol)[0] <= tol) for p in points]

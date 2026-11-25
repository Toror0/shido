"""Geometry measurements on a periodic structure (VESTA-like "measure" tool).

Atoms are given as text: ``"<site>[@<image>]"``.

- ``<site>`` is a site index (``"0"``, 0-based, the ``site`` column of
  ``atomic_positions.csv``), a CIF site label (``"Na1"``; the first site with that
  label), or element + 1-based ordinal (``"Si2"`` = the second Si site) when no label
  matches.
- ``<image>`` is the cell translation in lattice vectors, e.g. ``"Si1@1,0,0"`` or
  ``"Si1@(0 0 -1)"``. Without it, the first atom stays in the home cell and each
  later atom takes the periodic image nearest to the previous atom (so a two-atom
  distance is the minimum-image distance; ties go to the image nearest the home
  cell). For a best plane, images without ``@`` are taken nearest to the first atom.
  Give ``@`` explicitly when the same site appears twice.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import numpy as np

from shido.core.species import dominant_element

_IMAGE_RE = re.compile(r"^\s*([^@]+?)\s*(?:@\s*\(?\s*([-+\d\s,]+?)\s*\)?\s*)?$")


def _parse_image(text: str) -> tuple[int, int, int]:
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    if len(parts) != 3:
        raise ValueError(f"Image must have 3 integers (e.g. '1,0,0'), got {text!r}.")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def resolve_site_index(structure, site: str) -> int:
    """Site index from an index string, a CIF label, or element + ordinal."""
    name = str(site).strip()
    n = len(structure)
    if re.fullmatch(r"\d+", name):
        idx = int(name)
        if idx >= n:
            raise ValueError(f"Site index {idx} out of range (structure has {n} sites).")
        return idx
    for i, s in enumerate(structure):
        label = getattr(s, "label", None)
        if label is not None and str(label).lower() == name.lower():
            return i
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", name)
    if m:
        elem, k = m.group(1).lower(), int(m.group(2))
        hits = [i for i, s in enumerate(structure) if dominant_element(s).lower() == elem]
        if 1 <= k <= len(hits):
            return hits[k - 1]
    raise ValueError(f"Atom {site!r} not found (use a site index, a CIF label or e.g. 'Si2').")


def _parse_atom(structure, spec: str) -> tuple[int, tuple[int, int, int] | None]:
    m = _IMAGE_RE.match(str(spec))
    if not m:
        raise ValueError(f"Cannot read atom {spec!r} (use e.g. 'Na1' or 'Na1@1,0,0').")
    idx = resolve_site_index(structure, m.group(1))
    image = _parse_image(m.group(2)) if m.group(2) is not None else None
    return idx, image


def _nearest_image(structure, idx: int, ref_cart: np.ndarray) -> tuple[int, int, int]:
    lat = structure.lattice
    f = np.asarray(structure[idx].frac_coords, dtype=float)
    ref_f = lat.get_fractional_coords(ref_cart)
    base = np.round(ref_f - f)
    candidates = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                img = base + (dx, dy, dz)
                d = float(np.linalg.norm(lat.get_cartesian_coords(f + img) - ref_cart))
                # Ties (common in cubic cells) go to the image closest to the home cell.
                candidates.append((round(d, 6), float(np.abs(img).sum()), tuple(int(v) for v in img)))
    return min(candidates)[2]


def resolve_atoms(structure, specs: Sequence[str], chain: bool = True) -> list[dict[str, Any]]:
    """Resolve atom specs to site, image and Cartesian coordinates (A).

    chain=True: atoms without ``@`` take the image nearest to the previous atom;
    chain=False: nearest to the first atom.
    """
    lat = structure.lattice
    out: list[dict[str, Any]] = []
    for k, spec in enumerate(specs):
        idx, image = _parse_atom(structure, spec)
        if image is None:
            if k == 0:
                image = (0, 0, 0)
            else:
                ref = out[-1]["cart"] if chain else out[0]["cart"]
                image = _nearest_image(structure, idx, np.asarray(ref))
        frac = np.asarray(structure[idx].frac_coords, dtype=float) + np.asarray(image)
        cart = np.asarray(lat.get_cartesian_coords(frac), dtype=float)
        site = structure[idx]
        out.append(
            {
                "spec": str(spec),
                "site": idx,
                "label": str(getattr(site, "label", "") or ""),
                "element": dominant_element(site),
                "image": list(image),
                "frac": [round(float(v), 6) for v in frac],
                "cart": [float(v) for v in cart],
            }
        )
    return out


def distance(p1: Iterable[float], p2: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(p2, float) - np.asarray(p1, float)))


def angle(p1: Iterable[float], p2: Iterable[float], p3: Iterable[float]) -> float:
    """Angle p1-p2-p3 at p2, in degrees."""
    v1 = np.asarray(p1, float) - np.asarray(p2, float)
    v2 = np.asarray(p3, float) - np.asarray(p2, float)
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def dihedral(p1, p2, p3, p4) -> float:
    """Torsion angle p1-p2-p3-p4 in degrees, (-180, 180] (IUPAC sign)."""
    p = [np.asarray(x, float) for x in (p1, p2, p3, p4)]
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    ang = float(np.degrees(np.arctan2(y, x)))
    if ang <= -180.0 + 1e-9:
        ang = 180.0
    return ang


def _parse_hkl(hkl: Any) -> tuple[float, float, float]:
    if isinstance(hkl, str):
        parts = [p for p in re.split(r"[,\s()]+", hkl.strip()) if p]
        if len(parts) == 1 and re.fullmatch(r"\d{3}", parts[0]):  # "110" shorthand
            parts = list(parts[0])
        vals = [float(p) for p in parts]
    else:
        vals = [float(v) for v in hkl]
    if len(vals) != 3 or not any(vals):
        raise ValueError(f"Plane must be 3 Miller indices, not all zero (e.g. '1 1 0'), got {hkl!r}.")
    return tuple(vals)  # type: ignore[return-value]


def plane_normal(lattice, hkl) -> np.ndarray:
    """Unit Cartesian normal of (hkl): h a* + k b* + l c*."""
    h = np.asarray(_parse_hkl(hkl), float)
    n = h @ np.asarray(lattice.reciprocal_lattice_crystallographic.matrix, float)
    return n / np.linalg.norm(n)


def interplanar_angle(lattice, hkl1, hkl2) -> float:
    """Angle between the normals of (hkl1) and (hkl2), degrees in [0, 180]."""
    n1, n2 = plane_normal(lattice, hkl1), plane_normal(lattice, hkl2)
    return float(np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))))


def best_plane(points: Sequence[Iterable[float]], lattice=None) -> dict[str, Any]:
    """Least-squares plane through >= 3 points (SVD). Deviations in A."""
    pts = np.asarray([np.asarray(p, float) for p in points])
    if len(pts) < 3:
        raise ValueError("A best plane needs at least 3 atoms.")
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid)
    normal = vt[-1]
    # Fix the sign so the largest component is positive (stable output).
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    dev = (pts - centroid) @ normal
    res: dict[str, Any] = {
        "normal_cart": [round(float(v), 6) for v in normal],
        "centroid_cart": [round(float(v), 6) for v in centroid],
        "plane_d": round(float(np.dot(normal, centroid)), 6),
        "deviations": [round(float(v), 6) for v in dev],
        "rms_deviation": round(float(np.sqrt(np.mean(dev**2))), 6),
        "max_abs_deviation": round(float(np.max(np.abs(dev))), 6),
    }
    if lattice is not None:
        # n = h a* + k b* + l c*  =>  h_i = n . a_i (up to a common factor).
        hkl = np.asarray(lattice.matrix, float) @ normal
        hkl = hkl / np.max(np.abs(hkl))
        res["normal_hkl_approx"] = [round(float(v), 4) for v in hkl]
    return res


def measure_geometry(
    structure,
    atoms: Sequence[str] | None = None,
    planes: Sequence[Any] | None = None,
    best_plane_atoms: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Distance (2 atoms), angle (3), dihedral (4); angle between two (hkl); best plane.

    Returns a dict with the resolved atoms and the requested values (A, degrees).
    """
    if not atoms and not planes and not best_plane_atoms:
        raise ValueError("Give atoms (2-4), planes (2 hkl) and/or best_plane_atoms (>= 3).")
    out: dict[str, Any] = {}
    if atoms:
        if not 2 <= len(atoms) <= 4:
            raise ValueError(f"atoms must list 2, 3 or 4 atoms, got {len(atoms)}.")
        res = resolve_atoms(structure, atoms, chain=True)
        p = [r["cart"] for r in res]
        out["atoms"] = res
        out["distances"] = [round(distance(p[i], p[i + 1]), 6) for i in range(len(p) - 1)]
        if len(p) == 2:
            out["distance"] = out["distances"][0]
        if len(p) == 3:
            out["angle_deg"] = round(angle(*p), 4)
        if len(p) == 4:
            out["angles_deg"] = [round(angle(*p[0:3]), 4), round(angle(*p[1:4]), 4)]
            out["dihedral_deg"] = round(dihedral(*p), 4)
    if planes:
        if len(planes) != 2:
            raise ValueError(f"planes must list exactly 2 Miller indices, got {len(planes)}.")
        ang = interplanar_angle(structure.lattice, planes[0], planes[1])
        out["planes"] = [list(_parse_hkl(h)) for h in planes]
        out["interplanar_angle_deg"] = round(ang, 4)
        out["interplanar_acute_angle_deg"] = round(min(ang, 180.0 - ang), 4)
    if best_plane_atoms:
        res = resolve_atoms(structure, best_plane_atoms, chain=False)
        bp = best_plane([r["cart"] for r in res], lattice=structure.lattice)
        bp["atoms"] = res
        out["best_plane"] = bp
    return out

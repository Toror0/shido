"""Crystal shape (VESTA "Crystal shape" / morphology) from {hkl} and central distances.

Each face family ``{hkl}`` is expanded to its symmetry-equivalent (hkl) with the point
group of the structure (spglib operations via ``SpacegroupAnalyzer``; the lattice symmetry of
pymatgen ``get_symmetrically_equivalent_miller_indices`` only if that fails). A
plane (hkl) at central distance d is the half space ``n . x <= d`` with n the unit
vector along the reciprocal-lattice vector h a* + k b* + l c* (Cartesian, Å frame of
the structure). The crystal is the intersection of all half spaces (a convex
polyhedron; scipy ``HalfspaceIntersection``, with a brute-force fallback). Distances
are relative: the unit is whatever the user gives (only the ratios matter).

Faces whose plane lies outside the others have zero area and are reported as
``on_crystal = False``. An unbounded set of faces (e.g. only {001}) raises ValueError.

Output: `build_morphology()` -> `Morphology` (vertices, face polygons, areas, volume),
`write_morphology_csv()` -> ``morphology_faces.csv`` (one row per face) and
``morphology_families.csv`` (area per {hkl}). Drawing: `renderer.render_morphology`;
STL / HTML: `export3d.export_morphology_stl` / `export_morphology_html`.
"""

from __future__ import annotations

import csv
import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FACE_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]

_TOL = 1e-7


def _parse_hkl(text: str) -> tuple[int, int, int]:
    s = str(text).strip().strip("{}()[]").replace(",", " ")
    toks = s.split()
    if len(toks) == 1 and re.fullmatch(r"\d{3}", toks[0]):
        toks = list(toks[0])
    vals = [int(round(float(t))) for t in toks]
    if len(vals) == 4:  # Miller-Bravais h k i l -> h k l
        vals = [vals[0], vals[1], vals[3]]
    if len(vals) != 3 or not any(vals):
        raise ValueError(f"bad (hkl): {text!r} (give 3 integers, e.g. '1 0 0')")
    return int(vals[0]), int(vals[1]), int(vals[2])


def parse_faces(faces: Any) -> list[tuple[tuple[int, int, int], float]]:
    """``"1 0 0:1.0,1 1 1:0.9"`` / list of such strings / [(hkl, d)] / [{"hkl":..,"distance":..}]."""
    if faces is None:
        raise ValueError("Give at least one face, e.g. '1 0 0:1.0'.")
    items: list[Any]
    if isinstance(faces, str):
        items = [p for p in re.split(r"[,;]", faces) if p.strip()]
    elif isinstance(faces, dict):
        items = [faces]
    else:
        items = []
        for f in faces:
            if isinstance(f, str):
                items.extend(p for p in re.split(r"[,;]", f) if p.strip())
            else:
                items.append(f)
    out: list[tuple[tuple[int, int, int], float]] = []
    for it in items:
        if isinstance(it, str):
            if ":" in it:
                hkl_s, d_s = it.rsplit(":", 1)
            elif "@" in it:
                hkl_s, d_s = it.rsplit("@", 1)
            else:
                hkl_s, d_s = it, "1.0"
            hkl, d = _parse_hkl(hkl_s), float(d_s)
        elif isinstance(it, dict):
            raw = it.get("hkl", it.get("miller"))
            hkl = _parse_hkl(" ".join(str(v) for v in raw) if not isinstance(raw, str) else raw)
            d = float(it.get("distance", it.get("d", 1.0)))
        else:
            raw, d = it
            hkl = _parse_hkl(" ".join(str(v) for v in raw) if not isinstance(raw, str) else raw)
            d = float(d)
        if not np.isfinite(d) or d <= 0:
            raise ValueError(f"central distance must be > 0: {hkl} : {d}")
        out.append((hkl, d))
    if not out:
        raise ValueError("Give at least one face, e.g. '1 0 0:1.0'.")
    return out


def family_label(hkl: tuple[int, int, int]) -> str:
    return "{" + " ".join(str(v) for v in hkl) + "}"


@dataclass
class Morphology:
    lattice_matrix: np.ndarray
    vertices: np.ndarray                     # (n, 3) Cartesian, same unit as the distances
    faces: list[dict[str, Any]]              # one per (hkl): hkl, family, distance, normal, polygon, area
    families: list[dict[str, Any]]           # one per {hkl}: label, color, n_faces, n_on_crystal, area
    volume: float
    notes: list[str] = field(default_factory=list)

    @property
    def total_area(self) -> float:
        return float(sum(f["area"] for f in self.faces))

    @property
    def visible_faces(self) -> list[dict[str, Any]]:
        return [f for f in self.faces if f["on_crystal"]]

    def triangles(self) -> tuple[np.ndarray, list[int]]:
        """Fan triangles (outward winding) and the family index of each."""
        tris: list[list[int]] = []
        fam: list[int] = []
        for f in self.visible_faces:
            poly = f["polygon"]
            for i in range(1, len(poly) - 1):
                tris.append([poly[0], poly[i], poly[i + 1]])
                fam.append(f["family_index"])
        return np.asarray(tris, dtype=int).reshape(-1, 3), fam

    def to_polydata(self, scale: float = 1.0, family: int | None = None):
        import pyvista as pv

        cells: list[int] = []
        for f in self.visible_faces:
            if family is not None and f["family_index"] != family:
                continue
            cells.extend([len(f["polygon"]), *f["polygon"]])
        return pv.PolyData(self.vertices * scale, np.asarray(cells, dtype=np.int64))

    def report(self) -> dict[str, Any]:
        total = self.total_area
        return {
            "n_faces": len(self.visible_faces),
            "n_planes": len(self.faces),
            "n_vertices": int(len(self.vertices)),
            "volume": self.volume,
            "total_area": total,
            "families": [
                {k: v for k, v in fam.items()} | {"area_fraction": (fam["area"] / total if total else 0.0)}
                for fam in self.families
            ],
            "notes": list(self.notes),
        }


def _expand(structure: Any, hkl: tuple[int, int, int], expand: bool) -> list[tuple[int, int, int]]:
    if not expand or structure is None:
        return [hkl]
    # The crystal's point group (not the lattice's holohedry): zinc blende F-43m {111} is
    # a tetrahedron (4 faces), pyrite Pa-3 {210} a pyritohedron (12). A frac rotation R
    # (x' = R x) maps the plane h to h R^-1; over the whole group that is {h R}.
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        ops = SpacegroupAnalyzer(structure).get_symmetry_operations(cartesian=False)
        h = np.asarray(hkl, dtype=float)
        out: list[tuple[int, int, int]] = []
        for op in ops:
            m = tuple(int(v) for v in np.rint(h @ np.asarray(op.rotation_matrix, dtype=float)))
            if m not in out:
                out.append(m)
        if out:
            return out
    except Exception:  # noqa: BLE001 - fall back to the lattice symmetry below
        pass
    from pymatgen.core.surface import get_symmetrically_equivalent_miller_indices

    eq = get_symmetrically_equivalent_miller_indices(structure, hkl, return_hkil=False)
    out = [tuple(int(v) for v in m) for m in eq]
    return out or [hkl]


def _intersection_vertices(normals: np.ndarray, dists: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import HalfspaceIntersection

        hs = np.hstack([normals, -dists[:, None]])
        verts = HalfspaceIntersection(hs, np.zeros(3)).intersections
    except ImportError:  # pragma: no cover - scipy is installed with pymatgen
        cand = []
        for i, j, k in itertools.combinations(range(len(normals)), 3):
            a = normals[[i, j, k]]
            if abs(np.linalg.det(a)) < 1e-10:
                continue
            p = np.linalg.solve(a, dists[[i, j, k]])
            if np.all(normals @ p <= dists + 1e-9 * dists.max()):
                cand.append(p)
        verts = np.asarray(cand).reshape(-1, 3)
    return np.asarray(verts, dtype=float)


def _dedupe(points: np.ndarray, tol: float) -> np.ndarray:
    keep: list[np.ndarray] = []
    for p in points:
        if not any(np.linalg.norm(p - q) < tol for q in keep):
            keep.append(p)
    return np.asarray(keep).reshape(-1, 3)


def build_morphology(structure: Any, faces: Any, expand: bool = True) -> Morphology:
    """Convex crystal shape for ``faces`` ({hkl}: central distance) on the lattice of ``structure``.

    ``structure``: pymatgen Structure (its point group expands each {hkl}) or a Lattice
    (no expansion). ``expand=False`` uses only the given (hkl).
    """
    from pymatgen.core import Lattice, Structure

    parsed = parse_faces(faces)
    if isinstance(structure, Lattice):
        lattice, struct = structure, None
    elif isinstance(structure, Structure):
        lattice, struct = structure.lattice, structure
    else:
        lattice, struct = Lattice(np.asarray(structure, dtype=float)), None
    recip = lattice.reciprocal_lattice_crystallographic

    notes: list[str] = []
    planes: dict[tuple[float, float, float], dict[str, Any]] = {}
    families: list[dict[str, Any]] = []
    for hkl, d in parsed:
        label = family_label(hkl)
        fam_idx = len(families)
        families.append({"family": label, "hkl": list(hkl), "distance": d,
                         "color": FACE_COLORS[fam_idx % len(FACE_COLORS)]})
        for m in _expand(struct, hkl, expand):
            g = recip.get_cartesian_coords(m)
            n = g / np.linalg.norm(g)
            key = tuple(np.round(n, 6) + 0.0)
            old = planes.get(key)
            if old is not None:
                if d < old["distance"]:
                    notes.append(f"({' '.join(map(str, m))}) of {label} replaces the same plane of {old['family']} (closer).")
                    old.update(hkl=list(m), family=label, family_index=fam_idx, distance=d)
                continue
            planes[key] = {"hkl": list(m), "family": label, "family_index": fam_idx,
                           "distance": d, "normal": n}

    plist = list(planes.values())
    normals = np.array([p["normal"] for p in plist])
    dists = np.array([p["distance"] for p in plist])
    dmax = float(dists.max())
    big = 1e3 * dmax
    box_n = np.vstack([np.eye(3), -np.eye(3)])
    allv = _intersection_vertices(np.vstack([normals, box_n]), np.concatenate([dists, np.full(6, big)]))
    if len(allv) == 0 or not np.all(np.isfinite(allv)) or np.abs(allv).max() > 1e2 * dmax:
        raise ValueError("The faces do not enclose a finite crystal; add faces (e.g. {001} with {100}).")
    tol = 1e-6 * dmax
    verts = _dedupe(allv, tol)

    from scipy.spatial import ConvexHull

    volume = float(ConvexHull(verts).volume)
    face_rows: list[dict[str, Any]] = []
    for p in plist:
        n, d = p["normal"], p["distance"]
        on = np.where(np.abs(verts @ n - d) < 1e3 * tol)[0]
        row = {"hkl": p["hkl"], "family": p["family"], "family_index": p["family_index"],
               "distance": d, "normal": n, "polygon": [], "area": 0.0, "on_crystal": False}
        if len(on) >= 3:
            pts = verts[on]
            c = pts.mean(axis=0)
            u = pts[0] - c
            u = u - n * (u @ n)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            ang = np.arctan2((pts - c) @ v, (pts - c) @ u)
            order = on[np.argsort(ang)]
            poly = verts[order]
            area = 0.5 * float(np.cross(poly, np.roll(poly, -1, axis=0)).sum(axis=0) @ n)
            if area > _TOL * dmax * dmax:
                row.update(polygon=[int(i) for i in order], area=area, on_crystal=True)
        face_rows.append(row)

    for i, fam in enumerate(families):
        rows = [r for r in face_rows if r["family_index"] == i]
        fam["n_faces"] = len(rows)
        fam["n_on_crystal"] = sum(r["on_crystal"] for r in rows)
        fam["area"] = float(sum(r["area"] for r in rows))
        if fam["n_on_crystal"] == 0:
            notes.append(f"{fam['family']} at {fam['distance']:g} lies outside the other faces (area 0).")
    return Morphology(lattice.matrix.copy(), verts, face_rows, families, volume, notes)


def write_morphology_csv(morph: Morphology, out_dir: str | Path) -> list[Path]:
    """``morphology_faces.csv`` (each (hkl)) and ``morphology_families.csv`` (each {hkl})."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = morph.total_area or 1.0
    p1 = out / "morphology_faces.csv"
    with p1.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "h", "k", "l", "distance", "on_crystal", "n_vertices", "area", "area_fraction",
                    "normal_x", "normal_y", "normal_z"])
        for f in morph.faces:
            w.writerow([f["family"], *f["hkl"], f"{f['distance']:.6g}", int(f["on_crystal"]), len(f["polygon"]),
                        f"{f['area']:.6g}", f"{f['area'] / total:.6f}", *(f"{x:.6f}" for x in f["normal"])])
    p2 = out / "morphology_families.csv"
    with p2.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "distance", "n_equivalent", "n_on_crystal", "area", "area_fraction", "color"])
        for fam in morph.families:
            w.writerow([fam["family"], f"{fam['distance']:.6g}", fam["n_faces"], fam["n_on_crystal"],
                        f"{fam['area']:.6g}", f"{fam['area'] / total:.6f}", fam["color"]])
        w.writerow(["total", "", len(morph.faces), len(morph.visible_faces), f"{morph.total_area:.6g}", "1", ""])
        w.writerow(["volume", "", "", "", f"{morph.volume:.6g}", "", ""])
    return [p1.resolve(), p2.resolve()]


__all__ = ["Morphology", "build_morphology", "family_label", "parse_faces", "write_morphology_csv", "FACE_COLORS"]

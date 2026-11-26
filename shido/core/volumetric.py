"""Volumetric data (VESTA "Volumetric data"): read 3D grids and build isosurfaces.

Readers (all return a `VolumetricGrid`: lattice, atoms (pymatgen Structure) and a
(nx, ny, nz) array whose point (i, j, k) sits at fractional ``origin_frac + (i/nx,
j/ny, k/nz)``; the grid is periodic, i.e. it does not repeat the point at 1.0):

    CHGCAR / CHG / PARCHG / AECCAR*   VASP charge (pymatgen ``Chgcar``). The file holds
                                      rho * V_cell; ``data`` is divided by V_cell, so it
                                      is e/Å^3 (as VESTA shows it); ``raw_scale`` = V_cell.
    LOCPOT                            VASP local potential (pymatgen ``Locpot``), eV.
    ELFCAR                            VASP ELF (pymatgen ``Elfcar``), dimensionless 0..1.
    *.cube / *.cub                    Gaussian Cube. Values as stored (usually e/Bohr^3);
                                      Bohr lengths are converted to Å.
    *.xsf                             XCrySDen DATAGRID_3D (first block, or ``block=N``);
                                      the repeated end points of the general grid are
                                      dropped. Values as stored.
    *.grd                             VESTA grid (title, a b c alpha beta gamma, nx ny nz,
                                      values with x fastest). No atoms.

`n_electrons()` integrates the grid (mean * V_cell in the file's length unit), so a
CHGCAR / Cube of an electron density gives the number of electrons.

`isosurface()` contours the grid padded by one periodic layer (the 0-th plane copied
to index n), so a surface that crosses a cell face is not cut open one voxel short of
it; points are mapped to Å with the lattice, so oblique cells are handled exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Lattice, Structure

BOHR_TO_ANGSTROM: float = 0.529177210903

VOLUME_FORMATS: tuple[str, ...] = ("chgcar", "locpot", "elfcar", "cube", "xsf", "grd")
_VASP_PREFIX = {
    "CHGCAR": "chgcar", "CHG": "chgcar", "PARCHG": "chgcar", "AECCAR": "chgcar",
    "LOCPOT": "locpot", "ELFCAR": "elfcar",
}
_UNITS = {
    "chgcar": "e/Å^3 (file value / V_cell)",
    "locpot": "eV",
    "elfcar": "ELF (dimensionless)",
    "cube": "as stored (usually e/Bohr^3)",
    "xsf": "as stored",
    "grd": "as stored",
}


@dataclass
class VolumetricGrid:
    """A periodic 3D grid on a crystal lattice (see the module docstring)."""

    lattice: Lattice
    data: np.ndarray
    structure: Structure | None = None
    origin_frac: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fmt: str = "grd"
    unit: str = "as stored"
    length_unit_volume: float = 1.0  # V_cell in the unit the data are per (Å^3 or Bohr^3)
    raw_scale: float = 1.0  # file value = data * raw_scale
    path: str | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape)  # type: ignore[return-value]

    def n_electrons(self) -> float:
        """Integral over the cell: mean(data) * V_cell (in the data's length unit)."""
        return float(np.mean(self.data) * self.length_unit_volume)

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.fmt,
            "unit": self.unit,
            "grid": list(self.shape),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "mean": float(np.mean(self.data)),
            "integral": self.n_electrons(),
        }


# ---- detection ----------------------------------------------------------------------

def detect_volume_format(path: str | Path) -> str | None:
    """'chgcar' / 'locpot' / 'elfcar' / 'cube' / 'xsf' / 'grd' from the file name (XSF
    only when it holds a DATAGRID_3D), or None."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".cube", ".cub"):
        return "cube"
    if suffix == ".grd":
        return "grd"
    if suffix == ".xsf":
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return "xsf" if "DATAGRID_3D" in text.upper() else None
    name = p.name.upper()
    if suffix in (".cif", ".mcif", ".vasp", ".json", ".xyz"):
        return None
    for prefix in sorted(_VASP_PREFIX, key=len, reverse=True):  # CHGCAR before CHG
        if name.startswith(prefix):
            return _VASP_PREFIX[prefix]
    return None


def is_volume_file(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and detect_volume_format(p) is not None


# ---- readers ------------------------------------------------------------------------

def read_volume(path: str | Path, fmt: str | None = None, block: int = 0) -> VolumetricGrid:
    """Read a volumetric file (format from the name unless ``fmt`` is given)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Volume file not found: {p}")
    fmt = (fmt or detect_volume_format(p) or "").lower()
    if fmt not in VOLUME_FORMATS:
        raise ValueError(
            f"Unknown volumetric format for {p.name!r}: use CHGCAR / LOCPOT / ELFCAR "
            "(VASP names), *.cube, *.xsf (DATAGRID_3D) or *.grd."
        )
    if fmt in ("chgcar", "locpot", "elfcar"):
        grid = _read_vasp(p, fmt)
    elif fmt == "cube":
        grid = _read_cube(p)
    elif fmt == "xsf":
        grid = _read_xsf(p, block=block)
    else:
        grid = _read_grd(p)
    grid.path = str(p)
    if not np.all(np.isfinite(grid.data)):
        raise ValueError(f"{p.name}: the grid has non-finite values.")
    return grid


def _read_vasp(path: Path, fmt: str) -> VolumetricGrid:
    import warnings

    from pymatgen.io.vasp.outputs import Chgcar, Elfcar, Locpot

    cls = {"chgcar": Chgcar, "locpot": Locpot, "elfcar": Elfcar}[fmt]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = cls.from_file(str(path))
    structure = obj.structure
    raw = np.asarray(obj.data["total"], dtype=float)
    vol = float(structure.lattice.volume)
    scale = vol if fmt == "chgcar" else 1.0
    return VolumetricGrid(
        lattice=structure.lattice, data=raw / scale, structure=structure, fmt=fmt,
        unit=_UNITS[fmt], length_unit_volume=vol, raw_scale=scale,
    )


def _numbers(lines: list[str]) -> np.ndarray:
    return np.array(" ".join(lines).replace("D", "E").replace("d", "e").split(), dtype=float)


def _read_cube(path: Path) -> VolumetricGrid:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head = lines[2].split()
    natoms = int(head[0])
    origin = np.array([float(v) for v in head[1:4]])
    ns, vox = [], []
    for i in range(3):
        parts = lines[3 + i].split()
        ns.append(int(parts[0]))
        vox.append([float(v) for v in parts[1:4]])
    # n > 0: Bohr (the usual case); n < 0: Å
    to_ang = BOHR_TO_ANGSTROM if ns[0] > 0 else 1.0
    ns = [abs(n) for n in ns]
    vox_a = np.asarray(vox) * to_ang
    origin_a = origin * to_ang
    matrix = vox_a * np.asarray(ns)[:, None]
    lattice = Lattice(matrix)
    species, coords = [], []
    row = 6
    for _ in range(abs(natoms)):
        parts = lines[row].split()
        row += 1
        z = int(round(float(parts[0])))
        if z <= 0:
            continue  # ghost / point charge
        species.append(z)
        coords.append(np.array([float(v) for v in parts[2:5]]) * to_ang)
    if natoms < 0:
        row += 1  # MO line (number of orbitals + their indices)
    values = _numbers(lines[row:])
    n_tot = ns[0] * ns[1] * ns[2]
    if values.size < n_tot:
        raise ValueError(f"{path.name}: {values.size} values, expected {n_tot}.")
    if values.size > n_tot:  # several orbitals: keep the first
        values = values.reshape(n_tot, -1)[:, 0] if values.size % n_tot == 0 else values[:n_tot]
    data = values.reshape(ns[0], ns[1], ns[2])  # z fastest (C order)
    structure = None
    if species:
        structure = Structure(lattice, species, coords, coords_are_cartesian=True, to_unit_cell=True)
    vol_native = float(abs(np.linalg.det(np.asarray(vox) * np.asarray(ns)[:, None])))
    return VolumetricGrid(
        lattice=lattice, data=data, structure=structure,
        origin_frac=np.asarray(lattice.get_fractional_coords(origin_a), dtype=float),
        fmt="cube", unit=_UNITS["cube"], length_unit_volume=vol_native,
    )


def _read_xsf(path: Path, block: int = 0) -> VolumetricGrid:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines()]
    upper = [ln.upper() for ln in lines]
    structure = _xsf_structure(lines, upper)
    grids = [i for i, ln in enumerate(upper) if ln.startswith("BEGIN_DATAGRID_3D")]
    if not grids:
        raise ValueError(f"{path.name}: no DATAGRID_3D block.")
    if not 0 <= block < len(grids):
        raise ValueError(f"{path.name}: block {block} not found ({len(grids)} grid(s)).")
    i = grids[block] + 1
    ns = [int(v) for v in lines[i].split()[:3]]
    origin = np.array([float(v) for v in lines[i + 1].split()[:3]])
    span = np.array([[float(v) for v in lines[i + 2 + k].split()[:3]] for k in range(3)])
    end = next(j for j in range(i + 5, len(lines)) if upper[j].startswith("END_DATAGRID_3D"))
    values = _numbers(lines[i + 5:end])
    n_tot = ns[0] * ns[1] * ns[2]
    if values.size < n_tot:
        raise ValueError(f"{path.name}: {values.size} values, expected {n_tot}.")
    full = values[:n_tot].reshape(ns[2], ns[1], ns[0]).transpose(2, 1, 0)  # x fastest
    data = full[:-1, :-1, :-1].copy()  # general grid repeats the 1.0 end points
    lattice = structure.lattice if structure is not None else Lattice(span)
    if structure is not None and not np.allclose(span, lattice.matrix, atol=1e-3):
        lattice = Lattice(span)
    return VolumetricGrid(
        lattice=lattice, data=data, structure=structure,
        origin_frac=np.asarray(lattice.get_fractional_coords(origin), dtype=float),
        fmt="xsf", unit=_UNITS["xsf"], length_unit_volume=float(lattice.volume),
    )


def _xsf_structure(lines: list[str], upper: list[str]) -> Structure | None:
    try:
        i = upper.index("PRIMVEC")
        matrix = np.array([[float(v) for v in lines[i + 1 + k].split()[:3]] for k in range(3)])
        j = upper.index("PRIMCOORD")
        n = int(lines[j + 1].split()[0])
    except (ValueError, IndexError):
        return None
    species, coords = [], []
    for k in range(n):
        parts = lines[j + 2 + k].split()
        sym = parts[0]
        species.append(int(sym) if sym.isdigit() else sym)
        coords.append([float(v) for v in parts[1:4]])
    return Structure(Lattice(matrix), species, coords, coords_are_cartesian=True, to_unit_cell=True)


def _read_grd(path: Path) -> VolumetricGrid:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    a, b, c, al, be, ga = (float(v) for v in lines[1].split()[:6])
    ns = [int(v) for v in lines[2].split()[:3]]
    values = _numbers(lines[3:])
    n_tot = ns[0] * ns[1] * ns[2]
    if values.size < n_tot:
        raise ValueError(f"{path.name}: {values.size} values, expected {n_tot}.")
    data = values[:n_tot].reshape(ns[2], ns[1], ns[0]).transpose(2, 1, 0)
    lattice = Lattice.from_parameters(a, b, c, al, be, ga)
    return VolumetricGrid(
        lattice=lattice, data=data, structure=None, fmt="grd", unit=_UNITS["grd"],
        length_unit_volume=float(lattice.volume),
    )


# ---- writers (small; used by the tests and for hand-made data) ---------------------

def write_cube(
    path: str | Path,
    structure: Structure | None,
    data: np.ndarray,
    comment: str = "",
    lattice: Lattice | None = None,
    origin_frac: Any = None,
) -> Path:
    """Gaussian Cube in Bohr (data are written as given, z fastest). ``structure`` may be
    None (then ``lattice`` is needed and no atoms are written)."""
    p = Path(path)
    nx, ny, nz = data.shape
    lat = lattice if lattice is not None else structure.lattice  # type: ignore[union-attr]
    vox = lat.matrix / BOHR_TO_ANGSTROM / np.array([[nx], [ny], [nz]])
    o = (np.zeros(3) if origin_frac is None else np.asarray(origin_frac, dtype=float)) @ lat.matrix / BOHR_TO_ANGSTROM
    out = [comment or "cube written by Shido", "values"]
    out.append(f"{len(structure or []):5d} {o[0]:12.6f} {o[1]:12.6f} {o[2]:12.6f}")
    for n, v in zip((nx, ny, nz), vox):
        out.append(f"{n:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}")
    for site in structure or []:
        z = max(site.species.items(), key=lambda t: t[1])[0].Z
        x, y, w = site.coords / BOHR_TO_ANGSTROM
        out.append(f"{z:5d} {float(z):12.6f} {x:12.6f} {y:12.6f} {w:12.6f}")
    flat = np.asarray(data, dtype=float).reshape(-1)
    for i in range(0, flat.size, 6):
        out.append(" ".join(f"{v:13.5E}" for v in flat[i:i + 6]))
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


def write_chgcar(path: str | Path, structure: Structure, density: np.ndarray) -> Path:
    """VASP CHGCAR from a density in e/Å^3 (the file stores density * V_cell)."""
    from pymatgen.io.vasp.inputs import Poscar
    from pymatgen.io.vasp.outputs import Chgcar

    raw = np.asarray(density, dtype=float) * structure.lattice.volume
    Chgcar(Poscar(structure), {"total": raw}).write_file(str(path))
    return Path(path)


# ---- isosurfaces --------------------------------------------------------------------

def periodic_pad(data: np.ndarray) -> np.ndarray:
    """Copy the 0-th plane to index n on each axis -> (nx+1, ny+1, nz+1), covering 0..1."""
    return np.pad(np.asarray(data, dtype=float), ((0, 1), (0, 1), (0, 1)), mode="wrap")


def normalize_isovalues(isovalue: Any, grid: VolumetricGrid | None = None) -> list[float]:
    """'0.05,0.1' / 0.05 / [0.05, 0.1] -> [0.05, 0.1]; None -> 10 % of max |value|."""
    if isovalue is None or (isinstance(isovalue, str) and not isovalue.strip()):
        if grid is None:
            raise ValueError("isovalue is required.")
        level = 0.1 * float(np.max(np.abs(grid.data)))
        return [float(f"{level:.4g}")] if level > 0 else []
    if isinstance(isovalue, str):
        parts = [t for t in re.split(r"[,\s;]+", isovalue.strip()) if t]
    elif isinstance(isovalue, (int, float)):
        parts = [isovalue]
    else:
        parts = []
        for item in isovalue:
            parts.extend(normalize_isovalues(item) if isinstance(item, str) else [item])
    out: list[float] = []
    for t in parts:
        try:
            v = float(t)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"isovalue {t!r} is not a number.") from exc
        if not math.isfinite(v) or v == 0.0:
            raise ValueError(f"isovalue must be a finite non-zero number (got {t!r}).")
        out.append(v)
    return out


def isosurface(
    grid: VolumetricGrid,
    level: float,
    lattice_matrix: np.ndarray | None = None,
) -> Any:
    """pyvista PolyData of the ``level`` isosurface in Å (None if the level is not
    crossed). ``lattice_matrix`` (rows a, b, c) places it on another lattice (the
    drawn structure's), default the grid's own."""
    import pyvista as pv

    padded = periodic_pad(grid.data)
    lo, hi = float(padded.min()), float(padded.max())
    if not (lo < level < hi):
        return None
    nx, ny, nz = grid.shape
    img = pv.ImageData(
        dimensions=padded.shape,
        spacing=(1.0 / nx, 1.0 / ny, 1.0 / nz),
        origin=tuple(float(v) for v in grid.origin_frac),
    )
    img.point_data["v"] = padded.ravel(order="F")
    surf = img.contour([float(level)], scalars="v")
    if surf.n_points == 0:
        return None
    matrix = np.asarray(grid.lattice.matrix if lattice_matrix is None else lattice_matrix, dtype=float)
    surf.points = np.asarray(surf.points, dtype=float) @ matrix
    surf = surf.clean()
    if surf.n_points == 0:
        return None
    return surf.compute_normals(auto_orient_normals=False, consistent_normals=True)


def lattice_mismatch(a: Lattice, b: Lattice, tol: float = 0.01) -> str | None:
    """Warning text if the lattice parameters differ by more than ``tol`` (relative / deg)."""
    pa, pb = np.asarray(a.parameters, dtype=float), np.asarray(b.parameters, dtype=float)
    rel = np.abs(pa[:3] - pb[:3]) / np.maximum(pb[:3], 1e-9)
    ang = np.abs(pa[3:] - pb[3:])
    if np.any(rel > tol) or np.any(ang > 100 * tol):
        return (
            "volume lattice " + " ".join(f"{v:.3f}" for v in pb)
            + " differs from the structure " + " ".join(f"{v:.3f}" for v in pa)
            + "; the grid is drawn on the structure's cell."
        )
    return None


# ---- slices, contours and line profiles (batch 18) ---------------------------------

def interpolate_frac(grid: VolumetricGrid, frac: np.ndarray) -> np.ndarray:
    """Periodic trilinear interpolation of the grid at fractional points (..., 3)."""
    frac = np.asarray(frac, dtype=float)
    data = np.asarray(grid.data, dtype=float)
    ns = np.asarray(grid.shape, dtype=float)
    g = (frac - np.asarray(grid.origin_frac, dtype=float)) * ns  # grid index coordinates
    i0 = np.floor(g)
    t = g - i0
    i0 = i0.astype(np.int64)
    out = np.zeros(frac.shape[:-1], dtype=float)
    for dx in (0, 1):
        wx = t[..., 0] if dx else 1.0 - t[..., 0]
        ix = (i0[..., 0] + dx) % grid.shape[0]
        for dy in (0, 1):
            wy = t[..., 1] if dy else 1.0 - t[..., 1]
            iy = (i0[..., 1] + dy) % grid.shape[1]
            for dz in (0, 1):
                wz = t[..., 2] if dz else 1.0 - t[..., 2]
                iz = (i0[..., 2] + dz) % grid.shape[2]
                out += wx * wy * wz * data[ix, iy, iz]
    return out


def parse_slice_spec(spec: Any) -> tuple[tuple[int, int, int], float]:
    """'0 0 1@0.5' / '1,1,1@0.25' / ((h, k, l), pos) -> ((h, k, l), pos).

    The plane is h*x + k*y + l*z = pos in fractional coordinates, i.e. ``pos`` d_hkl
    from the origin along the (hkl) normal (as VESTA). No '@' -> pos 0."""
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and not isinstance(spec[0], (int, float, str)):
        hkl, pos = spec
    else:
        text = str(spec).strip()
        if "@" in text:
            hkl_text, pos_text = text.split("@", 1)
        else:
            hkl_text, pos_text = text, "0"
        hkl = [t for t in re.split(r"[,\s;()]+", hkl_text.strip()) if t]
        try:
            pos = float(pos_text)
        except ValueError as exc:
            raise ValueError(f"slice {spec!r}: position after '@' is not a number.") from exc
    try:
        h, k, l = (int(round(float(v))) for v in hkl)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"slice {spec!r}: give three integers h k l, e.g. '0 0 1@0.5'.") from exc
    if (h, k, l) == (0, 0, 0):
        raise ValueError(f"slice {spec!r}: (0 0 0) is not a plane.")
    if not math.isfinite(float(pos)):
        raise ValueError(f"slice {spec!r}: position must be finite.")
    return (h, k, l), float(pos)


def parse_line_spec(spec: Any) -> tuple[np.ndarray, np.ndarray]:
    """'x1 y1 z1 x2 y2 z2' (fractional) or six numbers -> (start, end)."""
    if isinstance(spec, str):
        vals = [t for t in re.split(r"[,\s;()]+", spec.strip()) if t]
    else:
        vals = list(np.ravel(spec))
    try:
        nums = [float(v) for v in vals]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line profile {spec!r}: six numbers are needed.") from exc
    if len(nums) != 6 or not all(math.isfinite(v) for v in nums):
        raise ValueError(f"line profile {spec!r}: give six numbers 'x1 y1 z1 x2 y2 z2' (fractional).")
    a, b = np.array(nums[:3]), np.array(nums[3:])
    if np.allclose(a, b):
        raise ValueError(f"line profile {spec!r}: start and end are the same point.")
    return a, b


def normalize_contours(contours: Any) -> list[float] | None:
    """None / 'auto' -> None (automatic levels); 'none' / [] -> [] (no lines);
    '0.05,0.1' / [0.05, 0.1] -> [0.05, 0.1]."""
    if contours is None:
        return None
    if isinstance(contours, str):
        low = contours.strip().lower()
        if low in ("", "auto"):
            return None
        if low in ("none", "off"):
            return []
        parts: list[Any] = [t for t in re.split(r"[,\s;]+", contours.strip()) if t]
    elif isinstance(contours, (int, float)):
        parts = [contours]
    else:
        parts = []
        for item in contours:
            parts.extend(normalize_contours(item) or [])
    out: list[float] = []
    for t in parts:
        try:
            v = float(t)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"contour level {t!r} is not a number.") from exc
        if not math.isfinite(v):
            raise ValueError(f"contour level {t!r} is not finite.")
        out.append(v)
    return sorted(set(out))


def _plane_basis(lattice: Lattice, hkl: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    """(G, n, u, v): G = h a* + k b* + l c* (no 2 pi; r.G = frac.hkl), unit normal n and an
    orthonormal in-plane pair u, v (u along the most in-plane lattice vector, v = n x u)."""
    m = np.asarray(lattice.matrix, dtype=float)
    g = np.linalg.inv(m) @ np.asarray(hkl, dtype=float)
    n = g / np.linalg.norm(g)
    best_score, u = -1.0, None
    for vec in m:
        proj = vec - np.dot(vec, n) * n
        norm = float(np.linalg.norm(proj))
        score = norm / float(np.linalg.norm(vec))
        if norm > 1e-9 and score > best_score + 1e-9:
            best_score, u = score, proj / norm
    if u is None:  # pragma: no cover - a lattice always has an in-plane component
        u = np.cross(n, [1.0, 0.0, 0.0])
        u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return g, n, u, v


def _cell_plane_polygon(m: np.ndarray, hkl: tuple[int, int, int], pos: float) -> np.ndarray:
    """Cartesian corners (k, 3) where the plane frac.hkl = pos cuts the unit cell (may be empty)."""
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
    hk = np.asarray(hkl, dtype=float)
    pts = []
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(corners[a] - corners[b])) != 1:
                continue  # only the 12 cell edges
            fa, fb = corners[a] @ hk - pos, corners[b] @ hk - pos
            if abs(fa) < 1e-12:
                pts.append(corners[a])
            if abs(fb) < 1e-12:
                pts.append(corners[b])
            if fa * fb < 0:
                pts.append(corners[a] + (corners[b] - corners[a]) * fa / (fa - fb))
    if not pts:
        return np.zeros((0, 3))
    return np.unique(np.round(np.asarray(pts) @ m, 9), axis=0)


def slice_plane(
    grid: VolumetricGrid,
    hkl: tuple[int, int, int],
    pos: float = 0.0,
    resolution: float | None = None,
    max_points: int = 400,
) -> dict[str, Any]:
    """2D map of the grid on the plane h x + k y + l z = pos (fractional).

    The map spans the bounding box (in the plane's u, v axes, Å) of where the plane cuts
    the unit cell (the whole projected cell if it does not cut it); values outside the
    cell come from the periodic grid. ``resolution`` = pixel size in Å (default half the
    smallest grid spacing). Returns u, v (1D, Å), values (nv, nu), the 3D point of
    (u=0, v=0), the unit vectors and the cut polygon in (u, v)."""
    m = np.asarray(grid.lattice.matrix, dtype=float)
    g, n, u_hat, v_hat = _plane_basis(grid.lattice, hkl)
    p0 = pos * g / float(np.dot(g, g))  # point of the plane closest to the origin
    poly = _cell_plane_polygon(m, hkl, pos)
    if len(poly) >= 3:
        ref = poly
    else:
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float) @ m
        ref = corners - np.outer((corners - p0) @ n, n)
    uv = np.column_stack(((ref - p0) @ u_hat, (ref - p0) @ v_hat))
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    if resolution is None:
        spacing = np.linalg.norm(m, axis=1) / np.asarray(grid.shape, dtype=float)
        resolution = float(spacing.min()) / 2.0
    if not resolution or resolution <= 0:
        raise ValueError("resolution must be a positive length in Å.")
    span = np.maximum(hi - lo, 1e-6)
    counts = np.clip(np.ceil(span / float(resolution) - 1e-3).astype(int) + 1, 2, max_points)
    us = np.linspace(lo[0], hi[0], counts[0])
    vs = np.linspace(lo[1], hi[1], counts[1])
    uu, vv = np.meshgrid(us, vs)  # (nv, nu)
    cart = p0 + uu[..., None] * u_hat + vv[..., None] * v_hat
    values = interpolate_frac(grid, cart @ np.linalg.inv(m))
    if len(poly) >= 3:
        puv = np.column_stack(((poly - p0) @ u_hat, (poly - p0) @ v_hat))
        c = puv.mean(axis=0)
        puv = puv[np.argsort(np.arctan2(puv[:, 1] - c[1], puv[:, 0] - c[0]))]
    else:
        puv = np.zeros((0, 2))
    return {
        "hkl": tuple(int(x) for x in hkl), "position": float(pos),
        "d_hkl": float(1.0 / np.linalg.norm(g)),
        "u": us, "v": vs, "values": values,
        "origin_cart": p0, "u_hat": u_hat, "v_hat": v_hat, "normal": n,
        "cell_polygon_uv": puv, "resolution": float(resolution),
    }


def line_profile(grid: VolumetricGrid, start_frac: Any, end_frac: Any, n_points: int = 201) -> dict[str, Any]:
    """Values along the straight line start -> end (fractional; may leave the cell)."""
    a, b = np.asarray(start_frac, dtype=float), np.asarray(end_frac, dtype=float)
    t = np.linspace(0.0, 1.0, max(2, int(n_points)))
    frac = a + t[:, None] * (b - a)
    m = np.asarray(grid.lattice.matrix, dtype=float)
    length = float(np.linalg.norm((b - a) @ m))
    return {"start": a, "end": b, "length": length, "distance": t * length,
            "frac": frac, "values": interpolate_frac(grid, frac)}


def _fmt_num(x: float) -> str:
    return f"{x:g}".replace("-", "m")


def slice_name(hkl: tuple[int, int, int], pos: float) -> str:
    """'slice_001_0.5' (minus signs as 'm')."""
    return "slice_" + "".join(_fmt_num(h) for h in hkl) + f"_{_fmt_num(pos)}"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_slice(
    sl: dict[str, Any],
    path: str | Path,
    contours: list[float] | None = None,
    unit: str = "",
    title: str | None = None,
    cmap: str = "viridis",
    dpi: int = 200,
) -> Path:
    """PNG of a slice: colour map, contour lines (None = 8 automatic levels, [] = none),
    the unit-cell cut as a dashed outline, axes in Å."""
    plt = _pyplot()
    us, vs, vals = sl["u"], sl["v"], sl["values"]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    mesh = ax.pcolormesh(us, vs, vals, shading="auto", cmap=cmap)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(unit or "value")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if contours is None:
        levels = list(np.linspace(lo, hi, 10)[1:-1]) if hi > lo else []
    else:
        levels = [c for c in contours if lo < c < hi]
    if levels:
        cs = ax.contour(us, vs, vals, levels=levels, colors="white", linewidths=0.8)
        if contours:
            ax.clabel(cs, fmt="%g", fontsize=7)
    poly = sl.get("cell_polygon_uv")
    if poly is not None and len(poly) >= 3:
        closed = np.vstack([poly, poly[:1]])
        ax.plot(closed[:, 0], closed[:, 1], "--", color="#ff6060", lw=1.0, label="unit cell")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    h, k, l = sl["hkl"]
    ax.set_title(title or f"({h} {k} {l}) slice at {sl['position']:g} d  (d = {sl['d_hkl']:.3f} Å)")
    ax.set_xlabel("u (Å)")
    ax.set_ylabel("v (Å)")
    ax.set_aspect("equal")
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_line_profile(prof: dict[str, Any], path: str | Path, unit: str = "", dpi: int = 200) -> Path:
    """PNG of a line profile (value against distance in Å)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(prof["distance"], prof["values"], color="#2f6fb3", lw=1.5)
    a, b = prof["start"], prof["end"]
    ax.set_title(f"line profile ({a[0]:g}, {a[1]:g}, {a[2]:g}) -> ({b[0]:g}, {b[1]:g}, {b[2]:g})", fontsize=10)
    ax.set_xlabel("distance from the start (Å)")
    ax.set_ylabel(unit or "value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def _as_list(x: Any) -> list[Any]:
    if x is None or (isinstance(x, str) and not x.strip()):
        return []
    return [x] if isinstance(x, str) else list(x)


def write_slice_outputs(
    grid: VolumetricGrid,
    slices: Any = None,
    line_profiles: Any = None,
    contours: Any = None,
    fig_dir: str | Path = ".",
    data_dir: str | Path = ".",
    resolution: float | None = None,
    n_points: int = 201,
) -> dict[str, Any]:
    """Slice maps (PNG in fig_dir + CSV u, v, x, y, z, value in data_dir) and line
    profiles (PNG + CSV). ``slices`` / ``line_profiles``: one spec or a list.
    Returns {"volume", "contours", "slices", "line_profiles", "files": {name: path}}."""
    import pandas as pd

    levels = normalize_contours(contours)
    fig_dir, data_dir = Path(fig_dir), Path(data_dir)
    m = np.asarray(grid.lattice.matrix, dtype=float)
    files: dict[str, str] = {}
    rep_slices: list[dict[str, Any]] = []
    rep_lines: list[dict[str, Any]] = []
    for spec in _as_list(slices):
        hkl, pos = parse_slice_spec(spec)
        sl = slice_plane(grid, hkl, pos, resolution=resolution)
        name = slice_name(hkl, pos)
        png = plot_slice(sl, fig_dir / f"{name}.png", contours=levels, unit=grid.unit)
        uu, vv = np.meshgrid(sl["u"], sl["v"])
        cart = sl["origin_cart"] + uu[..., None] * sl["u_hat"] + vv[..., None] * sl["v_hat"]
        data_dir.mkdir(parents=True, exist_ok=True)
        csv = data_dir / f"{name}.csv"
        pd.DataFrame({
            "u_A": uu.ravel(), "v_A": vv.ravel(),
            "x_A": cart[..., 0].ravel(), "y_A": cart[..., 1].ravel(), "z_A": cart[..., 2].ravel(),
            "value": sl["values"].ravel(),
        }).to_csv(csv, index=False, float_format="%.6g")
        files[f"{name}.png"], files[f"{name}.csv"] = str(png), str(csv)
        i_max = np.unravel_index(int(np.argmax(sl["values"])), sl["values"].shape)
        rep_slices.append({
            "hkl": list(hkl), "position": pos, "d_hkl_angstrom": sl["d_hkl"],
            "shape": list(sl["values"].shape), "pixel_angstrom": sl["resolution"],
            "min": float(np.min(sl["values"])), "max": float(np.max(sl["values"])),
            "max_at_cart": [float(c) for c in cart[i_max]],
            "u_axis_cart": [float(c) for c in sl["u_hat"]], "v_axis_cart": [float(c) for c in sl["v_hat"]],
            "png": str(png), "csv": str(csv),
        })
    for i, spec in enumerate(_as_list(line_profiles), start=1):
        a, b = parse_line_spec(spec)
        prof = line_profile(grid, a, b, n_points=n_points)
        name = f"line_profile_{i}"
        png = plot_line_profile(prof, fig_dir / f"{name}.png", unit=grid.unit)
        data_dir.mkdir(parents=True, exist_ok=True)
        csv = data_dir / f"{name}.csv"
        cart = prof["frac"] @ m
        pd.DataFrame({
            "distance_A": prof["distance"],
            "x_frac": prof["frac"][:, 0], "y_frac": prof["frac"][:, 1], "z_frac": prof["frac"][:, 2],
            "x_A": cart[:, 0], "y_A": cart[:, 1], "z_A": cart[:, 2], "value": prof["values"],
        }).to_csv(csv, index=False, float_format="%.6g")
        files[f"{name}.png"], files[f"{name}.csv"] = str(png), str(csv)
        rep_lines.append({
            "start_frac": a.tolist(), "end_frac": b.tolist(), "length_angstrom": prof["length"],
            "min": float(np.min(prof["values"])), "max": float(np.max(prof["values"])),
            "png": str(png), "csv": str(csv),
        })
    return {"volume": grid.summary(), "contours": levels, "slices": rep_slices,
            "line_profiles": rep_lines, "files": files}


# ---- model densities, Patterson, difference, export (batch 19) ----------------------
#
# Model density = Fourier synthesis of the calculated structure factors on the grid
# (as VESTA does): rho(r) = (1/V) sum_h F(h) exp(-2 pi i h.r) over every h the grid
# resolves, F(h) = sum_j occ_j f_j(s) exp(-B s^2) exp(2 pi i h.r_j), s = |h*|/2.
#   xray    f_j = pymatgen's X-ray form factor (as in core.structure_factors); the fit is
#           valid up to about s = 1.5-2 1/A and turns up again beyond its minimum, so
#           past that minimum it is continued by f_min * exp(-(s^2 - s_min^2)).
#           -> electrons / A^3; integral over the cell = sum occ * Z (F(000)).
#   neutron f_j = b_j (bound coherent scattering length, fm) -> fm / A^3; integral = sum b.
# The grid truncates the series, so a sharp model shows small Fourier ripples (also
# negative values), as any Fourier map does. B (A^2) smooths the atoms.
#
# Patterson P(u) = int rho(r) rho(r+u) dr (periodic autocorrelation via FFT): the origin
# is its maximum and its integral is (int rho)^2.

MODEL_KINDS: tuple[str, ...] = ("xray", "neutron")
EXPORT_FORMATS: tuple[str, ...] = ("cube", "grd")
DEFAULT_B_ISO: float = 0.5  # A^2
DEFAULT_GRID_SPACING: float = 0.1  # A
_GRID_MIN, _GRID_MAX = 16, 128


def auto_grid(lattice: Lattice, spacing: float = DEFAULT_GRID_SPACING) -> tuple[int, int, int]:
    """Grid points per axis for about ``spacing`` A (even, 16..128)."""
    out = []
    for length in lattice.abc:
        n = int(math.ceil(length / spacing))
        n += n % 2
        out.append(min(max(n, _GRID_MIN), _GRID_MAX))
    return tuple(out)  # type: ignore[return-value]


def parse_grid(grid: Any, lattice: Lattice) -> tuple[int, int, int]:
    """None / "" / "auto" -> auto_grid; 64 / "64" -> (64, 64, 64); "48,48,64" -> 3 values."""
    if grid is None or (isinstance(grid, str) and grid.strip().lower() in ("", "auto")):
        return auto_grid(lattice)
    if isinstance(grid, str):
        parts = [t for t in re.split(r"[,\sx]+", grid.strip()) if t]
    elif isinstance(grid, (int, float)):
        parts = [grid]
    else:
        parts = list(grid)
    try:
        ns = [int(p) for p in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grid {grid!r}: give one or three integers (e.g. 64 or '48,48,64').") from exc
    if len(ns) == 1:
        ns = ns * 3
    if len(ns) != 3 or min(ns) < 2 or max(ns) > 512:
        raise ValueError(f"grid {grid!r}: give one or three integers between 2 and 512.")
    return tuple(ns)  # type: ignore[return-value]


def _xray_f_clamped(element: str, s: np.ndarray) -> np.ndarray:
    from shido.core.structure_factors import atomic_scattering_factor

    probe = np.linspace(0.0, 2.5, 501)
    fp = atomic_scattering_factor(element, probe, "xray")
    i_min = int(np.argmin(fp))
    s_min, f_min = float(probe[i_min]), float(fp[i_min])
    f = np.asarray(atomic_scattering_factor(element, np.minimum(s, s_min), "xray"), dtype=float)
    tail = s > s_min
    f[tail] = f_min * np.exp(-(s[tail] ** 2 - s_min**2))
    return f


def model_density(
    structure: Structure,
    kind: str = "xray",
    grid: Any = None,
    b_iso: float | dict[str, float] | None = None,
) -> VolumetricGrid:
    """Model electron density (``kind`` "xray", e/A^3) or nuclear scattering density
    ("neutron", fm/A^3) of ``structure`` on a periodic grid (see the section notes).
    ``b_iso``: one B in A^2 for every element or {element: B}; default 0.5."""
    from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN

    kind = normalize_model_kind(kind)
    lattice = structure.lattice
    ns = parse_grid(grid, lattice)
    if b_iso is None:
        b_iso = DEFAULT_B_ISO
    b_values = list(b_iso.values()) if isinstance(b_iso, dict) else [b_iso]
    for b in b_values:
        if not np.isfinite(float(b)) or float(b) < 0:
            raise ValueError(f"b_iso (B in A^2) must be >= 0, got {b!r}.")
    if kind == "xray":
        from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS

        for idx, site in enumerate(structure):
            for sp in site.species:
                sym = getattr(sp, "symbol", str(sp))
                if sym not in ATOMIC_SCATTERING_PARAMS:
                    raise ValueError(f"No X-ray scattering factor for {sym!r} (site {idx}).")
    h = np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in ns), indexing="ij")
    hkl = np.stack(h, axis=-1)  # integer h, k, l per FFT index
    recip = np.asarray(lattice.reciprocal_lattice_crystallographic.matrix, dtype=float)
    s = 0.5 * np.linalg.norm(hkl @ recip, axis=-1)
    s2 = s**2
    F = np.zeros(ns, dtype=complex)
    f_cache: dict[str, np.ndarray] = {}
    for idx, site in enumerate(structure):
        phase = np.exp(2j * np.pi * (hkl @ np.asarray(site.frac_coords, dtype=float)))
        for sp, occ in site.species.items():
            sym = getattr(sp, "symbol", str(sp))
            if sym not in f_cache:
                if kind == "neutron":
                    if sym not in ATOMIC_SCATTERING_LEN:
                        raise ValueError(f"No neutron scattering length for {sym!r} (site {idx}).")
                    f = np.full(ns, float(ATOMIC_SCATTERING_LEN[sym]))
                else:
                    f = _xray_f_clamped(sym, s)
                b = float(b_iso.get(sym, DEFAULT_B_ISO)) if isinstance(b_iso, dict) else float(b_iso)
                f_cache[sym] = f * np.exp(-b * s2)
            F += float(occ) * f_cache[sym] * phase
    vol = float(lattice.volume)
    rho = np.fft.fftn(F).real / vol  # fftn: sum_h F(h) exp(-2 pi i h.k/n)
    unit = "e/Å^3 (model, X-ray)" if kind == "xray" else "fm/Å^3 (model, neutron)"
    return VolumetricGrid(
        lattice=lattice, data=rho, structure=structure, fmt=f"model-{kind}", unit=unit,
        length_unit_volume=vol,
    )


def normalize_model_kind(kind: Any) -> str:
    k = str(kind or "xray").strip().lower().replace("-", "")
    k = {"x": "xray", "xrays": "xray", "electron": "xray", "n": "neutron", "nuclear": "neutron"}.get(k, k)
    if k not in MODEL_KINDS:
        raise ValueError(f"model must be 'xray' or 'neutron', got {kind!r}.")
    return k


def patterson(grid: VolumetricGrid) -> VolumetricGrid:
    """Patterson function P(u) = int rho(r) rho(r+u) dr of a grid (periodic, via FFT)."""
    data = np.asarray(grid.data, dtype=float)
    a = np.fft.fftn(data)
    vol = float(grid.length_unit_volume)
    p = np.fft.ifftn(np.abs(a) ** 2).real * vol / data.size
    return VolumetricGrid(
        lattice=grid.lattice, data=p, structure=grid.structure, fmt="patterson",
        unit=f"Patterson of [{grid.unit}] (value^2 x volume)", length_unit_volume=vol,
    )


def volume_difference(a: VolumetricGrid, b: VolumetricGrid, tol: float = 0.01) -> VolumetricGrid:
    """a - b on the same grid. Raises ValueError if the grid sizes, the lattices (``tol``
    relative / 100*tol deg) or the grid origins differ. A Cube (per Bohr^3) and a grid per
    A^3 are put on a's length unit first."""
    if a.shape != b.shape:
        raise ValueError(f"grid sizes differ: {list(a.shape)} vs {list(b.shape)}.")
    msg = lattice_mismatch(a.lattice, b.lattice, tol=tol)
    if msg:
        raise ValueError("lattices differ: " + msg.split(";")[0] + ".")
    d_origin = np.asarray(a.origin_frac, dtype=float) - np.asarray(b.origin_frac, dtype=float)
    d_origin -= np.round(d_origin)  # periodic: origin 0 and 1 are the same point
    if np.any(np.abs(d_origin) * np.asarray(a.shape) > 1e-3):
        raise ValueError("grid origins differ.")
    # Length unit of each grid (1 for Å, 1/0.529 for Bohr): V_unit / V_cell(Å^3) = unit^-3.
    # Only the unit ratio is converted (not the small volume difference of two cells within
    # tol), and only for densities: LOCPOT (eV) and ELFCAR are not per volume.
    ua = float(a.length_unit_volume) / float(a.lattice.volume)
    ub = float(b.length_unit_volume) / float(b.lattice.volume)
    scale = ub / ua
    per_volume = a.fmt not in ("locpot", "elfcar") and b.fmt not in ("locpot", "elfcar")
    bdata = np.asarray(b.data, dtype=float) * (scale if per_volume and abs(scale - 1.0) > 1e-3 else 1.0)
    return VolumetricGrid(
        lattice=a.lattice, data=np.asarray(a.data, dtype=float) - bdata, structure=a.structure,
        origin_frac=np.asarray(a.origin_frac, dtype=float), fmt="difference", unit=a.unit,
        length_unit_volume=a.length_unit_volume,
    )


def write_grd(path: str | Path, grid: VolumetricGrid, title: str = "") -> Path:
    """VESTA *.grd: title, a b c alpha beta gamma, nx ny nz, values (x fastest)."""
    p = Path(path)
    a, b, c, al, be, ga = grid.lattice.parameters
    nx, ny, nz = grid.shape
    out = [title or "grd written by Shido",
           f"{a:.6f} {b:.6f} {c:.6f} {al:.4f} {be:.4f} {ga:.4f}", f"{nx} {ny} {nz}"]
    flat = np.asarray(grid.data, dtype=float).ravel(order="F")
    for i in range(0, flat.size, 6):
        out.append(" ".join(f"{v:.6E}" for v in flat[i:i + 6]))
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


def write_volume(grid: VolumetricGrid, path: str | Path, fmt: str | None = None, comment: str = "") -> Path:
    """Write a grid as Cube or VESTA grd (format from ``fmt`` or the suffix).
    Cube: Bohr; a density per A^3 (unit containing '/Å^3') is converted to per Bohr^3 so
    the file reads back with the same integral. grd: values as they are, no atoms."""
    p = Path(path)
    fmt = (fmt or p.suffix.lstrip(".")).lower()
    if fmt == "cub":
        fmt = "cube"
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"export format {fmt!r}: use one of {', '.join(EXPORT_FORMATS)}.")
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "grd":
        return write_grd(p, grid, title=comment)
    data = np.asarray(grid.data, dtype=float)
    if "/Å^3" in grid.unit:
        data = data * BOHR_TO_ANGSTROM**3
    return write_cube(p, grid.structure, data, comment=comment, lattice=grid.lattice,
                      origin_frac=grid.origin_frac)


def normalize_export_formats(formats: Any) -> list[str]:
    """'cube,grd' / ['cube'] / 'all' -> ['cube', 'grd']; None -> ['cube']."""
    if formats is None or (isinstance(formats, str) and not formats.strip()):
        return ["cube"]
    items = re.split(r"[,\s;]+", formats.strip()) if isinstance(formats, str) else list(formats)
    out: list[str] = []
    for t in items:
        t = str(t).strip().lower().lstrip(".")
        if not t:
            continue
        if t == "all":
            return list(EXPORT_FORMATS)
        t = "cube" if t == "cub" else t
        if t not in EXPORT_FORMATS:
            raise ValueError(f"export format {t!r}: use one of {', '.join(EXPORT_FORMATS)} or 'all'.")
        if t not in out:
            out.append(t)
    return out or ["cube"]


def build_volumes(
    source: Any,
    model: str | None = "xray",
    grid: Any = None,
    b_iso: float | dict[str, float] | None = None,
    patterson_map: bool = False,
    diff: Any = None,
    export_formats: Any = "cube",
    out_dir: str | Path = ".",
) -> dict[str, Any]:
    """Batch 19 driver. ``source``: a Structure (-> model density ``model``) or a
    VolumetricGrid (used as it is). Writes in ``out_dir``:
        model_density_<xray|neutron>.<fmt>   (structure source)
        patterson.<fmt>                       (patterson_map=True)
        diff.<fmt>                            (diff = another VolumetricGrid or Structure:
                                               source - diff; a Structure gives its model
                                               density on the same grid)
    Returns {"volumes": {name: summary + files}, "files": {name: path}}."""
    fmts = normalize_export_formats(export_formats)
    out = Path(out_dir)
    grids: dict[str, VolumetricGrid] = {}
    if isinstance(source, VolumetricGrid):
        base = source
        base_name = "input"
    else:
        kind = normalize_model_kind(model)
        base = model_density(source, kind, grid=grid, b_iso=b_iso)
        base_name = f"model_density_{kind}"
        grids[base_name] = base
    if patterson_map:
        grids["patterson"] = patterson(base)
    if diff is not None:
        other = diff
        if not isinstance(other, VolumetricGrid):
            kind = normalize_model_kind(model)
            other = model_density(other, kind, grid=base.shape, b_iso=b_iso)
        grids["diff"] = volume_difference(base, other)
    if not grids:
        raise ValueError("Nothing to make: give a structure (model density), patterson or diff.")
    files: dict[str, str] = {}
    report: dict[str, Any] = {}
    for name, g in grids.items():
        entry = g.summary()
        entry["unit"] = g.unit
        entry["files"] = []
        for fmt in fmts:
            p = write_volume(g, out / f"{name}.{fmt}", fmt, comment=f"{name} ({g.unit}) by Shido")
            files[p.name] = str(p)
            entry["files"].append(str(p))
        if name == "patterson":
            i_max = np.unravel_index(int(np.argmax(g.data)), g.shape)
            entry["max_at_index"] = [int(i) for i in i_max]
        report[name] = entry
    return {"source": base_name, "grid": list(base.shape), "export_formats": fmts,
            "volumes": report, "files": files}

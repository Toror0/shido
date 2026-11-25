"""Runtime scientific guardrails.

Every check raises :class:`ScientificValidationError` on violation so that an analysis or
figure built from broken input never reaches disk. The checks are deliberately simple,
deterministic and numeric (no heuristics that need interpretation):

* Miller index ``(0, 0, 0)`` is rejected.
* Lattice lengths / cell volume must be finite and ``> 1.0`` (Å / Å^3); angles finite and
  in (0, 180).
* The shortest interatomic distance (periodic, all images) must be ``>= 0.6 Å``.
  Exception: two *partially occupied* sites whose occupancies sum to <= 1 are a split-site
  disorder model from a CIF (only one of them is present at a time), not an overlap.
* Element fractions (``fractional_composition``) must be preserved through supercell /
  slab / lattice-deformation steps within ``atol = 1e-3``.
* A rendered RGB(A) array that is > 99.5 % background is treated as a camera / mesh failure.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

import numpy as np

from shido.core.species import site_occupancies

MIN_LATTICE_VALUE = 1.0  # Å for lengths, Å^3 for the volume
MIN_INTERATOMIC_DISTANCE = 0.6  # Å
COMPOSITION_ATOL = 1e-3
BLANK_BACKGROUND_RATIO = 0.995

# Above this many sites the dense N x N distance matrix is replaced by a neighbor search.
_DENSE_DISTANCE_LIMIT = 3000


class ScientificValidationError(ValueError):
    """Raised when an input or an output violates a crystallographic / numerical invariant.

    Subclass of ``ValueError`` so callers that already guard against bad input keep working.
    """


# --------------------------------------------------------------------------------------
# Input checks
# --------------------------------------------------------------------------------------
def validate_miller_index(hkl: Iterable[Any]) -> tuple[int, int, int]:
    """Return ``hkl`` as a tuple of 3 ints; reject (0, 0, 0), non-integers and wrong length."""
    try:
        vals = list(hkl)
    except TypeError as exc:
        raise ScientificValidationError(f"Miller index must be 3 integers, got {hkl!r}.") from exc
    if len(vals) != 3:
        raise ScientificValidationError(f"Miller index must have 3 integers, got {hkl!r}.")
    out: list[int] = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise ScientificValidationError(f"Miller index must be integers, got {hkl!r}.") from exc
        if not math.isfinite(fv) or abs(fv - round(fv)) > 1e-9:
            raise ScientificValidationError(f"Miller index must be integers, got {hkl!r}.")
        out.append(int(round(fv)))
    if out == [0, 0, 0]:
        raise ScientificValidationError("Miller index (0, 0, 0) does not define a plane.")
    return (out[0], out[1], out[2])


def validate_lattice(lattice_or_structure: Any) -> None:
    """Reject NaN/Inf, lattice lengths <= 1.0 Å, volume <= 1.0 Å^3 and degenerate angles."""
    lattice = getattr(lattice_or_structure, "lattice", lattice_or_structure)
    matrix = np.asarray(lattice.matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ScientificValidationError(f"Lattice matrix contains NaN/Inf or has wrong shape: {matrix!r}.")

    lengths = np.linalg.norm(matrix, axis=1)
    for name, val in zip("abc", lengths):
        if not math.isfinite(float(val)) or float(val) <= MIN_LATTICE_VALUE:
            raise ScientificValidationError(
                f"Lattice constant {name} = {float(val):.6g} Å is not > {MIN_LATTICE_VALUE} Å."
            )

    volume = abs(float(np.linalg.det(matrix)))
    if not math.isfinite(volume) or volume <= MIN_LATTICE_VALUE:
        raise ScientificValidationError(
            f"Cell volume = {volume:.6g} Å^3 is not > {MIN_LATTICE_VALUE} Å^3 (degenerate cell)."
        )

    for name, val in zip(("alpha", "beta", "gamma"), lattice.angles):
        fval = float(val)
        if not math.isfinite(fval) or not 0.0 < fval < 180.0:
            raise ScientificValidationError(f"Lattice angle {name} = {fval!r} deg is invalid.")


# --------------------------------------------------------------------------------------
# Physical consistency
# --------------------------------------------------------------------------------------
def _total_occupancy(site: Any) -> float:
    return float(sum(o for _, o in site_occupancies(site)))


def _shortest_lattice_translation(lattice: Any) -> float:
    """Shortest non-zero lattice translation (a site's distance to its own image).

    Searched over {-2..2}^3 combinations of the LLL-reduced basis, which contains the
    shortest vector even for strongly sheared (triclinic) cells.
    """
    matrix = np.asarray(lattice.lll_matrix, dtype=float)
    best = math.inf
    for n in itertools.product(range(-2, 3), repeat=3):
        if n == (0, 0, 0):
            continue
        best = min(best, float(np.linalg.norm(np.asarray(n, dtype=float) @ matrix)))
    return best


def _close_pairs(structure: Any, r: float) -> list[tuple[int, int, float]]:
    """All pairs (i < j, any periodic image) closer than ``r`` Å, including d = 0 duplicates.

    ``exclude_self=False`` is used on purpose: pymatgen's ``exclude_self`` also drops
    *different* sites sitting at exactly the same position, which is the worst overlap.
    """
    if len(structure) < 2:
        return []
    centers, points, _images, dists = structure.get_neighbor_list(r=float(r), exclude_self=False)
    pairs: dict[tuple[int, int], float] = {}
    for i, j, d in zip(centers, points, dists):
        i, j, d = int(i), int(j), float(d)
        if i == j:
            continue  # the site itself (d = 0) or its own image (handled by the lattice check)
        key = (min(i, j), max(i, j))
        if d < pairs.get(key, math.inf):
            pairs[key] = d
    return [(i, j, d) for (i, j), d in pairs.items()]


def min_interatomic_distance(structure: Any) -> tuple[float, int, int]:
    """Return the raw ``(d_min, i, j)`` over all periodic images (occupancy ignored).

    ``i == j`` means the shortest distance is a site to its own periodic image.
    """
    n = len(structure)
    if n == 0:
        return (math.inf, -1, -1)
    best = (_shortest_lattice_translation(structure.lattice), 0, 0)
    if n < 2:
        return best
    if n <= _DENSE_DISTANCE_LIMIT:
        dm = np.asarray(structure.distance_matrix, dtype=float)
        if not np.all(np.isfinite(dm)):
            raise ScientificValidationError("Interatomic distance matrix contains NaN/Inf.")
        np.fill_diagonal(dm, np.inf)
        i, j = np.unravel_index(int(np.argmin(dm)), dm.shape)
        if float(dm[i, j]) < best[0]:
            best = (float(dm[i, j]), int(min(i, j)), int(max(i, j)))
        return best
    # Large cells: search out to a few Å only (enough for any real nearest neighbor).
    for r in (3.0, 6.0):
        pairs = _close_pairs(structure, r)
        if pairs:
            i, j, d = min(pairs, key=lambda t: t[2])
            return (d, i, j) if d < best[0] else best
    return best


def validate_min_distance(structure: Any, threshold: float = MIN_INTERATOMIC_DISTANCE) -> float:
    """Raise if atoms (periodic images included) are closer than ``threshold`` Å.

    Partial occupancy is respected: the sites within ``threshold`` of each other form a
    cluster, and only a cluster holding more than one atom in total (sum of occupancies
    > 1) is an overlap. So two half-occupied split positions 0.3 Å apart pass, while
    three of them, or a full atom plus anything, are rejected. Returns the raw d_min.
    """
    if len(structure) == 0:
        return math.inf
    d_self = _shortest_lattice_translation(structure.lattice)
    if not math.isfinite(d_self) or d_self < threshold:
        raise ScientificValidationError(
            f"Atoms overlap their own periodic image (shortest lattice translation "
            f"{d_self:.4f} Å < {threshold} Å)."
        )

    occ = [_total_occupancy(site) for site in structure]
    close = _close_pairs(structure, threshold)
    if close:
        load = list(occ)
        nearest: dict[int, tuple[float, int]] = {}
        for i, j, d in close:
            load[i] += occ[j]
            load[j] += occ[i]
            if d < nearest.get(i, (math.inf, -1))[0]:
                nearest[i] = (d, j)
            if d < nearest.get(j, (math.inf, -1))[0]:
                nearest[j] = (d, i)
        bad = [k for k in nearest if load[k] > 1.0 + COMPOSITION_ATOL]
        if bad:
            i = min(bad, key=lambda k: nearest[k][0])
            d, j = nearest[i]
            raise ScientificValidationError(
                f"Atoms overlap: sites {i} ({site_occupancies(structure[i])[0][0]}) and "
                f"{j} ({site_occupancies(structure[j])[0][0]}) are {d:.4f} Å apart "
                f"(< {threshold} Å, combined occupancy {load[i]:.3f} > 1). "
                "Check coordinate conversion / periodic boundaries."
            )
    return min_interatomic_distance(structure)[0]


def validate_structure(structure: Any) -> float:
    """Full input gate: non-empty, finite coordinates, valid lattice, no overlapping atoms.

    Returns the minimum interatomic distance (Å).
    """
    if structure is None or len(structure) == 0:
        raise ScientificValidationError("Structure has no sites.")
    validate_lattice(structure)
    frac = np.asarray(structure.frac_coords, dtype=float)
    if not np.all(np.isfinite(frac)):
        raise ScientificValidationError("Atomic coordinates contain NaN/Inf.")
    for idx, site in enumerate(structure):
        occ = _total_occupancy(site)
        if not math.isfinite(occ) or occ <= 0.0 or occ > 1.0 + COMPOSITION_ATOL:
            raise ScientificValidationError(f"Site {idx} has invalid total occupancy {occ!r}.")
    return validate_min_distance(structure)


def _element_fractions(structure_or_comp: Any) -> dict[str, float]:
    comp = getattr(structure_or_comp, "composition", structure_or_comp)
    comp = comp.element_composition  # drop oxidation states
    return {str(el.symbol): float(v) for el, v in comp.fractional_composition.items()}


def validate_composition_conserved(
    before: Any,
    after: Any,
    atol: float = COMPOSITION_ATOL,
    context: str = "transformation",
) -> None:
    """Raise if element fractions differ by more than ``atol`` between ``before`` and ``after``."""
    fb = _element_fractions(before)
    fa = _element_fractions(after)
    for el in sorted(set(fb) | set(fa)):
        diff = abs(fb.get(el, 0.0) - fa.get(el, 0.0))
        if diff > atol:
            raise ScientificValidationError(
                f"{context}: composition not conserved for {el} "
                f"({fb.get(el, 0.0):.5f} -> {fa.get(el, 0.0):.5f}, |diff| = {diff:.2e} > {atol})."
            )


# --------------------------------------------------------------------------------------
# Output checks
# --------------------------------------------------------------------------------------
def _hex_to_rgb(color: Any) -> tuple[int, int, int]:
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        vals = [float(c) for c in color[:3]]
        if max(vals) <= 1.0:
            vals = [v * 255.0 for v in vals]
        return (int(round(vals[0])), int(round(vals[1])), int(round(vals[2])))
    h = str(color).lstrip("#")
    if len(h) != 6:
        return (255, 255, 255)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def background_ratio(
    image: np.ndarray,
    background: Any = "#FFFFFF",
    tol: int = 4,
    transparent: bool = False,
) -> float:
    """Fraction of pixels equal to the background (alpha ~0 when ``transparent``)."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4) or arr.size == 0:
        raise ScientificValidationError(f"Rendered image has unexpected shape {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if transparent and arr.shape[2] == 4:
        bg = arr[:, :, 3] <= tol
    else:
        ref = np.asarray(_hex_to_rgb(background), dtype=np.int16)
        bg = np.all(np.abs(arr[:, :, :3].astype(np.int16) - ref) <= tol, axis=2)
    return float(bg.mean())


def validate_blank_image(
    image: np.ndarray,
    background: Any = "#FFFFFF",
    threshold: float = BLANK_BACKGROUND_RATIO,
    transparent: bool = False,
    context: str = "render",
) -> float:
    """Raise if more than ``threshold`` of the pixels are background. Returns the ratio."""
    ratio = background_ratio(image, background=background, transparent=transparent)
    if ratio > threshold:
        raise ScientificValidationError(
            f"{context}: image is {ratio * 100:.2f} % background (> {threshold * 100:.1f} %). "
            "Camera focus is off or mesh generation failed."
        )
    return ratio

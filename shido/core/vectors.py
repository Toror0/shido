"""Atom vectors (VESTA "Vectors"): magnetic moments or displacements drawn as arrows.

Spec strings (CLI ``--vector``, MCP ``vectors``), any number of them:

    "Mn:0 0 4"     every site of element Mn
    "Mn1:0 0 4"    the site(s) whose CIF label is Mn1 (a label wins over an element)
    "#3:1 0 0"     site index 3 (0-based, as in atomic_positions.csv)
    "magmom"       the site property ``magmom`` (moments read from an mCIF by
                   pymatgen, or a pymatgen Structure / POSCAR with MAGMOM)

Components are along the crystal axes in mu_B (basis "crystal", the mCIF
``_atom_site_moment.crystalaxis_x/y/z`` convention):
    M = X a/|a| + Y b/|b| + Z c/|c|
With basis "cartesian" they are Cartesian x y z (pymatgen's frame: a along x).
``magmom`` values are already Cartesian (pymatgen converts the mCIF moments and
applies the magnetic symmetry operations, time reversal included). Explicit specs
override ``magmom`` for the sites they name; later specs override earlier ones.
Zero vectors draw no arrow.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from pymatgen.core import Structure

VECTOR_BASES: tuple[str, ...] = ("crystal", "cartesian")
_MAGMOM_KEYS = ("magmom", "auto", "file", "mcif")
_ZERO = 1e-8


def normalize_vector_basis(basis: str | None) -> str:
    if basis is None or not str(basis).strip():
        return "crystal"
    key = str(basis).strip().lower()
    key = {"crystalaxis": "crystal", "axes": "crystal", "abc": "crystal", "cart": "cartesian",
           "xyz": "cartesian"}.get(key, key)
    if key not in VECTOR_BASES:
        raise ValueError(f"Unknown vector basis {basis!r}. Choose one of: {', '.join(VECTOR_BASES)}.")
    return key


def _parse_one(spec: Any) -> dict[str, Any]:
    if isinstance(spec, dict):
        if spec.get("magmom"):  # already parsed
            return {"magmom": True}
        if "target" not in spec or "vector" not in spec:
            raise ValueError(f"vector spec dict needs 'target' and 'vector': {spec!r}")
        vec = [float(v) for v in spec["vector"]]
        if len(vec) != 3:
            raise ValueError(f"vector needs 3 components: {spec!r}")
        return {"target": str(spec["target"]).strip(), "vector": tuple(vec)}
    text = str(spec).strip()
    if text.lower() in _MAGMOM_KEYS:
        return {"magmom": True}
    if ":" not in text:
        raise ValueError(f"vector spec {spec!r}: use 'TARGET:X Y Z' (e.g. 'Mn:0 0 4') or 'magmom'.")
    target, _, rest = text.partition(":")
    target = target.strip()
    parts = [p for p in re.split(r"[\s,]+", rest.strip()) if p]
    try:
        vec = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"vector spec {spec!r}: components must be numbers.") from exc
    if not target or len(vec) != 3:
        raise ValueError(f"vector spec {spec!r}: use 'TARGET:X Y Z' with 3 numbers.")
    return {"target": target, "vector": tuple(vec)}


def normalize_vectors(vectors: Any) -> list[dict[str, Any]] | None:
    """Parse vector specs; None / [] / '' -> None. A string may hold several specs split by ';'.
    A dict {"Mn": [0, 0, 4]} is also accepted."""
    if vectors is None:
        return None
    if isinstance(vectors, dict) and not ("target" in vectors and "vector" in vectors):
        items: list[Any] = [{"target": k, "vector": v} for k, v in vectors.items()]
    elif isinstance(vectors, (str, dict)):
        items = [vectors]
    else:
        items = list(vectors)
    flat: list[Any] = []
    for item in items:
        if isinstance(item, str):
            flat.extend(p for p in item.split(";") if p.strip())
        else:
            flat.append(item)
    out = [_parse_one(s) for s in flat]
    return out or None


def _site_elements(site: Any) -> list[str]:
    return [str(getattr(el, "symbol", el)) for el in site.species.elements]


def _match_sites(structure: Structure, target: str) -> list[int]:
    if target.startswith("#"):
        try:
            idx = int(target[1:])
        except ValueError as exc:
            raise ValueError(f"vector target {target!r}: '#' needs a site index.") from exc
        if not 0 <= idx < len(structure):
            raise ValueError(f"vector target {target!r}: the structure has {len(structure)} sites.")
        return [idx]
    by_label = [i for i, s in enumerate(structure) if str(getattr(s, "label", "") or "") == target]
    if by_label:
        return by_label
    sym = target[:1].upper() + target[1:].lower()
    by_elem = [i for i, s in enumerate(structure) if sym in _site_elements(s)]
    if by_elem:
        return by_elem
    raise ValueError(f"vector target {target!r} matches no site label or element in the structure.")


def crystal_to_cartesian(structure: Structure, vec: Any) -> np.ndarray:
    """Components along the unit crystal axes a/|a|, b/|b|, c/|c| -> Cartesian."""
    m = np.asarray(structure.lattice.matrix, dtype=float)
    unit = m / np.linalg.norm(m, axis=1)[:, None]
    return np.asarray(vec, dtype=float) @ unit


def _magmom_cartesian(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(3)
    if hasattr(value, "get_moment"):
        return np.asarray(value.get_moment(), dtype=float)
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 1:  # collinear scalar: along Cartesian z (pymatgen's default saxis)
        return np.array([0.0, 0.0, float(arr[0])])
    if arr.size == 3:
        return arr
    raise ValueError(f"Unsupported magmom value {value!r}.")


def resolve_site_vectors(
    structure: Structure, vectors: Any, basis: str | None = None
) -> dict[int, np.ndarray] | None:
    """Cartesian vector (mu_B, or the user's unit) per site index; zero vectors dropped.

    None when no vectors are requested. Raises if 'magmom' is asked for but the
    structure has none, or a target matches nothing.
    """
    specs = normalize_vectors(vectors)
    if specs is None:
        return None
    basis = normalize_vector_basis(basis)
    out: dict[int, np.ndarray] = {}
    for spec in specs:
        if spec.get("magmom"):
            mags = structure.site_properties.get("magmom")
            if mags is None:
                raise ValueError(
                    "vectors 'magmom': the structure has no magnetic moments "
                    "(use an mCIF with _atom_site_moment.* or give 'Mn:0 0 4')."
                )
            for i, value in enumerate(mags):
                out[i] = _magmom_cartesian(value)
    for spec in specs:
        if spec.get("magmom"):
            continue
        vec = np.asarray(spec["vector"], dtype=float)
        cart = crystal_to_cartesian(structure, vec) if basis == "crystal" else vec
        for i in _match_sites(structure, spec["target"]):
            out[i] = cart
    return {i: v for i, v in out.items() if float(np.linalg.norm(v)) > _ZERO}


__all__ = [
    "VECTOR_BASES",
    "crystal_to_cartesian",
    "normalize_vector_basis",
    "normalize_vectors",
    "resolve_site_vectors",
]

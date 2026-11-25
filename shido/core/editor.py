"""Batch 22: edit atoms / sites and build a structure from a space group (VESTA "Edit Data").

* `apply_operations` applies a list of operations in order. Each operation is a dict
  with ``"op"``:

  - ``{"op": "add", "element": "Li", "coords": [x, y, z]}`` (+ ``"occupancy"``,
    ``"species": {"Mn": 0.5, "Ga": 0.5}``, ``"cartesian": true``, ``"label"``)
  - ``{"op": "remove", <selector>}``
  - ``{"op": "replace", <selector>, "to": "K"}`` (``"to"`` may be ``{"K": 0.5, "Na": 0.5}``;
    without a selector, ``"from": "Na"`` picks every Na site)
  - ``{"op": "move", <selector>, "coords": [x, y, z]}`` or ``"shift": [dx, dy, dz]``
    (+ ``"cartesian": true``)
  - ``{"op": "occupancy", <selector>, "value": 0.5}`` (single-species sites) or
    ``"species": {"Mn": 0.5, "Ga": 0.5}``
  - ``{"op": "merge", "tol": 0.01}``: sites closer than ``tol`` Å (periodic) become one.
    Same species -> the first is kept; different species -> occupancies are summed.

  ``<selector>`` is ``"site"`` (0-based index or list, counted in the structure *at that
  step*), ``"element"`` (every site holding that element) or ``"label"`` (site label).
* `create_structure` builds a structure from a space group, the asymmetric unit
  (element + fractional coordinates) and the lattice (`Structure.from_spacegroup`).
* `write_edited` runs the validator (`core.validator.validate_structure`) first, so an
  overlap / empty cell never reaches disk, then writes ``<stem>.cif`` (and POSCAR).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pymatgen.core import Composition, Lattice, Structure

from shido.core.validator import ScientificValidationError, validate_structure

DEFAULT_MERGE_TOL = 0.01  # Å

_OP_ALIASES = {
    "add": "add", "insert": "add",
    "remove": "remove", "delete": "remove",
    "replace": "replace", "substitute": "replace", "element": "replace",
    "move": "move", "coords": "move", "position": "move",
    "occupancy": "occupancy", "occ": "occupancy",
    "merge": "merge", "merge_duplicates": "merge",
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _species_dict(site: Any) -> dict[str, float]:
    return {str(getattr(sp, "symbol", sp)): float(o) for sp, o in site.species.items()}


def _parse_species(value: Any, occupancy: float | None = None) -> dict[str, float]:
    """"Na" / {"Mn": 0.5, "Ga": 0.5} / "Mn0.5Ga0.5" -> {element: occupancy}."""
    if isinstance(value, Mapping):
        out = {str(k): float(v) for k, v in value.items()}
    elif isinstance(value, str) and value.strip():
        comp = Composition(value.strip())
        out = {str(getattr(el, "symbol", el)): float(n) for el, n in comp.items()}
        if len(out) == 1 and occupancy is None and abs(next(iter(out.values())) - 1.0) > 1e-9:
            # "Na2" is not a site; treat a bare symbol only
            raise ValueError(f"species {value!r}: give one element or a dict of occupancies.")
    else:
        raise ValueError(f"species must be an element or a dict, got {value!r}.")
    if occupancy is not None:
        if len(out) != 1:
            raise ValueError("occupancy applies to a single element; use a species dict instead.")
        out = {k: float(occupancy) for k in out}
    for el, occ in out.items():
        if not math.isfinite(occ) or occ <= 0.0 or occ > 1.0 + 1e-6:
            raise ValueError(f"occupancy of {el} must be in (0, 1], got {occ!r}.")
    if sum(out.values()) > 1.0 + 1e-6:
        raise ValueError(f"total occupancy {sum(out.values()):.4g} > 1 for {out}.")
    return out


def _vec3(value: Any, name: str) -> list[float]:
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    try:
        vals = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be 3 numbers, got {value!r}.") from exc
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be 3 finite numbers, got {value!r}.")
    return vals


def _select(structure: Structure, op: Mapping[str, Any]) -> list[int]:
    """Site indices chosen by "site" / "sites" / "element" / "label" / "from"."""
    n = len(structure)
    idx: list[int] = []
    sites = op.get("site", op.get("sites"))
    if sites is not None:
        raw = sites if isinstance(sites, (list, tuple)) else [sites]
        for s in raw:
            i = int(s)
            if i < 0:
                i += n
            if not 0 <= i < n:
                raise ValueError(f"site index {s} is out of range (0..{n - 1}).")
            idx.append(i)
    el = op.get("element") if op.get("op") not in {"add"} else None
    el = el or op.get("from")
    if el is not None:
        el = str(el)
        idx += [i for i, site in enumerate(structure) if el in _species_dict(site)]
    label = op.get("label") if op.get("op") not in {"add"} else None
    if label is not None:
        idx += [i for i, site in enumerate(structure) if str(getattr(site, "label", "")) == str(label)]
    if sites is None and el is None and label is None:
        raise ValueError(f"operation {op!r} needs a selector: site, element or label.")
    out = sorted(set(idx))
    if not out:
        raise ValueError(f"operation {op!r} selected no site.")
    return out


def _rebuild(structure: Structure, keep: list[tuple[dict[str, float], Sequence[float], str | None]]) -> Structure:
    return Structure(
        structure.lattice,
        [sp for sp, _, _ in keep],
        [c for _, c, _ in keep],
        labels=[lb for _, _, lb in keep],
    )


def merge_duplicates(structure: Structure, tol: float = DEFAULT_MERGE_TOL) -> tuple[Structure, int]:
    """Merge sites closer than ``tol`` Å (periodic). Returns (new structure, removed count).

    Same species (same occupancies) -> keep the first site only. Different species ->
    one site with the occupancies summed (e.g. Mn 0.5 + Ga 0.5 -> Mn0.5Ga0.5). A sum
    over 1 raises, because those atoms really overlap.
    """
    n = len(structure)
    if n < 2 or tol <= 0:
        return structure.copy(), 0
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    centers, points, _img, dists = structure.get_neighbor_list(r=float(tol), exclude_self=False)
    for i, j, d in zip(centers, points, dists):
        i, j = int(i), int(j)
        if i != j and float(d) <= tol:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    keep = []
    for root in sorted(groups):
        members = groups[root]
        first = structure[members[0]]
        sp = _species_dict(first)
        others = [_species_dict(structure[k]) for k in members[1:]]
        if others and not all(_same_species(sp, o) for o in others):
            summed: dict[str, float] = {}
            for d in [sp] + others:
                for el, occ in d.items():
                    summed[el] = summed.get(el, 0.0) + occ
            total = sum(summed.values())
            if total > 1.0 + 1e-3:
                raise ScientificValidationError(
                    f"Cannot merge sites {members}: total occupancy {total:.3f} > 1 "
                    f"({summed}); these atoms overlap."
                )
            sp = summed
        keep.append((sp, first.frac_coords, getattr(first, "label", None)))
    return _rebuild(structure, keep), n - len(keep)


def _same_species(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    return set(a) == set(b) and all(abs(a[k] - b[k]) < 1e-6 for k in a)


# --------------------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------------------
def normalize_operations(operations: Any) -> list[dict[str, Any]]:
    """A list of dicts, one dict, a JSON string or a path to a JSON file -> list of dicts."""
    ops = operations
    if isinstance(ops, (str, Path)):
        text = str(ops).strip()
        p = Path(text)
        if not text.startswith(("[", "{")) and p.is_file():
            text = p.read_text(encoding="utf-8-sig")
        try:
            ops = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"operations: not valid JSON ({exc}).") from exc
    if isinstance(ops, Mapping):
        ops = ops.get("operations", [ops]) if "op" not in ops else [ops]
    if not isinstance(ops, (list, tuple)):
        raise ValueError("operations must be a list of {'op': ...} dicts.")
    out = []
    for k, op in enumerate(ops):
        if not isinstance(op, Mapping) or "op" not in op:
            raise ValueError(f"operation #{k} must be a dict with 'op', got {op!r}.")
        name = _OP_ALIASES.get(str(op["op"]).strip().lower())
        if name is None:
            raise ValueError(f"operation #{k}: unknown op {op['op']!r} (add, remove, replace, move, occupancy, merge).")
        d = dict(op)
        d["op"] = name
        out.append(d)
    return out


def apply_operation(structure: Structure, op: Mapping[str, Any]) -> tuple[Structure, str]:
    """Apply one normalized operation; returns (new structure, one-line log)."""
    s = structure.copy()
    kind = op["op"]
    cart = bool(op.get("cartesian", False))
    if kind == "add":
        species_in = op.get("species", op.get("element"))
        if species_in is None:
            raise ValueError("add needs 'element' (or 'species').")
        sp = _parse_species(species_in, op.get("occupancy"))
        xyz = _vec3(op.get("coords"), "coords")
        s.insert(len(s), sp, xyz, coords_are_cartesian=cart, label=op.get("label"))
        return s, f"add {sp} at {xyz}{' (cart)' if cart else ''} -> site {len(s) - 1}"
    if kind == "merge":
        tol = float(op.get("tol", DEFAULT_MERGE_TOL))
        new, removed = merge_duplicates(s, tol)
        return new, f"merge tol={tol}: {removed} duplicate site(s) removed"
    idx = _select(s, op)
    if kind == "remove":
        s.remove_sites(idx)
        return s, f"remove sites {idx}"
    if kind == "replace":
        if "to" not in op:
            raise ValueError("replace needs 'to'.")
        el_from = op.get("from") or (op.get("element") if "site" not in op and "sites" not in op else None)
        to = op["to"]
        for i in idx:
            cur = _species_dict(s[i])
            if el_from is not None and len(cur) > 1:
                # swap only that element on a mixed site, keep its occupancy
                if str(el_from) not in cur:
                    raise ValueError(f"site {i} has no {el_from} (it has {', '.join(cur)}).")
                occ = cur.pop(str(el_from))
                new_sp = _parse_species(to, None)
                if len(new_sp) != 1:
                    raise ValueError("replacing one element on a mixed site needs a single element as 'to'.")
                el_to = next(iter(new_sp))
                cur[el_to] = cur.get(el_to, 0.0) + occ
                sp = cur
            else:
                occ_total = sum(cur.values())
                sp = _parse_species(to, None)
                if not isinstance(to, Mapping) and len(sp) == 1:
                    # a plain element keeps the site's occupancy ("Mn0.5Fe0.5" is taken as written)
                    sp = {k: occ_total for k in sp}
            new_label = op.get("new_label") or max(sp, key=sp.get)  # the label follows the new element
            s.replace(i, sp, coords=s[i].frac_coords, label=new_label)
        return s, f"replace sites {idx} -> {to}"
    if kind == "move":
        if "coords" in op:
            xyz = _vec3(op["coords"], "coords")
            for i in idx:
                s.replace(i, s[i].species, coords=xyz, coords_are_cartesian=cart, label=getattr(s[i], "label", None))
            return s, f"move sites {idx} to {xyz}"
        if "shift" in op:
            dv = _vec3(op["shift"], "shift")
            s.translate_sites(idx, dv, frac_coords=not cart, to_unit_cell=True)
            return s, f"shift sites {idx} by {dv}"
        raise ValueError("move needs 'coords' or 'shift'.")
    if kind == "occupancy":
        for i in idx:
            cur = _species_dict(s[i])
            if "species" in op:
                sp = _parse_species(op["species"], None)
            elif "value" in op or "occupancy" in op:
                if len(cur) != 1:
                    raise ValueError(f"site {i} is mixed {cur}; give 'species' with every occupancy.")
                sp = _parse_species(next(iter(cur)), float(op.get("value", op.get("occupancy"))))
            else:
                raise ValueError("occupancy needs 'value' or 'species'.")
            s.replace(i, sp, coords=s[i].frac_coords, label=getattr(s[i], "label", None))
        return s, f"occupancy of sites {idx} -> {op.get('species', op.get('value', op.get('occupancy')))}"
    raise ValueError(f"unknown op {kind!r}")  # pragma: no cover


def apply_operations(
    structure: Structure,
    operations: Any,
    merge_tol: float | None = None,
) -> tuple[Structure, list[str]]:
    """Apply the operations in order; ``merge_tol`` (Å) merges duplicates at the end."""
    s = structure.copy()
    log: list[str] = []
    for k, op in enumerate(normalize_operations(operations)):
        try:
            s, line = apply_operation(s, op)
        except ScientificValidationError:
            raise
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"operation #{k} ({op.get('op')}): {exc}") from exc
        log.append(line)
    if merge_tol:
        s, removed = merge_duplicates(s, float(merge_tol))
        log.append(f"auto merge tol={merge_tol}: {removed} duplicate site(s) removed")
    if len(s) == 0:
        raise ScientificValidationError("Every site was removed; the structure is empty.")
    return s, log


# --------------------------------------------------------------------------------------
# create from a space group
# --------------------------------------------------------------------------------------
def lattice_from_params(spacegroup: str | int, lattice: Any) -> Lattice:
    """Lattice from 6 numbers [a b c alpha beta gamma], a 3x3 matrix, or a dict.

    Missing values in a dict (or a shorter list) follow the crystal system of the space
    group: cubic needs only a; tetragonal a, c; hexagonal/trigonal a, c (gamma 120);
    orthorhombic a, b, c; monoclinic a, b, c, beta.
    """
    from pymatgen.symmetry.groups import SpaceGroup

    if isinstance(lattice, Lattice):
        return lattice
    if isinstance(lattice, str):
        lattice = [float(v) for v in lattice.replace(",", " ").split()]
    arr = np.asarray(lattice, dtype=object) if not isinstance(lattice, Mapping) else None
    if arr is not None and arr.shape == (3, 3):
        return Lattice(np.asarray(lattice, dtype=float))
    sg = SpaceGroup.from_int_number(int(spacegroup)) if str(spacegroup).isdigit() else SpaceGroup(str(spacegroup))
    system = sg.crystal_system
    if isinstance(lattice, Mapping):
        given = {k: float(v) for k, v in lattice.items() if v is not None}
    else:
        vals = [float(v) for v in lattice]
        # a shorter list holds only the free parameters of the crystal system
        short = {"cubic": ("a",), "tetragonal": ("a", "c"), "hexagonal": ("a", "c"),
                 "trigonal": ("a", "c"), "orthorhombic": ("a", "b", "c"),
                 "monoclinic": ("a", "b", "c", "beta")}.get(system)
        keys = short if short is not None and len(vals) == len(short) else ("a", "b", "c", "alpha", "beta", "gamma")
        given = dict(zip(keys, vals))
    if "a" not in given:
        raise ValueError("lattice needs at least a.")
    a = given["a"]
    if system == "cubic":
        d = dict(b=a, c=a, alpha=90.0, beta=90.0, gamma=90.0)
    elif system == "tetragonal":
        d = dict(b=a, alpha=90.0, beta=90.0, gamma=90.0)
    elif system in {"hexagonal", "trigonal"}:
        d = dict(b=a, alpha=90.0, beta=90.0, gamma=120.0)
    elif system == "orthorhombic":
        d = dict(alpha=90.0, beta=90.0, gamma=90.0)
    elif system == "monoclinic":
        d = dict(alpha=90.0, gamma=90.0)
    else:
        d = {}
    d.update(given)
    missing = [k for k in ("a", "b", "c", "alpha", "beta", "gamma") if k not in d]
    if missing:
        raise ValueError(f"{system} space group {sg.symbol}: lattice is missing {', '.join(missing)}.")
    return Lattice.from_parameters(d["a"], d["b"], d["c"], d["alpha"], d["beta"], d["gamma"])


def create_structure(
    spacegroup: str | int,
    lattice: Any,
    sites: Sequence[Mapping[str, Any]] | Sequence[Sequence[Any]],
    merge_tol: float | None = DEFAULT_MERGE_TOL,
) -> Structure:
    """Space group + lattice + asymmetric unit -> full structure.

    ``sites``: [{"element": "Na", "coords": [0, 0, 0]}, ...] (+ "occupancy", "species",
    "label"), or [["Na", 0, 0, 0], ["Cl", 0.5, 0.5, 0.5], ...].
    """
    if isinstance(sites, str):
        sites = json.loads(sites)
    if not sites:
        raise ValueError("sites: give at least one atom of the asymmetric unit.")
    species, coords, labels = [], [], []
    for k, site in enumerate(sites):
        if isinstance(site, Mapping):
            sp_in = site.get("species", site.get("element"))
            sp = _parse_species(sp_in, site.get("occupancy"))
            xyz = _vec3(site.get("coords"), f"sites[{k}].coords")
            label = site.get("label")
        else:
            row = list(site)
            if len(row) not in (4, 5):
                raise ValueError(f"sites[{k}] must be [element, x, y, z(, occupancy)], got {site!r}.")
            sp = _parse_species(row[0], float(row[4]) if len(row) == 5 else None)
            xyz = _vec3(row[1:4], f"sites[{k}]")
            label = None
        species.append(Composition(sp))
        coords.append(xyz)
        labels.append(label)
    latt = lattice_from_params(spacegroup, lattice)
    sg = int(spacegroup) if str(spacegroup).isdigit() else str(spacegroup)
    try:
        struct = Structure.from_spacegroup(
            sg, latt, species, coords, tol=1e-4,
            labels=labels if any(lb is not None for lb in labels) else None,
        )
    except ValueError as exc:
        raise ValueError(f"from_spacegroup({spacegroup}): {exc}") from exc
    if merge_tol:
        struct, _ = merge_duplicates(struct, float(merge_tol))
    return struct


# --------------------------------------------------------------------------------------
# report / write
# --------------------------------------------------------------------------------------
def structure_summary(structure: Structure, symprec: float = 0.01) -> dict[str, Any]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    a, b, c = structure.lattice.abc
    al, be, ga = structure.lattice.angles
    out: dict[str, Any] = {
        "formula": structure.composition.reduced_formula,
        "composition": {str(k): round(float(v), 6) for k, v in structure.composition.get_el_amt_dict().items()},
        "n_sites": len(structure),
        "a": round(a, 6), "b": round(b, 6), "c": round(c, 6),
        "alpha": round(al, 4), "beta": round(be, 4), "gamma": round(ga, 4),
        "volume": round(float(structure.volume), 6),
        "density": round(float(structure.density), 6),
    }
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        out["spacegroup_symbol"] = sga.get_space_group_symbol()
        out["spacegroup_number"] = int(sga.get_space_group_number())
    except Exception:  # disordered sites etc.
        out["spacegroup_symbol"] = None
        out["spacegroup_number"] = None
    return out


def write_edited(
    structure: Structure,
    output_dir: str | Path,
    stem: str = "edited",
    poscar: bool = False,
    cif_symprec: float | None = None,
) -> dict[str, str]:
    """Validate, then write <stem>.cif (P1 unless cif_symprec) and optionally <stem>.vasp."""
    from pymatgen.io.cif import CifWriter

    validate_structure(structure)  # raises before anything is written
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    cif_path = out / f"{stem}.cif"
    structure = structure.copy().relabel_sites()  # CIF needs unique site labels
    try:
        CifWriter(structure, symprec=cif_symprec).write_file(str(cif_path))
    except Exception:
        if cif_symprec is None:
            raise
        CifWriter(structure).write_file(str(cif_path))
    files["cif"] = str(cif_path.resolve())
    if poscar:
        from pymatgen.io.vasp import Poscar

        from shido.analyzer import to_ordered_proxy  # lazy: analyzer imports core

        vasp = out / f"{stem}.vasp"
        Poscar(to_ordered_proxy(structure)).write_file(str(vasp))
        files["poscar"] = str(vasp.resolve())
    return files


def edit_report(before: Structure | None, after: Structure, log: Iterable[str]) -> dict[str, Any]:
    """Summary dict used by the CLI / MCP (before is None for create)."""
    min_d = validate_structure(after)
    rep: dict[str, Any] = {"operations": list(log), "output": structure_summary(after),
                           "min_distance": round(float(min_d), 4)}
    if before is not None:
        rep["input"] = structure_summary(before)
    return rep

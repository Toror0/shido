"""Miller-index surface cleaving, termination enumeration, and slab export."""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, SlabGenerator
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from shido.analyzer import compute_bonds, to_ordered_proxy
from shido.settings import get as _setting
from shido.renderer import pretty_formula, render_slab_views
from shido.core.output_paths import DATA_DIR, FIG_DIR, STRUCT_DIR  # noqa: E402
from shido.core.validator import (
    validate_composition_conserved,
    validate_miller_index,
    validate_structure,
)

# Human-friendly crystallographic plane presets (major Miller indices)
MAJOR_PLANES: list[dict[str, Any]] = [
    {
        "key": "111",
        "hkl": (1, 1, 1),
        "label": "(111) — FCC / ホイスラー最密面・双晶面",
        "aliases": ["111", "(111)", "1 1 1", "close-packed", "slip-fcc", "twin-fcc"],
    },
    {
        "key": "110",
        "hkl": (1, 1, 0),
        "label": "(110) — BCC / ホイスラー最密面・すべり面",
        "aliases": ["110", "(110)", "1 1 0", "slip-bcc", "bcc"],
    },
    {
        "key": "100",
        "hkl": (1, 0, 0),
        "label": "(100) — 立方晶基本面（キューブ面）",
        "aliases": ["100", "(100)", "1 0 0", "cube", "cubic"],
    },
    {
        "key": "001",
        "hkl": (0, 0, 1),
        "label": "(001) — 正方晶・六方晶の底面（基底面）",
        "aliases": ["001", "(001)", "0 0 1", "basal"],
    },
    {
        "key": "101",
        "hkl": (1, 0, 1),
        "label": "(101) — マルテンサイト変態の晶癖面候補",
        "aliases": ["101", "(101)", "1 0 1", "habit"],
    },
    {
        "key": "112",
        "hkl": (1, 1, 2),
        "label": "(112) — BCC双晶面・高指数すべり面",
        "aliases": ["112", "(112)", "1 1 2", "twin-bcc"],
    },
]


def resolve_miller_index(spec: str | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    """Parse a Miller index from a preset name ('111', 'habit', '(110)') or integer triplet."""
    if isinstance(spec, (tuple, list)) and len(spec) == 3:
        return validate_miller_index(spec)
    text = str(spec).strip().lower()
    for preset in MAJOR_PLANES:
        if text == preset["key"] or text in preset["aliases"]:
            return tuple(preset["hkl"])  # type: ignore[return-value]
    cleaned = text.strip("()[] ").replace(",", " ")
    parts = [p for p in cleaned.split() if p]
    if len(parts) == 3:
        return validate_miller_index(int(x) for x in parts)
    if len(cleaned) == 3 and cleaned.isdigit():
        return validate_miller_index(int(x) for x in cleaned)
    raise ValueError(
        f"Invalid plane specification '{spec}'. Use e.g. '111', '(110)', '1 0 1', or 'habit'."
    )

SUMMARY_COLUMNS: list[str] = [
    "termination_index",
    "miller_index",
    "d_hkl_angstrom",
    "surface_area_angstrom2",
    "num_layers",
    "num_sites",
    "formula",
    "reduced_formula",
    "top_surface_elements",
    "shift",
    "is_symmetric",
    "is_polar",
    "dipole_moment_eA",
    "dipole_per_area_e_per_A",
    "net_charge_e",
    "polarity_method",
    "cif_file",
    "poscar_file",
]


def _cluster_z_layers(slab: Slab, z_tol: float = 0.35) -> list[list[tuple[float, Any]]]:
    """Cluster slab sites into atomic layers along Cartesian z."""
    if len(slab) == 0:
        return []

    z_entries = sorted((float(site.coords[2]), site) for site in slab)
    layers: list[list[tuple[float, Any]]] = []

    for z_val, site in z_entries:
        if not layers or (z_val - layers[-1][0][0]) > z_tol:
            layers.append([(z_val, site)])
        else:
            layers[-1].append((z_val, site))

    return layers


def _layer_composition_signature(
    layer: list[tuple[float, Any]],
) -> tuple[tuple[str, float], ...]:
    """Return a canonical sorted (element, occupancy_sum) tuple for an atomic layer."""
    counts: Counter[str] = Counter()
    for _, site in layer:
        for el, occ in site.species.items():
            counts[el.symbol] += float(occ)
    return tuple(sorted((sym, round(val, 4)) for sym, val in counts.items()))


def _surface_layer_signatures(
    slab: Slab,
    z_tol: float = 0.35,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    """Return (bottom_layer_signature, top_layer_signature) for a slab."""
    layers = _cluster_z_layers(slab, z_tol=z_tol)
    if not layers:
        return (), ()
    return (
        _layer_composition_signature(layers[0]),
        _layer_composition_signature(layers[-1]),
    )


def count_atomic_layers(slab: Slab, z_tol: float = 0.35) -> tuple[int, str]:
    """Count distinct atomic layers along Cartesian z and identify top-layer elements.

    Sites are sorted by Cartesian z-coordinate and clustered so that all atoms
    within ``z_tol`` Angstrom of a layer's base z-coordinate belong to the same
    atomic layer.

    Args:
        slab: Pymatgen ``Slab`` (ideally orthogonalized along c).
        z_tol: Layer clustering tolerance in Angstrom.

    Returns:
        Tuple of ``(num_layers, top_layer_elements_csv)`` where
        ``top_layer_elements_csv`` is a comma-separated sorted list of unique
        element symbols present in the highest-z atomic layer.
    """
    layers = _cluster_z_layers(slab, z_tol=z_tol)
    if not layers:
        return 0, ""

    top_layer = layers[-1]
    sorted_top_layer_elements = sorted(
        {el.symbol for _, site in top_layer for el in site.species.elements}
    )
    return len(layers), ",".join(sorted_top_layer_elements)


# Polar threshold on the out-of-plane dipole per surface area (e/Angstrom). Same
# default as pymatgen ``Slab.is_polar(tol_dipole_per_unit_area=1e-3)``.
_DEFAULT_POLAR_DIPOLE_TOL = 1e-3


def _is_metallic(slab: Structure) -> bool:
    """True for metals and intermetallics: every element is a metal or a metalloid, and
    at least one is a metal (Co2MnSi, Ni2MnSb, Fe3Al... are not ionic crystals).

    Metalloids per pymatgen: B, Si, Ge, As, Sb, Te, Po.
    """
    elements = slab.composition.elements
    if not elements:
        return False
    is_metal = [bool(getattr(el, "is_metal", False)) for el in elements]
    is_metalloid = [bool(getattr(el, "is_metalloid", False)) for el in elements]
    return any(is_metal) and all(m or ml for m, ml in zip(is_metal, is_metalloid))


def guess_bulk_oxidation_states(structure: Structure) -> dict[str, float] | None:
    """Most likely oxidation state per element of the (charge-neutral) bulk, or None.

    Guessing on the bulk and reusing the result for every slab keeps the charges right
    for non-stoichiometric slabs (e.g. Na15Cl16), where a guess on the slab alone fails.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            guesses = structure.composition.oxi_state_guesses(max_sites=-50)
    except Exception:
        return None
    if not guesses:
        return None
    oxi = {str(el): float(v) for el, v in guesses[0].items()}
    return oxi if any(abs(v) > 1e-6 for v in oxi.values()) else None


def _slab_normal(slab: Structure) -> np.ndarray:
    """Unit surface normal (``Slab.normal``, else a x b)."""
    n = getattr(slab, "normal", None)
    if n is None:
        n = np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1])
    n = np.asarray(n, dtype=float)
    return n / max(float(np.linalg.norm(n)), 1e-12)


def evaluate_slab_polarity_detail(
    slab: Slab,
    bulk_oxidation_states: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Classify a slab as polar / non-polar with the ionic (Tasker) dipole test.

    - Metals and intermetallics (only metals / metalloids, at least one metal) are never
      polar: conduction electrons screen any surface dipole, so the test does not apply.
    - Charges: ``bulk_oxidation_states`` (element -> oxidation state guessed on the
      bulk, see :func:`guess_bulk_oxidation_states`) when given, else a guess on the
      slab itself. If none can be assigned (all zero), the test is not applicable.
    - A slab whose total charge is not zero (non-stoichiometric termination with bulk
      charges) is polar (``polarity_method="non_neutral"``).
    - Otherwise the dipole ``sum q_i (r_i . n - mean)`` along the surface normal ``n``
      (e*Angstrom) is divided by the surface area, and the slab is polar when
      ``|dipole / area| > slab.polar_dipole_tol`` (default 1e-3 e/Angstrom).

    Returns a dict with ``is_polar``, ``dipole_eA``, ``dipole_per_area_e_per_A``
    (NaN when not applicable), ``net_charge_e`` and ``polarity_method`` ("metallic",
    "no_oxidation_states", "non_neutral" or "oxidation_state_dipole").
    """
    result: dict[str, Any] = {
        "is_polar": False,
        "dipole_eA": float("nan"),
        "dipole_per_area_e_per_A": float("nan"),
        "net_charge_e": float("nan"),
        "polarity_method": "metallic",
    }
    if len(slab) == 0 or _is_metallic(slab):
        return result

    charges: list[float] = []
    if bulk_oxidation_states:
        for site in slab:
            charges.append(
                sum(
                    float(bulk_oxidation_states.get(str(getattr(sp, "symbol", sp)), 0.0)) * float(occ)
                    for sp, occ in site.species.items()
                )
            )
    else:
        try:
            slab_copy = slab.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                slab_copy.add_oxidation_state_by_guess()
            for site in slab_copy:
                q_site = 0.0
                for sp, occ in site.species.items():
                    oxi = getattr(sp, "oxi_state", None)
                    if oxi is not None:
                        q_site += float(oxi) * float(occ)
                charges.append(q_site)
        except Exception:
            charges = []

    if not charges or not any(abs(q) > 1e-6 for q in charges):
        result["polarity_method"] = "no_oxidation_states"
        return result

    normal = _slab_normal(slab)
    heights = [float(np.dot(np.asarray(site.coords, dtype=float), normal)) for site in slab]
    h_mean = sum(heights) / len(heights)
    dipole_eA = float(sum(q * (h - h_mean) for q, h in zip(charges, heights)))
    area = float(slab.surface_area) if hasattr(slab, "surface_area") else float(
        np.linalg.norm(np.cross(slab.lattice.matrix[0], slab.lattice.matrix[1]))
    )
    dipole_per_area = dipole_eA / area if area > 1e-12 else 0.0
    net_charge = float(sum(charges))
    tol = float(_setting("slab.polar_dipole_tol", _DEFAULT_POLAR_DIPOLE_TOL))
    non_neutral = abs(net_charge) > 1e-3
    result.update(
        is_polar=bool(non_neutral or abs(dipole_per_area) > tol),
        dipole_eA=round(dipole_eA, 5),
        dipole_per_area_e_per_A=round(dipole_per_area, 6),
        net_charge_e=round(net_charge, 6),
        polarity_method="non_neutral" if non_neutral else "oxidation_state_dipole",
    )
    return result


def evaluate_slab_polarity(slab: Slab) -> tuple[bool, float]:
    """Backward-compatible wrapper: ``(is_polar, dipole_eA)``.

    See :func:`evaluate_slab_polarity_detail`. ``dipole_eA`` is NaN when the
    ionic dipole test does not apply (metals / no oxidation states).
    """
    detail = evaluate_slab_polarity_detail(slab)
    return bool(detail["is_polar"]), float(detail["dipole_eA"])


def _polarity_label(is_polar: bool, detail: dict[str, Any]) -> str:
    """Short caption text for the polarity result."""
    method = detail.get("polarity_method")
    if method == "metallic":
        return "Non-polar (metal)"
    if method == "no_oxidation_states":
        return "Non-polar (no oxidation states)"
    if method == "non_neutral":
        return "Polar (charged termination)"
    return "Polar" if is_polar else "Non-polar"


def deduplicate_slabs(slabs: list[Slab], z_tol: float | None = None) -> list[Slab]:
    """Deduplicate symmetry-equivalent slabs while preserving distinct terminations.

    Uses ``StructureMatcher`` without primitive-cell reduction or scaling, and
    additionally verifies that top and bottom surface layer compositions match
    (either directly or under top/bottom inversion) before treating two slabs
    as duplicates.

    Args:
        slabs: Candidate ``Slab`` instances in generation order.
        z_tol: Layer clustering tolerance in Angstrom (default
            ``slab.layer_z_tol`` from settings).

    Returns:
        Order-preserving list of unique ``Slab`` instances.
    """
    if z_tol is None:
        z_tol = float(_setting("slab.layer_z_tol", 0.35))
    if len(slabs) <= 1:
        return list(slabs)

    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=False,
        scale=False,
    )

    unique_slabs: list[Slab] = []
    unique_sigs: list[tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]] = []

    for candidate in slabs:
        cand_bot, cand_top = _surface_layer_signatures(candidate, z_tol=z_tol)
        is_duplicate = False

        for existing, (ex_bot, ex_top) in zip(unique_slabs, unique_sigs):
            surface_match = (cand_bot == ex_bot and cand_top == ex_top) or (
                cand_bot == ex_top and cand_top == ex_bot
            )
            if not surface_match:
                continue
            try:
                if matcher.fit(existing, candidate):
                    is_duplicate = True
                    break
            except Exception:
                continue

        if not is_duplicate:
            unique_slabs.append(candidate)
            unique_sigs.append((cand_bot, cand_top))

    return unique_slabs


def cut_slabs(
    structure: Structure,
    miller_index: tuple[int, int, int],
    output_dir: str | Path,
    min_slab_size: float | None = None,
    min_vacuum_size: float | None = None,
    filter_polar: bool | None = None,
    deduplicate: bool | None = None,
    layer_z_tol: float | None = None,
    transparent_background: bool | None = None,
) -> dict[str, Any]:
    """Cleave a bulk crystal along a Miller plane and export filtered terminations.

    Safely converts disordered inputs to an ordered proxy, standardizes to the
    conventional cell, enumerates terminations via ``SlabGenerator``, optionally
    deduplicates symmetry-equivalent slabs and filters polar surfaces (with a
    safety fallback if all terminations are polar), exports each orthogonalized
    slab to CIF, renders annotated top/side views, and writes
    ``surface_summary.csv``.

    Args:
        structure: Input bulk pymatgen ``Structure``.
        miller_index: Tuple ``(h, k, l)`` specifying the cleavage plane.
        output_dir: Root directory where ``structures/``, ``images/``, and
            ``data/`` subdirectories will be created.
        min_slab_size: Minimum slab thickness in Angstrom.
        min_vacuum_size: Minimum vacuum padding thickness in Angstrom.
        filter_polar: If ``True``, exclude polar terminations (falls back to
            keeping all unique slabs if every termination is polar).
        deduplicate: If ``True``, remove symmetry-equivalent duplicate slabs.
        layer_z_tol: Atoms closer than this in z [Angstrom] count as one layer.
        transparent_background: Transparent PNG background (``None``: use
            ``render.transparent_background``).

        Any argument left as ``None`` is read from the settings
        (``slab.min_slab_size``, ``slab.min_vacuum_size``, ``slab.filter_polar``,
        ``slab.deduplicate``, ``slab.layer_z_tol``) at call time.

    Returns:
        Dictionary summarizing the cleavage results, filtering counts, and
        generated file paths.
    """
    if len(miller_index) != 3:
        raise ValueError(f"miller_index must have 3 integers, got {miller_index!r}")

    if min_slab_size is None:
        min_slab_size = float(_setting("slab.min_slab_size", 10.0))
    if min_vacuum_size is None:
        min_vacuum_size = float(_setting("slab.min_vacuum_size", 10.0))
    if filter_polar is None:
        filter_polar = bool(_setting("slab.filter_polar", True))
    if deduplicate is None:
        deduplicate = bool(_setting("slab.deduplicate", True))
    if layer_z_tol is None:
        layer_z_tol = float(_setting("slab.layer_z_tol", 0.35))

    h, k, l = validate_miller_index(miller_index)

    out_path = Path(output_dir).resolve()
    structures_dir = out_path / STRUCT_DIR
    images_dir = out_path / FIG_DIR
    data_dir = out_path / DATA_DIR
    structures_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    validate_structure(structure)
    work_struct = to_ordered_proxy(structure)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sga = SpacegroupAnalyzer(work_struct, symprec=0.1)
        conv_struct = sga.get_conventional_standard_structure()
        d_hkl = float(conv_struct.lattice.d_hkl((h, k, l)))

        slabgen = SlabGenerator(
            initial_structure=conv_struct,
            miller_index=(h, k, l),
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=True,
            lll_reduce=True,
            in_unit_planes=False,
        )
        raw_slabs = slabgen.get_slabs(ftol=0.1, tol=0.1)

    raw_terminations = len(raw_slabs)
    unique_slabs = (
        deduplicate_slabs(raw_slabs, z_tol=layer_z_tol) if deduplicate else list(raw_slabs)
    )
    after_dedup = len(unique_slabs)

    # Charges from the bulk (carried over to every slab, also non-stoichiometric ones)
    bulk_oxi = None if _is_metallic(conv_struct) else guess_bulk_oxidation_states(conv_struct)
    evaluated: list[tuple[Slab, bool, dict[str, Any]]] = []
    for slab in unique_slabs:
        detail = evaluate_slab_polarity_detail(slab, bulk_oxidation_states=bulk_oxi)
        evaluated.append((slab, bool(detail["is_polar"]), detail))

    polar_fallback = False
    if filter_polar:
        kept = [item for item in evaluated if not item[1]]
        polar_filtered_count = len(evaluated) - len(kept)
        if len(kept) == 0 and len(evaluated) > 0:
            polar_fallback = True
            kept = evaluated
            polar_filtered_count = 0
    else:
        kept = evaluated
        polar_filtered_count = 0

    hkl = f"{h}{k}{l}"
    rows: list[dict[str, Any]] = []
    cif_paths: list[str] = []
    poscar_paths: list[str] = []
    image_paths: list[str] = []

    for idx, (slab, is_polar, polar_detail) in enumerate(kept):
        ortho_slab = slab.get_orthogonal_c_slab()
        # Guardrail: the cut must keep the bulk stoichiometry and must not overlap atoms.
        validate_composition_conserved(conv_struct, ortho_slab, context=f"slab ({h}{k}{l}) term{idx}")
        validate_structure(ortho_slab)

        cif_path = (structures_dir / f"slab_{hkl}_term{idx}.cif").resolve()
        CifWriter(ortho_slab).write_file(str(cif_path))
        cif_paths.append(str(cif_path))

        poscar_path = (structures_dir / f"POSCAR_slab_{hkl}_term{idx}").resolve()
        Poscar(ortho_slab).write_file(str(poscar_path))
        poscar_paths.append(str(poscar_path))

        num_layers, top_elements = count_atomic_layers(ortho_slab, z_tol=layer_z_tol)
        slab_caption = (
            f"{pretty_formula(ortho_slab)} ({h}{k}{l}) term{idx} | "
            f"d={d_hkl:.3f} \u00c5 | Top: {top_elements} | "
            f"{_polarity_label(is_polar, polar_detail)}"
        )

        _, slab_bonds, _ = compute_bonds(ortho_slab)
        img_map = render_slab_views(
            ortho_slab,
            out_path,
            f"slab_{hkl}_term{idx}",
            render_bonds=slab_bonds,
            miller_index=(h, k, l),
            slab_caption=slab_caption,
            transparent_background=transparent_background,
        )
        if isinstance(img_map, dict):
            image_paths.extend(str(Path(p).resolve()) for p in img_map.values() if p)
        elif isinstance(img_map, (list, tuple)):
            image_paths.extend(str(Path(p).resolve()) for p in img_map if p)
        elif img_map:
            image_paths.append(str(Path(img_map).resolve()))

        shift_val = getattr(ortho_slab, "shift", getattr(slab, "shift", 0.0))

        rows.append(
            {
                "termination_index": idx,
                "miller_index": f"({h},{k},{l})",
                "d_hkl_angstrom": round(d_hkl, 5),
                "surface_area_angstrom2": round(float(ortho_slab.surface_area), 5),
                "num_layers": num_layers,
                "num_sites": len(ortho_slab),
                "formula": ortho_slab.composition.formula,
                "reduced_formula": ortho_slab.composition.reduced_formula,
                "top_surface_elements": top_elements,
                "shift": round(float(shift_val), 6),
                "is_symmetric": bool(ortho_slab.is_symmetric()),
                "is_polar": bool(is_polar),
                "dipole_moment_eA": polar_detail["dipole_eA"],
                "dipole_per_area_e_per_A": polar_detail["dipole_per_area_e_per_A"],
                "net_charge_e": polar_detail.get("net_charge_e", float("nan")),
                "polarity_method": polar_detail["polarity_method"],
                "cif_file": str(cif_path),
                "poscar_file": str(poscar_path),
            }
        )

    csv_path = (data_dir / "surface_summary.csv").resolve()
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(csv_path, index=False)

    return {
        "output_dir": str(out_path),
        "miller_index": [h, k, l],
        "d_hkl_angstrom": round(d_hkl, 5),
        "num_terminations": len(kept),
        "raw_terminations": raw_terminations,
        "after_dedup": after_dedup,
        "polar_filtered_count": polar_filtered_count,
        "polar_fallback": polar_fallback,
        "settings_used": {
            "min_slab_size": min_slab_size,
            "min_vacuum_size": min_vacuum_size,
            "filter_polar": filter_polar,
            "deduplicate": deduplicate,
            "layer_z_tol": layer_z_tol,
        },
        "surface_summary_csv": str(csv_path),
        "cif_files": cif_paths,
        "poscar_files": poscar_paths,
        "images": image_paths,
    }

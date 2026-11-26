"""Crystal structure analysis, CSV export, and Ni2MnGa demo CIF generation."""

from __future__ import annotations

import itertools
import math
import re
import shlex
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.analysis.local_env import CrystalNN, MinimumDistanceNN
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import logging

from shido.core.output_paths import DATA_DIR, STRUCT_DIR  # noqa: E402
from shido.core.species import dominant_element
from shido.core.validator import (
    ScientificValidationError,
    validate_composition_conserved,
    validate_structure,
)

_logger = logging.getLogger("shido")

# Suppress noisy pymatgen UserWarnings (e.g., CIF rounding / electronegativity fallbacks)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen(\..*)?")

BOND_COLUMNS = [
    "site_i",
    "element_i",
    "site_j",
    "element_j",
    "element_pair",
    "image",
    "distance_angstrom",
]

BOND_ANGLE_COLUMNS = [
    "site_center",
    "element_center",
    "site_a",
    "element_a",
    "site_b",
    "element_b",
    "angle_deg",
]

ATOMIC_POSITION_COLUMNS = [
    "site",
    "element",
    "occupancy",
    "frac_x",
    "frac_y",
    "frac_z",
    "cart_x",
    "cart_y",
    "cart_z",
    "wyckoff",
]

POSCAR_CELL_CHOICES = ("input", "conventional", "primitive")

# Guards against combinatorial blow-up of the bond-angle table on large cells.
MAX_ANGLE_NEIGHBORS_PER_SITE = 16
MAX_BOND_ANGLE_ROWS = 50_000

_UNCERTAINTY_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\(\d+\)$")


def sanitize_name(name: str) -> str:
    """Replace non-[A-Za-z0-9_-] characters with '_', strip '_', default 'structure'."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")
    return cleaned or "structure"


def _default_a(a: float | None) -> float:
    """Lattice parameter: the given value, else ``lattice.a_austenite`` from the settings (read at call time)."""
    if a is not None:
        return float(a)
    from shido.settings import get

    return float(get("lattice.a_austenite", 5.825))


def build_ni2mnga_structure(a: float | None = None) -> Structure:
    """Generate an L2_1 Heusler alloy Ni2MnGa cubic crystal structure (Fm-3m).

    ``a=None`` uses ``lattice.a_austenite`` from the settings.
    """
    a = _default_a(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Structure.from_spacegroup(
            sg="Fm-3m",
            lattice=Lattice.cubic(float(a)),
            species=["Mn", "Ga", "Ni"],
            coords=[
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
            ],
        )


def build_ni2mnga_b2_disordered(a: float | None = None) -> Structure:
    """Generate a B2-disordered Heusler alloy Ni2(Mn0.5Ga0.5)2 cubic structure (Fm-3m).

    ``a=None`` uses ``lattice.a_austenite`` from the settings.
    """
    a = _default_a(a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(float(a)),
            [{"Mn": 0.5, "Ga": 0.5}, {"Mn": 0.5, "Ga": 0.5}, "Ni"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )


def generate_demo_cif(
    output_path: str | Path,
    a: float | None = None,
) -> tuple[Structure, Path]:
    """Generate the Ni2MnGa L2_1 demo structure and write it to a CIF file (``a=None`` -> settings)."""
    structure = build_ni2mnga_structure(a=_default_a(a))
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        writer = CifWriter(structure, symprec=0.1)
        writer.write_file(str(out_path))
    return structure, out_path.resolve()


def _species_symbol(sp: Any) -> str:
    """Return a clean element/species symbol string."""
    return str(getattr(sp, "symbol", str(sp)))


def to_ordered_proxy(structure: Structure) -> Structure:
    """Return an ordered copy of `structure` that keeps the overall composition.

    Sites that share the same partial-occupancy set (e.g. all 4a/4b sites of a B2
    cell with Mn 0.5 / Ga 0.5) are grouped, and each species gets an integer number
    of those sites proportional to its occupancy (largest-remainder rounding),
    placed in an interleaved, deterministic order. A B2 Ni2MnGa cell therefore
    becomes Ni8Mn4Ga4 instead of losing Mn entirely. Vacancy fractions are spread
    over the species present. A group too small to split (e.g. a single 50:50 site
    in a primitive cell) falls back to the highest-occupancy species.
    """
    if structure.is_ordered:
        return structure.copy()

    groups: dict[tuple[tuple[str, float], ...], list[int]] = {}
    for idx, site in enumerate(structure):
        if site.is_ordered:
            continue
        items = list(site.species.items())
        if not items:
            raise ValueError("Encountered empty site with no species.")
        key = tuple(sorted((_species_symbol(sp), round(float(occ), 4)) for sp, occ in items))
        groups.setdefault(key, []).append(idx)

    assigned: dict[int, Any] = {}
    for idx_list in groups.values():
        species_occ = list(structure[idx_list[0]].species.items())
        total_occ = sum(float(o) for _, o in species_occ)
        n_sites = len(idx_list)
        exact = [(sp, float(o) / total_occ * n_sites) for sp, o in species_occ]
        counts = {id(sp): int(math.floor(x)) for sp, x in exact}
        leftover = n_sites - sum(counts.values())
        # Largest remainder, ties broken by symbol for determinism
        for sp, x in sorted(exact, key=lambda t: (-(t[1] - math.floor(t[1])), _species_symbol(t[0])))[:leftover]:
            counts[id(sp)] += 1
        targets = [(sp, counts[id(sp)]) for sp, _ in exact if counts[id(sp)] > 0]
        used = {id(sp): 0 for sp, _ in targets}
        # Interleave: give each site the species that is furthest behind its quota
        for k, site_idx in enumerate(idx_list):
            sp = min(
                targets,
                key=lambda t: (used[id(t[0])] / t[1] - (k + 1) / n_sites, _species_symbol(t[0])),
            )[0]
            used[id(sp)] += 1
            assigned[site_idx] = sp

    ordered_species: list[Any] = []
    frac_coords: list[np.ndarray] = []
    for idx, site in enumerate(structure):
        ordered_species.append(next(iter(site.species)) if site.is_ordered else assigned[idx])
        frac_coords.append(np.asarray(site.frac_coords, dtype=float))

    proxy = Structure(
        lattice=structure.lattice,
        species=ordered_species,
        coords=frac_coords,
        coords_are_cartesian=False,
    )
    _warn_if_proxy_composition_drifts(structure, proxy)
    return proxy


def _warn_if_proxy_composition_drifts(original: Structure, proxy: Structure, atol: float = 1e-3) -> None:
    """Log a warning when integer site assignment changed the element fractions.

    Not an error: a cell too small for the occupancies (e.g. one 50:50 site) cannot be
    ordered exactly. The warning makes the deviation visible instead of silent.
    """
    before = original.composition.element_composition.fractional_composition
    after = proxy.composition.element_composition.fractional_composition
    drift = {
        str(el.symbol): (float(before[el]), float(after[el]))
        for el in set(before.elements) | set(after.elements)
        if abs(float(before[el]) - float(after[el])) > atol
    }
    if drift:
        details = ", ".join(f"{el} {b:.4f}->{a:.4f}" for el, (b, a) in sorted(drift.items()))
        _logger.warning(
            "Ordered proxy changes the composition (cell too small for the occupancies): %s. "
            "Use a supercell for an exact ordered model.",
            details,
        )


def summarize_occupancies(structure: Structure) -> tuple[bool, int, str]:
    """Summarize site occupancies as (is_ordered, num_disordered_sites, summary_str)."""
    if structure.is_ordered:
        return True, 0, "ordered"

    disordered_labels: list[str] = []
    for site in structure:
        if not site.is_ordered:
            parts = [
                f"{_species_symbol(sp)}{float(occ):.2f}"
                for sp, occ in sorted(
                    site.species.items(),
                    key=lambda kv: (-float(kv[1]), _species_symbol(kv[0])),
                )
            ]
            disordered_labels.append("".join(parts) if parts else str(site.species_string))

    if not disordered_labels:
        return True, 0, "ordered"

    counts = Counter(disordered_labels)
    summary_str = ", ".join(
        f"{label} (x{count})"
        for label, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return False, len(disordered_labels), summary_str


def _parse_cif_parser(parser: CifParser) -> Structure:
    """Extract the first conventional structure from a CifParser across pymatgen versions."""
    if hasattr(parser, "parse_structures"):
        structures = parser.parse_structures(primitive=False)
    else:
        structures = parser.get_structures(primitive=False)
    if not structures:
        raise ValueError("CifParser returned no structures.")
    return structures[0]


def _tokenize_cif_line(line: str) -> list[str]:
    """Split a CIF data row respecting quoted values, with whitespace fallback."""
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def _sanitize_occupancy_token(token: str) -> tuple[str, bool]:
    """Sanitize broken CIF occupancy tokens such as '0.50(2)' or '1.05'."""
    changed = False
    cleaned = token.strip()

    match = _UNCERTAINTY_RE.match(cleaned)
    if match:
        cleaned = match.group(1)
        changed = True

    try:
        val = float(cleaned)
    except ValueError:
        return token, changed

    if val > 1.0:
        cleaned = "1.0"
        changed = True
    elif val < 0.0:
        cleaned = "0.0"
        changed = True

    return cleaned, changed


def _clean_cif_text(raw_text: str) -> tuple[str, list[str]]:
    """Clean malformed CIF text (BOM, CRLF/tabs, broken occupancies, corrupted trailing lines)."""
    notes: list[str] = []
    text = raw_text

    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
        notes.append("stripped UTF-8 BOM")

    if "\r" in text or "\t" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
        notes.append("normalized CRLF/tabs")

    raw_lines = [ln.rstrip() for ln in text.split("\n")]
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()

    cleaned_lines: list[str] = []
    in_loop = False
    in_loop_headers = False
    loop_headers: list[str] = []
    occ_idx: int | None = None
    sanitized_occ_count = 0
    dropped_rows = 0

    for line in raw_lines:
        stripped = line.strip()

        if not stripped:
            if in_loop and not in_loop_headers:
                in_loop = False
                loop_headers = []
                occ_idx = None
            cleaned_lines.append("")
            continue

        if stripped.startswith("#"):
            cleaned_lines.append(line)
            continue

        lower = stripped.lower()
        if lower == "loop_":
            in_loop = True
            in_loop_headers = True
            loop_headers = []
            occ_idx = None
            cleaned_lines.append("loop_")
            continue

        if lower.startswith("data_") or lower.startswith("save_"):
            in_loop = False
            in_loop_headers = False
            loop_headers = []
            occ_idx = None
            cleaned_lines.append(line)
            continue

        if stripped.startswith("_"):
            if in_loop and in_loop_headers:
                tag = stripped.split()[0]
                loop_headers.append(tag)
                if tag.lower() == "_atom_site_occupancy":
                    occ_idx = len(loop_headers) - 1
                cleaned_lines.append(line)
            else:
                in_loop = False
                in_loop_headers = False
                loop_headers = []
                occ_idx = None
                cleaned_lines.append(line)
            continue

        if in_loop:
            in_loop_headers = False
            tokens = _tokenize_cif_line(stripped)
            if loop_headers and len(tokens) < len(loop_headers):
                dropped_rows += 1
                continue
            if occ_idx is not None and occ_idx < len(tokens):
                new_tok, changed = _sanitize_occupancy_token(tokens[occ_idx])
                if changed:
                    tokens[occ_idx] = new_tok
                    sanitized_occ_count += 1
                    cleaned_lines.append("  " + " ".join(tokens))
                    continue
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    # Drop trailing empty or obviously corrupted non-CIF lines
    while cleaned_lines:
        tail = cleaned_lines[-1].strip()
        if not tail:
            cleaned_lines.pop()
            continue
        if (
            not tail.startswith(("_", "#", "data_", "loop_", ";"))
            and len(tail.split()) == 1
            and not re.match(r"^[+-]?\d", tail)
        ):
            cleaned_lines.pop()
            dropped_rows += 1
            continue
        break

    if sanitized_occ_count:
        notes.append(f"sanitized {sanitized_occ_count} occupancy value(s)")
    if dropped_rows:
        notes.append(f"dropped {dropped_rows} corrupted/truncated line(s)")

    return "\n".join(cleaned_lines) + "\n", notes


# ---------------------------------------------------------------------------------------
# Space-group setting (origin choice 1/2, hexagonal/rhombohedral axes) for CIFs that carry
# only a space-group symbol and no symmetry-operator loop.
#
# pymatgen's CifParser.get_symops() cannot look up symbols such as 'F d -3 m :2' and then
# falls back to _space_group_IT_number, which always yields origin choice 1. Choice-2
# coordinates (e.g. Si at 1/8,1/8,1/8) are then expanded with the wrong operators and give
# a wrong structure (16 instead of 8 Si) without any error. We therefore resolve the
# setting ourselves and inject an explicit `_symmetry_equiv_pos_as_xyz` loop built from
# spglib's Hall-setting database, which pymatgen always prefers.
#
# Groups with two origin choices in International Tables Vol. A (derived from spglib,
# standard axis setting): 48 Pnnn, 50 Pban, 59 Pmmn, 68 Ccce, 70 Fddd, 85 P4/n, 86 P4_2/n,
# 88 I4_1/a, 125 P4/nbm, 126 P4/nnc, 129 P4/nmm, 130 P4/ncc, 133 P4_2/nbc, 134 P4_2/nnm,
# 137 P4_2/nmc, 138 P4_2/ncm, 141 I4_1/amd, 142 I4_1/acd, 201 Pn-3, 203 Fd-3, 222 Pn-3n,
# 224 Pn-3m, 227 Fd-3m, 228 Fd-3c.  Rhombohedral groups with H/R axes: 146, 148, 155,
# 160, 161, 166, 167.
# ---------------------------------------------------------------------------------------
_SYMOP_TAGS = (
    "_symmetry_equiv_pos_as_xyz",
    "_space_group_symop_operation_xyz",
)
_HM_TAGS = (
    "_space_group_name_h-m_alt",
    "_symmetry_space_group_name_h-m",
    "_symmetry_space_group_name_h_m",
    "_space_group_name_h-m",
)
_HALL_TAGS = ("_space_group_name_hall", "_symmetry_space_group_name_hall")
_IT_NUMBER_TAGS = ("_space_group_it_number", "_symmetry_int_tables_number")
_COORD_CODE_TAGS = ("_space_group_it_coordinate_system_code",)
# Old (pre-2002) symbols of standard settings that differ from the current short symbol.
_OLD_HM_ALIASES = {"Ccca": 68, "Fd3m": 227, "Fd3c": 228, "Pn3n": 222, "Pn3m": 224, "Pn3": 201, "Fd3": 203}


def _norm_tag(tag: str) -> str:
    return tag.strip().lower().replace(".", "_").rstrip("_")


def _norm_hm(symbol: str) -> str:
    return re.sub(r"[\s_]", "", symbol)


_setting_table_cache: dict[str, Any] | None = None


def _setting_table() -> dict[str, Any]:
    """spglib Hall-setting table restricted to standard settings with choice 1/2/H/R."""
    global _setting_table_cache
    if _setting_table_cache is not None:
        return _setting_table_cache
    import spglib

    by_number: dict[int, dict[str, int]] = {}
    hm_to_number: dict[str, int] = {}
    hall_to_number: dict[str, tuple[int, str, int]] = {}
    for hall in range(1, 531):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sg_type = spglib.get_spacegroup_type(hall)
        if sg_type is None or sg_type.choice not in ("1", "2", "H", "R"):
            continue
        by_number.setdefault(sg_type.number, {})[sg_type.choice] = hall
        for name in (sg_type.international_short, sg_type.international_full):
            hm_to_number[_norm_hm(name)] = sg_type.number
        hall_to_number[" ".join(sg_type.hall_symbol.split())] = (sg_type.number, sg_type.choice, hall)
    for alias, number in _OLD_HM_ALIASES.items():
        hm_to_number.setdefault(alias, number)
    _setting_table_cache = {
        "by_number": {n: c for n, c in by_number.items() if len(c) == 2},
        "hm": hm_to_number,
        "hall": hall_to_number,
    }
    return _setting_table_cache


def _symops_xyz_for_hall(hall_number: int) -> list[str]:
    import spglib
    from pymatgen.core.operations import SymmOp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        data = spglib.get_symmetry_from_database(hall_number)
    ops: list[str] = []
    for rot, trans in zip(data["rotations"], data["translations"]):
        trans = np.mod(np.round(np.asarray(trans, dtype=float) * 24.0) / 24.0, 1.0)
        ops.append(SymmOp.from_rotation_and_translation(rot, trans).as_xyz_str())
    return ops


def _split_hm_choice(symbol: str) -> tuple[str, str | None]:
    """Split 'F d -3 m :2' / 'Fd-3m (origin choice 2)' / 'R -3 m H' into (base, choice)."""
    text = symbol.strip().strip("'\"").strip()
    choice: str | None = None
    patterns = (
        (r"\(?\s*origin\s*(?:choice|setting)?\s*([12])\s*\)?\s*$", None),
        (r"\(?\s*(?:setting|choice)\s*([12])\s*\)?\s*$", None),
        (r"\(?\s*hexagonal(?:\s+(?:axes|setting))?\s*\)?\s*$", "H"),
        (r"\(?\s*rhombohedral(?:\s+(?:axes|setting))?\s*\)?\s*$", "R"),
        (r":\s*([12HhRrSsZz])\s*$", None),
    )
    for pattern, fixed in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            choice = fixed or match.group(1)
            text = text[: match.start()].strip()
            break
    else:
        # Trailing bare 'S'/'Z' (Shoemaker) or 'H'/'R' after the symbol, e.g. 'R -3 m H'.
        match = re.search(r"\s([SZHR])\s*$", text)
        if match:
            choice = match.group(1)
            text = text[: match.start()].strip()
    if choice is not None:
        choice = {"S": "1", "Z": "2"}.get(choice.upper(), choice.upper())
    return text, choice


def _expected_composition(data: dict[str, str]) -> Any:
    from pymatgen.core import Composition

    formula = data.get("_chemical_formula_sum")
    z_text = data.get("_cell_formula_units_z")
    if not formula or not z_text:
        return None
    try:
        z_val = float(re.sub(r"\(\d+\)$", "", str(z_text).strip()))
        comp = Composition(str(formula).strip().strip("'\""))
    except Exception:
        return None
    if z_val <= 0 or comp.num_atoms <= 0:
        return None
    return comp * z_val


def _structure_from_text(text: str) -> Structure:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parser = CifParser.from_str(text, occupancy_tolerance=100.0, site_tolerance=1e-4)
        return _parse_cif_parser(parser)


def _has_overlap(structure: Structure, cutoff: float = 0.5) -> bool:
    if len(structure) < 2:
        return False
    centers, _, _, _ = structure.get_neighbor_list(cutoff)
    return len(centers) > 0


def _with_symops(block_text: str, ops: list[str]) -> str:
    lines = ["", "loop_", "_symmetry_equiv_pos_as_xyz"] + [f"  '{op}'" for op in ops]
    return block_text.rstrip("\n") + "\n" + "\n".join(lines) + "\n"


def _resolve_block_setting(block_text: str) -> tuple[str, list[str]]:
    """Inject the correct symmetry operators into one CIF data block when needed."""
    from pymatgen.io.cif import CifFile

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks = list(CifFile.from_str(block_text).data.values())
    except Exception:
        return block_text, []
    if not blocks:
        return block_text, []
    data = {_norm_tag(k): v for k, v in blocks[0].data.items()}
    if any(data.get(tag) for tag in _SYMOP_TAGS):
        return block_text, []

    def first(tags: tuple[str, ...]) -> str | None:
        for tag in tags:
            value = data.get(tag)
            if isinstance(value, list):
                value = value[0] if value else None
            if value and str(value).strip() not in ("?", "."):
                return str(value).strip()
        return None

    table = _setting_table()
    number: int | None = None
    choice: str | None = None
    hm_symbol = first(_HM_TAGS)

    # 1. Hall symbol fixes the setting completely.
    hall_symbol = first(_HALL_TAGS)
    if hall_symbol:
        key = " ".join(hall_symbol.strip("'\"").split())
        hit = table["hall"].get(key)
        if hit is None:
            ci = [v for k, v in table["hall"].items() if k.lower() == key.lower()]
            hit = ci[0] if len(ci) == 1 else None
        if hit is not None:
            number, choice, _ = hit

    # 2. H-M symbol with an explicit choice suffix, and/or the coordinate-system code.
    if number is None and hm_symbol:
        base, choice = _split_hm_choice(hm_symbol)
        number = table["hm"].get(_norm_hm(base))
        if number is None:
            # Symbol present but not a standard setting we know (e.g. 'A e a a'): leave it
            # to pymatgen rather than guessing axes from the IT number.
            return block_text, []
    if number is None:
        it_number = first(_IT_NUMBER_TAGS)
        if it_number:
            try:
                number = int(float(re.sub(r"\(\d+\)$", "", it_number)))
            except ValueError:
                number = None
    if number is None or number not in table["by_number"]:
        return block_text, []
    if choice is None:
        code = first(_COORD_CODE_TAGS)
        if code and code.strip("'\"").upper() in ("1", "2", "H", "R"):
            choice = code.strip("'\"").upper()

    settings = table["by_number"][number]
    label = hm_symbol.strip("'\"") if hm_symbol else f"No. {number}"
    if choice in settings:
        return _with_symops(block_text, _symops_xyz_for_hall(settings[choice])), []

    # 3. No choice given.
    if "H" in settings:
        # Rhombohedral groups: the cell metric decides (a=b=c, alpha=beta=gamma != 90 -> R).
        try:
            angles = [float(re.sub(r"\(\d+\)$", "", str(data[f"_cell_angle_{n}"]))) for n in ("alpha", "beta", "gamma")]
            choice = "R" if abs(angles[2] - 120.0) > 1.0 and max(angles) - min(angles) < 0.01 else "H"
        except Exception:
            choice = "H"
        return _with_symops(block_text, _symops_xyz_for_hall(settings[choice])), []

    candidates: dict[str, tuple[str, Structure | None]] = {}
    for opt in ("1", "2"):
        text = _with_symops(block_text, _symops_xyz_for_hall(settings[opt]))
        try:
            candidates[opt] = (text, _structure_from_text(text))
        except Exception:
            candidates[opt] = (text, None)

    expected = _expected_composition(data)
    ok: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for opt, (_, struct) in candidates.items():
        if struct is None:
            ok[opt], reasons[opt] = False, "could not be parsed"
            continue
        if _has_overlap(struct):
            ok[opt], reasons[opt] = False, "gives overlapping atoms"
            continue
        if expected is not None and not struct.composition.almost_equals(expected, rtol=0.02, atol=0.05):
            ok[opt], reasons[opt] = False, (
                f"gives {struct.composition.num_atoms:g} atoms, but formula x Z = {expected.num_atoms:g}"
            )
            continue
        ok[opt] = True

    good = [opt for opt in ("1", "2") if ok[opt]]
    if len(good) == 1:
        chosen = good[0]
        other = "2" if chosen == "1" else "1"
        return candidates[chosen][0], [
            f"Space group '{label}' has two origin choices and the CIF does not say which; "
            f"origin choice {chosen} was selected automatically (choice {other} {reasons[other]})."
        ]

    counts = ", ".join(
        f"choice {opt}: {s.composition.num_atoms:g} atoms" if s is not None else f"choice {opt}: unreadable"
        for opt, (_, s) in candidates.items()
    )
    return candidates["1"][0], [
        f"Space group '{label}' has two origin choices and the CIF gives neither symmetry "
        f"operators nor the choice; origin choice 1 was assumed but could not be verified "
        f"({counts}). Check the structure or add ':1'/':2' to the space-group symbol."
    ]


def _resolve_origin_choice(text: str) -> tuple[str, list[str]]:
    """Apply `_resolve_block_setting` to every data block of a CIF text."""
    if "data_" not in text.lower():
        return text, []
    try:
        blocks = _split_cif_blocks(text)
    except Exception:
        return text, []
    if not blocks:
        return text, []
    first_idx = text.lower().find("data_")
    line_start = text.rfind("\n", 0, first_idx) + 1
    preamble = text[:line_start]
    out: list[str] = []
    notes: list[str] = []
    changed = False
    for _name, block_text in blocks:
        try:
            new_text, block_notes = _resolve_block_setting(block_text)
        except Exception as exc:  # never break loading because of the setting check
            _logger.warning("Origin-choice check skipped: %s", exc)
            new_text, block_notes = block_text, []
        changed = changed or new_text != block_text
        out.append(new_text)
        notes.extend(block_notes)
    if not changed:
        return text, notes
    return preamble + "".join(out), notes


# Batch 9: structure files other than CIF (POSCAR/CONTCAR, XSF, CSSR).
STRUCTURE_SUFFIXES: tuple[str, ...] = (
    ".cif", ".mcif", ".vasp", ".poscar", ".contcar", ".xsf", ".cssr", ".vesta",
)
_POSCAR_SUFFIXES = (".vasp", ".poscar", ".contcar")
_POSCAR_NAMES = ("POSCAR", "CONTCAR")


def _is_poscar_name(path: Path) -> bool:
    return any(path.name.upper().startswith(n) for n in _POSCAR_NAMES)


def _sniff_structure_format(path: Path) -> str | None:
    """Guess 'cif' / 'xsf' / 'poscar' / 'cssr' from the first lines of the file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = [fh.readline() for _ in range(8)]
    except OSError:
        return None
    lines = [ln.strip().lstrip("﻿") for ln in head]
    words = [ln.split()[0].upper() for ln in lines if ln and not ln.startswith("#")]
    if lines and lines[0].startswith("#VESTA_FORMAT_VERSION"):
        return "vesta"
    if any(ln.lower().startswith("data_") for ln in lines):
        return "cif"
    if any(w in ("CRYSTAL", "PRIMVEC", "PRIMCOORD", "ATOMS", "SLAB", "POLYMER", "MOLECULE") for w in words):
        return "xsf"

    def floats(line: str, n: int) -> bool:
        parts = line.split()
        if len(parts) < n:
            return False
        try:
            [float(p) for p in parts[:n]]
            return True
        except ValueError:
            return False

    if len(lines) >= 5 and floats(lines[1], 1) and all(floats(lines[i], 3) for i in (2, 3, 4)):
        return "poscar"
    if len(lines) >= 5 and floats(lines[1], 3) and "SPGR" in lines[1].upper():
        return "cssr"
    return None


def detect_structure_format(path: str | Path) -> str | None:
    """'cif' / 'poscar' / 'xsf' / 'cssr' from the extension, the name (POSCAR*, CONTCAR*)
    or, for files without a known extension, the content. None if unknown."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".cif", ".mcif"):
        return "cif"
    if suffix in _POSCAR_SUFFIXES:
        return "poscar"
    if suffix in (".xsf", ".cssr", ".vesta"):
        return suffix[1:]
    if _is_poscar_name(p):
        return "poscar"
    if p.is_file():
        return _sniff_structure_format(p)
    return None


def is_structure_file(path: str | Path) -> bool:
    """True for files the batch should pick up: known extensions or POSCAR*/CONTCAR* names."""
    p = Path(path)
    return p.suffix.lower() in STRUCTURE_SUFFIXES or _is_poscar_name(p)


def structure_stem(path: str | Path) -> str:
    """Name for output folders: the stem for known extensions, else the whole name
    (so 'POSCAR_ca_0.85' stays 'POSCAR_ca_0.85' and 'POSCAR' stays 'POSCAR')."""
    p = Path(path)
    if p.suffix.lower() in STRUCTURE_SUFFIXES or not _is_poscar_name(p):
        return p.stem or p.name
    return p.name


def _read_cssr(text: str) -> Structure:
    """Lenient CSSR reader (labels like 'Na1' are reduced to the element)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    abc = [float(t) for t in lines[0].split()[-3:]]
    angles = [float(t) for t in lines[1].split()[:3]]
    head = lines[2].split()
    natoms = int(head[0])
    cartesian = len(head) > 1 and head[1] == "1"
    species: list[str] = []
    coords: list[list[float]] = []
    for line in lines[4:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        m = re.match(r"[A-Za-z][a-z]?", parts[1])
        if not m:
            raise ValueError(f"bad atom label {parts[1]!r}")
        species.append(m.group(0).capitalize())
        coords.append([float(v) for v in parts[2:5]])
        if len(species) == natoms:
            break
    if not species or len(species) != natoms:
        raise ValueError(f"expected {natoms} atoms, found {len(species)}")
    return Structure(Lattice.from_parameters(*abc, *angles), species, coords,
                     coords_are_cartesian=cartesian)


_FORMAT_LABEL = {"poscar": "POSCAR/CONTCAR", "xsf": "XSF", "cssr": "CSSR", "vesta": "VESTA"}


def _load_non_cif(path: Path, fmt: str) -> Structure:
    """Read a POSCAR/CONTCAR, XSF or CSSR file; ValueError with the format name on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if fmt == "poscar":
                from pymatgen.io.vasp import Poscar

                structure = Poscar.from_file(str(path), check_for_potcar=False).structure
            elif fmt == "xsf":
                from pymatgen.io.xcrysden import XSF

                structure = XSF.from_str(path.read_text(encoding="utf-8", errors="replace")).structure
            elif fmt == "vesta":
                from shido.core.vesta_io import read_vesta

                structure = read_vesta(path)[0]
            else:
                structure = _read_cssr(path.read_text(encoding="utf-8", errors="replace"))
        if structure is None or len(structure) == 0:
            raise ValueError("no atoms found")
    except Exception as exc:
        raise ValueError(
            f"Failed to load structure from '{path}' as {_FORMAT_LABEL[fmt]}: {exc}"
        ) from exc
    return structure


def load_structure_robust(cif_path: str | Path) -> tuple[Structure, list[str]]:
    """Load a crystal structure robustly with malformed-CIF recovery and disorder reporting.

    POSCAR/CONTCAR (also without extension), XSF and CSSR are recognised by extension,
    name or content (`detect_structure_format`) and read directly; CIF is unchanged.
    """
    path = Path(cif_path)
    warnings_list: list[str] = []
    structure: Structure | None = None

    fmt = detect_structure_format(path)
    if fmt in _FORMAT_LABEL:
        structure = _load_non_cif(path, fmt)
        is_ordered, num_disordered, occ_summary = summarize_occupancies(structure)
        if not is_ordered:
            warnings_list.append(
                f"Disordered/partial-occupancy structure ({num_disordered} disordered sites: {occ_summary})."
            )
        return structure, warnings_list

    resolved_text: str | None = None
    if path.suffix.lower() in (".cif", ".mcif", "") or path.name.lower().endswith(".cif"):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace").lstrip("﻿")
            new_text, setting_notes = _resolve_origin_choice(raw)
            if new_text != raw:
                resolved_text = new_text
            warnings_list.extend(setting_notes)
        except Exception:
            resolved_text = None

    # 1. Primary: lenient CifParser directly on file
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if resolved_text is not None:
                parser = CifParser.from_str(
                    resolved_text,
                    occupancy_tolerance=100.0,
                    site_tolerance=1e-4,
                )
            else:
                parser = CifParser(
                    str(path),
                    occupancy_tolerance=100.0,
                    site_tolerance=1e-4,
                )
            structure = _parse_cif_parser(parser)
    except Exception as primary_exc:
        # 2. Recovery: clean CIF text and parse via CifParser.from_str
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            cleaned_text, clean_notes = _clean_cif_text(raw_text)
            cleaned_text, setting_notes = _resolve_origin_choice(cleaned_text)
            for note in setting_notes:
                if note not in warnings_list:
                    warnings_list.append(note)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parser = CifParser.from_str(
                    cleaned_text,
                    occupancy_tolerance=100.0,
                    site_tolerance=1e-4,
                )
                structure = _parse_cif_parser(parser)
            detail = f" ({', '.join(clean_notes)})" if clean_notes else ""
            warnings_list.append(f"Recovered malformed CIF via text sanitization{detail}.")
        except Exception:
            # 3. Final fallback: generic pymatgen Structure.from_file
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    structure = Structure.from_file(str(path))
                warnings_list.append(
                    f"Loaded via Structure.from_file fallback (CifParser error: {primary_exc})."
                )
            except Exception as final_exc:
                raise ValueError(
                    f"Failed to load structure from '{path}': {final_exc}"
                ) from final_exc

    is_ordered, num_disordered, occ_summary = summarize_occupancies(structure)
    if not is_ordered:
        warnings_list.append(
            f"Disordered/partial-occupancy structure ({num_disordered} disordered sites: {occ_summary})."
        )

    return structure, warnings_list


def load_structure(cif_path: str | Path) -> Structure:
    """Load a crystal structure from a CIF (or other pymatgen-supported) file."""
    return load_structure_robust(cif_path)[0]


def _split_cif_blocks(text: str) -> list[tuple[str, str]]:
    """Split CIF text into ``[(block_name, block_text), ...]`` at ``data_`` lines.

    ``data_`` lines inside semicolon text fields are ignored.
    """
    blocks: list[tuple[str, list[str]]] = []
    in_text_field = False
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.startswith(";"):
            in_text_field = not in_text_field
        stripped = line.strip()
        if not in_text_field and stripped.lower().startswith("data_"):
            name = stripped[5:].split()[0] if stripped[5:].split() else ""
            blocks.append((name, [line]))
        elif blocks:
            blocks[-1][1].append(line)
    return [(name, "\n".join(lines) + "\n") for name, lines in blocks]


def _parse_cif_block_text(block_text: str) -> tuple[Structure, list[str]]:
    """Parse one CIF data block (lenient parser, then the text-sanitizing recovery)."""
    warnings_list: list[str] = []
    try:
        resolved_text, setting_notes = _resolve_origin_choice(block_text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parser = CifParser.from_str(resolved_text, occupancy_tolerance=100.0, site_tolerance=1e-4)
            structure = _parse_cif_parser(parser)
        warnings_list.extend(setting_notes)
    except Exception:
        cleaned_text, clean_notes = _clean_cif_text(block_text)
        cleaned_text, setting_notes = _resolve_origin_choice(cleaned_text)
        warnings_list.extend(setting_notes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parser = CifParser.from_str(cleaned_text, occupancy_tolerance=100.0, site_tolerance=1e-4)
            structure = _parse_cif_parser(parser)
        detail = f" ({', '.join(clean_notes)})" if clean_notes else ""
        warnings_list.append(f"Recovered malformed CIF via text sanitization{detail}.")
    return structure, warnings_list


def load_structures_all(cif_path: str | Path) -> list[tuple[Structure, list[str], str]]:
    """Load every data block of a (multi-block) CIF as ``[(structure, warnings, block_name)]``.

    Blocks without a readable structure (e.g. a header-only block) are skipped with a log
    warning. A file without ``data_`` blocks (POSCAR etc.) falls back to
    `load_structure_robust` and returns one entry named after the file stem.
    Raises ``ValueError`` if no block yields a structure.
    """
    path = Path(cif_path)
    raw_text = path.read_text(encoding="utf-8", errors="replace").lstrip("﻿")
    blocks = _split_cif_blocks(raw_text)
    if not blocks:
        structure, warns = load_structure_robust(path)
        return [(structure, warns, path.stem)]

    results: list[tuple[Structure, list[str], str]] = []
    errors: list[str] = []
    for idx, (name, block_text) in enumerate(blocks):
        label = name or f"block{idx + 1}"
        try:
            structure, warns = _parse_cif_block_text(block_text)
        except Exception as exc:
            errors.append(f"{label}: {exc}")
            _logger.warning("Skipped CIF data block '%s': %s", label, exc)
            continue
        is_ordered, num_disordered, occ_summary = summarize_occupancies(structure)
        if not is_ordered:
            warns.append(
                f"Disordered/partial-occupancy structure ({num_disordered} disordered sites: {occ_summary})."
            )
        results.append((structure, warns, label))

    if not results:
        raise ValueError(f"Failed to load any structure from '{path}': {'; '.join(errors)}")
    return results


def _site_element(site: Any) -> str:
    """Return the dominant element symbol for ordered or disordered sites."""
    return dominant_element(site)


_BOX_EPS = 1e-3


def _extract_nn_table(
    structure: Structure,
    nn_finder: Any,
    neighbor_map: dict[int, list[tuple[int, tuple[int, int, int], np.ndarray]]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[np.ndarray, np.ndarray, str, str]],
    list[tuple[np.ndarray, np.ndarray, str, str]],
    list[tuple[int, list[np.ndarray], str]],
]:
    """Collect deduplicated bonds, renderable bonds, and coordination polyhedra.

    When ``neighbor_map`` is given, it is filled with ``{site: [(j, image, cart), ...]}``
    (every neighbour, before deduplication) for the bond-angle table.
    """
    rows: list[dict[str, Any]] = []
    render_bonds_in_cell: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    render_bonds_near_cell: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    polyhedra_envs: list[tuple[int, list[np.ndarray], str]] = []
    seen_pairs: set[tuple[int, int, tuple[int, int, int]]] = set()
    render_keys: set[tuple[Any, ...]] = set()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for i, site_i in enumerate(structure):
            elem_i = _site_element(site_i)
            p_i = np.asarray(site_i.coords, dtype=float)
            nn_infos = nn_finder.get_nn_info(structure, i)

            nbr_coords: list[np.ndarray] = []
            for info in nn_infos:
                j = int(info["site_index"])
                img = tuple(int(x) for x in info["image"])
                nbr_site = info["site"]
                p_j = np.asarray(nbr_site.coords, dtype=float)
                nbr_coords.append(p_j)
                if neighbor_map is not None:
                    neighbor_map.setdefault(i, []).append((j, img, p_j))

                # Render every periodic copy of this bond whose two ends both lie in the
                # drawn unit cell [0, 1]^3 (boundary atoms included). Done before the
                # CSV dedup below, which keeps only one representative per bond.
                elem_j_r = _site_element(structure[j])
                f_i = np.asarray(site_i.frac_coords, dtype=float)
                f_j = np.asarray(nbr_site.frac_coords, dtype=float)
                for shift in itertools.product((-1, 0, 1), repeat=3):
                    t = np.asarray(shift, dtype=float)
                    a_f, b_f = f_i + t, f_j + t
                    if np.all((a_f > -_BOX_EPS) & (a_f < 1 + _BOX_EPS)) and np.all(
                        (b_f > -_BOX_EPS) & (b_f < 1 + _BOX_EPS)
                    ):
                        a_c = structure.lattice.get_cartesian_coords(a_f)
                        b_c = structure.lattice.get_cartesian_coords(b_f)
                        key = tuple(sorted((tuple(np.round(a_f, 3)), tuple(np.round(b_f, 3)))))
                        if key not in render_keys:
                            render_keys.add(key)
                            render_bonds_in_cell.append((a_c, b_c, elem_i, elem_j_r))

                neg_img = (-img[0], -img[1], -img[2])
                canonical_key = min((i, j, img), (j, i, neg_img))
                if canonical_key in seen_pairs:
                    continue
                seen_pairs.add(canonical_key)

                elem_j = _site_element(structure[j])
                dist = float(np.linalg.norm(p_j - p_i))
                elem_pair = "-".join(sorted((elem_i, elem_j)))

                rows.append(
                    {
                        "site_i": i,
                        "element_i": elem_i,
                        "site_j": j,
                        "element_j": elem_j,
                        "element_pair": elem_pair,
                        "image": str(img),
                        "distance_angstrom": round(dist, 6),
                    }
                )

                if img != (0, 0, 0):
                    frac_j = np.asarray(nbr_site.frac_coords, dtype=float)
                    if np.all((frac_j >= -0.25) & (frac_j <= 1.25)):
                        render_bonds_near_cell.append((p_i.copy(), p_j.copy(), elem_i, elem_j))

            if len(nbr_coords) >= 4:
                polyhedra_envs.append((i, nbr_coords, elem_i))

    return rows, render_bonds_in_cell, render_bonds_near_cell, polyhedra_envs


BondSpecs = dict[tuple[str, str], tuple[float, float]]

_BOND_SPEC_RE = re.compile(
    r"^\s*([A-Za-z][a-z]?)\s*-\s*([A-Za-z][a-z]?)\s*:\s*([^:]+?)\s*(?::\s*([^:]+?)\s*)?$"
)


def _norm_element(symbol: str) -> str:
    s = str(symbol).strip()
    return s[:1].upper() + s[1:].lower()


def parse_bond_specs(specs: Any) -> BondSpecs | None:
    """Parse VESTA-style bond specs ``"A-B:dmin:dmax"`` (or ``"A-B:dmax"``, dmin = 0).

    ``specs`` is a string, a list of strings, or an already-parsed dict. Returns
    ``{(A, B) sorted: (dmin, dmax)}`` in Angstrom, or None when nothing is given.
    The same pair given twice keeps the last range.
    """
    if specs is None:
        return None
    if isinstance(specs, dict):
        items = list(specs.items())
        out: BondSpecs = {}
        for key, rng in items:
            a, b = (_norm_element(x) for x in key)
            dmin, dmax = (float(x) for x in rng)
            out[tuple(sorted((a, b)))] = (dmin, dmax)  # type: ignore[assignment]
        specs_list: list[str] = []
    else:
        specs_list = [specs] if isinstance(specs, str) else [str(s) for s in specs]
        out = {}
    for raw in specs_list:
        for part in str(raw).split(","):
            if not part.strip():
                continue
            m = _BOND_SPEC_RE.match(part)
            if not m:
                raise ValueError(
                    f"Bad bond spec {part!r}: use 'A-B:dmin:dmax' or 'A-B:dmax' (e.g. 'Mn-Ga:0:3.0')."
                )
            a, b = _norm_element(m.group(1)), _norm_element(m.group(2))
            try:
                if m.group(4) is None:
                    dmin, dmax = 0.0, float(m.group(3))
                else:
                    dmin, dmax = float(m.group(3)), float(m.group(4))
            except ValueError as exc:
                raise ValueError(f"Bad distance in bond spec {part!r}.") from exc
            out[tuple(sorted((a, b)))] = (dmin, dmax)  # type: ignore[assignment]
    for pair, (dmin, dmax) in out.items():
        if not (math.isfinite(dmin) and math.isfinite(dmax)) or dmin < 0.0 or dmax <= dmin:
            raise ValueError(
                f"Bond spec {'-'.join(pair)}: need 0 <= dmin < dmax, got {dmin}..{dmax}."
            )
    return out or None


class _PairRangeNN:
    """Neighbour finder for explicit element-pair distance ranges (VESTA's 'search A-B bonds').

    Same idea as pymatgen ``CutOffDictNN`` plus a lower bound: j is a neighbour of i
    when the dominant elements (A, B) have a spec and dmin <= |r_ij| <= dmax.
    """

    def __init__(self, specs: BondSpecs) -> None:
        self.specs = dict(specs)
        self.r_max = max(dmax for _, dmax in self.specs.values())

    def get_nn_info(self, structure: Structure, n: int) -> list[dict[str, Any]]:
        elem_i = _site_element(structure[n])
        infos: list[dict[str, Any]] = []
        for nbr in structure.get_neighbors(structure[n], self.r_max + 1e-6):
            dist = float(nbr.nn_distance)
            if dist < 1e-6:
                continue
            rng = self.specs.get(tuple(sorted((elem_i, _site_element(structure[nbr.index])))))  # type: ignore[arg-type]
            if rng is None or not (rng[0] - 1e-6 <= dist <= rng[1] + 1e-6):
                continue
            infos.append(
                {
                    "site": nbr,
                    "image": tuple(int(round(x)) for x in nbr.image),
                    "weight": dist,
                    "site_index": int(nbr.index),
                }
            )
        infos.sort(key=lambda d: (d["weight"], d["site_index"], d["image"]))
        return infos


def compute_bonds(
    structure: Structure,
    bond_specs: Any = None,
) -> tuple[
    pd.DataFrame,
    list[tuple[np.ndarray, np.ndarray, str, str]],
    list[tuple[int, list[np.ndarray], str]],
]:
    """Compute chemical bonds and coordination environments using CrystalNN with MinimumDistanceNN fallback.

    ``bond_specs`` (see `parse_bond_specs`, e.g. ``["Na-Cl:0:3.0"]``) replaces the
    automatic search: only those element pairs within those ranges are bonds.
    """
    bonds_df, render_bonds, polyhedra_envs, _ = _compute_bonds_full(structure, bond_specs)
    return bonds_df, render_bonds, polyhedra_envs


def _compute_bonds_full(
    structure: Structure,
    bond_specs: Any = None,
) -> tuple[
    pd.DataFrame,
    list[tuple[np.ndarray, np.ndarray, str, str]],
    list[tuple[int, list[np.ndarray], str]],
    dict[int, list[tuple[int, tuple[int, int, int], np.ndarray]]],
]:
    """`compute_bonds` plus the per-site neighbour map (same neighbour finder)."""
    work_struct = to_ordered_proxy(structure)
    neighbor_map: dict[int, list[tuple[int, tuple[int, int, int], np.ndarray]]] = {}

    rows: list[dict[str, Any]] = []
    render_in_cell: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    render_near_cell: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    polyhedra_envs: list[tuple[int, list[np.ndarray], str]] = []

    specs = parse_bond_specs(bond_specs)
    if specs is not None:
        present = {_site_element(site) for site in work_struct}
        for pair in specs:
            if not set(pair) <= present:
                _logger.warning("Bond spec %s: element not in the structure, no such bonds.", "-".join(pair))
        # Explicit ranges: no automatic fallback (zero bonds is a valid answer).
        rows, render_in_cell, render_near_cell, polyhedra_envs = _extract_nn_table(
            work_struct, _PairRangeNN(specs), neighbor_map
        )
        if not rows:
            _logger.warning("No bonds match the bond specs %s.", sorted(specs.items()))
    else:
        try:
            rows, render_in_cell, render_near_cell, polyhedra_envs = _extract_nn_table(
                work_struct, CrystalNN(), neighbor_map
            )
        except Exception:
            rows = []

        if not rows:
            neighbor_map = {}
            try:
                rows, render_in_cell, render_near_cell, polyhedra_envs = _extract_nn_table(
                    work_struct, MinimumDistanceNN(), neighbor_map
                )
            except Exception:
                rows, render_in_cell, render_near_cell, polyhedra_envs = [], [], [], []
                neighbor_map = {}

    if rows:
        bonds_df = (
            pd.DataFrame(rows, columns=BOND_COLUMNS)
            .sort_values(
                by=["element_pair", "distance_angstrom", "site_i", "site_j"],
                ascending=[True, True, True, True],
            )
            .reset_index(drop=True)
        )
    else:
        bonds_df = pd.DataFrame(columns=BOND_COLUMNS)

    render_bonds = render_in_cell if render_in_cell else render_near_cell
    return bonds_df, render_bonds, polyhedra_envs, neighbor_map


def compute_bond_angles(
    structure: Structure,
    neighbor_map: dict[int, list[tuple[int, tuple[int, int, int], np.ndarray]]] | None = None,
    max_neighbors_per_site: int = MAX_ANGLE_NEIGHBORS_PER_SITE,
    max_rows: int = MAX_BOND_ANGLE_ROWS,
) -> tuple[pd.DataFrame, bool]:
    """Bond angles a-center-b between every pair of nearest neighbours of each site.

    Uses the same neighbours as `compute_bonds` (pass its ``neighbor_map`` to avoid a
    second neighbour search). Only the ``max_neighbors_per_site`` closest neighbours of a
    site are paired, and at most ``max_rows`` angles are kept. Elements follow
    `bond_lengths.csv` (ordered proxy, dominant element). Returns ``(df, truncated)``.
    """
    if neighbor_map is None:
        _, _, _, neighbor_map = _compute_bonds_full(structure)
    work_struct = to_ordered_proxy(structure)
    elements = [_site_element(site) for site in work_struct]

    rows: list[dict[str, Any]] = []
    truncated = False
    for c in sorted(neighbor_map):
        p_c = np.asarray(work_struct[c].coords, dtype=float)
        # Drop duplicate neighbour entries (same site and image), keep nearest first.
        uniq: dict[tuple[int, tuple[int, int, int]], np.ndarray] = {}
        for j, img, p_j in neighbor_map[c]:
            uniq.setdefault((int(j), tuple(int(x) for x in img)), np.asarray(p_j, dtype=float))
        nbrs = sorted(uniq.items(), key=lambda kv: (float(np.linalg.norm(kv[1] - p_c)), kv[0]))
        if len(nbrs) > max_neighbors_per_site:
            nbrs = nbrs[:max_neighbors_per_site]
            truncated = True
        for (key_a, p_a), (key_b, p_b) in itertools.combinations(nbrs, 2):
            v_a, v_b = p_a - p_c, p_b - p_c
            n_a, n_b = float(np.linalg.norm(v_a)), float(np.linalg.norm(v_b))
            if n_a < 1e-8 or n_b < 1e-8:
                continue
            cos_t = float(np.clip(np.dot(v_a, v_b) / (n_a * n_b), -1.0, 1.0))
            # Canonical order inside the pair: lower site index first.
            (ja, _), (jb, _) = sorted((key_a, key_b))
            rows.append(
                {
                    "site_center": int(c),
                    "element_center": elements[c],
                    "site_a": int(ja),
                    "element_a": elements[ja],
                    "site_b": int(jb),
                    "element_b": elements[jb],
                    "angle_deg": round(math.degrees(math.acos(cos_t)), 4),
                }
            )
            if len(rows) >= max_rows:
                truncated = True
                break
        if len(rows) >= max_rows:
            break

    if truncated:
        _logger.warning(
            "Bond-angle table truncated (max %d neighbours/site, max %d rows).",
            max_neighbors_per_site,
            max_rows,
        )
    if not rows:
        return pd.DataFrame(columns=BOND_ANGLE_COLUMNS), truncated
    df = (
        pd.DataFrame(rows, columns=BOND_ANGLE_COLUMNS)
        .sort_values(
            by=["element_center", "angle_deg", "site_center", "site_a", "site_b"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return df, truncated


def _occupancy_string(site: Any) -> str:
    """'Mn:0.5 Ga:0.5' (largest occupancy first, then symbol)."""
    items = sorted(
        ((_species_symbol(sp), float(occ)) for sp, occ in site.species.items()),
        key=lambda t: (-t[1], t[0]),
    )
    return " ".join(f"{sym}:{occ:.6g}" for sym, occ in items)


def _wyckoff_labels(structure: Structure) -> list[str]:
    """Wyckoff label (e.g. '4a') per site from the symmetrized structure; '' on failure."""
    for symprec in (0.1, 0.01):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sym = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetrized_structure()
            if len(sym) != len(structure):
                continue
            labels = [""] * len(structure)
            for idx_list, wy in zip(sym.equivalent_indices, sym.wyckoff_symbols):
                for idx in idx_list:
                    labels[idx] = str(wy)
            return labels
        except Exception:
            continue
    return [""] * len(structure)


def compute_atomic_positions(structure: Structure) -> pd.DataFrame:
    """Per-site table: element, occupancy, fractional and Cartesian (A) coordinates, Wyckoff."""
    wyckoff = _wyckoff_labels(structure)
    rows: list[dict[str, Any]] = []
    for i, site in enumerate(structure):
        f = np.asarray(site.frac_coords, dtype=float)
        c = np.asarray(structure.lattice.get_cartesian_coords(f), dtype=float)
        rows.append(
            {
                "site": i,
                "element": dominant_element(site),
                "occupancy": _occupancy_string(site),
                "frac_x": round(float(f[0]), 6),
                "frac_y": round(float(f[1]), 6),
                "frac_z": round(float(f[2]), 6),
                "cart_x": round(float(c[0]), 6),
                "cart_y": round(float(c[1]), 6),
                "cart_z": round(float(c[2]), 6),
                "wyckoff": wyckoff[i],
            }
        )
    return pd.DataFrame(rows, columns=ATOMIC_POSITION_COLUMNS)


def convert_cell(structure: Structure, cell: str = "input", symprec: float = 0.01) -> Structure:
    """Return the input, conventional standard, or primitive standard cell.

    The transformation is checked: element fractions must be conserved
    (`validate_composition_conserved`) and the site count must match the volume ratio,
    so atoms on cell boundaries are neither lost nor duplicated.
    Raises ``ValueError`` for an unknown ``cell`` and ``ScientificValidationError``
    when a check fails.
    """
    cell = str(cell).lower()
    if cell not in POSCAR_CELL_CHOICES:
        raise ValueError(f"poscar_cell must be one of {POSCAR_CELL_CHOICES}, got {cell!r}.")
    if cell == "input":
        return structure.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        if cell == "conventional":
            new = sga.get_conventional_standard_structure()
        else:
            new = sga.get_primitive_standard_structure()

    validate_composition_conserved(structure, new, context=f"{cell} cell conversion")
    expected = len(structure) * float(new.volume) / float(structure.volume)
    if abs(expected - len(new)) > 0.01 * max(expected, 1.0):
        raise ScientificValidationError(
            f"{cell} cell conversion: {len(new)} sites, expected {expected:.3f} from the "
            "volume ratio (boundary atoms lost or duplicated)."
        )
    return new


def _analyze_symmetry(structure: Structure) -> tuple[str, int, str]:
    """Extract spacegroup symbol, number, and crystal system with safe fallbacks."""
    work_struct = to_ordered_proxy(structure)
    for symprec in (0.1, 0.01):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sga = SpacegroupAnalyzer(work_struct, symprec=symprec)
                return (
                    str(sga.get_space_group_symbol()),
                    int(sga.get_space_group_number()),
                    str(sga.get_crystal_system()),
                )
        except Exception:
            continue
    return "P1", 1, "triclinic"


def analyze_structure_data(
    structure: Structure,
    output_dir: str | Path,
    parse_warnings: list[str] | None = None,
    poscar_cell: str = "input",
    bond_specs: Any = None,
) -> dict[str, Any]:
    """Analyze lattice/symmetry/bonds and export CSV summaries under <output_dir>/data/.

    ``bond_specs`` (e.g. ``["Mn-Ga:0:3.0"]``, see `parse_bond_specs`) replaces the automatic
    bond search; bond_lengths.csv, bond_angles.csv and the drawn bonds all use it.

    ``poscar_cell``: "input" (as given), "conventional" or "primitive" (standard cells
    via SpacegroupAnalyzer, composition- and site-count-checked) for the POSCAR only.
    """
    if str(poscar_cell).lower() not in POSCAR_CELL_CHOICES:
        raise ValueError(f"poscar_cell must be one of {POSCAR_CELL_CHOICES}, got {poscar_cell!r}.")
    validate_structure(structure)  # guardrail: lattice, NaN/Inf, overlapping atoms
    out_dir = Path(output_dir)
    data_dir = out_dir / DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = out_dir / STRUCT_DIR
    structures_dir.mkdir(parents=True, exist_ok=True)

    sg_symbol, sg_number, crystal_system = _analyze_symmetry(structure)
    is_ordered, disordered_sites, occupancy_info = summarize_occupancies(structure)
    warnings_str = "; ".join(str(w) for w in parse_warnings) if parse_warnings else ""
    lattice = structure.lattice
    c_over_a = round(float(lattice.c) / float(lattice.a), 5) if float(lattice.a) else 0.0

    summary_row: dict[str, Any] = {
        "formula": str(structure.composition.formula),
        "reduced_formula": str(structure.composition.reduced_formula),
        "num_sites": int(len(structure)),
        "spacegroup_symbol": sg_symbol,
        "spacegroup_number": sg_number,
        "crystal_system": crystal_system,
        "a_angstrom": round(float(lattice.a), 6),
        "b_angstrom": round(float(lattice.b), 6),
        "c_angstrom": round(float(lattice.c), 6),
        "c_over_a": c_over_a,
        "alpha_deg": round(float(lattice.alpha), 6),
        "beta_deg": round(float(lattice.beta), 6),
        "gamma_deg": round(float(lattice.gamma), 6),
        "volume_angstrom3": round(float(structure.volume), 6),
        "density_g_cm3": round(float(structure.density), 6),
        "is_ordered": bool(is_ordered),
        "disordered_sites": int(disordered_sites),
        "occupancy_info": str(occupancy_info),
        "parse_warnings": warnings_str,
    }

    lattice_summary_csv = data_dir / "lattice_summary.csv"
    pd.DataFrame([summary_row]).to_csv(lattice_summary_csv, index=False)

    bonds_df, render_bonds, polyhedra_envs, neighbor_map = _compute_bonds_full(structure, bond_specs)
    bond_lengths_csv = data_dir / "bond_lengths.csv"
    bonds_df.to_csv(bond_lengths_csv, index=False)

    angles_df, angles_truncated = compute_bond_angles(structure, neighbor_map)
    bond_angles_csv = data_dir / "bond_angles.csv"
    angles_df.to_csv(bond_angles_csv, index=False)

    positions_df = compute_atomic_positions(structure)
    atomic_positions_csv = data_dir / "atomic_positions.csv"
    positions_df.to_csv(atomic_positions_csv, index=False)

    poscar_structure = convert_cell(structure, poscar_cell)

    cif_file = structures_dir / "structure.cif"
    poscar_file = structures_dir / "POSCAR"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        CifWriter(structure).write_file(str(cif_file))
        Poscar(to_ordered_proxy(poscar_structure)).write_file(str(poscar_file))

    return {
        "output_dir": str(out_dir.resolve()),
        "lattice_summary_csv": str(lattice_summary_csv.resolve()),
        "bond_lengths_csv": str(bond_lengths_csv.resolve()),
        "cif_file": str(cif_file.resolve()),
        "poscar_file": str(poscar_file.resolve()),
        "summary": summary_row,
        "num_bonds": int(len(bonds_df)),
        "render_bonds": render_bonds,
        "polyhedra_envs": polyhedra_envs,
        "bond_angles_csv": str(bond_angles_csv.resolve()),
        "num_bond_angles": int(len(angles_df)),
        "bond_angles_truncated": bool(angles_truncated),
        "atomic_positions_csv": str(atomic_positions_csv.resolve()),
        "poscar_cell": str(poscar_cell).lower(),
        "poscar_num_sites": int(len(poscar_structure)),
    }

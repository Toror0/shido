"""Origin choice 1/2 and H/R axes for CIFs that carry only a space-group symbol.

pymatgen falls back to origin choice 1 for symbols such as 'F d -3 m :2', which silently
expands choice-2 coordinates into a wrong structure. ``analyzer`` resolves the setting and
injects explicit symmetry operators; these tests pin that behaviour down.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher

from shido import analyzer
from shido.core.validator import min_interatomic_distance

SI_A = 5.4309
SI_SI = SI_A * math.sqrt(3.0) / 4.0


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path


def _cubic_cell(a: float) -> str:
    return (
        f"_cell_length_a {a}\n_cell_length_b {a}\n_cell_length_c {a}\n"
        "_cell_angle_alpha 90.0\n_cell_angle_beta 90.0\n_cell_angle_gamma 90.0\n"
    )


_ATOM_LOOP = (
    "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
    "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
)


def _si_cif(header: str, xyz: str = "0.125 0.125 0.125", extra: str = "") -> str:
    return f"data_si\n{_cubic_cell(SI_A)}{header}\n{extra}{_ATOM_LOOP}Si1 Si {xyz} 1.0\n"


def _assert_diamond_si(structure) -> None:
    assert len(structure) == 8
    d, _, _ = min_interatomic_distance(structure)
    assert d == pytest.approx(SI_SI, abs=0.01)


# ---------------------------------------------------------------------------------------
# Spelling variants of "origin choice 2" (Fd-3m, Si at 1/8,1/8,1/8)
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "header",
    [
        "_space_group_name_H-M_alt 'F d -3 m :2'",
        "_symmetry_space_group_name_H-M 'F d -3 m:2'",
        "_symmetry_space_group_name_H-M 'Fd-3m:2'\n_symmetry_Int_Tables_number 227",
        "_symmetry_space_group_name_H-M 'F d -3 m (origin choice 2)'",
        "_symmetry_space_group_name_H-M 'F d -3 m origin choice 2'",
        "_symmetry_space_group_name_H-M 'F d -3 m Z'",
        "_symmetry_space_group_name_H-M 'F d 3 m :2'",
        "_space_group_name_H-M_alt 'F d -3 m'\n_space_group.IT_coordinate_system_code 2",
        "_space_group_IT_number 227\n_space_group.IT_coordinate_system_code '2'",
        "_space_group_name_Hall '-F 4vw 2vw 3'",
        "_space_group_name_Hall '-F 4vw 2vw 3'\n_space_group_name_H-M_alt 'F d -3 m'",
    ],
)
def test_fd3m_origin_choice_2_spellings(tmp_path: Path, header: str):
    s, warns = analyzer.load_structure_robust(_write(tmp_path, "si.cif", _si_cif(header)))
    _assert_diamond_si(s)
    assert not any("origin choice" in w for w in warns)


@pytest.mark.parametrize(
    "header",
    [
        "_space_group_name_H-M_alt 'F d -3 m :1'",
        "_symmetry_space_group_name_H-M 'F d -3 m S'",
        "_space_group_name_Hall 'F 4d 2 3 -1d'",
    ],
)
def test_fd3m_origin_choice_1_spellings(tmp_path: Path, header: str):
    s, _ = analyzer.load_structure_robust(_write(tmp_path, "si.cif", _si_cif(header, "0 0 0")))
    _assert_diamond_si(s)


def test_load_structures_all_uses_origin_choice(tmp_path: Path):
    text = _si_cif("_space_group_name_H-M_alt 'F d -3 m :2'") + "\n" + _si_cif(
        "_space_group_name_H-M_alt 'F d -3 m :1'", "0 0 0"
    ).replace("data_si", "data_si_b")
    results = analyzer.load_structures_all(_write(tmp_path, "two.cif", text))
    assert len(results) == 2
    for structure, _, _ in results:
        _assert_diamond_si(structure)
    assert StructureMatcher().fit(results[0][0], results[1][0])


def test_explicit_symops_are_left_alone(tmp_path: Path):
    """A CIF with its own operator loop must not be touched (no warning, same result)."""
    ops = analyzer._symops_xyz_for_hall(526)  # Fd-3m origin choice 2
    loop = "loop_\n_symmetry_equiv_pos_as_xyz\n" + "".join(f"'{op}'\n" for op in ops)
    cif = _si_cif("_symmetry_space_group_name_H-M 'F d -3 m'", extra=loop)
    s, warns = analyzer.load_structure_robust(_write(tmp_path, "si.cif", cif))
    _assert_diamond_si(s)
    assert not any("origin choice" in w for w in warns)


# ---------------------------------------------------------------------------------------
# Other groups with two origin choices
# ---------------------------------------------------------------------------------------
def _fese_cif(symbol: str, fe: str, se: str) -> str:
    return (
        "data_fese\n_cell_length_a 3.7734\n_cell_length_b 3.7734\n_cell_length_c 5.5258\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        f"_symmetry_space_group_name_H-M '{symbol}'\n_symmetry_Int_Tables_number 129\n"
        f"{_ATOM_LOOP}Fe1 Fe {fe} 1.0\nSe1 Se {se} 1.0\n"
    )


def test_p4nmm_fese_origin_choices(tmp_path: Path):
    """P4/nmm (No. 129) FeSe: Fe 2a, Se 2c -> 4 atoms in both origin choices."""
    z_se = 0.2672
    s2, _ = analyzer.load_structure_robust(
        _write(tmp_path, "o2.cif", _fese_cif("P 4/n m m :2", "0.75 0.25 0", f"0.25 0.25 {z_se}"))
    )
    s1, _ = analyzer.load_structure_robust(
        _write(tmp_path, "o1.cif", _fese_cif("P 4/n m m :1", "0 0 0", f"0 0.5 {z_se}"))
    )
    assert len(s2) == 4
    assert len(s1) == 4
    assert StructureMatcher().fit(s1, s2)
    d, _, _ = min_interatomic_distance(s2)
    assert 2.3 < d < 2.5  # Fe-Se


@pytest.mark.parametrize(
    ("symbol", "xyz"),
    [("P n -3 n :1", "0 0 0"), ("P n -3 n :2", "0.25 0.25 0.25"), ("P n -3 n", "0.25 0.25 0.25")],
)
def test_pn3n_2a_site(tmp_path: Path, symbol: str, xyz: str):
    """Pn-3n (No. 222) 2a: (0,0,0) in choice 1, (1/4,1/4,1/4) in choice 2 -> 2 atoms."""
    cif = (
        f"data_x\n{_cubic_cell(6.0)}_symmetry_space_group_name_H-M '{symbol}'\n"
        "_chemical_formula_sum 'Cs'\n_cell_formula_units_Z 2\n"
        f"{_ATOM_LOOP}Cs1 Cs {xyz} 1.0\n"
    )
    s, _ = analyzer.load_structure_robust(_write(tmp_path, "x.cif", cif))
    assert len(s) == 2


# ---------------------------------------------------------------------------------------
# Choice not given: decide automatically, otherwise warn
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(("xyz", "chosen"), [("0.125 0.125 0.125", "2"), ("0 0 0", "1")])
def test_unspecified_choice_resolved_by_formula_and_z(tmp_path: Path, xyz: str, chosen: str):
    extra = "_chemical_formula_sum 'Si'\n_cell_formula_units_Z 8\n"
    cif = _si_cif("_symmetry_space_group_name_H-M 'F d -3 m'", xyz, extra=extra)
    s, warns = analyzer.load_structure_robust(_write(tmp_path, "si.cif", cif))
    _assert_diamond_si(s)
    assert any(f"origin choice {chosen} was selected automatically" in w for w in warns)


def test_unspecified_choice_without_z_warns(tmp_path: Path):
    cif = _si_cif("_symmetry_space_group_name_H-M 'F d -3 m'\n_symmetry_Int_Tables_number 227")
    s, warns = analyzer.load_structure_robust(_write(tmp_path, "si.cif", cif))
    assert any("two origin choices" in w and "could not be verified" in w for w in warns)
    # The same warning reaches the multi-block loader.
    results = analyzer.load_structures_all(_write(tmp_path, "si2.cif", cif))
    assert any("could not be verified" in w for w in results[0][1])


def test_it_number_only_unspecified_choice_warns(tmp_path: Path):
    cif = _si_cif("_space_group_IT_number 227")
    _, warns = analyzer.load_structure_robust(_write(tmp_path, "si.cif", cif))
    assert any("two origin choices" in w for w in warns)


def test_single_origin_group_has_no_warning(tmp_path: Path):
    cif = (
        f"data_nacl\n{_cubic_cell(5.64)}_symmetry_space_group_name_H-M 'F m -3 m'\n"
        f"{_ATOM_LOOP}Na1 Na 0 0 0 1.0\nCl1 Cl 0.5 0.5 0.5 1.0\n"
    )
    s, warns = analyzer.load_structure_robust(_write(tmp_path, "nacl.cif", cif))
    assert len(s) == 8
    assert not any("origin" in w for w in warns)


# ---------------------------------------------------------------------------------------
# Rhombohedral groups: hexagonal vs rhombohedral axes
# ---------------------------------------------------------------------------------------
def _r3m_cif(symbol: str, rhombohedral_cell: bool) -> str:
    if rhombohedral_cell:
        cell = (
            "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 4.0\n"
            "_cell_angle_alpha 60\n_cell_angle_beta 60\n_cell_angle_gamma 60\n"
        )
    else:
        cell = (
            "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 9.798\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 120\n"
        )
    return f"data_r\n{cell}_symmetry_space_group_name_H-M '{symbol}'\n{_ATOM_LOOP}Cu1 Cu 0 0 0 1.0\n"


@pytest.mark.parametrize(
    ("symbol", "rhombo", "n_atoms"),
    [
        ("R -3 m :H", False, 3),
        ("R -3 m :R", True, 1),
        ("R -3 m", True, 1),
        ("R -3 m", False, 3),
        ("R -3 m R", True, 1),
    ],
)
def test_r3m_axes(tmp_path: Path, symbol: str, rhombo: bool, n_atoms: int):
    s, _ = analyzer.load_structure_robust(_write(tmp_path, "r.cif", _r3m_cif(symbol, rhombo)))
    assert len(s) == n_atoms
    d, _, _ = min_interatomic_distance(s)
    assert d == pytest.approx(4.0, abs=0.01)

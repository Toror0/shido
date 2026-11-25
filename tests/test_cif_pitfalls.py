"""Crystallographic CIF-loading pitfall ("mean") tests inspired by VESTA's bug history.

Every test writes a self-contained CIF string into ``tmp_path`` and loads it via
``analyzer.load_structure_robust`` (and, where relevant, exports POSCAR via
``analyzer.analyze_structure_data``). Assertions express the crystallographically
correct outcome rather than the current implementation's quirks.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from shido import analyzer
from shido.core.species import dominant_element, site_occupancies
from shido.core.validator import min_interatomic_distance, validate_structure


def _write_cif(tmp_path: Path, filename: str, content: str) -> Path:
    path = tmp_path / filename
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------
# 1. Multiline semicolon-delimited text fields and inline / block '#' comments
# ---------------------------------------------------------------------------------------
def test_multiline_semicolon_text_and_hash_comments(tmp_path: Path):
    """CIF with ';' multiline text (containing '#' and tag-like lines) and '#' comments."""
    cif = """
    # Leading comment before data block
    data_nacl_comments
    # Comment inside header
    _publ_section_abstract
    ;
    Multiline text field for NaCl.
    # A hash line inside a semicolon text field is literal text, not a comment.

    loop_
    _not_a_real_tag 123
    ;
    _cell_length_a   5.6400  # inline comment on lattice parameter
    _cell_length_b   5.6400
    # comment between lattice parameters
    _cell_length_c   5.6400
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'F m -3 m'
    _space_group_IT_number 225

    # Comment right before atom loop
    loop_
    _atom_site_label
    _atom_site_type_symbol
    # Comment inside loop header
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    # Comment between loop header and rows
    Na1 Na 0.0 0.0 0.0 1.0  # inline comment on data row
    Cl1 Cl 0.5 0.5 0.5 1.0
    # Trailing comment at EOF
    """
    path = _write_cif(tmp_path, "multiline_comments.cif", cif)
    struct, _warnings = analyzer.load_structure_robust(path)

    assert len(struct) == 8
    assert struct.lattice.a == pytest.approx(5.64, abs=1e-4)
    elem_comp = struct.composition.element_composition
    assert elem_comp["Na"] == pytest.approx(4.0, abs=1e-4)
    assert elem_comp["Cl"] == pytest.approx(4.0, abs=1e-4)


# ---------------------------------------------------------------------------------------
# 2. Huge standard uncertainties (esd) in coordinates and occupancies
# ---------------------------------------------------------------------------------------
def test_huge_coordinate_esd_does_not_merge_sites(tmp_path: Path):
    """Huge coordinate esd like 0.25(900) must not cause nearby distinct sites to merge."""
    cif = """
    data_huge_coord_esd
    _cell_length_a 10.0
    _cell_length_b 10.0
    _cell_length_c 10.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'P 1'
    _space_group_IT_number 1
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Fe1 Fe 0.25(900) 0.25(900) 0.25(900) 1.0
    Fe2 Fe 0.55(800) 0.25(800) 0.25(800) 1.0
    """
    path = _write_cif(tmp_path, "huge_coord_esd.cif", cif)
    struct, _warnings = analyzer.load_structure_robust(path)

    assert len(struct) == 2
    d_min, _, _ = min_interatomic_distance(struct)
    assert d_min == pytest.approx(3.0, abs=1e-3)


def test_huge_occupancy_esd_keeps_sites_and_occupancy(tmp_path: Path):
    """Huge occupancy esd like 0.5(50) must parse as occupancy 0.5 without dropping sites."""
    cif = """
    data_huge_occ_esd
    _cell_length_a 6.0
    _cell_length_b 6.0
    _cell_length_c 6.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'P 1'
    _space_group_IT_number 1
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Fe1 Fe 0.0 0.0 0.0 0.5(50)
    O1  O  0.5 0.5 0.5 0.5(50)
    """
    path = _write_cif(tmp_path, "huge_occ_esd.cif", cif)
    struct, _warnings = analyzer.load_structure_robust(path)

    assert len(struct) == 2
    for site in struct:
        total_occ = sum(occ for _, occ in site_occupancies(site))
        assert total_occ == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------------------
# 3. New ('A e m 2') vs old ('A b m 2') orthorhombic space-group symbol (No. 39)
# ---------------------------------------------------------------------------------------
def test_spacegroup_aem2_and_abm2_give_same_structure(tmp_path: Path):
    """Space group No. 39 written as 'A e m 2' (new) or 'A b m 2' (old) must both give 8 atoms."""
    template = """
    data_sg39_{tag}
    _cell_length_a 6.0
    _cell_length_b 7.0
    _cell_length_c 8.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt '{hm}'
    _space_group_IT_number 39
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Si1 Si 0.15 0.20 0.25 1.0
    """
    path_new = _write_cif(tmp_path, "aem2.cif", template.format(tag="new", hm="A e m 2"))
    path_old = _write_cif(tmp_path, "abm2.cif", template.format(tag="old", hm="A b m 2"))

    s_new, _ = analyzer.load_structure_robust(path_new)
    s_old, _ = analyzer.load_structure_robust(path_old)

    assert len(s_old) == 8
    assert len(s_new) == 8
    assert StructureMatcher().fit(s_new, s_old)


# ---------------------------------------------------------------------------------------
# 4. Origin choice 1 vs 2 (Fd-3m diamond Si: 8 atoms, Si-Si = 2.35 Å)
# ---------------------------------------------------------------------------------------
def test_fd3m_silicon_origin_choice_1_and_2(tmp_path: Path):
    """Diamond Si in Fd-3m origin choice 1 (0,0,0) and choice 2 (1/8,1/8,1/8) must match."""
    a = 5.4309
    expected_si_si = a * math.sqrt(3.0) / 4.0  # ~2.3516 Å

    cif_origin1 = f"""
    data_si_origin1
    _cell_length_a {a}
    _cell_length_b {a}
    _cell_length_c {a}
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'F d -3 m :1'
    _space_group_IT_number 227
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Si1 Si 0.0 0.0 0.0 1.0
    """

    cif_origin2 = f"""
    data_si_origin2
    _cell_length_a {a}
    _cell_length_b {a}
    _cell_length_c {a}
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'F d -3 m :2'
    _space_group_IT_number 227
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Si1 Si 0.125 0.125 0.125 1.0
    """

    s1, _ = analyzer.load_structure_robust(_write_cif(tmp_path, "si_o1.cif", cif_origin1))
    s2, _ = analyzer.load_structure_robust(_write_cif(tmp_path, "si_o2.cif", cif_origin2))

    assert len(s1) == 8
    assert len(s2) == 8

    d1, _, _ = min_interatomic_distance(s1)
    d2, _, _ = min_interatomic_distance(s2)
    assert d1 == pytest.approx(expected_si_si, abs=0.01)
    assert d2 == pytest.approx(expected_si_si, abs=0.01)
    assert StructureMatcher().fit(s1, s2)


# ---------------------------------------------------------------------------------------
# 5. Boundary atoms (coords 0 and 1.0) on load and POSCAR export
# ---------------------------------------------------------------------------------------
def test_boundary_atoms_p1_load_and_poscar_export(tmp_path: Path):
    """Atoms at fractional coords 0.0 and 1.0 must not be dropped or duplicated in load or POSCAR."""
    a = 4.0
    expected_nn = a / math.sqrt(2.0)
    cif = f"""
    data_boundary_p1
    _cell_length_a {a}
    _cell_length_b {a}
    _cell_length_c {a}
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'P 1'
    _space_group_IT_number 1
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Cu1 Cu 0.0 0.0 0.0 1.0
    Cu2 Cu 0.5 0.5 0.0 1.0
    Cu3 Cu 0.5 1.0 0.5 1.0
    Cu4 Cu 1.0 0.5 0.5 1.0
    """
    path = _write_cif(tmp_path, "boundary_p1.cif", cif)
    struct, _ = analyzer.load_structure_robust(path)

    assert len(struct) == 4
    d_load, _, _ = min_interatomic_distance(struct)
    assert d_load == pytest.approx(expected_nn, abs=1e-3)

    res = analyzer.analyze_structure_data(struct, tmp_path / "analysis_p1")
    poscar_path = Path(res["poscar_file"])
    assert poscar_path.name == "POSCAR"
    assert poscar_path.exists()

    poscar_struct = Structure.from_file(poscar_path)
    assert len(poscar_struct) == 4
    d_poscar, _, _ = min_interatomic_distance(poscar_struct)
    assert d_poscar == pytest.approx(expected_nn, abs=1e-3)
    assert StructureMatcher().fit(struct, poscar_struct)


def test_boundary_atom_at_1_1_1_with_symmetry_and_poscar(tmp_path: Path):
    """FCC Cu placed at (1.0, 1.0, 1.0) must expand to 4 atoms and export 4 atoms to POSCAR."""
    a = 3.61
    expected_nn = a / math.sqrt(2.0)
    cif = f"""
    data_cu_111
    _cell_length_a {a}
    _cell_length_b {a}
    _cell_length_c {a}
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'F m -3 m'
    _space_group_IT_number 225
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Cu1 Cu 1.0 1.0 1.0 1.0
    """
    path = _write_cif(tmp_path, "cu_111.cif", cif)
    struct, _ = analyzer.load_structure_robust(path)

    assert len(struct) == 4
    d_load, _, _ = min_interatomic_distance(struct)
    assert d_load == pytest.approx(expected_nn, abs=1e-3)

    res = analyzer.analyze_structure_data(struct, tmp_path / "analysis_cu")
    poscar_struct = Structure.from_file(Path(res["poscar_file"]))
    assert len(poscar_struct) == 4
    d_poscar, _, _ = min_interatomic_distance(poscar_struct)
    assert d_poscar == pytest.approx(expected_nn, abs=1e-3)
    assert StructureMatcher().fit(struct, poscar_struct)


# ---------------------------------------------------------------------------------------
# 6. Missing Uiso/Biso; missing _atom_site_type_symbol (label only, e.g. 'Fe1')
# ---------------------------------------------------------------------------------------
def test_cif_without_uiso_or_biso(tmp_path: Path):
    """CIF omitting Uiso/Biso thermal displacement columns must load cleanly."""
    cif = """
    data_fe_no_uiso
    _cell_length_a 2.8665
    _cell_length_b 2.8665
    _cell_length_c 2.8665
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'I m -3 m'
    _space_group_IT_number 229
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Fe1 Fe 0.0 0.0 0.0
    """
    path = _write_cif(tmp_path, "no_uiso.cif", cif)
    struct, _ = analyzer.load_structure_robust(path)

    assert len(struct) == 2
    assert struct.composition.reduced_formula == "Fe"
    assert struct.lattice.a == pytest.approx(2.8665, abs=1e-4)


def test_cif_without_type_symbol_uses_label_only(tmp_path: Path):
    """CIF without _atom_site_type_symbol must infer elements from _atom_site_label (e.g. 'Fe1')."""
    cif = """
    data_feo_label_only
    _cell_length_a 4.30
    _cell_length_b 4.30
    _cell_length_c 4.30
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'F m -3 m'
    _space_group_IT_number 225
    loop_
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Fe1 0.0 0.0 0.0
    O1  0.5 0.5 0.5
    """
    path = _write_cif(tmp_path, "label_only.cif", cif)
    struct, _ = analyzer.load_structure_robust(path)

    assert len(struct) == 8
    elem_comp = struct.composition.element_composition
    assert elem_comp["Fe"] == pytest.approx(4.0, abs=1e-4)
    assert elem_comp["O"] == pytest.approx(4.0, abs=1e-4)


# ---------------------------------------------------------------------------------------
# 7. Mixed B_ij (_atom_site_aniso_B_11 ...) and U_ij (_atom_site_aniso_U_11 ...)
# ---------------------------------------------------------------------------------------
def test_mixed_aniso_bij_and_uij_does_not_crash(tmp_path: Path):
    """CIF containing both _atom_site_aniso_B_ij and _atom_site_aniso_U_ij must not crash."""
    cif = """
    data_mixed_aniso
    _cell_length_a 5.64
    _cell_length_b 5.64
    _cell_length_c 5.64
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'F m -3 m'
    _space_group_IT_number 225
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Na1 Na 0.0 0.0 0.0 1.0
    Cl1 Cl 0.5 0.5 0.5 1.0
    loop_
    _atom_site_aniso_label
    _atom_site_aniso_U_11
    _atom_site_aniso_U_22
    _atom_site_aniso_U_33
    _atom_site_aniso_U_12
    _atom_site_aniso_U_13
    _atom_site_aniso_U_23
    _atom_site_aniso_B_11
    _atom_site_aniso_B_22
    _atom_site_aniso_B_33
    _atom_site_aniso_B_12
    _atom_site_aniso_B_13
    _atom_site_aniso_B_23
    Na1 0.015 0.015 0.015 0.0 0.0 0.0 . . . . . .
    Cl1 . . . . . . 1.20 1.20 1.20 0.0 0.0 0.0
    """
    path = _write_cif(tmp_path, "mixed_aniso.cif", cif)
    struct, _ = analyzer.load_structure_robust(path)

    assert len(struct) == 8
    elem_comp = struct.composition.element_composition
    assert elem_comp["Na"] == pytest.approx(4.0, abs=1e-4)
    assert elem_comp["Cl"] == pytest.approx(4.0, abs=1e-4)


# ---------------------------------------------------------------------------------------
# 8. Occupancy slightly > 1 (1.002) and oxidation state in label ('Fe3+')
# ---------------------------------------------------------------------------------------
def test_occupancy_slightly_above_one_is_physical(tmp_path: Path):
    """Refinement occupancy 1.002 must load and be clamped/normalized to physical <= 1.0."""
    cif = """
    data_occ_over_one
    _cell_length_a 4.0
    _cell_length_b 4.0
    _cell_length_c 4.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'P 1'
    _space_group_IT_number 1
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Fe1 Fe 0.0 0.0 0.0 1.002
    O1  O  0.5 0.5 0.5 1.000
    """
    path = _write_cif(tmp_path, "occ_1002.cif", cif)
    struct, _ = analyzer.load_structure_robust(path)

    assert len(struct) == 2
    for site in struct:
        total_occ = sum(occ for _, occ in site_occupancies(site))
        assert total_occ == pytest.approx(1.0, abs=1e-3)
    # Downstream scientific validator must also accept the loaded structure.
    validate_structure(struct)


def test_oxidation_state_in_label_only(tmp_path: Path):
    """CIF with oxidation state in _atom_site_label ('Fe3+', 'O2-') and no type_symbol."""
    cif = """
    data_ox_label_only
    _cell_length_a 4.0
    _cell_length_b 4.0
    _cell_length_c 4.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'P 1'
    _space_group_IT_number 1
    loop_
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Fe3+ 0.0 0.0 0.0 1.0
    O2-  0.5 0.5 0.5 1.0
    """
    path = _write_cif(tmp_path, "ox_label_only.cif", cif)
    struct, _ = analyzer.load_structure_robust(path)

    assert len(struct) == 2
    elems = {dominant_element(site) for site in struct}
    assert elems == {"Fe", "O"}


def test_oxidation_state_in_label_and_type_symbol(tmp_path: Path):
    """CIF with oxidation state 'Fe3+' / 'O2-' in both label and type_symbol."""
    cif = """
    data_ox_both
    _cell_length_a 4.0
    _cell_length_b 4.0
    _cell_length_c 4.0
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'P 1'
    _space_group_IT_number 1
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
    Fe3+ Fe3+ 0.0 0.0 0.0 1.0
    O2-  O2-  0.5 0.5 0.5 1.0
    """
    path = _write_cif(tmp_path, "ox_both.cif", cif)
    struct, _ = analyzer.load_structure_robust(path)

    assert len(struct) == 2
    elems = {dominant_element(site) for site in struct}
    assert elems == {"Fe", "O"}

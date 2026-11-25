"""v3 numeric outputs: bond angles, atomic positions, POSCAR cell choice, multi-block CIF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from shido.analyzer import (
    ATOMIC_POSITION_COLUMNS,
    BOND_ANGLE_COLUMNS,
    analyze_structure_data,
    build_ni2mnga_b2_disordered,
    build_ni2mnga_structure,
    compute_bond_angles,
    convert_cell,
    load_structures_all,
)
from shido.core.validator import validate_composition_conserved

TETRAHEDRAL = 109.4712206


def _si() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def test_si_bond_angles_all_tetrahedral(out_dir):
    res = analyze_structure_data(_si(), out_dir)
    df = pd.read_csv(res["bond_angles_csv"])
    assert list(df.columns) == BOND_ANGLE_COLUMNS
    # 8 sites x C(4, 2) = 48 angles
    assert len(df) == 48 == res["num_bond_angles"]
    assert np.allclose(df["angle_deg"], TETRAHEDRAL, atol=0.01)
    assert not res["bond_angles_truncated"]


def test_nacl_cl_na_cl_is_90_and_180(out_dir):
    res = analyze_structure_data(_nacl(), out_dir)
    df = pd.read_csv(res["bond_angles_csv"])
    na = df[df["element_center"] == "Na"]
    assert set(na["element_a"]) == {"Cl"} and set(na["element_b"]) == {"Cl"}
    angles = na["angle_deg"].to_numpy()
    assert np.all(np.isclose(angles, 90.0, atol=0.01) | np.isclose(angles, 180.0, atol=0.01))
    # octahedron: per center 12 x 90 deg and 3 x 180 deg
    assert np.isclose(angles, 90.0, atol=0.01).sum() == 12 * 4
    assert np.isclose(angles, 180.0, atol=0.01).sum() == 3 * 4
    # sorted by element_center then angle
    assert df["element_center"].is_monotonic_increasing
    for _, grp in df.groupby("element_center"):
        assert grp["angle_deg"].is_monotonic_increasing


def test_atomic_positions_cartesian_matches_lattice(out_dir):
    s = build_ni2mnga_structure(5.825)
    res = analyze_structure_data(s, out_dir)
    df = pd.read_csv(res["atomic_positions_csv"], keep_default_na=False)
    assert list(df.columns) == ATOMIC_POSITION_COLUMNS
    assert len(df) == len(s)
    frac = df[["frac_x", "frac_y", "frac_z"]].to_numpy(dtype=float)
    cart = df[["cart_x", "cart_y", "cart_z"]].to_numpy(dtype=float)
    assert np.allclose(cart, s.lattice.get_cartesian_coords(frac), atol=1e-5)
    assert np.allclose(frac, s.frac_coords, atol=1e-5)
    assert set(df["wyckoff"]) == {"4a", "4b", "8c"}
    assert set(df["occupancy"]) == {"Ni:1", "Mn:1", "Ga:1"}


def test_poscar_cell_primitive_and_conventional_ni2mnga(out_dir):
    s = build_ni2mnga_structure(5.825)
    assert len(s) == 16
    prim = convert_cell(s, "primitive")
    conv = convert_cell(s, "conventional")
    assert len(prim) == 4
    assert len(conv) == 16
    validate_composition_conserved(s, prim)
    validate_composition_conserved(s, conv)
    assert prim.composition.reduced_formula == conv.composition.reduced_formula == s.composition.reduced_formula

    for cell, n in (("input", 16), ("conventional", 16), ("primitive", 4)):
        res = analyze_structure_data(s, out_dir / cell, poscar_cell=cell)
        assert res["poscar_cell"] == cell
        assert res["poscar_num_sites"] == n
        written = Poscar.from_file(res["poscar_file"]).structure
        assert len(written) == n
        validate_composition_conserved(s, written)


def test_poscar_cell_rejects_unknown_value(out_dir):
    with pytest.raises(ValueError):
        analyze_structure_data(build_ni2mnga_structure(5.825), out_dir, poscar_cell="hexagonal")


def test_b2_disordered_csvs_are_written(out_dir):
    s = build_ni2mnga_b2_disordered(5.825)
    res = analyze_structure_data(s, out_dir)
    pos = pd.read_csv(res["atomic_positions_csv"], keep_default_na=False)
    ang = pd.read_csv(res["bond_angles_csv"])
    assert len(pos) == len(s)
    mixed = pos[pos["occupancy"].str.contains(" ")]
    assert len(mixed) == 8
    assert set(mixed["occupancy"]) == {"Ga:0.5 Mn:0.5"}
    assert set(mixed["element"]) <= {"Mn", "Ga"}
    assert len(ang) > 0 and ang["angle_deg"].between(0, 180).all()


def test_bond_angle_caps_limit_rows():
    df, truncated = compute_bond_angles(build_ni2mnga_structure(5.825), max_neighbors_per_site=3, max_rows=10)
    assert truncated
    assert len(df) <= 10


_TWO_BLOCK_CIF = """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
  Na1 Na 0.0 0.0 0.0 1.0
  Cl1 Cl 0.5 0.5 0.5 1.0

data_Cu
_cell_length_a 3.615
_cell_length_b 3.615
_cell_length_c 3.615
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
  Cu1 Cu 0.0 0.0 0.0 1.0
  Cu2 Cu 0.5 0.5 0.0 1.0
  Cu3 Cu 0.5 0.0 0.5 1.0
  Cu4 Cu 0.0 0.5 0.5 1.0
"""


def test_multi_block_cif_loads_all(tmp_path: Path):
    path = tmp_path / "two_blocks.cif"
    path.write_text(_TWO_BLOCK_CIF, encoding="utf-8")
    items = load_structures_all(path)
    assert [name for _, _, name in items] == ["NaCl", "Cu"]
    (s1, w1, _), (s2, w2, _) = items
    assert s1.composition.reduced_formula == "NaCl" and len(s1) == 2
    assert s2.composition.reduced_formula == "Cu" and len(s2) == 4
    assert isinstance(w1, list) and isinstance(w2, list)

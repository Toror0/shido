"""Batch 3: cell transformation (transform_cell / `server.py transform`)."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido.core.cell_transform import (
    apply_transform,
    atoms_in_range,
    parse_matrix,
    transform_cell,
    write_transformed,
)

F_TO_P = "0 1/2 1/2  1/2 0 1/2  1/2 1/2 0"
P_TO_F = "-1 1 1  1 -1 1  1 1 -1"


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _cu() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.615), ["Cu"], [[0, 0, 0]])


def _mg() -> Structure:
    return Structure.from_spacegroup(
        "P6_3/mmc", Lattice.hexagonal(3.209, 5.211), ["Mg"], [[1 / 3, 2 / 3, 0.25]]
    )


def _same_composition(a: Structure, b: Structure) -> None:
    fa = a.composition.fractional_composition
    fb = b.composition.fractional_composition
    assert fa.almost_equals(fb)


def test_parse_matrix_forms():
    assert np.allclose(parse_matrix("2 2 1"), np.diag([2, 2, 1]))
    assert np.allclose(parse_matrix(F_TO_P)[0], [0, 0.5, 0.5])
    assert np.allclose(parse_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.eye(3))
    with pytest.raises(ValueError):
        parse_matrix("1 2")


def test_cu_f_to_p_and_back():
    cu = _cu()
    prim, rep = transform_cell(cu, transform=F_TO_P)
    assert len(prim) == 1
    assert rep["det_P_total"] == pytest.approx(0.25)
    assert prim.density == pytest.approx(cu.density, rel=1e-6)
    assert prim.lattice.a == pytest.approx(3.615 / np.sqrt(2), abs=1e-6)
    assert prim.lattice.alpha == pytest.approx(60.0, abs=1e-4)
    back, rep2 = transform_cell(prim, transform=P_TO_F)
    assert len(back) == 4
    assert rep2["det_P_total"] == pytest.approx(4.0)
    assert back.lattice.abc == pytest.approx((3.615,) * 3, abs=1e-6)
    assert back.lattice.angles == pytest.approx((90.0,) * 3, abs=1e-4)


def test_nacl_supercell_volume_ratio_and_density():
    nacl = _nacl()
    sc, rep = transform_cell(nacl, supercell="2 2 1")
    assert len(sc) == 32
    assert rep["volume_ratio"] == pytest.approx(4.0)
    assert rep["output"]["density_g_cm3"] == pytest.approx(rep["input"]["density_g_cm3"])
    _same_composition(nacl, sc)


def test_nacl_niggli_and_primitive_are_rhombohedral_60deg():
    for cell in ("niggli", "primitive"):
        s, rep = transform_cell(_nacl(), cell=cell)
        out = rep["output"]
        assert out["n_sites"] == 2
        assert (out["a"], out["b"], out["c"]) == pytest.approx((3.988082,) * 3, abs=1e-5)
        assert (out["alpha"], out["beta"], out["gamma"]) == pytest.approx((60.0,) * 3, abs=1e-3)
        assert out["spacegroup_number"] == 225


def test_primitive_to_standard_conventional():
    prim = _nacl().get_primitive_structure()
    s, rep = transform_cell(prim, cell="standard")
    assert len(s) == 8
    assert s.lattice.abc == pytest.approx((5.64,) * 3, abs=1e-5)


def test_hexagonal_gamma60_setting_reduced_to_120():
    g60, _ = apply_transform(_mg(), parse_matrix("1 1 0  0 1 0  0 0 1"))
    assert g60.lattice.gamma == pytest.approx(60.0, abs=1e-6)
    for cell in ("niggli", "standard"):
        s, rep = transform_cell(g60, cell=cell)
        assert s.lattice.gamma == pytest.approx(120.0, abs=1e-4)
        assert s.lattice.a == pytest.approx(3.209, abs=1e-6)
        assert s.lattice.c == pytest.approx(5.211, abs=1e-6)
        assert rep["output"]["spacegroup_number"] == 194


def test_negative_determinant_warns():
    s, rep = transform_cell(_nacl(), transform="1 0 0  0 1 0  0 0 -1")
    assert len(s) == 8
    assert rep["det_P_total"] == pytest.approx(-1.0)
    assert any("left-handed" in w for w in rep["warnings"])


def test_origin_shift_moves_cl_to_origin():
    s, _ = transform_cell(_nacl(), origin_shift="1/2 1/2 1/2")
    at_origin = [site for site in s if np.allclose(site.frac_coords, 0.0, atol=1e-6)]
    assert len(at_origin) == 1 and at_origin[0].specie.symbol == "Cl"


def test_non_lattice_axis_is_rejected():
    with pytest.raises(ValueError, match="not a lattice translation"):
        transform_cell(_nacl(), transform="1/2 1 1")
    with pytest.raises(ValueError, match="singular"):
        transform_cell(_nacl(), transform="1 1 0")


def test_write_transformed_files_round_trip(tmp_path):
    s, _ = transform_cell(_cu(), transform=F_TO_P)
    files = write_transformed(s, tmp_path, stem="Cu_prim")
    back = Structure.from_file(files["cif"])
    assert len(back) == 1
    assert back.volume == pytest.approx(s.volume, rel=1e-4)
    poscar = Structure.from_file(files["poscar"])
    assert len(poscar) == 1


def test_atoms_in_range_counts():
    nacl = _nacl()
    # 0..1 incl. boundaries: Na 8 corners + 6 faces = 14, Cl 12 edges + 1 centre = 13.
    assert len(atoms_in_range(nacl, [0, 1, 0, 1, 0, 1])) == 27
    # -1..2: 3x3x3 cells -> (2*3+1)^3 = 343 fcc+fcc lattice points on a 1/2 grid.
    assert len(atoms_in_range(nacl, [-1, 2, -1, 2, -1, 2])) == 343


def test_transform_cli(tmp_path, capsys):
    from shido.server import main

    cif = tmp_path / "Cu.cif"
    _cu().to(filename=str(cif))
    rc = main(["transform", str(cif), "--transform", F_TO_P, "--out-dir", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "4 -> 1 sites" in out
    assert (tmp_path / "out" / "Cu_transformed.cif").is_file()
    assert (tmp_path / "out" / "Cu_transformed.vasp").is_file()

"""Displacement parameters (core.adp): CIF reading, U_cif -> U_cart, principal axes, probability."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pymatgen.analysis.diffraction.core import get_ustar
from pymatgen.core import Lattice, Structure

from shido import analyzer
from shido.core.adp import (
    DISPLACEMENT_COLUMNS,
    analyze_displacement,
    attach_ucif_site_properties,
    ellipsoid_axes,
    probability_scale,
    read_cif_adp,
    site_displacement_tensors,
    ucif_to_ucart,
    ucif_to_ustar,
    u_equivalent,
    write_displacement_report,
)
from shido.core.structure_factors import compute_structure_factors
from shido.core.validator import ScientificValidationError

COD = Path(__file__).parent / "data" / "cod"


def _cif(tmp_path: Path, name: str, cell: str, symops: list[str], atoms: str, extra: str = "") -> Path:
    text = (
        f"data_{name}\n{cell}\nloop_\n_symmetry_equiv_pos_as_xyz\n" + "\n".join(f"'{s}'" for s in symops)
        + "\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
        "_atom_site_fract_z\n_atom_site_occupancy\n_atom_site_U_iso_or_equiv\n" + atoms + "\n" + extra + "\n"
    )
    p = tmp_path / f"{name}.cif"
    p.write_text(text, encoding="utf-8")
    return p


def _cell(a, b, c, al=90, be=90, ga=90) -> str:
    return (f"_cell_length_a {a}\n_cell_length_b {b}\n_cell_length_c {c}\n"
            f"_cell_angle_alpha {al}\n_cell_angle_beta {be}\n_cell_angle_gamma {ga}")


def test_probability_scale():
    assert probability_scale(50) == pytest.approx(1.5382, abs=1e-4)
    assert probability_scale(0.5) == pytest.approx(1.5382, abs=1e-4)
    assert probability_scale(99) == pytest.approx(math.sqrt(11.3449), abs=1e-3)
    with pytest.raises(ScientificValidationError):
        probability_scale(100)


def test_isotropic_u_gives_equal_axes_in_triclinic_cell(tmp_path):
    """U_iso only: three equal principal axes, U_eq = U_iso, even in a triclinic cell."""
    p = _cif(tmp_path, "iso", _cell(5.1, 6.2, 7.3, 80, 95, 110), ["x,y,z", "-x,-y,-z"],
             "Na1 Na 0.1 0.2 0.3 1 0.0123(4)\nCl1 Cl 0.4 0.3 0.2 1 0.0200")
    df, rep = analyze_displacement(p, 50)
    assert rep["n_isotropic"] == 2 and rep["n_anisotropic"] == 0
    for _, r in df.iterrows():
        assert r["U1"] == pytest.approx(r["U3"], rel=1e-9)
        assert r["U_eq"] == pytest.approx(r["U_iso_cif"], rel=1e-9)
        assert r["semi_axis1"] == pytest.approx(1.5382 * math.sqrt(r["U_iso_cif"]), rel=1e-4)
    # the CIF U tensor of an isotropic atom is not diagonal in a triclinic cell
    assert abs(df.loc[0, "U12"]) > 1e-5


def test_b_iso_and_b_aniso_are_converted(tmp_path):
    cif = (
        f"data_b\n{_cell(4, 4, 4)}\nloop_\n_symmetry_equiv_pos_as_xyz\nx,y,z\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "_atom_site_B_iso_or_equiv\nA1 0 0 0 0.789568\nA2 0.5 0.5 0.5 1.0\n"
        "loop_\n_atom_site_aniso_label\n_atom_site_aniso_B_11\n_atom_site_aniso_B_22\n_atom_site_aniso_B_33\n"
        "_atom_site_aniso_B_12\n_atom_site_aniso_B_13\n_atom_site_aniso_B_23\nA2 1.5 0.75 0.75 0 0 0\n"
    )
    p = tmp_path / "b.cif"
    p.write_text(cif, encoding="utf-8")
    df, _ = analyze_displacement(p)
    a1, a2 = df.iloc[0], df.iloc[1]
    assert a1["source"] == "Biso" and a1["U_eq"] == pytest.approx(0.01, rel=1e-5)
    assert a2["source"] == "B" and a2["U11"] == pytest.approx(1.5 / (8 * math.pi**2))
    assert a2["U3"] == pytest.approx(1.5 / (8 * math.pi**2))


def test_beta_ij_is_converted(tmp_path):
    """beta = 2 pi^2 U*: orthorhombic, U11 = beta11 / (2 pi^2 a*^2)."""
    a, b, c = 4.0, 5.0, 6.0
    u = (0.01, 0.02, 0.03)
    betas = [2 * math.pi**2 * u[i] / L**2 for i, L in enumerate((a, b, c))]
    cif = (
        f"data_beta\n{_cell(a, b, c)}\nloop_\n_symmetry_equiv_pos_as_xyz\nx,y,z\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nX1 0 0 0\n"
        "loop_\n_atom_site_aniso_label\n_atom_site_aniso_beta_11\n_atom_site_aniso_beta_22\n"
        "_atom_site_aniso_beta_33\n_atom_site_aniso_beta_12\n_atom_site_aniso_beta_13\n_atom_site_aniso_beta_23\n"
        f"X1 {betas[0]} {betas[1]} {betas[2]} 0 0 0\n"
    )
    p = tmp_path / "beta.cif"
    p.write_text(cif, encoding="utf-8")
    r = analyze_displacement(p)[0].iloc[0]
    assert r["source"] == "beta"
    assert (r["U11"], r["U22"], r["U33"]) == pytest.approx(u, rel=1e-9)
    assert (r["U1"], r["U2"], r["U3"]) == pytest.approx(u, rel=1e-9)


def test_cubic_symmetry_keeps_u11_u22_u33(tmp_path):
    """Cubic site with U11=U22=U33: every symmetry image keeps the same diagonal tensor."""
    ops = ["x,y,z", "-x,-y,z", "-x,y,-z", "x,-y,-z", "z,x,y", "z,-x,-y", "-z,-x,y", "-z,x,-y",
           "y,z,x", "-y,z,-x", "y,-z,-x", "-y,-z,x", "-x,-y,-z", "x,y,-z", "x,-y,z", "-x,y,z",
           "-z,-x,-y", "-z,x,y", "z,x,-y", "z,-x,y", "-y,-z,-x", "y,-z,x", "-y,z,x", "y,z,-x"]
    aniso = ("loop_\n_atom_site_aniso_label\n_atom_site_aniso_U_11\n_atom_site_aniso_U_22\n_atom_site_aniso_U_33\n"
             "_atom_site_aniso_U_12\n_atom_site_aniso_U_13\n_atom_site_aniso_U_23\nM1 0.012 0.012 0.012 0 0 0\n")
    p = _cif(tmp_path, "cubic", _cell(5, 5, 5), ops, "M1 Fe 0.25 0.1 0.1 1 0.012", aniso)
    s = Structure.from_spacegroup("Pm-3", Lattice.cubic(5), ["Fe"], [[0.25, 0.1, 0.1]])
    tensors, labels, _ = site_displacement_tensors(s, p)
    assert len(s) == 24 and all(t is not None for t in tensors) and set(labels) == {"M1"}
    for t in tensors:
        assert np.allclose(t, 0.012 * np.eye(3), atol=1e-12)
    r = analyze_displacement(p, structure=s)[0].iloc[0]
    assert r["U11"] == pytest.approx(r["U22"]) == pytest.approx(r["U33"]) == pytest.approx(0.012)
    assert r["n_sites_in_cell"] == 24


def test_symmetry_rotates_the_tensor(tmp_path):
    """4-fold axis along z: the image of an x-elongated atom is elongated along y."""
    aniso = ("loop_\n_atom_site_aniso_label\n_atom_site_aniso_U_11\n_atom_site_aniso_U_22\n_atom_site_aniso_U_33\n"
             "_atom_site_aniso_U_12\n_atom_site_aniso_U_13\n_atom_site_aniso_U_23\nA1 0.03 0.01 0.01 0 0 0\n")
    ops = ["x,y,z", "-y,x,z", "-x,-y,z", "y,-x,z"]
    p = _cif(tmp_path, "tet", _cell(4, 4, 3), ops, "A1 Na 0.3 0 0 1 0.0167", aniso)
    s = Structure.from_spacegroup("P4", Lattice.tetragonal(4, 3), ["Na"], [[0.3, 0, 0]])
    tensors, _, _ = site_displacement_tensors(s, p)
    assert len(s) == 4
    for site, t in zip(s, tensors):
        y = site.frac_coords[1] % 1.0
        big = 1 if 0.1 < y < 0.9 else 0  # atoms on the y axis are elongated along y
        assert t[big, big] == pytest.approx(0.03) and t[1 - big, 1 - big] == pytest.approx(0.01)
        assert t[2, 2] == pytest.approx(0.01)


def test_monoclinic_u_eq_equals_trace_over_three(tmp_path):
    """U_eq = trace(U_cart)/3 = (1/3)[U22 + (U11 + U33 + 2 U13 cos(beta)) / sin^2(beta)]."""
    beta = 105.3
    u11, u22, u33, u13 = 0.021, 0.015, 0.032, 0.004
    aniso = ("loop_\n_atom_site_aniso_label\n_atom_site_aniso_U_11\n_atom_site_aniso_U_22\n_atom_site_aniso_U_33\n"
             f"_atom_site_aniso_U_12\n_atom_site_aniso_U_13\n_atom_site_aniso_U_23\nC1 {u11} {u22} {u33} 0 {u13} 0\n")
    p = _cif(tmp_path, "mono", _cell(7.1, 5.3, 9.4, 90, beta, 90), ["x,y,z", "-x,y+1/2,-z"],
             "C1 C 0.1 0.2 0.3 1 .", aniso)
    r = analyze_displacement(p)[0].iloc[0]
    b = math.radians(beta)
    expected = (u22 + (u11 + u33 + 2 * u13 * math.cos(b)) / math.sin(b) ** 2) / 3
    assert r["U_eq"] == pytest.approx(expected, rel=1e-9)
    assert r["U_eq"] == pytest.approx((r["U1"] + r["U2"] + r["U3"]) / 3, rel=1e-9)
    lat = Lattice.from_parameters(7.1, 5.3, 9.4, 90, beta, 90)
    ucif = np.array([[u11, 0, u13], [0, u22, 0], [u13, 0, u33]])
    assert u_equivalent(ucif_to_ucart(ucif, lat)) == pytest.approx(expected, rel=1e-9)


def test_non_positive_definite_is_flagged():
    semi, _, ok = ellipsoid_axes(np.diag([0.01, 0.02, -0.001]), 50)
    assert not ok and np.isnan(semi[0]) and semi[2] == pytest.approx(1.5382 * math.sqrt(0.02), rel=1e-4)


def test_cod_triclinic_shelx_matches_u_equiv():
    """COD 1100710 (P-1, SHELXL): computed U_eq equals the file's U_iso_or_equiv for Uani atoms."""
    cif = COD / "1100710.cif"
    s, _ = analyzer.load_structure_robust(cif)
    df, rep = analyze_displacement(cif, 50, structure=s)
    assert rep["n_anisotropic"] == 11 and rep["n_isotropic"] == 8 and not rep["non_positive_definite"]
    uani = df[df["adp_type"] == "Uani"]
    assert np.allclose(uani["U_eq"], uani["U_iso_cif"], atol=6e-4)
    pt = df[df["label"] == "Pt1"].iloc[0]
    assert pt["U11"] == pytest.approx(0.00663) and pt["U23"] == pytest.approx(0.00043)
    assert pt["U_eq"] == pytest.approx(0.00631, abs=2e-5)
    tensors, _, notes = site_displacement_tensors(s, cif)
    assert all(t is not None for t in tensors) and not notes


@pytest.mark.parametrize("cod_id, n_aniso", [("1000030", 8), ("1008785", 5), ("9001735", 12)])
def test_cod_aniso_values_as_in_file(cod_id, n_aniso):
    cif = COD / f"{cod_id}.cif"
    adp = read_cif_adp(cif)
    atoms = {a["label"]: a for a in adp["atoms"]}
    assert sum(a["adp_type"] == "Uani" for a in atoms.values()) == n_aniso
    s, _ = analyzer.load_structure_robust(cif)
    tensors, _, _ = site_displacement_tensors(s, adp=adp)
    assert all(t is not None for t in tensors)
    if cod_id == "1000030":  # YBCO: U_cif given directly, orthorhombic
        assert np.allclose(np.diag(atoms["Ba1"]["u_cif"]), [0.0078, 0.0096, 0.0198])
    if cod_id == "1008785":  # urea, column order U11 U12 U13 U22 U23 U33
        assert atoms["N1"]["u_cif"][2, 2] == pytest.approx(_raw(cif, "N1")[5])


def _raw(cif: Path, label: str) -> list[float]:
    for line in cif.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] == label and len(parts) == 7:
            return [float(x.split("(")[0]) for x in parts[1:]]
    raise AssertionError(label)


def test_cod_without_adp_reports_it():
    df, rep = analyze_displacement(COD / "1000068.cif")
    assert rep["n_without_adp"] == len(df) and rep["warnings"]


def test_attach_ucif_matches_pymatgen_and_structure_factors(tmp_path):
    """U11_cif ... site properties: pymatgen's U* = ours, and isotropic U gives F as with B = 8 pi^2 U."""
    cif = COD / "1100710.cif"
    s, _ = analyzer.load_structure_robust(cif)
    s2 = attach_ucif_site_properties(s, cif)
    tensors, _, _ = site_displacement_tensors(s, cif)
    ustar = get_ustar(s2)
    a = s.lattice.matrix.T
    assert np.allclose([a @ u @ a.T for u in ustar], tensors, atol=1e-10)

    p = _cif(tmp_path, "nacl", _cell(5.64, 5.64, 5.64), ["x,y,z"],
             "Na1 Na 0 0 0 1 0.01\nCl1 Cl 0.5 0.5 0.5 1 0.01")
    nacl = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    df_u, rep_u = compute_structure_factors(attach_ucif_site_properties(nacl, p), dmin=1.5)
    b = 8 * math.pi**2 * 0.01
    df_b, _ = compute_structure_factors(nacl, dmin=1.5, debye_waller_factors={"Na": b, "Cl": b})
    assert rep_u["anisotropic_U"]
    assert np.allclose(df_u["F_abs"], df_b["F_abs"], rtol=1e-9)


def test_write_report_and_cli(tmp_path, capsys):
    cif = COD / "1100710.cif"
    df, rep = analyze_displacement(cif)
    files = write_displacement_report(df, rep, tmp_path / "w")
    assert Path(files["csv"]).name == "displacement.csv"
    assert json.loads(Path(files["json"]).read_text(encoding="utf-8"))["radius_scale"] == pytest.approx(1.5382, abs=1e-4)
    assert list(df.columns) == DISPLACEMENT_COLUMNS

    from shido.server import main

    out = tmp_path / "cli"
    rc = main(["displacement", str(cif), "--probability", "30", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "displacement.csv").is_file()
    text = capsys.readouterr().out
    assert "Pt1" in text and "30" in text


def test_ucif_to_ustar_diagonal_orthorhombic():
    lat = Lattice.orthorhombic(2, 4, 5)
    u = ucif_to_ustar(np.diag([0.04, 0.16, 0.25]), lat)
    assert np.allclose(np.diag(u), [0.01, 0.01, 0.01])

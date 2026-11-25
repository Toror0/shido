"""Batch 11: polyhedron metrics, bond valence sum, Madelung (core/polyhedra_metrics.py)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido.core.polyhedra_metrics import analyze_polyhedra, polyhedron_metrics


def _nacl(a: float = 5.640) -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _cscl(a: float = 4.123) -> Structure:
    return Structure(Lattice.cubic(a), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _caf2(a: float = 5.463) -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Ca", "F"], [[0, 0, 0], [0.25, 0.25, 0.25]])


OCTA = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)


def test_regular_octahedron():
    m = polyhedron_metrics(np.zeros(3), 2.0 * OCTA)
    assert m["coordination"] == 6
    assert m["volume_A3"] == pytest.approx(4.0 / 3.0 * 2.0**3)
    assert m["quadratic_elongation"] == pytest.approx(1.0, abs=1e-12)
    assert m["bond_angle_variance_deg2"] == pytest.approx(0.0, abs=1e-9)
    assert m["distortion_index"] == pytest.approx(0.0, abs=1e-12)
    assert m["ECoN"] == pytest.approx(6.0)


def test_regular_tetrahedron():
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    m = polyhedron_metrics(np.zeros(3), t)
    assert m["quadratic_elongation"] == pytest.approx(1.0, abs=1e-12)
    assert m["bond_angle_variance_deg2"] == pytest.approx(0.0, abs=1e-9)


def test_distorted_octahedron():
    # Stretch two trans bonds by 10 %: Baur D = (1/6)(2*|1.1-l|+4*|1-l|)/l, l = 6.2/6.
    pts = OCTA.copy()
    pts[4:] *= 1.1
    m = polyhedron_metrics(np.zeros(3), pts)
    l_av = 6.2 / 6.0
    assert m["distortion_index"] == pytest.approx((2 * (1.1 - l_av) + 4 * (l_av - 1.0)) / 6.0 / l_av)
    assert m["quadratic_elongation"] > 1.0
    assert m["bond_angle_variance_deg2"] == pytest.approx(0.0, abs=1e-9)  # angles still 90
    assert m["ECoN"] < 6.0


def test_nacl_polyhedra_bvs_and_madelung():
    df, rep = analyze_polyhedra(_nacl(), madelung=True)
    na = df[df.element == "Na"].iloc[0]
    cl = df[df.element == "Cl"].iloc[0]
    assert na.coordination == 6 and na.ligands == "Cl6"
    assert na.volume_A3 == pytest.approx(4.0 / 3.0 * (5.640 / 2) ** 3)
    assert na.quadratic_elongation == pytest.approx(1.0)
    assert na.bond_angle_variance_deg2 == pytest.approx(0.0, abs=1e-8)
    assert na.bvs == pytest.approx(1.0, abs=0.05)
    assert cl.bvs == pytest.approx(-1.0, abs=0.05)
    assert na.bvs_method == "brese_okeeffe"
    mad = rep["madelung"]
    assert mad["madelung_constant"] == pytest.approx(1.747565, abs=1e-4)
    assert mad["r0_A"] == pytest.approx(2.820)
    # E per formula unit = -M k / r0
    assert mad["energy_per_formula_unit_eV"] == pytest.approx(-1.747565 * 14.399645 / 2.820, rel=1e-4)
    assert na.site_potential_V == pytest.approx(-cl.site_potential_V)


@pytest.mark.parametrize(
    "builder,expected",
    [(_cscl, 1.762675), (_caf2, 2.519392)],
)
def test_madelung_constants(builder, expected):
    _, rep = analyze_polyhedra(builder(), madelung=True)
    assert rep["madelung"]["madelung_constant"] == pytest.approx(expected, abs=1e-4)


def test_caf2_cube_and_tetrahedron():
    df, _ = analyze_polyhedra(_caf2())
    ca = df[df.element == "Ca"].iloc[0]
    f = df[df.element == "F"].iloc[0]
    assert ca.coordination == 8 and f.coordination == 4
    for row in (ca, f):
        assert row.quadratic_elongation == pytest.approx(1.0)
        assert row.bond_angle_variance_deg2 == pytest.approx(0.0, abs=1e-8)
    assert ca.bvs == pytest.approx(2.0, abs=0.1)


def test_bond_specs_are_used():
    # Only Na-Cl up to 2.5 A: nothing in NaCl (d = 2.82), so no polyhedra and BVS 0.
    df, _ = analyze_polyhedra(_nacl(), bond_specs=["Na-Cl:0:2.5"])
    assert (df.coordination == 0).all()
    assert math.isclose(float(df.bvs.abs().sum()), 0.0)
    df2, _ = analyze_polyhedra(_nacl(), bond_specs=["Na-Cl:0:3.0"])
    assert (df2.coordination == 6).all()


def test_madelung_given_oxidation_states_and_missing():
    _, rep = analyze_polyhedra(_nacl(), madelung=True, oxidation_states={"Na": 1, "Cl": -1})
    assert rep["madelung"]["oxidation_source"] == "given"
    _, rep2 = analyze_polyhedra(_nacl(), madelung=True, oxidation_states={"Na": 1})
    assert "error" in rep2["madelung"]


def test_polyhedra_cli(tmp_path, capsys):
    from shido.server import main

    cif = tmp_path / "NaCl.cif"
    _nacl().to(filename=str(cif))
    out = tmp_path / "out"
    rc = main(["polyhedra", str(cif), "--madelung", "--out-dir", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "Madelung constant = 1.747" in text
    assert (out / "polyhedra.csv").is_file()
    rep = json.loads((out / "polyhedra.json").read_text(encoding="utf-8"))
    assert rep["madelung"]["madelung_constant"] == pytest.approx(1.7476, abs=1e-3)
    assert len(rep["sites"]) == 8

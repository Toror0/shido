"""Structure factor table (core.structure_factors): hand formulas, extinctions, XRD consistency."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido.core.structure_factors import (
    SF_COLUMNS,
    atomic_scattering_factor,
    compute_structure_factors,
    write_structure_factor_report,
)
from shido.core.validator import ScientificValidationError
from shido.xrd import compute_xrd


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _si() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])


def _cu() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.615), ["Cu"], [[0, 0, 0]])


def _row(df, hkl: str):
    rows = df[df["hkl"] == hkl]
    assert len(rows) == 1, f"{hkl} not found"
    return rows.iloc[0]


def test_nacl_111_200_hand_formula():
    df, rep = compute_structure_factors(_nacl(), dmin=1.0)
    assert list(df.columns[: len(SF_COLUMNS)]) == SF_COLUMNS
    r111, r200 = _row(df, "(1 1 1)"), _row(df, "(2 0 0)")
    # f from the same coefficients at s = 1/(2d)
    for r, sign in ((r111, -1.0), (r200, 1.0)):
        s = 1.0 / (2.0 * r["d_angstrom"])
        f_na = float(atomic_scattering_factor("Na", s))
        f_cl = float(atomic_scattering_factor("Cl", s))
        assert r["f_Na"] == pytest.approx(f_na, abs=1e-4)
        assert r["F_real"] == pytest.approx(4.0 * (f_na + sign * f_cl), abs=1e-3)
        assert r["F_imag"] == pytest.approx(0.0, abs=1e-6)
    assert r111["phase_deg"] == pytest.approx(180.0)  # f_Na < f_Cl
    assert r200["phase_deg"] == pytest.approx(0.0)
    assert r111["multiplicity"] == 8 and r200["multiplicity"] == 6
    assert r111["F_squared"] == pytest.approx(r111["F_abs"] ** 2, rel=1e-4)
    assert rep["spacegroup_symbol"] == "Fm-3m" and rep["F_unit"] == "electrons"


def test_nacl_primitive_cell_same_abs_f():
    conv, _ = compute_structure_factors(_nacl(), dmin=1.5)
    prim, _ = compute_structure_factors(_nacl().get_primitive_structure(), dmin=1.5)
    allowed = conv[conv["extinct"] == ""]
    # same set of |F| (x 1/4: the primitive cell holds one formula unit instead of four)
    a = sorted(np.round(allowed["F_abs"] / 4.0, 3))
    b = sorted(np.round(prim["F_abs"], 3))
    assert a == pytest.approx(b, abs=2e-3)
    assert (prim["extinct"] == "").all()


def test_si_200_systematic_222_structure_zero():
    df, _ = compute_structure_factors(_si(), dmin=1.0)
    r200, r222, r111 = _row(df, "(2 0 0)"), _row(df, "(2 2 2)"), _row(df, "(1 1 1)")
    assert r200["F_abs"] == 0.0 and r200["extinct"] == "systematic"
    # allowed by Fd-3m but zero with spherical Si atoms on 8a (observed weakly from bonding)
    assert r222["F_abs"] < 1e-6 and r222["extinct"] == "structure"
    f = float(atomic_scattering_factor("Si", 1.0 / (2.0 * r111["d_angstrom"])))
    assert r111["F_abs"] == pytest.approx(4.0 * math.sqrt(2.0) * f, abs=1e-3)  # |4f(1 - i)|
    assert r111["extinct"] == ""


def test_cu_100_extinct_and_drop_option():
    df, rep = compute_structure_factors(_cu(), dmin=1.2)
    r100 = _row(df, "(1 0 0)")
    assert r100["F_abs"] == 0.0 and r100["extinct"] == "systematic"
    assert rep["n_extinct_systematic"] > 0
    only, _ = compute_structure_factors(_cu(), dmin=1.2, include_extinct=False)
    assert (only["extinct"] == "").all()
    assert set(only["hkl"]) == {"(1 1 1)", "(2 0 0)", "(2 2 0)"}


def test_hcp_glide_absence_and_four_index():
    ti = Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(2.95, 4.68), ["Ti"], [[1 / 3, 2 / 3, 0.25]])
    df, _ = compute_structure_factors(ti, dmin=1.3)
    assert _row(df, "(0 0 0 1)")["extinct"] == "systematic"
    assert _row(df, "(1 1 -2 1)")["extinct"] == "systematic"  # hh-2hl: l = 2n
    assert _row(df, "(1 0 -1 1)")["extinct"] == ""


def test_debye_waller_reduces_f():
    s = 1.0 / (2.0 * 2.82)
    df0, _ = compute_structure_factors(_nacl(), dmin=2.0)
    dfb, _ = compute_structure_factors(_nacl(), dmin=2.0, debye_waller_factors={"Na": 1.0, "Cl": 1.0})
    ratio = _row(dfb, "(2 0 0)")["F_abs"] / _row(df0, "(2 0 0)")["F_abs"]
    assert ratio == pytest.approx(math.exp(-1.0 * s * s), rel=1e-4)


def test_partial_occupancy_weighted():
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.615), [{"Cu": 0.5, "Au": 0.5}], [[0, 0, 0]])
    df, _ = compute_structure_factors(s, dmin=1.5)
    r = _row(df, "(1 1 1)")
    assert r["F_real"] == pytest.approx(4.0 * 0.5 * (r["f_Cu"] + r["f_Au"]), abs=1e-3)


def _xrd_ratio_check(structure, radiation, tmp_path):
    xrd = compute_xrd(structure, tmp_path, make_figure=False, two_theta_range=(10, 90), radiation=radiation)
    df, _ = compute_structure_factors(structure, dmin=0.5, radiation=radiation)
    df = df[(df["two_theta_deg"] >= 10) & (df["two_theta_deg"] <= 90) & (df["extinct"] == "")]
    # families at the same 2θ ((3 3 3)+(5 1 1)) make one powder peak
    summed = df.groupby(df["two_theta_deg"].round(2))["intensity"].sum()
    summed = summed / summed.max() * 100.0
    assert len(summed) == xrd["n_peaks"]
    for p in xrd["peaks"]:
        key = min(summed.index, key=lambda t: abs(t - p["two_theta_deg"]))
        assert abs(key - p["two_theta_deg"]) < 0.01
        assert summed[key] == pytest.approx(p["intensity_rel"], abs=0.01)


@pytest.mark.parametrize("make", [_nacl, _si, _cu])
def test_intensity_ratio_matches_xrd(make, tmp_path):
    _xrd_ratio_check(make(), "xray", tmp_path)


def test_neutron_matches_nd_and_units(tmp_path):
    _xrd_ratio_check(_nacl(), "neutron", tmp_path)
    df, rep = compute_structure_factors(_si(), dmin=1.0, radiation="neutron")
    assert rep["F_unit"] == "fm"
    b = rep["scattering"]["Si"]
    assert _row(df, "(2 2 0)")["F_real"] == pytest.approx(8.0 * b, abs=1e-3)
    assert _row(df, "(2 0 0)")["extinct"] == "systematic"


def test_write_report_and_unreachable(tmp_path):
    df, rep = compute_structure_factors(_cu(), dmin=0.7, wavelength="CuKa")
    assert df["two_theta_deg"].isna().any()  # d < λ/2
    assert any("λ/2" in w for w in rep["warnings"])
    files = write_structure_factor_report(df, rep, tmp_path)
    data = json.loads(open(files["json"], encoding="utf-8").read())
    assert data["n_families"] == len(df)
    assert files["csv"].endswith("structure_factors.csv")


def test_bad_inputs():
    with pytest.raises(ScientificValidationError):
        compute_structure_factors(_cu(), dmin=0.0)
    with pytest.raises(ScientificValidationError):
        compute_structure_factors(_cu(), radiation="electron")


def test_server_cli_structure_factors(tmp_path, capsys):
    from shido import server

    cif = tmp_path / "NaCl.cif"
    _nacl().to(filename=str(cif))
    out = tmp_path / "sf"
    rc = server.main(["structure-factors", str(cif), "--dmin", "1.5", "--b", "Na=0.5,Cl=0.5", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "structure_factors.csv").is_file() and (out / "structure_factors.json").is_file()
    assert "(2 0 0)" in capsys.readouterr().out

"""Powder XRD (xrd.compute_xrd): Bragg positions, extinction rules, intensity trends, outputs."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from shido.analyzer import build_ni2mnga_b2_disordered
from shido.core.validator import ScientificValidationError
from shido.xrd import XRD_PEAK_COLUMNS, compute_xrd

CU_A = 3.615
CUKA = 1.54184  # pymatgen 'CuKa' (weighted Kα1/Kα2)


def _cu() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(CU_A), ["Cu"], [[0, 0, 0]])


def _si() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _bragg_2theta(a: float, hkl: tuple[int, int, int], lam: float = CUKA) -> float:
    d = a / math.sqrt(sum(i * i for i in hkl))
    return math.degrees(2.0 * math.asin(lam / (2.0 * d)))


def _peak(result: dict, hkl: str) -> dict | None:
    return next((p for p in result["peaks"] if hkl in p["hkl"].split(", ")), None)


def test_cu_fcc_bragg_positions(tmp_path):
    res = compute_xrd(_cu(), tmp_path, wavelength="CuKa", two_theta_range=(10, 90))
    assert res["wavelength_angstrom"] == pytest.approx(CUKA)
    first = res["peaks"][0]
    assert first["hkl"] == "(1 1 1)"
    for hkl_text, hkl, approx in (("(1 1 1)", (1, 1, 1), 43.3), ("(2 0 0)", (2, 0, 0), 50.4), ("(2 2 0)", (2, 2, 0), 74.1)):
        p = _peak(res, hkl_text)
        assert p is not None, hkl_text
        assert p["two_theta_deg"] == pytest.approx(_bragg_2theta(CU_A, hkl), abs=0.05)
        assert p["two_theta_deg"] == pytest.approx(approx, abs=0.2)
        assert p["d_angstrom"] == pytest.approx(CU_A / math.sqrt(sum(i * i for i in hkl)), abs=1e-3)
    assert _peak(res, "(1 1 1)")["multiplicity"] == 8
    assert _peak(res, "(2 0 0)")["multiplicity"] == 6
    assert _peak(res, "(2 2 0)")["multiplicity"] == 12
    assert max(p["intensity_rel"] for p in res["peaks"]) == pytest.approx(100.0)


def test_fcc_extinction_no_100_110(tmp_path):
    res = compute_xrd(_cu(), tmp_path, two_theta_range=(5, 150))
    for forbidden in ("(1 0 0)", "(1 1 0)", "(2 1 0)", "(2 1 1)"):
        assert _peak(res, forbidden) is None, forbidden
    # nothing between the (100) and (110) positions and the first allowed (111)
    assert res["peaks"][0]["two_theta_deg"] > _bragg_2theta(CU_A, (1, 1, 0))


def test_si_diamond_no_200(tmp_path):
    res = compute_xrd(_si(), tmp_path)
    assert _peak(res, "(1 1 1)") is not None
    assert _peak(res, "(2 0 0)") is None
    assert _peak(res, "(2 2 2)") is None  # also forbidden for spherical atoms
    assert res["peaks"][0]["hkl"] == "(1 1 1)"


def test_nacl_200_strongest(tmp_path):
    res = compute_xrd(_nacl(), tmp_path)
    p111, p200 = _peak(res, "(1 1 1)"), _peak(res, "(2 0 0)")
    assert p111 is not None and p200 is not None
    assert p200["intensity_rel"] == pytest.approx(100.0)
    # literature (PDF 05-0628): I(111) ~ 13, I(220) ~ 55 relative to (200)
    assert p111["intensity_rel"] < 20.0
    assert p111["intensity_rel"] / p200["intensity_rel"] < 0.2
    assert 40.0 < _peak(res, "(2 2 0)")["intensity_rel"] < 80.0


def test_ni2mnga_b2_partial_occupancy(tmp_path):
    b2 = build_ni2mnga_b2_disordered()
    assert any(not s.is_ordered for s in b2)
    res = compute_xrd(b2, tmp_path)
    assert res["partial_occupancy"] is True
    assert res["n_peaks"] > 0
    assert "occupancy" in res["occupancy_treatment"].lower()
    # B2 disorder (Mn/Ga mixed) removes the L2_1 (111) superlattice peak; (220) is the main line
    assert _peak(res, "(1 1 1)") is None
    assert _peak(res, "(2 2 0)")["intensity_rel"] == pytest.approx(100.0)


def test_outputs_written(tmp_path):
    res = compute_xrd(_nacl(), tmp_path)
    csv = tmp_path / "XRD" / "xrd_peaks.csv"
    png = tmp_path / "XRD" / "figure_xrd.png"
    svg = tmp_path / "XRD" / "figure_xrd.svg"
    for f in (csv, png, svg):
        assert f.exists() and f.stat().st_size > 0, f
    df = pd.read_csv(csv)
    assert list(df.columns) == XRD_PEAK_COLUMNS
    assert len(df) == res["n_peaks"]
    assert df["intensity_rel"].max() == pytest.approx(100.0)
    assert res["figure_png"].endswith("figure_xrd.png")


def test_bad_inputs_rejected(tmp_path):
    with pytest.raises(ScientificValidationError):
        compute_xrd(_cu(), tmp_path, wavelength="XxKa")
    with pytest.raises(ScientificValidationError):
        compute_xrd(_cu(), tmp_path, wavelength=-1.0)
    with pytest.raises(ScientificValidationError):
        compute_xrd(_cu(), tmp_path, two_theta_range=(90, 10))

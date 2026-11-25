"""Powder diffraction extensions (batch 12): peak profiles, neutron, overlay with measured data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from shido.core.validator import ScientificValidationError
from shido.xrd import caglioti_fwhm, compute_xrd, load_observed_pattern, profile_curve


def _si() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _peak(result: dict, hkl: str) -> dict | None:
    return next((p for p in result["peaks"] if hkl in p["hkl"].split(", ")), None)


def _fwhm_of(x: np.ndarray, y: np.ndarray) -> float:
    """Full width at half maximum of a single peak sampled on x (linear interpolation)."""
    i = int(np.argmax(y))
    half = y[i] / 2.0
    j = i
    while y[j] > half:
        j -= 1
    left = x[j] + (half - y[j]) * (x[j + 1] - x[j]) / (y[j + 1] - y[j])
    k = i
    while y[k] > half:
        k += 1
    right = x[k - 1] + (half - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1])
    return right - left


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian", "pseudo_voigt"])
@pytest.mark.parametrize("fwhm", [0.05, 0.3])
def test_profile_area_and_fwhm(shape, fwhm):
    x = np.arange(0.0, 400.0, 0.001)
    y = profile_curve(x, [200.0], [37.0], fwhm, shape)
    assert np.trapezoid(y, x) == pytest.approx(37.0, rel=2e-3)  # area = stick intensity
    assert x[np.argmax(y)] == pytest.approx(200.0, abs=0.001)
    assert _fwhm_of(x, y) == pytest.approx(fwhm, rel=1e-2)


def test_caglioti_formula():
    u, v, w = 0.02, -0.01, 0.01
    t = math.tan(math.radians(30.0))
    assert caglioti_fwhm(60.0, u, v, w) == pytest.approx(math.sqrt(u * t * t + v * t + w))
    with pytest.raises(ScientificValidationError):
        caglioti_fwhm([20.0, 60.0], 0.0, 0.0, -0.01)


def test_default_output_unchanged(tmp_path):
    """No new options -> same keys, same CSV, no profile files."""
    a = compute_xrd(_nacl(), tmp_path / "a", make_figure=False)
    b = compute_xrd(_nacl(), tmp_path / "b", make_figure=False, radiation="xray", profile=None)
    assert a["peaks"] == b["peaks"]
    assert set(a) == set(b)
    assert not {"profile", "observed", "radiation", "profile_csv"} & set(a)
    assert not (tmp_path / "a" / "XRD" / "xrd_profile.csv").exists()


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian", "pseudo_voigt"])
def test_curve_maxima_match_sticks(tmp_path, shape):
    res = compute_xrd(_si(), tmp_path, profile=shape, fwhm=0.1, step=0.005, make_figure=False)
    prof = pd.read_csv(res["profile_csv"])
    x, y = prof["two_theta_deg"].to_numpy(), prof["intensity"].to_numpy()
    for p in res["peaks"]:
        m = np.abs(x - p["two_theta_deg"]) < 0.3
        assert x[m][np.argmax(y[m])] == pytest.approx(p["two_theta_deg"], abs=0.02)
    assert prof["intensity_rel"].max() == pytest.approx(100.0, abs=1e-3)
    assert res["profile"]["shape"] == shape


def test_profile_area_preserved_between_shapes(tmp_path):
    """Integrated intensity of the whole curve does not depend on the peak shape (Gaussian, narrow)."""
    g = compute_xrd(_si(), tmp_path / "g", two_theta_range=(20, 100), profile="gaussian", fwhm=0.1,
                    make_figure=False)
    g2 = compute_xrd(_si(), tmp_path / "g2", two_theta_range=(20, 100), profile="gaussian", fwhm=0.4,
                     make_figure=False)
    ag = np.trapezoid(*pd.read_csv(g["profile_csv"])[["intensity", "two_theta_deg"]].to_numpy().T)
    ag2 = np.trapezoid(*pd.read_csv(g2["profile_csv"])[["intensity", "two_theta_deg"]].to_numpy().T)
    total = sum(p["intensity_rel"] for p in g["peaks"])
    assert ag == pytest.approx(total, rel=5e-3)
    assert ag2 == pytest.approx(total, rel=5e-3)


def test_caglioti_widths_used(tmp_path):
    uvw = (0.02, -0.01, 0.01)
    res = compute_xrd(_si(), tmp_path, profile="gaussian", caglioti=uvw, make_figure=False)
    expected = caglioti_fwhm([p["two_theta_deg"] for p in res["peaks"]], *uvw)
    assert np.allclose(res["profile"]["peak_fwhm_deg"], expected, atol=1e-4)
    with pytest.raises(ScientificValidationError):
        compute_xrd(_si(), tmp_path, caglioti=(0, 0, -1), make_figure=False)
    with pytest.raises(ScientificValidationError):
        compute_xrd(_si(), tmp_path, profile="triangle", make_figure=False)


def test_neutron_si_forbidden_and_files(tmp_path):
    res = compute_xrd(_si(), tmp_path, radiation="neutron", wavelength=1.54)
    assert res["radiation"] == "neutron"
    assert _peak(res, "(2 0 0)") is None and _peak(res, "(2 2 2)") is None
    assert _peak(res, "(1 1 1)") is not None
    assert res["xrd_peaks_csv"].endswith("neutron_peaks.csv")
    assert (tmp_path / "XRD" / "figure_neutron.png").is_file()


def test_neutron_nacl_ratio_differs_from_xray(tmp_path):
    """b(Na)=3.63 fm, b(Cl)=9.577 fm (both positive): (111) ∝ (b_Cl-b_Na)^2, (200) ∝ (b_Na+b_Cl)^2."""
    xr = compute_xrd(_nacl(), tmp_path / "x", wavelength=1.54, make_figure=False)
    nd = compute_xrd(_nacl(), tmp_path / "n", radiation="neutron", wavelength=1.54, make_figure=False)
    r_x = _peak(xr, "(1 1 1)")["intensity_rel"] / _peak(xr, "(2 0 0)")["intensity_rel"]
    r_n = _peak(nd, "(1 1 1)")["intensity_rel"] / _peak(nd, "(2 0 0)")["intensity_rel"]
    assert r_n > 3.0 * r_x
    # NDCalculator: I = M |F|^2 L with L = 1/(sin^2θ cosθ) (no polarization factor), so
    # I111/I200 / (M111 L111 / M200 L200) = ((b_Cl - b_Na)/(b_Cl + b_Na))^2 (no form-factor fall-off).
    b_na, b_cl = 3.63, 9.577
    th1 = math.radians(_peak(nd, "(1 1 1)")["two_theta_deg"] / 2)
    th2 = math.radians(_peak(nd, "(2 0 0)")["two_theta_deg"] / 2)

    def lp(th):
        return 1.0 / (math.sin(th) ** 2 * math.cos(th))

    geom = (8 * lp(th1)) / (6 * lp(th2))
    assert r_n / geom == pytest.approx(((b_cl - b_na) / (b_cl + b_na)) ** 2, rel=0.03)


def test_observed_self_comparison_zero_difference(tmp_path):
    ref = compute_xrd(_nacl(), tmp_path / "ref", profile="pseudo_voigt", fwhm=0.2, make_figure=False)
    prof = pd.read_csv(ref["profile_csv"])
    obs = tmp_path / "self.csv"
    pd.DataFrame({"2theta": prof["two_theta_deg"], "I": prof["intensity"]}).to_csv(obs, index=False)
    res = compute_xrd(_nacl(), tmp_path / "cmp", observed=obs, fwhm=0.2)
    o = res["observed"]
    assert o["scale"] == pytest.approx(1.0, rel=1e-6)
    assert o["background"] == pytest.approx(0.0, abs=1e-6)
    assert o["r_factor_percent"] < 1e-4
    cmp_df = pd.read_csv(o["compare_csv"])
    assert np.abs(cmp_df["difference"]).max() < 1e-6 * cmp_df["observed"].max()
    assert res["two_theta_range"] == [pytest.approx(10.0), pytest.approx(90.0)]
    assert (tmp_path / "cmp" / "XRD" / "figure_xrd_observed.png").is_file()


def test_observed_scale_and_background_recovered(tmp_path):
    ref = compute_xrd(_si(), tmp_path / "ref", profile="gaussian", fwhm=0.15, make_figure=False)
    prof = pd.read_csv(ref["profile_csv"])
    obs = tmp_path / "si.xy"
    lines = ["# measured", "2theta intensity"]
    lines += [f"{a:.5f}\t{250 * b + 40:.6f}" for a, b in zip(prof["two_theta_deg"], prof["intensity"])]
    obs.write_text("\n".join(lines), encoding="utf-8")
    x, y = load_observed_pattern(obs)
    assert x.size == len(prof)
    res = compute_xrd(_si(), tmp_path / "cmp", observed=obs, profile="gaussian", fwhm=0.15, make_figure=False)
    assert res["observed"]["scale"] == pytest.approx(250.0, rel=1e-4)
    assert res["observed"]["background"] == pytest.approx(40.0, abs=0.01)


def test_cli_options(tmp_path):
    from shido import server

    parser = server._build_arg_parser()
    opts = server._bulk_options_from_args(parser.parse_args(["demo"]))
    assert opts["xrd"] is False and opts["xrd_options"] == {}
    args = parser.parse_args(["demo", "--xrd-profile", "--caglioti", "0.02,-0.01,0.01", "--radiation", "neutron"])
    opts = server._bulk_options_from_args(args)
    assert opts["xrd"] is True
    assert opts["xrd_options"] == {
        "radiation": "neutron", "profile": "pseudo_voigt", "caglioti": (0.02, -0.01, 0.01)
    }
    assert server._bulk_options_from_args(parser.parse_args(["demo", "--fwhm", "0.2"]))["xrd_options"] == {"fwhm": 0.2}


def test_cli_single_cif_profile_and_observed(tmp_path, capsys):
    from pymatgen.io.cif import CifWriter

    from shido import server

    cif = tmp_path / "NaCl.cif"
    CifWriter(_nacl()).write_file(str(cif))
    ref = compute_xrd(_nacl(), tmp_path / "ref", profile="gaussian", fwhm=0.2, make_figure=False)
    prof = pd.read_csv(ref["profile_csv"])
    obs = tmp_path / "obs.xy"
    obs.write_text("\n".join(f"{a} {b}" for a, b in zip(prof["two_theta_deg"], prof["intensity"])), encoding="utf-8")
    base = tmp_path / "cli"
    rc = server.main([str(cif), "--out-dir", str(base), "--no-polyhedra",
                      "--xrd-profile", "gaussian", "--fwhm", "0.2", "--xrd-observed", str(obs)])
    assert rc == 0
    out = base / "output_NaCl"
    assert (out / "XRD" / "xrd_profile.csv").is_file()
    assert (out / "XRD" / "figure_xrd_observed.png").is_file()
    assert "XRD vs observed PNG" in capsys.readouterr().out


def test_mcp_tool_has_new_arguments():
    import asyncio

    from shido import server

    tools = asyncio.run(server.create_mcp_server().list_tools())
    tool = next(t for t in tools if t.name == "compute_xrd_pattern")
    props = (getattr(tool, "input_schema", None) or getattr(tool, "inputSchema"))["properties"]
    for name in ("radiation", "profile", "fwhm", "caglioti_u", "caglioti_v", "caglioti_w", "eta", "observed_path"):
        assert name in props, name


def test_observed_bad_file(tmp_path):
    bad = tmp_path / "bad.xy"
    bad.write_text("no numbers here\n", encoding="utf-8")
    with pytest.raises(ScientificValidationError):
        compute_xrd(_si(), tmp_path, observed=bad)
    with pytest.raises(ScientificValidationError):
        compute_xrd(_si(), tmp_path, observed=tmp_path / "missing.xy")

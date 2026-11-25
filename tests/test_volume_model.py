"""Batch 19: model electron / nuclear density, Patterson, difference, Cube / grd export."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido import server
from shido.core import volumetric as V
from shido.core.output_paths import describe_file


def _nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _si() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])


def test_electron_count_nacl_and_si():
    g = V.model_density(_nacl(), "xray", grid=48)
    assert g.shape == (48, 48, 48)
    assert g.n_electrons() == pytest.approx(4 * (11 + 17), rel=0.01)
    # highest peak on a Cl site (0, 0, 1/2) -> index (0, 0, 24)
    assert np.unravel_index(int(np.argmax(g.data)), g.shape) in {(0, 0, 24), (0, 24, 0), (24, 0, 0), (24, 24, 24)}
    si = V.model_density(_si(), "xray", grid="40,40,40")
    assert si.n_electrons() == pytest.approx(8 * 14, rel=0.01)


def test_neutron_density_integral_is_sum_of_b():
    from pymatgen.analysis.diffraction.neutron import ATOMIC_SCATTERING_LEN

    g = V.model_density(_nacl(), "neutron", grid=32)
    expect = 4 * (ATOMIC_SCATTERING_LEN["Na"] + ATOMIC_SCATTERING_LEN["Cl"])
    assert g.n_electrons() == pytest.approx(expect, rel=0.01)
    assert "fm" in g.unit


def test_patterson_origin_is_max_and_integral():
    g = V.model_density(_nacl(), "xray", grid=32)
    p = V.patterson(g)
    assert np.unravel_index(int(np.argmax(p.data)), p.shape) == (0, 0, 0)
    assert p.n_electrons() == pytest.approx(112.0**2, rel=0.01)
    # centrosymmetric: P(u) = P(-u)
    assert np.allclose(p.data, np.roll(p.data[::-1, ::-1, ::-1], 1, axis=(0, 1, 2)), atol=1e-6 * p.data.max())


def test_difference_self_is_zero_and_checks():
    g = V.model_density(_nacl(), "xray", grid=24)
    d = V.volume_difference(g, g)
    assert np.max(np.abs(d.data)) == 0.0
    with pytest.raises(ValueError, match="grid sizes"):
        V.volume_difference(g, V.model_density(_nacl(), "xray", grid=20))
    other = Structure.from_spacegroup("Fm-3m", Lattice.cubic(6.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="lattices"):
        V.volume_difference(g, V.model_density(other, "xray", grid=24))


def test_write_and_read_back(tmp_path):
    g = V.model_density(_nacl(), "xray", grid=(20, 20, 24))
    cube = V.write_volume(g, tmp_path / "m.cube")
    grd = V.write_volume(g, tmp_path / "m.grd")
    c = V.read_volume(cube)
    assert c.shape == g.shape and c.structure is not None and len(c.structure) == 8
    assert np.allclose(c.data / V.BOHR_TO_ANGSTROM**3, g.data, rtol=1e-4, atol=1e-4)
    assert c.n_electrons() == pytest.approx(112.0, rel=1e-4)
    r = V.read_volume(grd)
    assert r.shape == g.shape
    assert np.allclose(r.data, g.data, rtol=1e-5, atol=1e-6)
    assert np.allclose(r.lattice.abc, g.lattice.abc)
    # a Cube read back minus the model on the same grid: zero (units put on one length unit)
    assert np.max(np.abs(V.volume_difference(g, c).data)) < 1e-3


def test_cli_outputs_usable_by_slice(tmp_path):
    cif = tmp_path / "NaCl.cif"
    _nacl().to(filename=str(cif))
    out = tmp_path / "out"
    rc = server.main(["volume-model", str(cif), "--grid", "24", "--patterson",
                      "--volume-diff", str(cif), "--export-volume", "cube,grd", "--out-dir", str(out)])
    assert rc == 0
    for name in ("model_density_xray", "patterson", "diff"):
        assert (out / f"{name}.cube").is_file() and (out / f"{name}.grd").is_file()
    assert np.max(np.abs(V.read_volume(out / "diff.grd").data)) < 1e-9
    # the written Cube feeds batch 18 directly
    rc = server.main(["slice", str(out / "model_density_xray.cube"), "--slice", "0 0 1@0",
                      "--out-dir", str(tmp_path / "sl")])
    assert rc == 0 and (tmp_path / "sl" / "slice_001_0.png").is_file()


def test_default_run_folder_mcp_and_notes(tmp_path):
    cif = tmp_path / "NaCl.cif"
    _nacl().to(filename=str(cif))
    rep = server._volume_model_impl(cif, "neutron", 16, None, False, None, "cube", None)
    run = Path(rep["output_dir"])
    assert (run / "数値" / "model_density_neutron.cube").is_file()
    assert rep["index_html"] and Path(rep["index_html"]).is_file()
    html = Path(rep["index_html"]).read_text(encoding="utf-8")
    assert "核密度" in html
    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    props = (getattr(tools["volume_model"], "input_schema", None) or getattr(tools["volume_model"], "inputSchema"))["properties"]
    assert {"input_path", "model", "grid", "b_iso", "patterson", "diff_path", "export_formats", "output_dir"} <= set(props)
    for name in ("slice_001_0.5.png", "line_profile_1.csv", "model_density_xray.cube", "patterson.grd", "diff.cube"):
        assert describe_file(name), name

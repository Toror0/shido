"""Batch 18: 2D sections, contour lines and line profiles of volumetric data."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pymatgen.core import Lattice, Structure

from shido import server
from shido.core import volumetric as V

SIGMA = 0.5  # Å


def _gauss_cube(tmp_path, lattice: Lattice, center, n=(24, 24, 24), name="g.cube"):
    """One periodic Gaussian (peak 1) at fractional ``center``, written as a Cube file."""
    s = Structure(lattice, ["H"], [center])
    f = np.stack(np.meshgrid(*(np.arange(k) / k for k in n), indexing="ij"), -1)
    rho = np.zeros(n)
    for shift in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3):
        d = (f - np.asarray(center) - shift) @ lattice.matrix
        rho += np.exp(-np.sum(d**2, -1) / (2 * SIGMA**2))
    return V.write_cube(tmp_path / name, s, rho), s


def test_parse_specs():
    assert V.parse_slice_spec("0 0 1@0.5") == ((0, 0, 1), 0.5)
    assert V.parse_slice_spec("1,1,1@-0.25") == ((1, 1, 1), -0.25)
    assert V.parse_slice_spec("1 1 0") == ((1, 1, 0), 0.0)
    with pytest.raises(ValueError):
        V.parse_slice_spec("0 0 0@0.5")
    with pytest.raises(ValueError):
        V.parse_slice_spec("0 1@0.5")
    a, b = V.parse_line_spec("0 0.5 0.5 1 0.5 0.5")
    assert a.tolist() == [0, 0.5, 0.5] and b.tolist() == [1, 0.5, 0.5]
    with pytest.raises(ValueError):
        V.parse_line_spec("0 0 0 1 1")
    assert V.normalize_contours(None) is None
    assert V.normalize_contours("none") == []
    assert V.normalize_contours(["0.1,0.05", "0.2"]) == [0.05, 0.1, 0.2]


def test_interpolation_is_exact_on_grid_and_periodic(tmp_path):
    cube, _ = _gauss_cube(tmp_path, Lattice.cubic(4.0), [0.5, 0.5, 0.5], n=(8, 10, 12))
    g = V.read_volume(cube)
    i, j, k = 3, 7, 5
    f = np.array([i / 8, j / 10, k / 12])
    assert V.interpolate_frac(g, f) == pytest.approx(g.data[i, j, k], rel=1e-9)
    assert V.interpolate_frac(g, f + [1, -2, 3]) == pytest.approx(g.data[i, j, k], rel=1e-9)


def test_cubic_slice_max_at_center_and_symmetric_profile(tmp_path):
    cube, _ = _gauss_cube(tmp_path, Lattice.cubic(5.0), [0.5, 0.5, 0.5])
    g = V.read_volume(cube)
    sl = V.slice_plane(g, (0, 0, 1), 0.5)
    vals = sl["values"]
    iv, iu = np.unravel_index(np.argmax(vals), vals.shape)
    p = sl["origin_cart"] + sl["u"][iu] * sl["u_hat"] + sl["v"][iv] * sl["v_hat"]
    assert np.allclose(p, [2.5, 2.5, 2.5], atol=sl["resolution"])
    assert vals.max() == pytest.approx(1.0, abs=0.02)
    assert len(sl["cell_polygon_uv"]) == 4  # (001) cuts the cell in a square
    # a plane far from the centre is weaker
    assert V.slice_plane(g, (0, 0, 1), 0.0)["values"].max() < 0.1

    prof = V.line_profile(g, [0, 0.5, 0.5], [1, 0.5, 0.5], n_points=201)
    y = prof["values"]
    assert np.allclose(y, y[::-1], atol=1e-9)
    assert int(np.argmax(y)) == 100
    assert prof["length"] == pytest.approx(5.0, abs=1e-4)


def test_oblique_cell_positions(tmp_path):
    lat = Lattice.from_parameters(5.0, 6.0, 5.5, 80, 105, 115)
    center = np.array([0.3, 0.6, 0.4])
    cube, _ = _gauss_cube(tmp_path, lat, center, n=(30, 36, 33), name="tri.cube")
    g = V.read_volume(cube)
    lat = g.lattice  # as read back (the Cube stores 6 decimals in Bohr)
    for hkl, pos in [((0, 0, 1), 0.4), ((1, 1, 1), 1.3), ((1, 0, 0), 0.3), ((2, 1, 0), 1.2)]:
        sl = V.slice_plane(g, hkl, pos, resolution=0.05)
        vals = sl["values"]
        iv, iu = np.unravel_index(np.argmax(vals), vals.shape)
        p = sl["origin_cart"] + sl["u"][iu] * sl["u_hat"] + sl["v"][iv] * sl["v_hat"]
        # the point lies on the plane and is the Gaussian's centre (modulo the lattice)
        frac = p @ np.linalg.inv(lat.matrix)
        assert frac @ np.array(hkl) == pytest.approx(pos, abs=1e-9)
        diff = frac - center
        diff -= np.round(diff)
        assert np.linalg.norm(diff @ lat.matrix) < 0.12, (hkl, pos)
        assert vals.max() == pytest.approx(1.0, abs=0.03)
        assert abs(np.dot(sl["u_hat"], sl["normal"])) < 1e-12
        assert abs(np.dot(sl["v_hat"], sl["normal"])) < 1e-12
    # line through the centre along a: symmetric about it and peaked there
    prof = V.line_profile(g, center - [0.5, 0, 0], center + [0.5, 0, 0], n_points=301)
    assert np.allclose(prof["values"], prof["values"][::-1], atol=0.01)
    assert int(np.argmax(prof["values"])) == 150
    assert prof["length"] == pytest.approx(5.0, abs=1e-4)


def test_cli_writes_png_and_csv(tmp_path):
    cube, _ = _gauss_cube(tmp_path, Lattice.hexagonal(4.0, 6.0), [1 / 3, 2 / 3, 0.5], name="hex.cube")
    out = tmp_path / "out"
    rc = server.main([
        "slice", str(cube), "--slice", "0 0 1@0.5", "--slice", "1 1 0@0.5",
        "--contours", "0.2", "--contours", "0.5,0.8",
        "--line-profile", "0 0 0.5 0.6667 1.3333 0.5", "--out-dir", str(out),
    ])
    assert rc == 0
    for name in ("slice_001_0.5", "slice_110_0.5", "line_profile_1"):
        png = out / f"{name}.png"
        assert png.is_file()
        im = np.asarray(Image.open(png).convert("L"), dtype=float)
        assert im.std() > 5  # not blank
        df = pd.read_csv(out / f"{name}.csv")
        assert "value" in df.columns and len(df) > 10
    lp = pd.read_csv(out / "line_profile_1.csv")
    y = lp["value"].to_numpy()
    assert np.allclose(y, y[::-1], atol=0.01)  # the line passes through the peak at its middle


def test_default_run_folder_and_mcp(tmp_path):
    cube, _ = _gauss_cube(tmp_path, Lattice.cubic(4.0), [0.5, 0.5, 0.5], n=(16, 16, 16))
    rep = server._slice_volume_impl(cube, ["0 0 1@0.5"], None, ["0 0 0 1 1 1"], None, None)
    run = rep["output_dir"]
    assert (tmp_path / "出力").exists() or run  # conftest moves 出力 into tmp
    from pathlib import Path

    run = Path(run)
    assert (run / "図" / "slice_001_0.5.png").is_file()
    assert (run / "数値" / "slice_001_0.5.csv").is_file()
    assert (run / "図" / "line_profile_1.png").is_file()
    assert rep["index_html"] and Path(rep["index_html"]).is_file()

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    props = (getattr(tools["slice_volume"], "input_schema", None) or getattr(tools["slice_volume"], "inputSchema"))["properties"]
    assert {"volume_path", "slice", "contours", "line_profile", "output_dir"} <= set(props)
    with pytest.raises(ValueError):
        server._slice_volume_impl(cube, None, None, None, None, tmp_path / "x")

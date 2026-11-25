"""Batch 10: user (hkl) lattice planes on the bulk views (--plane-show / show_planes)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from PIL import Image
from pymatgen.core import Lattice, Structure

from shido import renderer
from shido import server
from shido.analyzer import compute_bonds
from conftest import SMALL_WINDOW, png_background_ratio


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _hex() -> Structure:
    return Structure(Lattice.hexagonal(3.2, 5.2), ["Mg", "Mg"], [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]])


def _triclinic() -> Structure:
    lat = Lattice.from_parameters(4.1, 5.3, 6.2, 78.0, 96.0, 108.0)
    return Structure(lat, ["Si", "O"], [[0.1, 0.2, 0.3], [0.6, 0.5, 0.8]])


def _render(structure, path, **kw):
    _, bonds, envs = compute_bonds(structure, ["Na-Cl:0:3.0"])
    return renderer.render_single_view(
        structure=structure,
        output_path=path,
        camera_dir=[1.0, 0.36, 0.24],
        view_up=[0.0, 0.0, 1.0],
        render_bonds=bonds,
        polyhedra_envs=envs,
        window_size=SMALL_WINDOW,
        **kw,
    )


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _polygon(structure, hkl, d):
    built = renderer._compute_hkl_plane_mesh(structure.lattice, hkl, d_offset=d)
    assert built is not None
    return np.asarray(built[0].points, dtype=float)


def _area(pts):
    c = pts.mean(axis=0)
    return 0.5 * float(
        np.linalg.norm(sum(np.cross(pts[i] - c, pts[(i + 1) % len(pts)] - c) for i in range(len(pts))))
    )


# ---- parsing -----------------------------------------------------------------------

def test_parse_plane_specs():
    sp = renderer.normalize_planes(["1 1 1@0.5:#ff0000:0.4"])[0]
    assert sp == {"hkl": (1, 1, 1), "d": 0.5, "color": "#ff0000", "opacity": 0.4}
    assert renderer.normalize_planes("111")[0]["hkl"] == (1, 1, 1)
    assert renderer.normalize_planes("(1-10)")[0]["hkl"] == (1, -1, 0)
    assert renderer.normalize_planes("-1,1,0::0.3")[0] == {
        "hkl": (-1, 1, 0), "d": None, "color": None, "opacity": 0.3
    }
    assert renderer.normalize_planes([{"hkl": [0, 0, 1], "d": 0.5, "color": "red"}])[0]["d"] == 0.5
    assert renderer.normalize_planes([1, 1, 0])[0]["hkl"] == (1, 1, 0)
    assert renderer.normalize_planes(None) is None and renderer.normalize_planes([]) is None
    for bad in ("0 0 0", "1 1", "1 1 1:notacolour", "1 1 1::1.5", "1 1 1:red:0.5:x", "1.5 1 0"):
        with pytest.raises(ValueError):
            renderer.normalize_planes([bad])


# ---- geometry (independent of the renderer's own check) ----------------------------

@pytest.mark.parametrize(
    "structure, hkl, d, n_vertices",
    [
        (_nacl(), (1, 1, 1), 1.0, 3),
        (_nacl(), (1, 1, 1), 1.5, 6),
        (_nacl(), (1, 1, 0), 1.0, 4),
        (_nacl(), (1, 1, 0), 0.5, 4),
        (_nacl(), (-1, 1, 0), 0.0, 4),
        (_hex(), (0, 0, 1), 0.5, 4),
        (_triclinic(), (2, -1, 3), 1.3, None),
    ],
)
def test_plane_polygon_on_plane_and_in_cell(structure, hkl, d, n_vertices):
    pts = _polygon(structure, hkl, d)
    if n_vertices is not None:
        assert len(pts) == n_vertices
    frac = structure.lattice.get_fractional_coords(pts)
    # h*x + k*y + l*z = d for every vertex, and every vertex inside [0, 1]^3
    assert np.allclose(frac @ np.asarray(hkl, dtype=float), d, atol=1e-9)
    assert np.all(frac > -1e-9) and np.all(frac < 1.0 + 1e-9)
    # also the Cartesian form G.r = d with G from the reciprocal lattice
    recip = structure.lattice.reciprocal_lattice_crystallographic.matrix
    assert np.allclose(pts @ (np.asarray(hkl, dtype=float) @ recip), d, atol=1e-9)


def test_plane_polygon_areas():
    a = 5.64
    assert _area(_polygon(_nacl(), (1, 1, 0), 1.0)) == pytest.approx(a * a * np.sqrt(2))
    assert _area(_polygon(_nacl(), (1, 1, 1), 1.0)) == pytest.approx(np.sqrt(3) / 2 * a * a)
    h = _hex()
    ab = float(np.linalg.norm(np.cross(h.lattice.matrix[0], h.lattice.matrix[1])))
    assert _area(_polygon(h, (0, 0, 1), 0.5)) == pytest.approx(ab)


def test_plane_outside_cell_raises(out_dir):
    with pytest.raises(ValueError, match="does not cut"):
        _render(_nacl(), out_dir / "bad.png", planes=["1 1 1@3.5"])
    with pytest.raises(ValueError, match="does not cut"):
        _render(_nacl(), out_dir / "bad2.png", planes=["1 1 1@0"])  # only touches the origin


# ---- rendering ---------------------------------------------------------------------

def test_default_pixels_unchanged_and_planes_drawn(out_dir):
    s = _nacl()
    a = _pixels(_render(s, out_dir / "a.png"))
    b = _pixels(_render(s, out_dir / "b.png", planes=None))
    c = _pixels(_render(s, out_dir / "c.png", planes=[]))
    assert np.array_equal(a, b) and np.array_equal(a, c)
    p = _render(s, out_dir / "p.png", planes=["1 1 1@1.5:#ff0000:0.6", "1 1 0"])
    assert png_background_ratio(p) < 0.995
    px = _pixels(p).astype(int)
    assert not np.array_equal(a, px)
    reddish = (px[..., 0] > 150) & (px[..., 1] < 120) & (px[..., 2] < 120)
    reddish0 = (a[..., 0] > 150) & (a[..., 1] < 120) & (a[..., 2] < 120)
    assert reddish.sum() > reddish0.sum() + 200


def test_bulk_views_pass_planes(tmp_path, monkeypatch):
    seen = []

    def fake_single(**kw):
        seen.append(kw)
        p = tmp_path / kw["output_path"].name
        Image.new("RGB", (20, 20), (255, 255, 255)).save(p)
        return p

    monkeypatch.setattr(renderer, "render_single_view", fake_single)
    monkeypatch.setattr(renderer, "_compose_overview_figure", lambda **kw: tmp_path / "o.png")
    monkeypatch.setattr(renderer, "_compose_overview_svg", lambda **kw: tmp_path / "o.svg")
    renderer.render_bulk_views(_nacl(), tmp_path)
    assert all(kw["planes"] is None for kw in seen)
    seen.clear()
    renderer.render_bulk_views(_nacl(), tmp_path, planes=["1 1 1"], view_dir=[1, 1, 0])
    assert len(seen) == 4 and all(kw["planes"][0]["hkl"] == (1, 1, 1) for kw in seen)


def test_cli_and_mcp_expose_planes(monkeypatch, tmp_path):
    parser = server._build_arg_parser()
    assert server._bulk_options_from_args(parser.parse_args(["demo"]))["show_planes"] is None
    args = parser.parse_args(["demo", "--plane-show", "1 1 1", "--plane-show", "-1 1 0@0:red:0.5"])
    assert server._bulk_options_from_args(args)["show_planes"] == ["1 1 1", "-1 1 0@0:red:0.5"]

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("analyze_structure", "batch_analyze_cifs", "render_view"):
        t = tools[name]
        props = (getattr(t, "input_schema", None) or getattr(t, "inputSchema"))["properties"]
        assert "show_planes" in props

    seen = {}

    def fake_render(structure, output_dir, **kw):
        seen.update(kw)
        return {"figure_overview": None}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    server.run_bulk_analysis(_nacl(), tmp_path, show_planes=["0 0 1"])
    assert seen["planes"] == ["0 0 1"]

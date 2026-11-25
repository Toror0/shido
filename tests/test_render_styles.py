"""Display styles (--style): ball-stick (default) / space-filling / stick / wireframe."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from PIL import Image

from shido import renderer
from shido import server
from shido.analyzer import build_ni2mnga_structure, compute_bonds
from conftest import SMALL_WINDOW, png_background_ratio


def _render(structure, path, **kw):
    _, bonds, _ = compute_bonds(structure)
    a_vec, _, c_vec = structure.lattice.matrix
    return renderer.render_single_view(
        structure=structure,
        output_path=path,
        camera_dir=a_vec + 0.36 * structure.lattice.matrix[1] + 0.24 * c_vec,
        view_up=c_vec,
        render_bonds=bonds,
        window_size=SMALL_WINDOW,
        **kw,
    )


@pytest.fixture()
def counters(monkeypatch):
    """Count bond-cylinder / sphere mesh builds and wireframe line calls."""
    calls = {"cyl": 0, "sph": 0, "lines": 0}
    orig_cyl, orig_sph, orig_bi = (
        renderer._merge_cylinders,
        renderer._merge_spheres,
        renderer._add_bicolor_bonds,
    )

    def cyl(*a, **k):
        calls["cyl"] += 1
        return orig_cyl(*a, **k)

    def sph(*a, **k):
        calls["sph"] += 1
        return orig_sph(*a, **k)

    def bi(plotter, bonds, as_lines=False, **k):
        if as_lines:
            calls["lines"] += 1
        return orig_bi(plotter, bonds, as_lines=as_lines, **k)

    monkeypatch.setattr(renderer, "_merge_cylinders", cyl)
    monkeypatch.setattr(renderer, "_merge_spheres", sph)
    monkeypatch.setattr(renderer, "_add_bicolor_bonds", bi)
    return calls


@pytest.mark.parametrize("style", renderer.RENDER_STYLES)
def test_each_style_renders_non_blank_png(style, out_dir):
    path = _render(build_ni2mnga_structure(), out_dir / f"{style}.png", style=style)
    assert path.is_file()
    assert png_background_ratio(path) < 0.995


def test_space_filling_has_no_bond_mesh(out_dir, counters):
    _render(build_ni2mnga_structure(), out_dir / "sf.png", style="space-filling")
    assert counters["cyl"] == 0
    assert counters["lines"] == 0
    assert counters["sph"] > 0


def test_stick_and_wireframe_meshes(out_dir, counters):
    s = build_ni2mnga_structure()
    _render(s, out_dir / "stick.png", style="stick")
    assert counters["cyl"] > 0 and counters["sph"] > 0
    counters.update(cyl=0, sph=0, lines=0)
    _render(s, out_dir / "wire.png", style="wireframe")
    assert counters["sph"] == 0  # no atom spheres
    assert counters["cyl"] == 0  # bonds are lines, not cylinders
    assert counters["lines"] == 1


def test_default_style_is_unchanged_ball_stick(out_dir):
    s = build_ni2mnga_structure()
    a = np.asarray(Image.open(_render(s, out_dir / "none.png")).convert("RGB"))
    b = np.asarray(Image.open(_render(s, out_dir / "bs.png", style="ball-stick")).convert("RGB"))
    assert np.array_equal(a, b)


def test_style_radii():
    s = build_ni2mnga_structure()
    r = {
        st: max(a[2] for a in renderer._collect_atoms_to_draw(s, style=st))
        for st in renderer.RENDER_STYLES
    }
    assert r["space-filling"] > r["ball-stick"] > r["stick"]
    assert r["stick"] == pytest.approx(renderer._stick_radius())


def test_normalize_style_aliases_and_errors():
    assert renderer.normalize_style(None) == "ball-stick"
    assert renderer.normalize_style("Space_Filling") == "space-filling"
    assert renderer.normalize_style("ball-and-stick") == "ball-stick"
    assert renderer.normalize_style("wire") == "wireframe"
    with pytest.raises(ValueError):
        renderer.normalize_style("bogus")


def test_cli_and_mcp_expose_style(monkeypatch, tmp_path):
    parser = server._build_arg_parser()
    assert parser.parse_args(["demo"]).style == "ball-stick"
    args = parser.parse_args(["demo", "--style", "stick"])
    assert server._bulk_options_from_args(args)["style"] == "stick"
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--style", "bogus"])

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}

    def _props(name):
        t = tools[name]
        return (getattr(t, "input_schema", None) or getattr(t, "inputSchema"))["properties"]

    assert "style" in _props("analyze_structure")
    assert "style" in _props("batch_analyze_cifs")

    seen = {}

    def fake_render(structure, output_dir, **kw):
        seen["style"] = kw.get("style")
        return {"figure_overview": None}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    server.run_bulk_analysis(build_ni2mnga_structure(), tmp_path, style="wireframe")
    assert seen["style"] == "wireframe"

"""Batch 7 (--image-format) and the drawing range (--range x0 x1 y0 y1 z0 z1)."""

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

WIDE = [-0.5, 1.5, -0.5, 1.5, -0.5, 1.5]


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


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


# ---- range -------------------------------------------------------------------------

def test_range_atom_count_and_default():
    s = _nacl()
    assert len(renderer._collect_atoms_to_draw(s)) == 27  # {0, .5, 1}^3
    assert len(renderer._collect_atoms_to_draw(s, frac_range=[0, 1, 0, 1, 0, 1])) == 27
    wide = renderer._collect_atoms_to_draw(s, frac_range=WIDE)
    assert len(wide) == 125  # {-.5, 0, .5, 1, 1.5}^3, every point is Na or Cl
    labels = renderer.atom_label_candidates(s, labels="element", frac_range=WIDE)
    assert len(labels) == 125 and all(np.allclose(c[0], a[0]) for c, a in zip(labels, wide))
    assert renderer.normalize_frac_range(None) is None
    for bad in ([0, 1, 0, 1], [1, 0, 0, 1, 0, 1]):
        with pytest.raises(ValueError):
            renderer.normalize_frac_range(bad)


def test_range_replicates_bonds_between_drawn_atoms():
    s = _nacl()
    _, bonds, envs = compute_bonds(s, ["Na-Cl:0:3.0"])
    atoms = [a[0] for a in renderer._collect_atoms_to_draw(s, frac_range=WIDE)]
    rb, polys = renderer._replicate_in_range(s, atoms, WIDE, bonds, envs)
    # Nearest-neighbour pairs on a 5x5x5 grid: 3 axes x 4 x 25.
    assert len(rb) == 300
    assert all(abs(np.linalg.norm(e - st) - 2.82) < 1e-6 for st, e, _, _ in rb)
    # Polyhedra only where all 6 ligands are drawn: the 3x3x3 interior points.
    assert len(polys) == 27 and all(len(lig) == 6 for _, lig, _ in polys)


def test_range_default_pixels_unchanged_and_wide_differs(out_dir):
    s = _nacl()
    a = _pixels(_render(s, out_dir / "a.png"))
    b = _pixels(_render(s, out_dir / "b.png", frac_range=None, image_formats=None))
    assert np.array_equal(a, b)
    w = _render(s, out_dir / "w.png", frac_range=WIDE)
    assert png_background_ratio(w) < 0.995
    assert not np.array_equal(a, _pixels(w))


# ---- image formats -----------------------------------------------------------------

def test_normalize_image_formats():
    assert renderer.normalize_image_formats(None) == []
    assert renderer.normalize_image_formats("png") == []
    assert renderer.normalize_image_formats("TIF, jpg,pdf") == ["tiff", "jpeg", "pdf"]
    assert renderer.normalize_image_formats(["eps", "svg,eps"]) == ["eps", "svg"]
    with pytest.raises(ValueError):
        renderer.normalize_image_formats("webp")


def test_each_format_is_written(out_dir):
    png = _render(_nacl(), out_dir / "v.png", image_formats="tiff,jpeg,pdf,eps,svg")
    w, h = Image.open(png).size
    for ext, fmt in ((".tiff", "TIFF"), (".jpg", "JPEG"), (".eps", "EPS")):
        p = png.with_suffix(ext)
        assert p.is_file() and p.stat().st_size > 1000
        with Image.open(p) as im:
            assert im.format == fmt
            if fmt != "EPS":
                assert im.size == (w, h)
    pdf = png.with_suffix(".pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
    eps = png.with_suffix(".eps").read_bytes()
    assert eps.startswith(b"%!PS")
    # print size follows render.dpi (1 px = 72/dpi pt), not PIL's 1 px = 1 pt
    dpi = renderer._dpi_pair()[0]
    assert f"%%BoundingBox: 0 0 {int(np.ceil(w * 72 / dpi))} ".encode() in eps
    svg = png.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<svg" in svg and "data:image/png;base64," in svg


def test_transparent_tiff_keeps_alpha(out_dir):
    png = _render(_nacl(), out_dir / "t.png", transparent_background=True, image_formats="tiff,jpeg")
    with Image.open(png.with_suffix(".tiff")) as im:
        assert im.mode == "RGBA"
    with Image.open(png.with_suffix(".jpg")) as im:
        assert im.mode == "RGB"


def test_bulk_views_list_extra_formats(tmp_path, monkeypatch):
    seen = []

    def fake_single(**kw):
        seen.append(kw)
        p = tmp_path / kw["output_path"].name
        Image.new("RGB", (20, 20), (255, 255, 255)).save(p)
        renderer.export_image_formats(p, kw["image_formats"])
        return p

    def fake_overview(**kw):
        p = tmp_path / "figure_overview.png"
        Image.new("RGB", (20, 20), (255, 255, 255)).save(p)
        return p

    monkeypatch.setattr(renderer, "render_single_view", fake_single)
    monkeypatch.setattr(renderer, "_compose_overview_figure", fake_overview)
    monkeypatch.setattr(renderer, "_compose_overview_svg", lambda **kw: tmp_path / "figure_overview.svg")
    s = _nacl()

    res = renderer.render_bulk_views(s, tmp_path)
    assert set(res) == {
        "view_a_axis", "view_c_axis", "view_isometric", "figure_overview", "figure_overview_svg"
    }
    assert all(kw["frac_range"] is None and kw["image_formats"] == [] for kw in seen)

    seen.clear()
    res = renderer.render_bulk_views(s, tmp_path, image_formats="tiff,svg", frac_range=WIDE)
    assert all(kw["frac_range"] == WIDE for kw in seen)
    assert res["view_a_axis_tiff"].endswith("view_a_axis.tiff")
    assert res["view_isometric_svg"].endswith("view_isometric.svg")
    assert res["figure_overview_tiff"].endswith("figure_overview.tiff")
    assert res["figure_overview_svg"].endswith("figure_overview.svg")  # editable one kept


def test_cli_and_mcp_expose_range_and_format(monkeypatch, tmp_path):
    parser = server._build_arg_parser()
    opts = server._bulk_options_from_args(parser.parse_args(["demo"]))
    assert opts["frac_range"] is None and opts["image_formats"] is None
    args = parser.parse_args(
        ["demo", "--range", "-0.5", "1.5", "-0.5", "1.5", "-0.5", "1.5", "--image-format", "tiff,pdf"]
    )
    opts = server._bulk_options_from_args(args)
    assert opts["frac_range"] == WIDE and opts["image_formats"] == "tiff,pdf"

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("analyze_structure", "batch_analyze_cifs", "render_view"):
        t = tools[name]
        props = (getattr(t, "input_schema", None) or getattr(t, "inputSchema"))["properties"]
        assert {"frac_range", "image_formats"} <= set(props)

    seen = {}

    def fake_render(structure, output_dir, **kw):
        seen.update(kw)
        return {"figure_overview": None}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    server.run_bulk_analysis(_nacl(), tmp_path, frac_range=WIDE, image_formats="eps")
    assert seen["frac_range"] == WIDE and seen["image_formats"] == "eps"

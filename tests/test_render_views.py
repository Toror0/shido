"""View direction / projection / atom labels (--view-dir, --view-normal, --perspective, --labels)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from PIL import Image
from pymatgen.core import Lattice, Structure

from shido import renderer
from shido import server
from shido.analyzer import build_ni2mnga_structure, compute_bonds
from conftest import SMALL_WINDOW, png_background_ratio


def _render(structure, path, view_dir=(1, 0.36, 0.24), **kw):
    _, bonds, _ = compute_bonds(structure)
    cam, up, _ = renderer.resolve_view_direction(structure, view_dir=view_dir)
    return renderer.render_single_view(
        structure=structure,
        output_path=path,
        camera_dir=cam,
        view_up=up,
        render_bonds=bonds,
        window_size=SMALL_WINDOW,
        **kw,
    )


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGB"))


def test_cubic_direction_equals_plane_normal():
    s = build_ni2mnga_structure()
    for uvw in ((0, 0, 1), (1, 1, 0), (1, 1, 1)):
        d, up_d, lab_d = renderer.resolve_view_direction(s, view_dir=uvw)
        n, up_n, lab_n = renderer.resolve_view_direction(s, view_normal=uvw)
        assert np.allclose(d, n) and np.allclose(up_d, up_n)
        assert abs(float(np.dot(d, up_d))) < 1e-9
    assert np.allclose(renderer.resolve_view_direction(s, view_dir=(0, 0, 1))[0], [0, 0, 1])
    assert lab_d == "[111]" and lab_n == "(111) normal"


def test_noncubic_normal_differs_from_direction():
    # Monoclinic cell: [101] and the (101) normal are different directions.
    s = Structure(Lattice.monoclinic(4.0, 5.0, 6.0, 110.0), ["Cu"], [[0, 0, 0]])
    d = renderer.resolve_view_direction(s, view_dir=(1, 0, 1))[0]
    n = renderer.resolve_view_direction(s, view_normal=(1, 0, 1))[0]
    assert not np.allclose(d, n, atol=1e-3)
    # The (h k l) normal is perpendicular to in-plane vectors, e.g. [1 0 -1] for (101).
    in_plane = s.lattice.get_cartesian_coords([1, 0, -1])
    assert abs(float(np.dot(n, in_plane))) < 1e-9


def test_resolve_view_direction_errors():
    s = build_ni2mnga_structure()
    with pytest.raises(ValueError):
        renderer.resolve_view_direction(s)
    with pytest.raises(ValueError):
        renderer.resolve_view_direction(s, view_dir=(1, 0, 0), view_normal=(1, 0, 0))
    with pytest.raises(ValueError):
        renderer.resolve_view_direction(s, view_dir=(0, 0, 0))


def test_label_count_equals_drawn_atoms():
    s = build_ni2mnga_structure()
    atoms = renderer._collect_atoms_to_draw(s)
    for mode in ("site", "element"):
        cands = renderer.atom_label_candidates(s, labels=mode)
        assert len(cands) == len(atoms)
        assert all(np.allclose(c[0], a[0]) for c, a in zip(cands, atoms))
    assert {t for _, t in renderer.atom_label_candidates(s, labels="element")} == {"Ni", "Mn", "Ga"}
    assert renderer.atom_label_candidates(s, labels=None) == []
    assert renderer.normalize_labels("none") is None
    with pytest.raises(ValueError):
        renderer.normalize_labels("index")


def test_default_view_is_unchanged(out_dir):
    s = build_ni2mnga_structure()
    a = _pixels(_render(s, out_dir / "none.png"))
    b = _pixels(_render(s, out_dir / "explicit.png", perspective=False, labels=None))
    c = _pixels(_render(s, out_dir / "none_str.png", labels="none"))
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_labels_are_drawn(out_dir, monkeypatch):
    drawn = []
    orig = renderer._draw_atom_labels

    def spy(draw, labels, width, ps):
        n = orig(draw, labels, width, ps)
        drawn.append((len(labels), n))
        return n

    monkeypatch.setattr(renderer, "_draw_atom_labels", spy)
    s = build_ni2mnga_structure()
    plain = _pixels(_render(s, out_dir / "plain.png", view_dir=(1, 1, 0)))
    lab = _pixels(_render(s, out_dir / "lab.png", view_dir=(1, 1, 0), labels="element"))
    assert not np.array_equal(plain, lab)
    (visible, n_drawn), = drawn
    # [110] of L2_1: every column front atom is visible -> 15 labels, none overlapping.
    assert 0 < n_drawn <= visible <= len(renderer._collect_atoms_to_draw(s))
    assert n_drawn >= 10


def test_perspective_renders_without_scale_bar(out_dir, monkeypatch):
    calls = []
    orig = renderer._select_scale_bar_length
    monkeypatch.setattr(
        renderer, "_select_scale_bar_length", lambda *a, **k: calls.append(1) or orig(*a, **k)
    )
    s = build_ni2mnga_structure()
    par = _render(s, out_dir / "par.png")
    assert len(calls) == 1
    per = _render(s, out_dir / "per.png", perspective=True)
    assert len(calls) == 1  # no scale bar in perspective
    assert png_background_ratio(per) < 0.995
    assert not np.array_equal(_pixels(par), _pixels(per))


def test_bulk_views_add_custom_view(tmp_path, monkeypatch):
    seen = []

    def fake_single(**kw):
        seen.append(kw)
        return tmp_path / kw["output_path"].name

    monkeypatch.setattr(renderer, "render_single_view", fake_single)
    monkeypatch.setattr(renderer, "_compose_overview_figure", lambda **kw: tmp_path / "o.png")
    monkeypatch.setattr(renderer, "_compose_overview_svg", lambda **kw: tmp_path / "o.svg")
    s = build_ni2mnga_structure()

    res = renderer.render_bulk_views(s, tmp_path)
    assert "view_custom" not in res and len(seen) == 3
    assert all(kw["perspective"] is False and kw["labels"] is None for kw in seen)

    seen.clear()
    res = renderer.render_bulk_views(s, tmp_path, view_normal=(1, 1, 1), perspective=True, labels="site")
    assert res["view_custom"].endswith("view_custom.png") and len(seen) == 4
    assert seen[-1]["orientation_label"] == "(111) normal"
    assert all(kw["perspective"] is True and kw["labels"] == "site" for kw in seen)


def test_cli_and_mcp_expose_view_options(monkeypatch, tmp_path):
    parser = server._build_arg_parser()
    opts = server._bulk_options_from_args(parser.parse_args(["demo"]))
    assert opts["view_dir"] is None and opts["view_normal"] is None
    assert opts["perspective"] is False and opts["labels"] is None
    args = parser.parse_args(
        ["demo", "--view-dir", "1", "1", "0", "--perspective", "--labels", "site"]
    )
    opts = server._bulk_options_from_args(args)
    assert opts["view_dir"] == [1.0, 1.0, 0.0] and opts["perspective"] and opts["labels"] == "site"
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--view-dir", "1", "0", "0", "--view-normal", "1", "0", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--labels", "bogus"])

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}

    def _props(name):
        t = tools[name]
        return (getattr(t, "input_schema", None) or getattr(t, "inputSchema"))["properties"]

    for name in ("analyze_structure", "batch_analyze_cifs", "render_view"):
        assert {"view_dir", "view_normal", "perspective", "labels"} <= set(_props(name))

    seen = {}

    def fake_render(structure, output_dir, **kw):
        seen.update(kw)
        return {"figure_overview": None, "view_custom": "x.png"}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    res = server.run_bulk_analysis(
        build_ni2mnga_structure(), tmp_path, view_dir=[1, 1, 0], perspective=True, labels="element"
    )
    assert seen["view_dir"] == [1, 1, 0] and seen["perspective"] is True
    assert seen["labels"] == "element" and res["view_custom"] == "x.png"

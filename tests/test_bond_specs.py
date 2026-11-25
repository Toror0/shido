"""Bond specs (--bond "A-B:dmin:dmax"), polyhedra centre (--polyhedra-center), --bond-color."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pymatgen.core import Lattice, Structure

from shido import renderer
from shido import server
from shido.analyzer import (
    _compute_bonds_full,
    analyze_structure_data,
    build_ni2mnga_structure,
    compute_bonds,
    parse_bond_specs,
)
from conftest import SMALL_WINDOW, png_background_ratio


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _coordination(neighbor_map) -> dict[int, int]:
    return {i: len({(j, img) for j, img, _ in v}) for i, v in neighbor_map.items()}


def _render(structure, path, bond_specs=None, **kw):
    _, bonds, envs = compute_bonds(structure, bond_specs)
    a_vec, b_vec, c_vec = structure.lattice.matrix
    return renderer.render_single_view(
        structure=structure,
        output_path=path,
        camera_dir=a_vec + 0.36 * b_vec + 0.24 * c_vec,
        view_up=c_vec,
        render_bonds=bonds,
        polyhedra_envs=envs,
        show_polyhedra=True,
        window_size=SMALL_WINDOW,
        **kw,
    )


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGB"))


def test_parse_bond_specs():
    assert parse_bond_specs(None) is None
    assert parse_bond_specs([]) is None
    assert parse_bond_specs("Mn-Ga:0:3.0") == {("Ga", "Mn"): (0.0, 3.0)}
    assert parse_bond_specs(["na-cl:2.9", "Ni-Mn:2.0:2.6"]) == {
        ("Cl", "Na"): (0.0, 2.9),
        ("Mn", "Ni"): (2.0, 2.6),
    }
    assert parse_bond_specs("Na-Cl:3, Na-Na:4") == {("Cl", "Na"): (0.0, 3.0), ("Na", "Na"): (0.0, 4.0)}
    for bad in ("NaCl:3", "Na-Cl", "Na-Cl:3:2", "Na-Cl:-1:2", "Na-Cl:x"):
        with pytest.raises(ValueError):
            parse_bond_specs(bad)


def test_nacl_pair_range_gives_octahedra():
    s = _nacl()
    df, render_bonds, envs, nmap = _compute_bonds_full(s, ["Na-Cl:0:3.0"])
    assert set(_coordination(nmap).values()) == {6}
    assert set(df["element_pair"]) == {"Cl-Na"}
    assert np.allclose(df["distance_angstrom"], 2.82)
    assert render_bonds and len(envs) == len(s)


def test_nacl_range_too_short_gives_no_bonds():
    df, render_bonds, envs, nmap = _compute_bonds_full(_nacl(), ["Na-Cl:0:2.0"])
    assert len(df) == 0 and render_bonds == [] and envs == [] and nmap == {}


def test_min_distance_and_same_element_pair():
    s = _nacl()
    # Na-Na at a/sqrt2 = 3.99 A: 12 neighbours, Cl gets none.
    _, _, _, nmap = _compute_bonds_full(s, ["Na-Na:3.5:4.2"])
    cn = _coordination(nmap)
    assert {s[i].specie.symbol for i in cn} == {"Na"} and set(cn.values()) == {12}
    # Lower bound above the Na-Cl distance removes them.
    df, _, _, _ = _compute_bonds_full(s, ["Na-Cl:2.9:3.5"])
    assert len(df) == 0


def test_ni2mnga_mn_ga_only():
    df, _, envs, _ = _compute_bonds_full(build_ni2mnga_structure(), ["Mn-Ga:0:3.0"])
    assert set(df["element_pair"]) == {"Ga-Mn"}
    assert {elem for _, _, elem in envs} == {"Mn", "Ga"}


def test_csv_outputs_follow_bond_specs(tmp_path):
    res = analyze_structure_data(_nacl(), tmp_path, bond_specs=["Na-Cl:0:3.0"])
    bonds = pd.read_csv(res["bond_lengths_csv"])
    angles = pd.read_csv(res["bond_angles_csv"])
    assert res["num_bonds"] == len(bonds) and set(bonds["element_pair"]) == {"Cl-Na"}
    # Octahedron: 90 and 180 degrees only, centred on both Na and Cl.
    assert set(np.round(angles["angle_deg"], 3)) == {90.0, 180.0}
    assert set(angles["element_center"]) == {"Na", "Cl"}
    res0 = analyze_structure_data(_nacl(), tmp_path / "none", bond_specs=["Na-Cl:2.0"])
    assert res0["num_bonds"] == 0 and res0["render_bonds"] == []


def test_default_render_is_unchanged(out_dir):
    s = build_ni2mnga_structure()
    a = _pixels(_render(s, out_dir / "default.png"))
    b = _pixels(_render(s, out_dir / "explicit.png", polyhedra_center=None, bond_color=None))
    c = _pixels(_render(s, out_dir / "single.png", bond_color="single"))
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_polyhedra_center_and_split_change_image(out_dir, monkeypatch):
    s = build_ni2mnga_structure()
    centers = []
    orig = renderer._add_polyhedra

    def spy(plotter, structure, envs, center_elements=None):
        centers.append(center_elements)
        return orig(plotter, structure, envs, center_elements=center_elements)

    monkeypatch.setattr(renderer, "_add_polyhedra", spy)
    base = _pixels(_render(s, out_dir / "base.png", ["Mn-Ga:0:3.0"]))
    mn = _pixels(_render(s, out_dir / "mn.png", ["Mn-Ga:0:3.0"], polyhedra_center="mn"))
    split = _render(s, out_dir / "split.png", ["Mn-Ga:0:3.0"], bond_color="split")
    assert centers == [None, ["Mn"], None]
    assert not np.array_equal(base, mn)
    assert not np.array_equal(base, _pixels(split))
    assert png_background_ratio(split) < 0.995


def test_add_polyhedra_center_selection(monkeypatch):
    s = build_ni2mnga_structure()
    _, _, envs = compute_bonds(s, ["Mn-Ga:0:3.0", "Ni-Mn:0:3.0"])
    drawn = []
    monkeypatch.setattr(renderer, "_safe_add_mesh", lambda plotter, mesh, **kw: drawn.append(kw["color"]))
    n_mn = sum(1 for _, _, e in envs if e == "Mn")
    renderer._add_polyhedra(None, s, envs, center_elements=["Mn"])
    assert len(drawn) == n_mn and set(drawn) == {renderer._element_color("Mn")}
    drawn.clear()
    renderer._add_polyhedra(None, s, envs, center_elements=["Fe"])  # absent: nothing, no error
    assert drawn == []


def test_normalizers():
    assert renderer.normalize_polyhedra_center(None) is None
    assert renderer.normalize_polyhedra_center("mn, ga") == ["Mn", "Ga"]
    assert renderer.normalize_polyhedra_center(["Ni", "Ni"]) == ["Ni"]
    assert renderer.normalize_bond_color(None) is None
    assert renderer.normalize_bond_color("Split") == "split"
    with pytest.raises(ValueError):
        renderer.normalize_bond_color("rainbow")


def test_cli_and_mcp_expose_bond_options(monkeypatch, tmp_path):
    parser = server._build_arg_parser()
    opts = server._bulk_options_from_args(parser.parse_args(["demo"]))
    assert opts["bonds"] is None and opts["polyhedra_center"] is None and opts["bond_color"] is None
    args = parser.parse_args(
        ["demo", "--bond", "Mn-Ga:0:3.0", "--bond", "Ni-Mn:2.6", "--polyhedra-center", "Mn",
         "--bond-color", "split"]
    )
    opts = server._bulk_options_from_args(args)
    assert opts["bonds"] == ["Mn-Ga:0:3.0", "Ni-Mn:2.6"]
    assert opts["polyhedra_center"] == "Mn" and opts["bond_color"] == "split"
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--bond-color", "rainbow"])

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("analyze_structure", "batch_analyze_cifs", "render_view"):
        t = tools[name]
        props = (getattr(t, "input_schema", None) or getattr(t, "inputSchema"))["properties"]
        assert {"bonds", "polyhedra_center", "bond_color"} <= set(props)

    seen = {}

    def fake_render(structure, output_dir, **kw):
        seen.update(kw)
        return {"figure_overview": None}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    res = server.run_bulk_analysis(
        _nacl(), tmp_path, bonds=["Na-Cl:0:3.0"], polyhedra_center="Na", bond_color="split"
    )
    assert seen["polyhedra_center"] == "Na" and seen["bond_color"] == "split"
    assert res["num_bonds"] == len(pd.read_csv(res["bond_lengths_csv"]))
    assert all({e1, e2} == {"Na", "Cl"} for _, _, e1, e2 in seen["render_bonds"])

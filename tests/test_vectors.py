"""Batch 14: atom vectors / magnetic moments drawn as arrows (--vector / vectors)."""

from __future__ import annotations

import asyncio
import json
import re

import numpy as np
import pytest
from PIL import Image
from pymatgen.core import Lattice, Structure

from shido import renderer
from shido import server
from shido.analyzer import build_ni2mnga_structure, compute_bonds, load_structure_robust
from conftest import SMALL_WINDOW, png_background_ratio
from shido.core.vectors import crystal_to_cartesian, normalize_vectors, resolve_site_vectors
from shido.export3d import export_html

_MCIF = """data_afm_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 6.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_space_group_magn.name_BNS "P1"
loop_
_space_group_symop_magn_operation.id
_space_group_symop_magn_operation.xyz
1 x,y,z,+1
loop_
_space_group_symop_magn_centering.id
_space_group_symop_magn_centering.xyz
1 x,y,z,+1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mn1 Mn 0.25 0.25 0.25
Mn2 Mn 0.75 0.75 0.75
O1 O 0.5 0.0 0.5
loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
Mn1 0 2 0
Mn2 0 -2 0
"""


def _hex_mn() -> Structure:
    s = Structure(Lattice.hexagonal(4.0, 6.0), ["Mn", "Mn", "O"],
                  [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.5, 0.0, 0.5]])
    for site, lab in zip(s, ["Mn1", "Mn2", "O1"]):
        site.label = lab
    return s


def _render(structure, path, **kw):
    _, bonds, envs = compute_bonds(structure)
    return renderer.render_single_view(
        structure=structure, output_path=path, camera_dir=[1.0, 0.36, 0.24],
        view_up=[0.0, 0.0, 1.0], render_bonds=bonds, polyhedra_envs=envs,
        window_size=SMALL_WINDOW, **kw,
    )


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


# ---- parsing / basis ----------------------------------------------------------------

def test_parse_specs():
    assert normalize_vectors(None) is None and normalize_vectors([]) is None
    assert normalize_vectors("Mn:0 0 4") == [{"target": "Mn", "vector": (0.0, 0.0, 4.0)}]
    assert normalize_vectors("Mn1:1,0,0; #2:0 0 -1")[1] == {"target": "#2", "vector": (0.0, 0.0, -1.0)}
    assert normalize_vectors({"Mn": [0, 0, 4]})[0]["vector"] == (0.0, 0.0, 4.0)
    assert normalize_vectors(["magmom"]) == [{"magmom": True}]
    parsed = normalize_vectors(["magmom", "Mn:0 0 4"])
    assert normalize_vectors(parsed) == parsed  # render_bulk_views hands parsed specs on
    for bad in ("Mn", "Mn:0 0", "Mn:a b c"):
        with pytest.raises(ValueError):
            normalize_vectors(bad)
    with pytest.raises(ValueError, match="matches no site"):
        resolve_site_vectors(_hex_mn(), ["Fe:0 0 1"])
    with pytest.raises(ValueError, match="no magnetic moments"):
        resolve_site_vectors(_hex_mn(), ["magmom"])


def test_crystal_axes_to_cartesian_hexagonal():
    s = _hex_mn()
    # along b of a hexagonal cell (gamma = 120): (-1/2, sqrt(3)/2, 0) per mu_B
    np.testing.assert_allclose(crystal_to_cartesian(s, [0, 2, 0]), [-1.0, np.sqrt(3.0), 0.0], atol=1e-9)
    np.testing.assert_allclose(crystal_to_cartesian(s, [0, 0, 3]), [0, 0, 3], atol=1e-9)
    # a + b (unit axes) is along x+y bisector at 60 deg: length 1 (|a/|a| + b/|b|| = 1 for 120 deg)
    v = crystal_to_cartesian(s, [1, 1, 0])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    vecs = resolve_site_vectors(s, ["Mn1:0 2 0", "Mn2:0 -2 0"])
    assert set(vecs) == {0, 1}
    np.testing.assert_allclose(vecs[0], -vecs[1])
    cart = resolve_site_vectors(s, ["Mn:0 2 0"], basis="cartesian")
    np.testing.assert_allclose(cart[0], [0, 2, 0])
    # later spec wins; zero vector drops the arrow
    assert set(resolve_site_vectors(s, ["Mn:0 0 1", "#1:0 0 0"])) == {0}


def test_arrow_geometry_matches_input():
    s = _hex_mn()
    arrows = renderer.vector_arrows(s, ["Mn1:0 2 0"], scale=0.5, include_boundary_atoms=False)
    assert len(arrows) == 1
    start, end = arrows[0]
    np.testing.assert_allclose(0.5 * (start + end), s[0].coords, atol=1e-9)  # through the atom
    np.testing.assert_allclose(end - start, 0.5 * np.array([-1.0, np.sqrt(3.0), 0.0]), atol=1e-9)
    mesh = renderer.build_arrow_mesh(arrows, radius=0.05)
    pts = np.asarray(mesh.points)
    u = (end - start) / np.linalg.norm(end - start)
    proj = (pts - start) @ u
    assert abs(proj.min()) < 1e-6 and abs(proj.max() - np.linalg.norm(end - start)) < 1e-6
    radial = np.linalg.norm((pts - start) - np.outer(proj, u), axis=1)
    assert radial.max() <= 2.2 * 0.05 + 1e-6


def test_arrow_count_equals_drawn_target_atoms():
    s = build_ni2mnga_structure()
    arrows = renderer.vector_arrows(s, ["Mn:0 0 4"])
    n_mn = sum(1 for idx, _ in renderer._drawn_site_images(s) if s[idx].species_string == "Mn")
    assert len(arrows) == n_mn > 4  # boundary images of the Mn sites get arrows too
    for start, end in arrows:
        np.testing.assert_allclose(end - start, [0, 0, 4.0], atol=1e-9)  # 4 mu_B * 1.0 A (default) along c
    rng = renderer.vector_arrows(s, ["Mn:0 0 4"], frac_range=[0, 2, 0, 1, 0, 1])
    n_rng = sum(1 for idx, _ in renderer._drawn_site_images(s, frac_range=[0, 2, 0, 1, 0, 1])
                if s[idx].species_string == "Mn")
    assert len(rng) == n_rng > len(arrows)


# ---- mCIF ---------------------------------------------------------------------------

def test_mcif_moments_read_as_in_file(tmp_path):
    p = tmp_path / "afm.mcif"
    p.write_text(_MCIF, encoding="utf-8")
    s, _ = load_structure_robust(p)
    vecs = resolve_site_vectors(s, ["magmom"])
    mn = [i for i, site in enumerate(s) if site.species_string == "Mn"]
    assert sorted(vecs) == sorted(mn)  # O has zero moment -> no arrow
    b_unit = s.lattice.matrix[1] / np.linalg.norm(s.lattice.matrix[1])
    got = sorted((float(np.dot(vecs[i], b_unit)) for i in mn))
    np.testing.assert_allclose(got, [-2.0, 2.0], atol=1e-6)  # crystalaxis_y = +-2 along b
    for i in mn:
        np.testing.assert_allclose(np.linalg.norm(vecs[i]), 2.0, atol=1e-6)
    # an explicit spec overrides the file for the sites it names
    over = resolve_site_vectors(s, ["magmom", "Mn1:0 0 1"])
    i1 = next(i for i in mn if s[i].label == "Mn1")
    np.testing.assert_allclose(over[i1], [0, 0, 1], atol=1e-9)


# ---- rendering ----------------------------------------------------------------------

def test_default_pixels_unchanged_and_arrows_drawn(out_dir):
    s = build_ni2mnga_structure()
    a = _pixels(_render(s, out_dir / "a.png"))
    b = _pixels(_render(s, out_dir / "b.png", vectors=None))
    c = _pixels(_render(s, out_dir / "c.png", vectors=[]))
    assert np.array_equal(a, b) and np.array_equal(a, c)
    p = _render(s, out_dir / "v.png", vectors=["Mn:0 0 4"], vector_color="#00ff00")
    assert png_background_ratio(p) < 0.995
    px = _pixels(p)
    green = (px[..., 1] > 150) & (px[..., 0] < 100) & (px[..., 2] < 100)
    green0 = (a[..., 1] > 150) & (a[..., 0] < 100) & (a[..., 2] < 100)
    assert green.sum() > green0.sum() + 100
    with pytest.raises(ValueError):
        _render(s, out_dir / "bad.png", vectors=["Mn:0 0 4"], vector_scale=0)


def test_bulk_views_pass_vectors(tmp_path, monkeypatch):
    seen = []

    def fake_single(**kw):
        seen.append(kw)
        p = tmp_path / kw["output_path"].name
        Image.new("RGB", (20, 20), (255, 255, 255)).save(p)
        return p

    monkeypatch.setattr(renderer, "render_single_view", fake_single)
    monkeypatch.setattr(renderer, "_compose_overview_figure", lambda **kw: tmp_path / "o.png")
    monkeypatch.setattr(renderer, "_compose_overview_svg", lambda **kw: tmp_path / "o.svg")
    s = build_ni2mnga_structure()
    renderer.render_bulk_views(s, tmp_path)
    assert all("vectors" not in kw for kw in seen)
    seen.clear()
    renderer.render_bulk_views(s, tmp_path, vectors="Mn:0 0 4", vector_scale=0.3, view_dir=[1, 1, 0])
    assert len(seen) == 4
    assert all(kw["vectors"][0]["vector"] == (0.0, 0.0, 4.0) and kw["vector_scale"] == 0.3 for kw in seen)


def test_html_has_arrows_only_when_asked(out_dir):
    s = build_ni2mnga_structure()

    def scene(path):
        text = path.read_text(encoding="utf-8")
        m = re.search(r'<script type="application/json" id="scene">(.*?)</script>', text, re.S)
        return json.loads(m.group(1).replace("<\\/", "</"))

    d0 = scene(export_html(s, out_dir / "a.html"))
    assert "arrows" not in d0
    d1 = scene(export_html(s, out_dir / "b.html", vectors=["Mn:0 0 4"], vector_scale=0.5))
    heads = np.asarray(d1["arrows"]["heads"]).reshape(-1, 6)
    shafts = np.asarray(d1["arrows"]["shafts"]).reshape(-1, 6)
    # same atoms as the PNG views (4 Mn + their corner / face images)
    assert len(heads) == len(shafts) == len(renderer.vector_arrows(s, ["Mn:0 0 4"])) == 14
    total = (heads[:, 3:] - shafts[:, :3])
    np.testing.assert_allclose(total, np.tile([0, 0, 2.0], (len(total), 1)), atol=1e-3)


def test_cli_and_mcp_expose_vectors(monkeypatch, tmp_path):
    parser = server._build_arg_parser()
    assert "vectors" not in server._bulk_options_from_args(parser.parse_args(["demo"]))
    args = parser.parse_args(["demo", "--vector", "Mn:0 0 4", "--vector", "Ni1:0 0 -1",
                              "--vector-scale", "0.4", "--vector-color", "red", "--vector-basis", "cartesian"])
    opts = server._bulk_options_from_args(args)
    assert opts["vectors"] == ["Mn:0 0 4", "Ni1:0 0 -1"]
    assert opts["vector_scale"] == 0.4 and opts["vector_color"] == "red"
    assert opts["vector_basis"] == "cartesian"

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("analyze_structure", "batch_analyze_cifs", "render_view"):
        t = tools[name]
        props = (getattr(t, "input_schema", None) or getattr(t, "inputSchema"))["properties"]
        assert {"vectors", "vector_basis", "vector_scale", "vector_color", "vector_radius"} <= set(props)

    seen = {}

    def fake_render(structure, output_dir, **kw):
        seen.update(kw)
        return {"figure_overview": None}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    server.run_bulk_analysis(build_ni2mnga_structure(), tmp_path)
    assert "vectors" not in seen
    server.run_bulk_analysis(build_ni2mnga_structure(), tmp_path, vectors=["Mn:0 0 4"])
    assert seen["vectors"] == ["Mn:0 0 4"]

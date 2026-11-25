"""Batch 16: displacement-ellipsoid drawing (renderer style 'ellipsoid')."""

from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido import analyzer
from shido import renderer
from shido import server
from conftest import SMALL_WINDOW, png_background_ratio

COD = Path(__file__).parent / "data" / "cod"


def _iso_structure() -> tuple[Structure, list[np.ndarray]]:
    s = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return s, [0.0123 * np.eye(3), 0.0200 * np.eye(3)]


def _render(s: Structure, path: Path, **kw) -> Path:
    return renderer.render_single_view(
        s, path, camera_dir=[1.0, 0.36, 0.24], view_up=[0, 0, 1], style="ellipsoid", window_size=SMALL_WINDOW, **kw
    )


def test_isotropic_u_gives_sphere_of_1_5382_sqrt_u():
    s, tensors = _iso_structure()
    shapes, warns = renderer.ellipsoid_shapes(s, tensors, 50)
    assert not warns and len(shapes) == 8 + 1  # Na at the corners (8 images) + Cl at the centre
    for sh in shapes:
        u = 0.0123 if s[sh["site"]].specie.symbol == "Na" else 0.0200
        assert sh["kind"] == "ellipsoid"
        assert np.allclose(sh["semi"], 1.5382 * math.sqrt(u), rtol=1e-4)
    # the mesh is a sphere of that radius
    sh = shapes[0]
    mesh = renderer.build_ellipsoid_mesh(sh["center"], sh["semi"], sh["axes"], 24)
    r = np.linalg.norm(np.asarray(mesh.points) - sh["center"], axis=1)
    assert np.allclose(r, 1.5382 * math.sqrt(0.0123), rtol=1e-4)
    # 99 % is larger than 50 %
    big, _ = renderer.ellipsoid_shapes(s, tensors, 99)
    assert big[0]["radius"] > shapes[0]["radius"]


def test_anisotropic_semi_axes_and_principal_ellipses():
    s = Structure(Lattice.cubic(5.0), ["Cu"], [[0.5, 0.5, 0.5]])
    u = np.diag([0.01, 0.02, 0.04])
    shapes, _ = renderer.ellipsoid_shapes(s, [u], 50)
    sh = shapes[0]
    assert np.allclose(sh["semi"], 1.5382 * np.sqrt([0.01, 0.02, 0.04]), rtol=1e-4)
    rings = renderer.principal_ellipse_points(sh["center"], sh["semi"], sh["axes"])
    assert len(rings) == 3
    for ring in rings:  # each point of a principal ellipse lies on the ellipsoid surface
        d = (ring - sh["center"]) @ sh["axes"]
        assert np.allclose(np.sum((d / sh["semi"]) ** 2, axis=1), 1.0)


def test_non_positive_definite_warns_and_draws_sphere(caplog):
    s, tensors = _iso_structure()
    tensors[1] = np.diag([0.01, 0.02, -0.003])
    with caplog.at_level(logging.WARNING):
        shapes, warns = renderer.ellipsoid_shapes(s, tensors, 50)
    assert warns and "Non-positive-definite" in warns[0] and "Cl" in warns[0]
    assert "Non-positive-definite" in caplog.text
    cl = [sh for sh in shapes if sh["site"] == 1][0]
    u_eq = (0.01 + 0.02 - 0.003) / 3
    assert cl["kind"] == "npd" and np.allclose(cl["semi"], 1.5382 * math.sqrt(u_eq), rtol=1e-4)


def test_atoms_without_adp_are_small_spheres():
    s, tensors = _iso_structure()
    shapes, warns = renderer.ellipsoid_shapes(s, [tensors[0], None], 50)
    cl = [sh for sh in shapes if sh["site"] == 1][0]
    assert cl["kind"] == "none" and cl["radius"] == renderer.ELLIPSOID_NO_ADP_RADIUS
    assert any("without ADPs" in w for w in warns)


def test_render_isotropic_not_blank(tmp_path):
    s, tensors = _iso_structure()
    p = _render(s, tmp_path / "iso.png", adp=tensors)
    assert png_background_ratio(p) < 0.995


def test_render_cod_cif_not_blank(tmp_path):
    cif = COD / "1100710.cif"
    s, _ = analyzer.load_structure_robust(cif)
    p = _render(s, tmp_path / "cod.png", adp=str(cif), probability=50)
    assert png_background_ratio(p) < 0.995


def test_style_name_and_default_unchanged():
    assert renderer.normalize_style("ORTEP") == "ellipsoid"
    assert "ellipsoid" in renderer.RENDER_STYLES
    assert server._ellipsoid_kwargs("ball-stick", "x.cif", 50) == {}
    assert server._ellipsoid_kwargs("ellipsoid", "x.cif", 30) == {"adp": "x.cif", "probability": 30}


def test_cli_and_mcp_expose_ellipsoid(monkeypatch, tmp_path):
    parser = server._build_arg_parser()
    opts = server._bulk_options_from_args(parser.parse_args(["demo"]))
    assert "adp" not in opts and "probability" not in opts
    opts = server._bulk_options_from_args(
        parser.parse_args(["demo", "--style", "ellipsoid", "--probability", "30"])
    )
    assert opts["style"] == "ellipsoid" and opts["probability"] == 30.0 and opts["adp"] is None

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("analyze_structure", "render_view"):
        assert "probability" in tools[name].input_schema["properties"]

    seen: dict = {}

    def fake_render(structure, output_dir, **kw):
        seen.clear()
        seen.update(kw)
        return {"figure_overview": None}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    s, _ = _iso_structure()
    server.run_bulk_analysis(s, tmp_path)
    assert "adp" not in seen
    server.run_bulk_analysis(s, tmp_path, style="ellipsoid", adp="a.cif", probability=30)
    assert seen["adp"] == "a.cif" and seen["probability"] == 30

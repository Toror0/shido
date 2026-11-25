"""Batch 17: volumetric data (CHGCAR / Cube / XSF / grd) and isosurfaces (--volume)."""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest
from PIL import Image
from pymatgen.core import Lattice, Structure

from shido import renderer
from shido import server
from shido.analyzer import compute_bonds
from conftest import SMALL_WINDOW, png_background_ratio
from shido.core import volumetric as V

N = 24
SIGMA = 0.45  # Å
ELECTRONS = (2.0, 8.0)  # Na, Cl (toy numbers)


def _nacl_like(lattice: Lattice | None = None) -> Structure:
    return Structure(lattice or Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _gauss_density(s: Structure, n: int = N, sigma: float = SIGMA) -> np.ndarray:
    """Periodic sum of normalised Gaussians (e/Å^3) on an n^3 grid; integral = sum(ELECTRONS)."""
    f = np.stack(np.meshgrid(*(np.arange(n) / n,) * 3, indexing="ij"), -1)
    rho = np.zeros((n, n, n))
    for site, ne in zip(s, ELECTRONS):
        for shift in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3):
            d = (f - site.frac_coords - shift) @ s.lattice.matrix
            rho += ne * np.exp(-np.sum(d**2, -1) / (2 * sigma**2)) / (2 * np.pi * sigma**2) ** 1.5
    return rho


@pytest.fixture()
def files(tmp_path):
    s = _nacl_like()
    rho = _gauss_density(s)
    chg = V.write_chgcar(tmp_path / "CHGCAR", s, rho)
    cube = V.write_cube(tmp_path / "nacl.cube", s, rho * V.BOHR_TO_ANGSTROM**3)  # e/Bohr^3
    return s, rho, chg, cube


def _radius_at(level: float, ne: float, sigma: float = SIGMA) -> float:
    """Radius where one Gaussian of ne electrons equals level."""
    peak = ne / (2 * np.pi * sigma**2) ** 1.5
    return sigma * math.sqrt(2 * math.log(peak / level))


# ---- reading ------------------------------------------------------------------------

def test_detect_formats(tmp_path):
    assert V.detect_volume_format("CHGCAR") == "chgcar"
    assert V.detect_volume_format("CHG") == "chgcar"
    assert V.detect_volume_format("PARCHG.0001") == "chgcar"
    assert V.detect_volume_format("LOCPOT") == "locpot"
    assert V.detect_volume_format("ELFCAR") == "elfcar"
    assert V.detect_volume_format("a.cube") == "cube"
    assert V.detect_volume_format("a.grd") == "grd"
    assert V.detect_volume_format("a.cif") is None
    assert V.detect_volume_format("POSCAR") is None
    x = tmp_path / "s.xsf"
    x.write_text("CRYSTAL\nPRIMVEC\n1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
    assert V.detect_volume_format(x) is None


def test_chgcar_and_cube_read_back(files):
    s, rho, chg, cube = files
    for path, fmt, scale in ((chg, "chgcar", 1.0), (cube, "cube", V.BOHR_TO_ANGSTROM**3)):
        g = V.read_volume(path)
        assert g.fmt == fmt and g.shape == (N, N, N)
        np.testing.assert_allclose(g.lattice.matrix, s.lattice.matrix, atol=1e-4)
        assert [str(x.specie) for x in g.structure] == ["Na", "Cl"]
        np.testing.assert_allclose(g.structure.frac_coords, s.frac_coords, atol=1e-5)
        np.testing.assert_allclose(g.data, rho * scale, rtol=1e-4, atol=1e-8)
        assert g.n_electrons() == pytest.approx(sum(ELECTRONS), rel=1e-4)
    g = V.read_volume(chg)
    assert g.raw_scale == pytest.approx(s.volume)  # CHGCAR stores rho * V


def test_xsf_and_grd_read_back(tmp_path):
    s = _nacl_like(Lattice.from_parameters(4.0, 4.4, 5.0, 80, 95, 110))
    rho = _gauss_density(s, n=8)
    full = V.periodic_pad(rho)  # XSF general grid repeats the end points
    flat = full.transpose(2, 1, 0).reshape(-1)  # x fastest
    m = s.lattice.matrix
    lines = ["CRYSTAL", "PRIMVEC", *(" ".join(f"{v:.8f}" for v in r) for r in m), "PRIMCOORD", "2 1"]
    lines += [f"{x.specie.Z} " + " ".join(f"{v:.8f}" for v in x.coords) for x in s]
    lines += ["BEGIN_BLOCK_DATAGRID_3D", "rho", "BEGIN_DATAGRID_3D_rho", "9 9 9", "0 0 0"]
    lines += [" ".join(f"{v:.8f}" for v in r) for r in m]
    lines += [" ".join(f"{v:.8e}" for v in flat[i:i + 5]) for i in range(0, flat.size, 5)]
    lines += ["END_DATAGRID_3D", "END_BLOCK_DATAGRID_3D"]
    x = tmp_path / "s.xsf"
    x.write_text("\n".join(lines) + "\n", encoding="utf-8")
    g = V.read_volume(x)
    assert g.fmt == "xsf" and g.shape == (8, 8, 8) and len(g.structure) == 2
    np.testing.assert_allclose(g.data, rho, rtol=1e-6)
    np.testing.assert_allclose(g.structure.frac_coords, s.frac_coords, atol=1e-6)

    a, b, c, al, be, ga = s.lattice.parameters
    vals = rho.transpose(2, 1, 0).reshape(-1)
    grd = ["title", f"{a} {b} {c} {al} {be} {ga}", "8 8 8"]
    grd += [" ".join(f"{v:.8e}" for v in vals[i:i + 6]) for i in range(0, vals.size, 6)]
    p = tmp_path / "s.grd"
    p.write_text("\n".join(grd) + "\n", encoding="utf-8")
    g2 = V.read_volume(p)
    assert g2.structure is None and g2.shape == (8, 8, 8)
    np.testing.assert_allclose(g2.data, rho, rtol=1e-6)
    np.testing.assert_allclose(g2.lattice.parameters, s.lattice.parameters, atol=1e-6)
    assert g2.n_electrons() == pytest.approx(sum(ELECTRONS), rel=1e-3)


def test_normalize_isovalues(files):
    g = V.read_volume(files[2])
    assert V.normalize_isovalues("0.05,0.1") == [0.05, 0.1]
    assert V.normalize_isovalues(0.2) == [0.2]
    assert V.normalize_isovalues(["0.1", 0.3]) == [0.1, 0.3]
    assert V.normalize_isovalues(None, g) == [pytest.approx(0.1 * g.data.max(), rel=1e-3)]
    for bad in ("abc", "0", "nan"):
        with pytest.raises(ValueError):
            V.normalize_isovalues(bad)


# ---- isosurfaces --------------------------------------------------------------------

def test_isosurface_radius_matches_gaussian(files):
    s, _, chg, _ = files
    g = V.read_volume(chg)
    level = 0.5
    surf = V.isosurface(g, level)
    centre = s[1].coords  # Cl in the middle of the cell: a closed sphere
    pts = np.asarray(surf.points)
    near = pts[np.linalg.norm(pts - centre, axis=1) < 1.5]
    r = np.linalg.norm(near - centre, axis=1)
    assert len(near) > 100
    assert np.mean(r) == pytest.approx(_radius_at(level, ELECTRONS[1]), abs=0.05)
    assert np.std(r) < 0.05


def test_oblique_cell_surface_stays_on_atom(tmp_path):
    s = _nacl_like(Lattice.from_parameters(4.0, 4.5, 5.0, 75, 100, 115))
    rho = _gauss_density(s)
    g = V.read_volume(V.write_cube(tmp_path / "obl.cube", s, rho))
    surf = V.isosurface(g, 0.5)
    centre = s[1].coords
    pts = np.asarray(surf.points)
    near = pts[np.linalg.norm(pts - centre, axis=1) < 1.5]
    np.testing.assert_allclose(near.mean(axis=0), centre, atol=0.03)
    assert np.mean(np.linalg.norm(near - centre, axis=1)) == pytest.approx(
        _radius_at(0.5, ELECTRONS[1]), abs=0.05
    )


def test_periodic_seam_closes(files):
    """Na sits on the corner: its sphere is cut by the cell faces. With the periodic
    padding the surface reaches the faces exactly and the pieces at 0 and 1 match."""
    s, _, chg, _ = files
    g = V.read_volume(chg)
    surf = V.isosurface(g, 0.5)
    frac = np.asarray(surf.points) @ np.linalg.inv(s.lattice.matrix)
    assert frac.max() == pytest.approx(1.0, abs=1e-9) and frac.min() == pytest.approx(0.0, abs=1e-9)
    edges = surf.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
    )
    ef = np.asarray(edges.points) @ np.linalg.inv(s.lattice.matrix)
    on_face = np.any((np.abs(ef) < 1e-6) | (np.abs(ef - 1) < 1e-6), axis=1)
    assert edges.n_points > 0 and on_face.all()  # open only on the cell faces
    # the rim on x = 0 and on x = 1 is the same curve
    lo = np.round(ef[np.abs(ef[:, 0]) < 1e-6][:, 1:], 6)
    hi = np.round(ef[np.abs(ef[:, 0] - 1) < 1e-6][:, 1:], 6)
    assert {tuple(p) for p in lo} == {tuple(p) for p in hi}
    # without the padding the corner sphere would stop one voxel short of the faces
    unpadded = V.VolumetricGrid(lattice=g.lattice, data=g.data)
    import pyvista as pv

    img = pv.ImageData(dimensions=g.shape, spacing=(1 / N,) * 3)
    img.point_data["v"] = unpadded.data.ravel(order="F")
    assert np.asarray(img.contour([0.5], scalars="v").points).max() < 1 - 0.5 / N


def test_middle_sphere_is_closed(files):
    g = V.read_volume(files[2])
    surf = V.isosurface(g, 2.0)  # only the Cl sphere (Na peak is lower)
    edges = surf.extract_feature_edges(
        boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False
    )
    assert edges.n_points == 0
    assert V.isosurface(g, 1e6) is None


# ---- rendering ----------------------------------------------------------------------

def _render(structure, path, **kw):
    _, bonds, envs = compute_bonds(structure)
    return renderer.render_single_view(
        structure=structure, output_path=path, camera_dir=[1.0, 0.36, 0.24],
        view_up=[0.0, 0.0, 1.0], render_bonds=bonds, polyhedra_envs=envs,
        window_size=SMALL_WINDOW, **kw,
    )


def _pixels(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(int)


def test_default_pixels_unchanged_and_isosurface_drawn(files, out_dir):
    s, _, chg, _ = files
    a = _pixels(_render(s, out_dir / "a.png"))
    b = _pixels(_render(s, out_dir / "b.png", volume=None, isovalue=0.5))
    assert np.array_equal(a, b)
    p = _render(s, out_dir / "v.png", volume=chg, isovalue="0.5", iso_color="#00ff00", iso_opacity=1.0)
    assert png_background_ratio(p) < 0.995
    px = _pixels(p)
    green = (px[..., 1] > 120) & (px[..., 0] < 90) & (px[..., 2] < 90)
    green0 = (a[..., 1] > 120) & (a[..., 0] < 90) & (a[..., 2] < 90)
    assert green.sum() > green0.sum() + 500
    with pytest.raises(ValueError):
        _render(s, out_dir / "bad.png", volume=chg, iso_opacity=0)


def test_both_signs_two_colours(files):
    s, rho, _, _ = files
    f = np.arange(N) / N
    wave = np.cos(2 * np.pi * f)[:, None, None] * np.ones((N, N, N))  # +1 .. -1
    g = V.VolumetricGrid(lattice=s.lattice, data=wave, structure=s)
    meshes = renderer.isosurface_meshes(s, g, "0.3", both_signs=True, color="#ff0000",
                                        color_negative="#0000ff")
    assert [m[2] for m in meshes] == [0.3, -0.3]
    assert [m[1] for m in meshes] == ["#ff0000", "#0000ff"]
    multi = renderer.isosurface_meshes(s, g, [0.3, 0.6], color="#ff0000,#00ff00")
    assert [m[1] for m in multi] == ["#ff0000", "#00ff00"]


def test_bulk_views_pass_volume(files, tmp_path, monkeypatch):
    s, _, chg, _ = files
    seen = []

    def fake_single(**kw):
        seen.append(kw)
        p = tmp_path / kw["output_path"].name
        Image.new("RGB", (20, 20), (255, 255, 255)).save(p)
        return p

    monkeypatch.setattr(renderer, "render_single_view", fake_single)
    monkeypatch.setattr(renderer, "_compose_overview_figure", lambda **kw: tmp_path / "o.png")
    monkeypatch.setattr(renderer, "_compose_overview_svg", lambda **kw: tmp_path / "o.svg")
    renderer.render_bulk_views(s, tmp_path)
    assert all("volume" not in kw for kw in seen)
    seen.clear()
    renderer.render_bulk_views(s, tmp_path, volume=chg, isovalue="0.5", iso_both_signs=True)
    assert len(seen) == 3
    assert all(isinstance(kw["volume"], V.VolumetricGrid) and kw["isovalue"] == "0.5"
               and kw["iso_both_signs"] for kw in seen)


def test_cli_and_mcp_expose_volume(files, monkeypatch, tmp_path):
    s, _, chg, cube = files
    parser = server._build_arg_parser()
    assert "volume" not in server._bulk_options_from_args(parser.parse_args(["demo"]))
    args = parser.parse_args(["demo", "--volume", str(chg), "--isovalue", "0.05,0.1",
                              "--iso-color", "red", "--iso-opacity", "0.4", "--iso-both-signs"])
    opts = server._bulk_options_from_args(args)
    assert opts["volume"] == str(chg) and opts["isovalue"] == "0.05,0.1"
    assert opts["iso_color"] == "red" and opts["iso_opacity"] == 0.4 and opts["iso_both_signs"]

    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    for name in ("analyze_structure", "render_view"):
        t = tools[name]
        props = (getattr(t, "input_schema", None) or getattr(t, "inputSchema"))["properties"]
        assert {"volume_path", "isovalue", "iso_color", "iso_opacity", "iso_both_signs"} <= set(props)

    seen = {}

    def fake_render(structure, output_dir, **kw):
        seen.clear()
        seen.update(kw)
        return {"figure_overview": None}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    res = server.run_bulk_analysis(s, tmp_path)
    assert "volume" not in seen and "volume" not in res
    res = server.run_bulk_analysis(s, tmp_path, volume=str(cube), isovalue=0.05)
    assert isinstance(seen["volume"], V.VolumetricGrid)
    assert res["volume"]["grid"] == [N, N, N] and res["volume"]["isovalues"] == [0.05]

    # a CHGCAR alone: its own structure and its own isosurfaces
    struct, warns, own = server._load_structure_or_volume(chg)
    assert own == str(chg) and struct.composition.reduced_formula == "NaCl"
    calls = []
    monkeypatch.setattr(server, "run_bulk_analysis", lambda st, d, **kw: calls.append((st, kw)) or {})
    monkeypatch.setattr(server, "_print_bulk_result", lambda r: None)
    assert server.main([str(chg), "--isovalue", "0.5", "--out-dir", str(tmp_path)]) == 0
    st, kw = calls[0]
    assert len(st) == 2 and kw["volume"] == str(chg) and kw["isovalue"] == "0.5"

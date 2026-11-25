"""Edge-case ("mean") tests for the runtime guardrails and defensive rendering.

Each test builds its own pathological structure, so no external CIF is needed.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido import renderer
from shido import theme_manager
from shido.analyzer import analyze_structure_data, compute_bonds, to_ordered_proxy
from conftest import SMALL_WINDOW, png_background_ratio
from shido.core.species import dominant_element, site_occupancies
from shido.core.validator import (
    ScientificValidationError,
    validate_blank_image,
    validate_composition_conserved,
    validate_lattice,
    validate_miller_index,
    validate_min_distance,
    validate_structure,
)
from shido.slab_cutter import cut_slabs, resolve_miller_index

HEX_RE = re.compile(r"^#[0-9A-F]{6}$")


def _render(structure, path, **kw):
    """render_single_view along +a with a small window (bonds computed unless given)."""
    if "render_bonds" not in kw:
        _, kw["render_bonds"], _ = compute_bonds(structure)
    a_vec, _, c_vec = structure.lattice.matrix
    return renderer.render_single_view(
        structure=structure,
        output_path=path,
        camera_dir=kw.pop("camera_dir", a_vec),
        view_up=kw.pop("view_up", c_vec),
        window_size=SMALL_WINDOW,
        **kw,
    )


# ---------------------------------------------------------------------------------------
# 1. Disordered alloy: Mn0.5Ga0.5 partial site
# ---------------------------------------------------------------------------------------
def _b2_nimnga() -> Structure:
    return Structure(
        Lattice.cubic(2.91),
        [{"Mn": 0.5, "Ga": 0.5}, "Ni"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def test_disordered_alloy_dominant_color_render(out_dir, caplog):
    s = _b2_nimnga()
    site = s[0]
    with pytest.raises(AttributeError):
        _ = site.specie  # what the helper protects against

    # Dominant element: tie -> alphabetical (deterministic); clear majority -> majority.
    assert dominant_element(site) == "Ga"
    assert site_occupancies(site) == [("Ga", 0.5), ("Mn", 0.5)]
    majority = Structure(Lattice.cubic(2.91), [{"Mn": 0.6, "Ga": 0.4}], [[0, 0, 0]])
    assert dominant_element(majority[0]) == "Mn"

    colors = theme_manager.resolve_structure_colors(s)
    assert set(colors) == {"Mn", "Ga", "Ni"}
    assert all(HEX_RE.match(c) for c in colors.values())

    with caplog.at_level(logging.WARNING, logger="crystal_mcp"):
        img = _render(s, out_dir / "b2.png")
    assert img.exists() and img.stat().st_size > 0
    assert any("partially occupied" in r.getMessage() for r in caplog.records)

    # Vacancy-containing site (total occupancy 0.9 < 1) is drawn too.
    vac = Structure(
        Lattice.cubic(2.91), [{"Mn": 0.45, "Ga": 0.45}, "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    assert _render(vac, out_dir / "vac.png").exists()

    # Full analysis path (CSV + bonds) also survives the disordered site.
    res = analyze_structure_data(s, out_dir / "analysis")
    assert res["summary"]["is_ordered"] is False


# ---------------------------------------------------------------------------------------
# 2. Overlapping atoms (0.04 Å)
# ---------------------------------------------------------------------------------------
def _overlap_structure() -> Structure:
    a = 5.0
    return Structure(
        Lattice.cubic(a),
        ["Fe", "Fe"],
        [[0.25, 0.25, 0.25], [0.25 + 0.04 / a, 0.25, 0.25]],
    )


def test_overlapping_atoms_blocked(out_dir):
    s = _overlap_structure()
    with pytest.raises(ScientificValidationError, match="overlap"):
        validate_structure(s)
    with pytest.raises(ScientificValidationError):
        _render(s, out_dir / "overlap.png", render_bonds=[])
    assert not (out_dir / "overlap.png").exists()
    with pytest.raises(ScientificValidationError):
        analyze_structure_data(s, out_dir / "analysis")


def test_overlap_across_periodic_boundary_blocked():
    # 0.04 Å apart only through the periodic boundary (x = 0.001 and x = 0.993).
    a = 5.0
    s = Structure(Lattice.cubic(a), ["Fe", "Fe"], [[0.001, 0, 0], [1 - 0.03 / a, 0, 0]])
    with pytest.raises(ScientificValidationError):
        validate_structure(s)


def test_split_site_is_not_an_overlap():
    # CIF split-site disorder: two half-occupied positions 0.3 Å apart hold one atom.
    a = 5.0
    s = Structure(
        Lattice.cubic(a),
        [{"O": 0.5}, {"O": 0.5}, "Ti"],
        [[0.25, 0.25, 0.25], [0.25 + 0.3 / a, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    validate_structure(s)  # must not raise


# ---------------------------------------------------------------------------------------
# 3. Zero bonds and over-dense bonds
# ---------------------------------------------------------------------------------------
class _SpyPlotter:
    def __init__(self):
        self.meshes = []

    def add_mesh(self, mesh, **kw):
        assert mesh is not None and mesh.n_points > 0
        self.meshes.append(mesh)


def test_zero_bonds_isolated_atom(out_dir):
    s = Structure(Lattice.cubic(12.0), ["Ar"], [[0.5, 0.5, 0.5]])
    spy = _SpyPlotter()
    renderer._add_bonds(spy, [])
    renderer._add_bonds(spy, [(np.zeros(3), np.zeros(3), "Ar", "Ar")])  # zero-length only
    assert spy.meshes == []  # nothing (and no empty mesh) handed to add_mesh

    assert _render(s, out_dir / "iso_nobonds.png", render_bonds=[]).exists()
    assert _render(s, out_dir / "iso_none.png", render_bonds=None).exists()
    assert _render(s, out_dir / "iso_auto.png").exists()  # whatever compute_bonds finds


def _dense_fcc_bonds() -> tuple[Structure, list]:
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.61), ["Cu"], [[0, 0, 0]])
    s.make_supercell([4, 4, 4])  # 256 atoms
    centers, points, _, dists = s.get_neighbor_list(r=3.7)  # 1st (2.55) + 2nd (3.61) shell
    bonds = []
    seen = set()
    for i, j, d in zip(centers, points, dists):
        key = (min(i, j), max(i, j), round(float(d), 3))
        if key in seen:
            continue
        seen.add(key)
        p_i = s[int(i)].coords
        vec = s[int(j)].coords - p_i
        vec -= s.lattice.get_cartesian_coords(np.round(s.lattice.get_fractional_coords(vec)))
        bonds.append((p_i, p_i + vec, "Cu", "Cu"))
    return s, bonds


def test_dense_bonds_limited_to_nearest(out_dir, caplog):
    s, bonds = _dense_fcc_bonds()
    assert len(bonds) > renderer.MAX_RENDER_BONDS

    with caplog.at_level(logging.WARNING, logger="crystal_mcp"):
        limited = renderer.limit_render_bonds(bonds)
    assert 0 < len(limited) <= renderer.MAX_RENDER_BONDS
    lengths = np.array([np.linalg.norm(b[1] - b[0]) for b in limited])
    nn = 3.61 / np.sqrt(2.0)
    assert np.all(np.abs(lengths - nn) < 0.01)  # only the 1st-neighbor shell survives
    assert any("nearest-neighbor" in r.getMessage() for r in caplog.records)

    img = _render(s, out_dir / "dense.png", render_bonds=bonds)
    assert img.exists() and img.stat().st_size > 0


# ---------------------------------------------------------------------------------------
# 4. Extreme slab: 3 x 3 x 80 Å cell with ~70 Å vacuum
# ---------------------------------------------------------------------------------------
def _vacuum_slab() -> Structure:
    c = 80.0
    species, coords = [], []
    for n in range(6):  # 6 layers, 2 Å apart, centred in the cell
        z = (35.0 + 2.0 * n) / c
        xy = (0.0, 0.0) if n % 2 == 0 else (0.5, 0.5)
        species.append("Cu")
        coords.append([xy[0], xy[1], z])
    return Structure(Lattice.from_parameters(3.0, 3.0, c, 90, 90, 90), species, coords)


def test_extreme_vacuum_slab_camera(out_dir):
    s = _vacuum_slab()
    a_vec, b_vec, c_vec = s.lattice.matrix

    atoms = renderer._collect_atoms_to_draw(s)
    cam = a_vec / np.linalg.norm(a_vec)
    up = renderer._orthonormal_up(cam, c_vec)
    focal, scale, apx = renderer._compute_parallel_framing(s, atoms, cam, up, SMALL_WINDOW)
    # Framed on the ~10 Å of atoms, not on the 80 Å cell (old framing: scale ~ 47 Å).
    assert scale < 15.0
    z_atoms = [a[0][2] for a in atoms]
    assert min(z_atoms) - 1e-6 <= focal[2] <= max(z_atoms) + 1e-6
    # A Cu atom is drawn with a diameter of many pixels (old framing: ~4 px at 480 px).
    radius_px = atoms[0][2] / apx
    assert radius_px > 15

    for name, cam_dir, view_up in (("side", a_vec, c_vec), ("top", c_vec, b_vec)):
        path = _render(s, out_dir / f"slab_{name}.png", camera_dir=cam_dir, view_up=view_up)
        assert path.exists()
        assert png_background_ratio(path) < 0.95


def test_blank_image_detected():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    blank[0, 0] = (10, 10, 10)  # one stray pixel: 99.9975 % background
    with pytest.raises(ScientificValidationError, match="background"):
        validate_blank_image(blank, background="#FFFFFF")
    ok = blank.copy()
    ok[50:80, 50:80] = (200, 30, 30)
    assert validate_blank_image(ok, background="#FFFFFF") < 0.995
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)  # fully transparent
    with pytest.raises(ScientificValidationError):
        validate_blank_image(rgba, transparent=True)


# ---------------------------------------------------------------------------------------
# 5. Strongly triclinic cell and invalid Miller indices
# ---------------------------------------------------------------------------------------
def _triclinic() -> Structure:
    lat = Lattice.from_parameters(4.0, 4.5, 5.0, 68.0, 113.0, 81.0)
    return Structure(lat, ["Fe", "Al"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


TRICLINIC_HKLS = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0),
    (1, 1, 1), (-1, -1, -1), (2, 1, -1), (1, 2, 3), (0, 1, -1),
]


@pytest.mark.parametrize("hkl", TRICLINIC_HKLS)
def test_triclinic_plane_polygon_exact(hkl):
    lat = _triclinic().lattice
    d_lo = float(sum(min(x, 0) for x in hkl))
    d_hi = float(sum(max(x, 0) for x in hkl))
    built = None
    for d_off in (d_lo + 1.0, 0.5 * (d_lo + d_hi)):
        if d_lo < d_off <= d_hi:
            built = renderer._compute_hkl_plane_mesh(lat, hkl, d_offset=d_off)
            if built is not None:
                break
    assert built is not None, f"no polygon for {hkl}"
    mesh, edges = built
    pts = np.asarray(mesh.points)
    assert mesh.n_cells == 1 and len(pts) >= 3 and edges.n_lines == len(pts)

    # Independent check in fractional space: h*u + k*v + l*w = d, inside [0,1]^3, on a face.
    frac = lat.get_fractional_coords(pts)
    np.testing.assert_allclose(frac @ np.asarray(hkl, float), d_off, atol=1e-9)
    assert np.all(frac > -1e-9) and np.all(frac < 1 + 1e-9)
    assert np.all(np.any(np.isclose(frac, 0.0, atol=1e-9) | np.isclose(frac, 1.0, atol=1e-9), axis=1))
    # Cartesian check: the polygon is perpendicular to the reciprocal vector G_hkl.
    g = np.asarray(hkl, float) @ lat.reciprocal_lattice_crystallographic.matrix
    edges_vec = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    np.testing.assert_allclose(edges_vec @ g, 0.0, atol=1e-9)


def test_triclinic_planes_render_and_slab_cut(out_dir):
    s = _triclinic()
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=SMALL_WINDOW)
    try:
        renderer.add_crystal_planes(plotter, s, TRICLINIC_HKLS[:4])
        assert len(plotter.renderer.actors) >= 8  # fill + edge per plane
    finally:
        plotter.close()

    res = cut_slabs(s, (1, 1, 0), out_dir / "slab", min_slab_size=6.0, min_vacuum_size=8.0)
    assert res and (out_dir / "slab" / "数値" / "surface_summary.csv").exists()
    assert list((out_dir / "slab" / "図").glob("*.png"))


@pytest.mark.parametrize("bad", [(0, 0, 0), [0, 0, 0], (0.0, 0.0, 0.0)])
def test_zero_miller_index_rejected(bad, out_dir):
    with pytest.raises(ValueError):
        validate_miller_index(bad)
    with pytest.raises(ValueError):
        renderer._compute_hkl_plane_mesh(_triclinic().lattice, bad)
    with pytest.raises(ValueError):
        cut_slabs(_triclinic(), bad, out_dir / "never")
    # and it is the scientific error type
    with pytest.raises(ScientificValidationError):
        validate_miller_index(bad)


@pytest.mark.parametrize("spec", ["000", "(0 0 0)", "0,0,0"])
def test_zero_miller_string_rejected(spec):
    with pytest.raises(ValueError):
        resolve_miller_index(spec)


@pytest.mark.parametrize("bad", [(1, 0), (1, 0.5, 0), ("a", 0, 1), (float("nan"), 1, 0)])
def test_malformed_miller_index_rejected(bad):
    with pytest.raises(ValueError):
        validate_miller_index(bad)


# ---------------------------------------------------------------------------------------
# Lattice / composition guardrails
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([float("nan"), 3.0, 3.0]),
        np.diag([float("inf"), 3.0, 3.0]),
        np.diag([1.0, 3.0, 3.0]),          # a = 1.0 Å (not > 1.0)
        np.diag([3.0, 3.0, 0.1]),          # V = 0.9 Å^3
        np.array([[3.0, 0, 0], [3.0, 1e-4, 0], [0, 0, 3.0]]),  # (almost) coplanar
    ],
)
def test_bad_lattice_rejected(matrix):
    with pytest.raises(ScientificValidationError):
        validate_lattice(Lattice(matrix))


def test_composition_guard():
    a = Structure(Lattice.cubic(4.0), ["Ni", "Ni", "Mn", "Ga"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0]])
    sc = a.copy()
    sc.make_supercell([2, 1, 1])
    validate_composition_conserved(a, sc)
    broken = sc.copy()
    broken.remove_sites([0])
    with pytest.raises(ScientificValidationError, match="composition"):
        validate_composition_conserved(a, broken)


def test_empty_structure_rejected():
    with pytest.raises(ScientificValidationError):
        validate_structure(Structure(Lattice.cubic(4.0), [], []))


# ---------------------------------------------------------------------------------------
# 6. Ten-element high-entropy alloy: color exhaustion
# ---------------------------------------------------------------------------------------
HEA = ["Ti", "Zr", "Hf", "Nb", "Ta", "Mo", "W", "Cr", "V", "Al"]


@pytest.mark.parametrize("theme", theme_manager.list_themes())
def test_hea_ten_colors_every_theme(theme):
    cm = theme_manager.ColorManager(theme_name=theme)
    colors = cm.resolve_element_colors(HEA)
    assert set(colors) == set(HEA)
    assert all(HEX_RE.match(c) for c in colors.values()), colors
    assert len(set(colors.values())) == 10, colors


def test_hea_colors_when_all_reserves_exhausted(monkeypatch):
    """Tiny reserve palette, no theme colors, no Jmol table: still 10 valid, distinct colors."""
    cm = theme_manager.ColorManager()
    monkeypatch.setattr(cm, "theme", {"elements": {"Ti": "not-a-color"}, "fallback_order": ["#112233", "bad", "#112233"]})
    monkeypatch.setattr(theme_manager, "JMOL_FALLBACK_COLORS", {})
    colors = cm.resolve_element_colors({el: 1.0 for el in HEA})
    assert len(colors) == 10
    assert all(HEX_RE.match(c) for c in colors.values()), colors
    assert len(set(colors.values())) == 10


def test_hea_structure_renders(out_dir):
    a = 3.1
    s = Structure(
        Lattice.from_parameters(a * 5, a * 2, a, 90, 90, 90),
        HEA,
        [[i / 5.0, (i // 5) / 2.0, 0.0] for i in range(10)],
    )
    colors = theme_manager.resolve_structure_colors(s)
    assert len(colors) == 10 and all(HEX_RE.match(c) for c in colors.values())
    assert _render(s, out_dir / "hea.png").exists()


# ---------------------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------------------
def test_three_half_sites_cluster_is_overlap():
    a = 5.0
    s = Structure(
        Lattice.cubic(a),
        [{"O": 0.5}, {"O": 0.5}, {"O": 0.5}, "Ti"],
        [
            [0.25, 0.25, 0.25],
            [0.25 + 0.2 / a, 0.25, 0.25],
            [0.25 + 0.4 / a, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )
    with pytest.raises(ScientificValidationError):
        validate_structure(s)


def test_sheared_lattice_self_image_overlap():
    s = Structure(Lattice([[3, 0, 0], [5.7, 0.3, 0], [0, 0, 3]]), ["Fe"], [[0, 0, 0]])
    with pytest.raises(ScientificValidationError):
        validate_structure(s)


def test_large_cell_custom_threshold():
    s = Structure(Lattice.cubic(2.5), ["Fe"], [[0, 0, 0]])
    s.make_supercell([16, 16, 16])  # 4096 atoms
    assert validate_min_distance(s) == pytest.approx(2.5, abs=1e-6)
    with pytest.raises(ScientificValidationError):
        validate_min_distance(s, threshold=3.0)


def test_empty_plane_palette_no_zero_division(monkeypatch):
    cm = theme_manager.ColorManager()
    monkeypatch.setattr(cm, "theme", {"planes": {"palette": []}})
    styles = cm.resolve_plane_styles(num_planes=3)
    assert len(styles) == 3


def test_oxidation_states_merged_in_colors():
    s = Structure(
        Lattice.cubic(4.0),
        ["Fe2+", "Fe3+", "O2-"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]],
    )
    colors = theme_manager.resolve_structure_colors(s)
    assert set(colors) == {"Fe", "O"}


def test_proxy_composition_drift_warns(caplog):
    s = Structure(
        Lattice.cubic(2.9),
        [{"Mn": 0.5, "Ga": 0.5}, "Ni"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    with caplog.at_level(logging.WARNING, logger="crystal_mcp"):
        to_ordered_proxy(s)
    assert any("changes the composition" in r.getMessage() for r in caplog.records)

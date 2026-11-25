"""Batch 21: several structures overlaid in one picture (shift / rotation / opacity)."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido import renderer
from shido.analyzer import compute_bonds
from conftest import SMALL_WINDOW, png_background_ratio


def _nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _water() -> Structure:
    return Structure(Lattice.cubic(6.0), ["O", "H", "H"],
                     [[3.0, 3.0, 3.0], [3.96, 3.0, 3.0], [2.76, 3.93, 3.0]], coords_are_cartesian=True)


class _Grab:
    """frame_hook that keeps the framing of the still."""

    def __call__(self, plotter, info):
        self.info = info


def _render(tmp_path, name, structure, **kw):
    grab = _Grab()
    _, b, p = compute_bonds(structure[0] if isinstance(structure, list) else structure)
    img = renderer.render_single_view(
        structure, tmp_path / name, camera_dir=[1, 0.36, 0.24], view_up=[0, 0, 1],
        render_bonds=b, polyhedra_envs=p, window_size=SMALL_WINDOW, frame_hook=grab, **kw,
    )
    return img, grab.info


def test_rotation_parsing():
    r = renderer.parse_overlay_rotation("0 0 1 90")
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(renderer.parse_overlay_rotation(None), np.eye(3))
    assert np.allclose(renderer.parse_overlay_rotation(matrix=np.eye(3).ravel()), np.eye(3))
    with pytest.raises(ValueError):
        renderer.parse_overlay_rotation("0 0 1")
    with pytest.raises(ValueError):
        renderer.parse_overlay_rotation(matrix=[2, 0, 0, 0, 1, 0, 0, 0, 1])  # not a rotation


def test_same_structure_overlays_exactly():
    s = _nacl()
    spec = renderer.normalize_overlays([s])[0]
    geom = renderer.overlay_geometry(s, spec)
    base = renderer._collect_atoms_to_draw(s)
    assert len(geom["atoms"]) == len(base)
    for (p, occ, r), (q, occ2, r2) in zip(geom["atoms"], base):
        assert np.allclose(p, q, atol=1e-9) and occ == occ2 and r == pytest.approx(r2)


def test_rotate_180_twice_is_identity():
    s = _water()
    for about in ("origin", "center"):
        spec = renderer.normalize_overlays({"structure": s, "rotate": "1 1 0 180", "about": about})[0]
        rot, t, c = renderer.overlay_transform(spec, s)
        pts = s.cart_coords
        once = renderer.apply_overlay_transform(pts, rot, t, c)
        assert not np.allclose(once, pts)
        twice = renderer.apply_overlay_transform(once, rot, t, c)
        assert np.allclose(twice, pts, atol=1e-9)


def test_frac_shift_uses_base_cell():
    base, mol = _nacl(), _water()
    spec = renderer.normalize_overlays({"structure": mol, "shift": [0, 0, 1], "shift_unit": "frac"})[0]
    _, t, _ = renderer.overlay_transform(spec, base)
    assert np.allclose(t, [0, 0, 5.64])


def test_render_identical_frame_and_shift_widens_it(tmp_path):
    s = _nacl()
    img0, info0 = _render(tmp_path, "plain.png", s)
    img1, info1 = _render(tmp_path, "same.png", [s, s])
    assert info1["parallel_scale"] == pytest.approx(info0["parallel_scale"], rel=1e-9)
    assert np.allclose(info1["cell_center"], info0["cell_center"])
    img2, info2 = _render(tmp_path, "shift.png", s, overlays=[{"structure": s, "shift": [0, 0, 12.0]}])
    # the framed box now spans 12 A more along c (view_up): centre moves up ~6 A, scale grows
    assert info2["parallel_scale"] > info0["parallel_scale"] + 4.0
    assert float(np.dot(info2["cell_center"] - info0["cell_center"], [0, 0, 1])) == pytest.approx(6.0, abs=0.5)
    for img in (img0, img1, img2):
        assert png_background_ratio(img) < 0.995


def test_render_molecule_on_surface_translucent(tmp_path):
    img, info = _render(
        tmp_path, "overlay.png", _nacl(),
        overlays=[{"structure": _water(), "shift": [0, 0, 5.0], "rotate": "0 0 1 45", "about": "center",
                   "opacity": 0.5, "show_cell": True},
                  {"structure": _water(), "shift": [0.5, 0.5, 1.6], "shift_unit": "frac", "color": "#8844cc"}],
    )
    assert png_background_ratio(img) < 0.995
    assert len(info["atoms"]) > len(info["atoms_drawn"])


def test_cli_and_specs(tmp_path):
    from shido import server

    f = tmp_path / "w.cif"
    _water().to(filename=str(f))
    args = server._build_arg_parser().parse_args(
        ["x.cif", "--overlay", str(f), "--overlay-shift", "0", "0", "5", "--overlay-rotate", "0 0 1 90",
         "--overlay-opacity", "0.5"]
    )
    opts = server._bulk_options_from_args(args)
    assert opts["overlays"] == [{"path": str(f), "shift": [0.0, 0.0, 5.0], "rotate": "0 0 1 90", "opacity": 0.5}]
    assert "overlays" not in server._bulk_options_from_args(server._build_arg_parser().parse_args(["x.cif"]))
    spec = renderer.normalize_overlays(opts["overlays"])[0]
    assert spec["structure"].composition.reduced_formula == "H2O"
    with pytest.raises(ValueError):
        server._overlay_specs([str(f)], shifts=[[0, 0, 1], [0, 0, 2]])

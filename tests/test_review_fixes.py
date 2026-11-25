"""Regression tests for the review findings (agy / Gemini Argon, 2026-11-25)."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido.core.editor import apply_operation, lattice_from_params
from shido.core.molecules import find_molecules, restrict_drawing
from shido.core.morphology import _expand, build_morphology
from shido.core.vesta_io import read_vesta, spaced_hm, write_vesta
from shido.core.volumetric import VolumetricGrid, model_density, volume_difference


def _nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


# 1 -- morphology: {hkl} expands with the crystal's point group, not the lattice holohedry
def test_morphology_uses_point_group_not_lattice():
    zns = Structure.from_spacegroup("F-43m", Lattice.cubic(5.41), ["Zn", "S"], [[0, 0, 0], [0.25] * 3])
    # zinc blende (-43m): {111} is a tetrahedron
    assert len(_expand(zns, (1, 1, 1), True)) == 4
    m = build_morphology(zns, "1 1 1:1.0")
    assert m.report()["n_faces"] == 4
    assert m.volume == pytest.approx(8 * np.sqrt(3), rel=1e-6)  # regular tetrahedron, inradius 1
    # pyrite (m-3): {210} is the pyritohedron (12 faces), not the tetrahexahedron (24)
    fes2 = Structure.from_spacegroup("Pa-3", Lattice.cubic(5.42), ["Fe", "S"], [[0, 0, 0], [0.385] * 3])
    assert len(_expand(fes2, (2, 1, 0), True)) == 12
    # centrosymmetric m-3m keeps the octahedron
    assert len(_expand(_nacl(), (1, 1, 1), True)) == 8
    # a primitive cell of zinc blende gives the same 4 planes (same normals)
    prim = zns.get_primitive_structure()
    assert build_morphology(prim, _prim_111(zns, prim)).report()["n_faces"] == 4


def _prim_111(conv: Structure, prim: Structure) -> str:
    # conventional (111) direction expressed as Miller indices of the primitive cell
    g = conv.lattice.reciprocal_lattice_crystallographic.get_cartesian_coords([1, 1, 1])
    h = prim.lattice.reciprocal_lattice_crystallographic.get_fractional_coords(g)
    h = np.rint(h / np.abs(h[np.abs(h) > 1e-6]).min()).astype(int)
    return " ".join(str(v) for v in h) + ":1.0"


# 2 -- volume difference: no volume-ratio scaling for same units; periodic origins
def test_volume_difference_same_unit_no_rescale_and_periodic_origin():
    la, lb = Lattice.cubic(4.0), Lattice.cubic(4.01)
    ga = VolumetricGrid(lattice=la, data=np.ones((4, 4, 4)), length_unit_volume=la.volume)
    gb = VolumetricGrid(lattice=lb, data=np.ones((4, 4, 4)), length_unit_volume=lb.volume)
    assert np.abs(volume_difference(ga, gb).data).max() == pytest.approx(0.0, abs=1e-12)
    gc = VolumetricGrid(lattice=la, data=np.ones((4, 4, 4)), origin_frac=np.array([1.0, 0, 0]),
                        length_unit_volume=la.volume)
    assert np.abs(volume_difference(ga, gc).data).max() == pytest.approx(0.0, abs=1e-12)


def test_volume_difference_bohr_vs_angstrom_still_converted():
    bohr = 0.529177210903
    la = Lattice.cubic(4.0)
    # a: per Bohr^3 (Cube), b: per Å^3 -> b is put on Bohr^3: x bohr^3
    ga = VolumetricGrid(lattice=la, data=np.ones((4, 4, 4)), length_unit_volume=la.volume / bohr**3)
    gb = VolumetricGrid(lattice=la, data=np.ones((4, 4, 4)), length_unit_volume=la.volume)
    assert volume_difference(ga, gb).data.max() == pytest.approx(1 - bohr**3, rel=1e-6)
    # LOCPOT (eV) is not per volume: no Bohr conversion
    gl = VolumetricGrid(lattice=la, data=np.ones((4, 4, 4)), fmt="locpot", length_unit_volume=la.volume)
    assert volume_difference(ga, gl).data.max() == pytest.approx(0.0, abs=1e-12)


# 3 -- cut-off planes with whole molecules: each periodic image by its own centre
def test_cutoff_plane_per_image_with_frac_range():
    h2 = Structure(Lattice.cubic(6.0), ["H", "H"], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 0.74 / 6]])
    info = {0: ([("H", 1.0)], 0.3), 1: ([("H", 1.0)], 0.3)}
    atoms, *_ = restrict_drawing(h2, [], None, None, cutoff_planes="1 0 0@1.0", whole_molecules=True,
                                 frac_range=[0, 2, 0, 1, 0, 1], site_info=info)
    xs = sorted(round(float(a[0][0]), 3) for a in atoms)
    assert xs == [3.0, 3.0]  # the image at x = 1.5 (9 Å) is beyond x <= 1


def test_cutoff_plane_cuts_extended_network_atom_by_atom():
    si = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
    mols = find_molecules(si)
    assert len(mols) == 1 and mols[0]["extended"]  # diamond: one covalent network
    info = {i: ([("Si", 1.0)], 1.0) for i in range(len(si))}
    atoms, *_ = restrict_drawing(si, [], None, None, cutoff_planes="1 0 0@0.3", whole_molecules=True,
                                 site_info=info)
    assert 0 < len(atoms) < len(si)
    fx = [si.lattice.get_fractional_coords(a[0])[0] for a in atoms]
    assert max(fx) <= 0.3 + 1e-6


# 4 -- VESTA: titles that look like keywords, setting suffix, duplicated STRUC entries
@pytest.mark.parametrize("title", ["H2O", "C60", "CELLP"])
def test_vesta_keyword_like_title(tmp_path, title):
    p = tmp_path / "t.vesta"
    write_vesta(_nacl(), p, title=title)
    s, info = read_vesta(p)
    assert info["title"] == title and len(s) == 8


def test_spaced_hm_drops_setting_suffix():
    assert spaced_hm("Fd-3m:2") == "F d -3 m"


def test_vesta_duplicate_equivalent_struc_entries(tmp_path):
    p = tmp_path / "d.vesta"
    write_vesta(_nacl(), p)
    text = p.read_text(encoding="utf-8")
    lines = text.split("\n")
    i = lines.index("STRUC")
    # write Na again at an equivalent position (0.5 0.5 0): occupancy must stay 1
    extra = ["  9 Na        Na9  1.0000   0.500000   0.500000   0.000000    4a     m-3m",
             "                            0.000000   0.000000   0.000000  0.00"]
    lines[i + 1:i + 1] = extra
    p.write_text("\n".join(lines), encoding="utf-8")
    s, _ = read_vesta(p)
    assert len(s) == 8 and s.is_ordered


# 5 -- editor: composition string for replace, short lattice lists, missing 'from'
def test_replace_with_composition_string():
    fe = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s, _ = apply_operation(fe.copy(), {"op": "replace", "site": 0, "to": "Mn0.5Fe0.5"})
    assert dict(s[0].species.get_el_amt_dict()) == pytest.approx({"Mn": 0.5, "Fe": 0.5})


def test_replace_missing_from_is_value_error():
    mix = Structure(Lattice.cubic(3), [{"Fe": 0.5, "Co": 0.5}], [[0, 0, 0]])
    with pytest.raises(ValueError, match="no Ni"):
        apply_operation(mix.copy(), {"op": "replace", "site": 0, "from": "Ni", "to": "Mn"})


@pytest.mark.parametrize("sg, vals, want", [
    ("P4/mmm", [3.0, 5.0], (3.0, 3.0, 5.0, 90, 90, 90)),
    (194, [3.0, 5.0], (3.0, 3.0, 5.0, 90, 90, 120)),
    (14, [3.0, 4.0, 5.0, 100.0], (3.0, 4.0, 5.0, 90, 100, 90)),
    (225, [5.64], (5.64, 5.64, 5.64, 90, 90, 90)),
    (62, [5.0, 6.0, 7.0], (5.0, 6.0, 7.0, 90, 90, 90)),
])
def test_lattice_short_lists_follow_crystal_system(sg, vals, want):
    assert lattice_from_params(sg, vals).parameters == pytest.approx(want)


# 6 -- rotation animation: labels of cut-off atoms are not drawn
def test_animation_labels_follow_cutoff(tmp_path, monkeypatch):
    from shido import renderer as R
    from shido.core import animation
    from shido.core.molecules import plane_keep_mask

    seen: list[list] = []
    orig = R._project_visible_labels

    def spy(plotter, candidates, atoms, *a, **kw):
        seen.append(list(candidates))
        return orig(plotter, candidates, atoms, *a, **kw)

    monkeypatch.setattr(R, "_project_visible_labels", spy)
    s = _nacl()
    animation.render_rotation_animation(
        s, tmp_path / "a.gif", camera_dir=[1.0, 0.36, 0.24], view_up=[0.0, 0.0, 1.0], frames=2,
        size=120, window_size=(300, 300), labels="element", cutoff_planes="1 0 0@0.3",
    )
    assert len(seen) >= 3  # the still + 2 frames
    for cands in seen:
        assert cands and plane_keep_mask(s, [c[0] for c in cands], "1 0 0@0.3").all()


def test_write_animation_rejects_bad_fps(tmp_path):
    from PIL import Image

    from shido.core.animation import write_animation

    with pytest.raises(ValueError, match="fps"):
        write_animation([Image.new("RGB", (8, 8))], tmp_path / "x.gif", "gif", fps=0)


# 8 -- find_molecules: a bond to the own image is listed once
def test_self_image_bonds_listed_once():
    po = Structure(Lattice.cubic(3.35), ["Po"], [[0, 0, 0]])
    bonds = find_molecules(po, tolerance=1.25)[0]["bonds"]
    keys = {tuple(int(v) for v in b[2]) for b in bonds}
    assert len(bonds) == len(keys) == 3  # +a, +b, +c (not also -a, -b, -c)
    for k in keys:
        assert tuple(-v for v in k) not in keys


# 9 -- model density: unknown element and negative B are clear errors
def test_model_density_input_checks():
    from pymatgen.core.periodic_table import DummySpecies

    with pytest.raises(ValueError, match="B"):
        model_density(_nacl(), grid=8, b_iso=-5.0)
    with pytest.raises(ValueError, match="B"):
        model_density(_nacl(), grid=8, b_iso={"Na": -1.0})
    dum = Structure(Lattice.cubic(4), [DummySpecies("X")], [[0, 0, 0]])
    with pytest.raises(ValueError, match="scattering"):
        model_density(dum, grid=8)


# 7 -- ellipsoid mesh: a left-handed axes matrix (eigh) must not turn the faces inside out
def test_ellipsoid_mesh_faces_outward_for_left_handed_axes():
    from shido.renderer import build_ellipsoid_mesh

    for axes in (np.eye(3), np.diag([1.0, 1.0, -1.0])):
        m = build_ellipsoid_mesh(np.zeros(3), np.array([1.0, 2.0, 3.0]), axes)
        n = m.compute_normals(cell_normals=False, auto_orient_normals=False)
        pts, nn = np.asarray(n.points), np.asarray(n.point_data["Normals"])
        assert np.all(np.sum(pts * nn, axis=1) > 0)

"""Batch 4: measure_geometry (distance, angle, dihedral, interplanar angle, best plane)."""

from __future__ import annotations

import math

import pytest
from pymatgen.core import Lattice, Structure

from shido.core.geometry import (
    dihedral,
    interplanar_angle,
    measure_geometry,
    resolve_site_index,
)

A_SI = 5.431
A_NACL = 5.640
A_CU = 3.615


def _si() -> Structure:
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    coords = fcc + [(x + 0.25, y + 0.25, z + 0.25) for x, y, z in fcc]
    return Structure(Lattice.cubic(A_SI), ["Si"] * 8, coords)


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(A_NACL), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _cu() -> Structure:
    coords = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    return Structure(Lattice.cubic(A_CU), ["Cu"] * 4, coords)


def test_si_tetrahedral_angle_with_explicit_images():
    # site 4 = (1/4,1/4,1/4); site 7 = (3/4,3/4,1/4) shifted by (-1,-1,0) -> (-1/4,-1/4,1/4)
    res = measure_geometry(_si(), atoms=["4", "0", "7@-1,-1,0"])
    assert res["angle_deg"] == pytest.approx(math.degrees(math.acos(-1 / 3)), abs=1e-3)
    assert res["angle_deg"] == pytest.approx(109.47, abs=0.01)
    assert res["distances"][0] == pytest.approx(A_SI * math.sqrt(3) / 4, abs=1e-6)


def test_si_nearest_image_is_chosen_without_at():
    # No image given: site 7 should be taken nearest to site 0 -> same 109.47 deg angle.
    res = measure_geometry(_si(), atoms=["4", "0", "7"])
    assert res["atoms"][2]["image"] == [-1, -1, 0]
    assert res["angle_deg"] == pytest.approx(109.4712, abs=1e-3)


def test_nacl_distances_and_images():
    s = _nacl()
    na = resolve_site_index(s, "Na1")
    assert list(s[na].frac_coords) == [0, 0, 0]
    # Minimum-image distances from the origin Na to each Cl site: a/2 (x3) and a*sqrt(3)/2 (x1).
    d = sorted(measure_geometry(s, atoms=[str(na), f"Cl{k}"])["distance"] for k in range(1, 5))
    assert d[:3] == pytest.approx([A_NACL / 2] * 3)
    assert d[3] == pytest.approx(A_NACL * math.sqrt(3) / 2)
    # Same Na with image (0,0,1): distance = a.
    assert measure_geometry(s, atoms=[f"{na}", f"{na}@0,0,1"])["distance"] == pytest.approx(A_NACL)
    # Bracket / space syntax also works.
    assert measure_geometry(s, atoms=["Na1", "Na1@(1 1 0)"])["distance"] == pytest.approx(
        A_NACL * math.sqrt(2)
    )


def test_cu_nearest_neighbour_and_angle():
    # fcc Cu: nearest neighbour a/sqrt(2); the angle between two neighbours in one face is 60 deg.
    s = _cu()
    res = measure_geometry(s, atoms=["Cu2@0,0,0", "Cu1", "Cu3@0,0,0"])
    assert res["distances"][0] == pytest.approx(A_CU / math.sqrt(2), abs=1e-6)
    assert res["angle_deg"] == pytest.approx(60.0, abs=1e-6)


def test_interplanar_angles_cubic():
    lat = Lattice.cubic(4.0)
    assert interplanar_angle(lat, "1 0 0", "1 1 0") == pytest.approx(45.0)
    assert interplanar_angle(lat, "100", "111") == pytest.approx(math.degrees(math.acos(1 / math.sqrt(3))))
    res = measure_geometry(_nacl(), planes=["(1 0 0)", "-1 0 0"])
    assert res["interplanar_angle_deg"] == pytest.approx(180.0)
    assert res["interplanar_acute_angle_deg"] == pytest.approx(0.0)


def test_interplanar_angle_hexagonal_uses_reciprocal_lattice():
    # Hexagonal: (100) and (010) normals are a* and b*, 60 deg apart (not the 120 deg of a, b).
    lat = Lattice.hexagonal(3.0, 5.0)
    assert interplanar_angle(lat, "1 0 0", "0 1 0") == pytest.approx(60.0)


def test_dihedral_planar_cis_trans():
    assert dihedral((1, 1, 0), (0, 0, 0), (1, 0, 0), (2, 1, 0)) == pytest.approx(0.0, abs=1e-9)
    assert dihedral((1, 1, 0), (0, 0, 0), (1, 0, 0), (2, -1, 0)) == pytest.approx(180.0)
    assert abs(dihedral((1, 1, 0), (0, 0, 0), (1, 0, 0), (1, 0, 1))) == pytest.approx(90.0)


def test_dihedral_in_cubic_cell_face():
    # Four Cu atoms on one cube face (z = 0 plane through face centre) are coplanar -> 0 or 180.
    s = Structure(Lattice.cubic(A_CU), ["Cu"], [[0, 0, 0]])
    res = measure_geometry(s, atoms=["0@1,0,0", "0@0,0,0", "0@0,1,0", "0@1,1,0"])
    assert abs(res["dihedral_deg"]) == pytest.approx(0.0, abs=1e-9)
    res2 = measure_geometry(s, atoms=["0@0,1,0", "0@0,0,0", "0@1,0,0", "0@1,-1,0"])
    assert abs(res2["dihedral_deg"]) == pytest.approx(180.0)


def test_best_plane_nacl_111_layer():
    # Na atoms of a (111) layer of NaCl: zero deviation, normal parallel to [111].
    s = _nacl()
    na_sites = [i for i, site in enumerate(s) if site.specie.symbol == "Na"]
    # In-cell Na at (0,0,0) and the three face centres (1/2,1/2,0) etc. lie on x+y+z = 0 or 1.
    on_plane = []
    for i in na_sites:
        f = s[i].frac_coords
        if abs(sum(f) - 1.0) < 1e-6:
            on_plane.append(str(i))
    on_plane.append(f"{na_sites[0]}@1,0,0")  # (1,0,0) image of the origin Na
    res = measure_geometry(s, best_plane_atoms=[f"{p}" for p in on_plane])
    bp = res["best_plane"]
    assert bp["rms_deviation"] == pytest.approx(0.0, abs=1e-9)
    assert bp["normal_hkl_approx"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)


def test_best_plane_deviation_value():
    # Four points at z = +-0.1 alternately: plane z = 0, deviations +-0.1, rms 0.1.
    s = Structure(Lattice.cubic(10.0), ["H"] * 4, [[0, 0, 0.01], [0.1, 0, -0.01], [0.1, 0.1, 0.01], [0, 0.1, -0.01]])
    bp = measure_geometry(s, best_plane_atoms=["0", "1", "2", "3"])["best_plane"]
    assert bp["rms_deviation"] == pytest.approx(0.1, abs=1e-6)
    assert bp["normal_cart"] == pytest.approx([0, 0, 1], abs=1e-6)


def test_errors():
    s = _nacl()
    with pytest.raises(ValueError):
        measure_geometry(s)
    with pytest.raises(ValueError):
        measure_geometry(s, atoms=["Na1"])
    with pytest.raises(ValueError):
        measure_geometry(s, atoms=["Xx9", "Na1"])
    with pytest.raises(ValueError):
        measure_geometry(s, planes=["0 0 0", "1 0 0"])


def test_cli_measure(tmp_path, capsys):
    from shido.server import main

    cif = tmp_path / "nacl.cif"
    _nacl().to(filename=str(cif))
    rc = main(["measure", str(cif), "--atoms", "Na1", "Na1@0,0,1", "--planes", "1 0 0", "1 1 0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "5.64" in out
    assert "45" in out

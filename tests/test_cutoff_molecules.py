"""Batch 23: cut-off planes, molecules (connected components) and hydrogen bonds."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido import renderer
from shido.analyzer import compute_bonds
from conftest import SMALL_WINDOW, png_background_ratio
from shido.core.molecules import (
    find_hbonds,
    find_molecules,
    molecule_atoms,
    parse_cutoff_plane,
    parse_hbond_spec,
    plane_keep_mask,
    restrict_drawing,
    write_hbonds_csv,
    write_molecules_csv,
)


def _si() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])


def _water_dimer(shift: float = -2.5) -> Structure:
    """Two H2O in a 10 Å cell, one linear O-H...O bond (H...O 1.90 Å, D...A 2.86 Å).
    ``shift`` moves the first molecule across the cell edge (unwrapping is tested)."""
    a, b = 0.96 * math.cos(math.radians(52.5)), 0.96 * math.sin(math.radians(52.5))
    hoh = math.radians(104.5)
    o1 = np.array([2.0, 5.0, 5.0])
    h1 = o1 + [0.96, 0, 0]
    h2 = o1 + [0.96 * math.cos(hoh), 0.96 * math.sin(hoh), 0]
    o2 = o1 + [2.86, 0, 0]
    h3, h4 = o2 + [a, 0, b], o2 + [a, 0, -b]
    carts = np.array([o1, h1, h2, o2, h3, h4]) + [shift, 0, 0]
    return Structure(Lattice.cubic(10.0), ["O", "H", "H", "O", "H", "H"], carts,
                     coords_are_cartesian=True, to_unit_cell=True)


def _co2() -> Structure:
    """Pa-3 CO2 (dry ice, a = 5.624 Å): 4 molecules per cell."""
    return Structure.from_spacegroup(
        "Pa-3", Lattice.cubic(5.624), ["C", "O"], [[0, 0, 0], [0.1185, 0.1185, 0.1185]]
    )


def test_parse_specs():
    hkl, d = parse_cutoff_plane("1 1 1@0.5")
    assert list(hkl) == [1, 1, 1] and d == 0.5
    assert parse_cutoff_plane("0 0 1")[1] == 0.5
    spec = parse_hbond_spec("O-H...O:2.5:120")
    assert spec["donor"] == {"O"} and spec["acceptor"] == {"O"} and spec["dmax"] == 2.5
    with pytest.raises(ValueError):
        parse_hbond_spec("OH O")
    with pytest.raises(ValueError):
        parse_cutoff_plane("0 0 0@1")


def test_si_cut_111_keeps_one_side():
    si = _si()
    atoms = renderer._collect_atoms_to_draw(si)
    _, bonds, polys = compute_bonds(si)
    kept, kb, _, _ = restrict_drawing(si, atoms, bonds, polys, cutoff_planes=["1 1 1@1.5"])
    assert 0 < len(kept) < len(atoms)
    frac = si.lattice.get_fractional_coords(np.array([a[0] for a in kept]))
    assert np.all(frac.sum(axis=1) <= 1.5 + 1e-3)
    dropped = [a for a in atoms if not plane_keep_mask(si, [a[0]], ["1 1 1@1.5"])[0]]
    assert len(kept) + len(dropped) == len(atoms)
    for s, e, _, _ in kb:  # every bond has both ends on the kept side
        assert plane_keep_mask(si, [s, e], ["1 1 1@1.5"]).all()


def test_water_molecules_and_hbond(tmp_path):
    s = _water_dimer()
    mols = find_molecules(s)
    assert len(mols) == 2
    assert sorted(m["formula"] for m in mols) == ["H2O", "H2O"]
    assert not any(m["extended"] for m in mols)
    for m in mols:  # unwrapped: every O-H within 1.0 Å
        carts = [s.lattice.get_cartesian_coords(s[k].frac_coords + o) for k, o in zip(m["sites"], m["offsets"])]
        o = carts[[s[k].specie.symbol for k in m["sites"]].index("O")]
        assert max(np.linalg.norm(c - o) for c in carts) < 1.0
        assert np.all((m["centroid_frac"] >= 0) & (m["centroid_frac"] < 1))
    hb = find_hbonds(s, ["O-H...O:2.5:120"])
    assert len(hb) == 1
    assert hb[0]["d_HA"] == pytest.approx(1.90, abs=1e-3)
    assert hb[0]["d_DA"] == pytest.approx(2.86, abs=1e-3)
    assert hb[0]["angle_DHA"] == pytest.approx(180.0, abs=0.1)
    assert find_hbonds(s, ["O-H...O:1.8:120"]) == []
    p1 = write_molecules_csv(s, mols, tmp_path / "molecules.csv")
    p2 = write_hbonds_csv(s, hb, tmp_path / "hbonds.csv")
    assert len(list(csv.DictReader(p1.open(encoding="utf-8-sig")))) == 2
    assert len(list(csv.DictReader(p2.open(encoding="utf-8-sig")))) == 1


def test_co2_and_si_components():
    mols = find_molecules(_co2())
    assert len(mols) == 4 and {m["formula"] for m in mols} == {"CO2"}
    si = find_molecules(_si())
    assert len(si) == 1 and si[0]["extended"]


def test_whole_molecule_atoms_in_range():
    s = _water_dimer()
    mols = find_molecules(s)
    assert len(molecule_atoms(s, mols)) == 6
    assert len(molecule_atoms(s, mols, [0, 2, 0, 1, 0, 1])) == 12


def test_render_cut_molecules_hbond(tmp_path):
    si = _si()
    _, b, p = compute_bonds(si)
    kw = dict(camera_dir=[1, 1, 1], view_up=[0, 0, 1], window_size=SMALL_WINDOW)
    full = renderer.render_single_view(si, tmp_path / "si_full.png", render_bonds=b, polyhedra_envs=p, **kw)
    cut = renderer.render_single_view(si, tmp_path / "si_cut.png", render_bonds=b, polyhedra_envs=p,
                                      cutoff_planes=["1 1 1@1.5"], **kw)
    assert png_background_ratio(cut) < 0.995 and png_background_ratio(full) < 0.995
    w = _water_dimer()
    _, wb, wp = compute_bonds(w)
    img = renderer.render_single_view(
        w, tmp_path / "water.png", render_bonds=wb, polyhedra_envs=wp, whole_molecules=True,
        hbonds=["O-H...O:2.5:120"], camera_dir=[0, 0, 1], view_up=[0, 1, 0], window_size=SMALL_WINDOW,
    )
    assert png_background_ratio(img) < 0.995


def test_bulk_analysis_writes_csvs(tmp_path):
    from shido import server

    res = server.run_bulk_analysis(_water_dimer(), tmp_path, molecules=True, hbonds=["O-H...O:2.5:120"],
                                   whole_molecules=True)
    assert res["num_molecules"] == 2 and res["num_hbonds"] == 1
    assert res["molecules_csv"].endswith("molecules.csv") and res["hbonds_csv"].endswith("hbonds.csv")

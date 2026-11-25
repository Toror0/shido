"""3D export: printable STL (mm) and offline HTML viewer."""

from __future__ import annotations

import re

import numpy as np
import pytest
import pyvista as pv
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator

from shido.analyzer import build_ni2mnga_structure
from shido.core.validator import ScientificValidationError
from shido.export3d import build_print_model, export_html, export_stl, split_stl_paths


@pytest.fixture(scope="module")
def ni2mnga() -> Structure:
    return build_ni2mnga_structure()


@pytest.fixture(scope="module")
def slab_111(ni2mnga: Structure) -> Structure:
    gen = SlabGenerator(ni2mnga, (1, 1, 1), 8.0, 15.0, center_slab=False)
    return gen.get_slabs()[0].get_orthogonal_c_slab()


def test_stl_closed_and_sized(ni2mnga, out_dir):
    path = export_stl(ni2mnga, out_dir / "ni2mnga.stl")
    assert path.exists() and path.stat().st_size > 0
    mesh = pv.read(path)
    assert mesh.n_cells > 0
    assert mesh.n_open_edges == 0
    b = np.asarray(mesh.bounds).reshape(3, 2)
    size = b[:, 1] - b[:, 0]
    # 1 Å = 10 mm: a = 5.8 Å cell plus one atom radius each side.
    a_mm = ni2mnga.lattice.a * 10.0
    assert np.all(size >= a_mm) and np.all(size <= a_mm + 30.0)
    assert np.all(size <= 180.0)
    assert b[2, 0] == pytest.approx(0.0, abs=1e-3)  # sits on the bed


def test_stl_bonds_thick_enough_and_connected(ni2mnga):
    model = build_print_model(ni2mnga)
    assert model.bond_radius_mm >= 1.2
    assert model.n_bonds > 0
    assert model.n_removed_floating == 0


def test_stl_shrinks_to_max_size(ni2mnga, out_dir, caplog):
    model = build_print_model(ni2mnga, max_size_mm=40.0)
    assert max(model.size_mm) <= 40.0 + 1e-6
    assert model.scale_mm_per_angstrom < 10.0
    assert any("shrunk" in w for w in model.warnings)
    mesh = pv.read(export_stl(ni2mnga, out_dir / "small.stl", max_size_mm=40.0))
    b = np.asarray(mesh.bounds).reshape(3, 2)
    assert np.all(b[:, 1] - b[:, 0] <= 40.0 + 1e-3)


def test_stl_no_size_limit_by_default(ni2mnga):
    big = build_print_model(ni2mnga, scale=40.0)
    assert big.scale_mm_per_angstrom == pytest.approx(40.0)
    assert max(big.size_mm) > 180.0


def test_stl_split_by_element(ni2mnga, out_dir):
    main = export_stl(ni2mnga, out_dir / "split.stl", split_by_element=True)
    elems = sorted(el.symbol for el in ni2mnga.composition.elements)
    per_elem = sorted(p for p in out_dir.glob("split_*.stl") if p.stem != "split_bonds")
    assert [p.stem.split("_", 1)[1] for p in per_elem] == elems
    for p in per_elem:
        assert pv.read(p).n_open_edges == 0
    assert (out_dir / "split_bonds.stl").exists()
    assert set(split_stl_paths(main, ni2mnga)) == set(per_elem) | {out_dir / "split_bonds.stl"}
    # All parts share one coordinate system: their union spans the main model.
    b_main = np.asarray(pv.read(main).bounds)
    b_parts = np.asarray([pv.read(p).bounds for p in out_dir.glob("split_*.stl")])
    assert b_parts[:, 0::2].min(axis=0) == pytest.approx(b_main[0::2], abs=1e-3)
    assert b_parts[:, 1::2].max(axis=0) == pytest.approx(b_main[1::2], abs=1e-3)


def test_stl_slab_size_follows_atoms_not_vacuum(slab_111, out_dir):
    c_len = slab_111.lattice.c
    model = build_print_model(slab_111)
    assert any("Vacuum" in w for w in model.warnings)
    z_atoms = np.asarray(slab_111.cart_coords)[:, 2]
    # Slab thickness (atoms) is well below the cell height (atoms + vacuum).
    thickness = float(z_atoms.max() - z_atoms.min())
    assert thickness < c_len - 10.0
    assert model.size_mm[2] < (thickness + 3.0) * 10.0
    assert model.size_mm[2] > thickness * 10.0
    assert model.n_removed_floating == 0
    mesh = pv.read(export_stl(slab_111, out_dir / "slab.stl"))
    assert mesh.n_open_edges == 0
    b = np.asarray(mesh.bounds).reshape(3, 2)
    assert b[2, 1] - b[2, 0] == pytest.approx(model.size_mm[2], abs=0.05)


def test_stl_removes_floating_atoms(ni2mnga):
    # No bonds: only atoms touching the unit-cell frame stay attached.
    model = build_print_model(ni2mnga, render_bonds=[])
    assert model.n_bonds == 0
    assert model.n_removed_floating > 0
    assert any("floating" in w for w in model.warnings)


def test_stl_rejects_bad_input(out_dir):
    bad = Structure(Lattice.cubic(3.0), ["Ni", "Ni"], [[0, 0, 0], [0.01, 0, 0]])
    with pytest.raises(ScientificValidationError):
        export_stl(bad, out_dir / "bad.stl")
    assert not (out_dir / "bad.stl").exists()


@pytest.mark.parametrize("which", ["bulk", "slab"])
def test_html_offline(which, ni2mnga, slab_111, out_dir):
    s = ni2mnga if which == "bulk" else slab_111
    path = export_html(s, out_dir / f"{which}.html")
    text = path.read_text(encoding="utf-8")
    assert path.stat().st_size < 5 * 1024 * 1024
    assert not re.search(r"<script[^>]+src=", text, flags=re.I)
    assert "http://" not in text and "https://" not in text
    assert "<link" not in text.lower()
    assert "Content-Security-Policy" in text
    for el in ("Ni", "Mn", "Ga"):
        assert f'"element":"{el}"' in text
    assert "webgl2" in text

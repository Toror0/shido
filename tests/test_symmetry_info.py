"""Batch 5: symmetry information table (get_symmetry_info / `server.py symmetry`)."""

from __future__ import annotations

import csv
import json

import pytest
from pymatgen.core import Lattice, Structure

from shido.analyzer import build_ni2mnga_structure
from shido.core.symmetry_info import (
    compute_symmetry_info,
    operation_to_xyz,
    write_symmetry_report,
)


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _si() -> Structure:
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    coords = fcc + [(x + 0.25, y + 0.25, z + 0.25) for x, y, z in fcc]
    return Structure(Lattice.cubic(5.431), ["Si"] * 8, coords)


def _orbit(info, element):
    return next(o for o in info["orbits"] if o["element"] == element)


def test_nacl_fm3m_general_positions_and_sites():
    info = compute_symmetry_info(_nacl())
    assert info["spacegroup_number"] == 225
    assert info["spacegroup_symbol"] == "Fm-3m"
    # ITA: Fm-3m general position 192e = 48 point operations x 4 centring vectors.
    assert info["n_operations"] == 192
    assert info["n_operations_standard"] == 192
    na, cl = _orbit(info, "Na"), _orbit(info, "Cl")
    assert (na["wyckoff"], na["site_symmetry"], na["n_in_cell"]) == ("4a", "m-3m", 4)
    assert (cl["wyckoff"], cl["site_symmetry"], cl["n_in_cell"]) == ("4b", "m-3m", 4)
    xyz = {op["xyz"] for op in info["operations"]}
    assert {"x,y,z", "-x,-y,-z", "z,x,y", "x,y+1/2,z+1/2"} <= xyz
    assert len(xyz) == 192


def test_nacl_primitive_keeps_standard_multiplicity():
    info = compute_symmetry_info(_nacl().get_primitive_structure())
    assert info["n_operations"] == 48
    na = _orbit(info, "Na")
    assert na["n_in_cell"] == 1
    assert na["multiplicity"] == 4 and na["wyckoff"] == "4a"


def test_si_fd3m_8a():
    info = compute_symmetry_info(_si())
    assert info["spacegroup_number"] == 227
    assert info["spacegroup_symbol"] == "Fd-3m"
    assert len(info["orbits"]) == 1
    o = info["orbits"][0]
    assert (o["wyckoff"], o["site_symmetry"], o["n_in_cell"]) == ("8a", "-43m", 8)


def test_ni2mnga_l21_8c_4a_4b():
    info = compute_symmetry_info(build_ni2mnga_structure(5.82))
    assert info["spacegroup_number"] == 225
    got = {o["element"]: (o["wyckoff"], o["site_symmetry"]) for o in info["orbits"]}
    assert got == {"Ni": ("8c", "-43m"), "Mn": ("4a", "m-3m"), "Ga": ("4b", "m-3m")}


def test_operation_to_xyz():
    assert operation_to_xyz([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [0.5, 0, 0.75]) == "-y+1/2,x,z+3/4"
    assert operation_to_xyz([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1.0, 0, -0.5]) == "x,y,z+1/2"


def test_write_report_files(tmp_path):
    info = compute_symmetry_info(_nacl())
    paths = write_symmetry_report(info, tmp_path)
    data = json.loads(open(paths["json"], encoding="utf-8").read())
    assert data["spacegroup_symbol"] == "Fm-3m"
    with open(paths["sites_csv"], encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8 and rows[0]["site_symmetry"] == "m-3m"
    with open(paths["operations_csv"], encoding="utf-8") as fh:
        assert sum(1 for _ in csv.DictReader(fh)) == 192


def test_cli_symmetry(tmp_path, capsys):
    from shido.server import main

    cif = tmp_path / "nacl.cif"
    _nacl().to(filename=str(cif))
    out = tmp_path / "out"
    assert main(["symmetry", str(cif), "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Fm-3m" in printed
    assert (out / "symmetry_sites.csv").is_file()
    assert main(["symmetry", str(tmp_path / "missing.cif")]) == 1

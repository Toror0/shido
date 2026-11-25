"""Batch 8: data export (XYZ / P1 CIF / XTL / PDB / VRML / multi-structure CIF)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser

from shido import server
from shido.core.exporters import (
    export_structures,
    parse_formats,
    pdb_orthogonal_matrix,
    write_multi_cif,
)


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _mg() -> Structure:
    return Structure.from_spacegroup(
        "P6_3/mmc", Lattice.hexagonal(3.209, 5.211), ["Mg"], [[1 / 3, 2 / 3, 0.25]]
    )


def _b2_ni2mnga() -> Structure:
    return Structure(
        Lattice.cubic(5.82),
        ["Ni"] * 8 + [{"Mn": 0.5, "Ga": 0.5}] * 8,
        [[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)]
        + [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
           [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0]],
    )


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    n = int(lines[0])
    rows = [ln.split() for ln in lines[2:2 + n]]
    assert len(rows) == n
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def _read_pdb(path: Path) -> tuple[tuple[float, ...], list[str], np.ndarray, list[float]]:
    cell = None
    els, xyz, occ = [], [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("CRYST1"):
            cell = tuple(float(ln[i:j]) for i, j in ((6, 15), (15, 24), (24, 33), (33, 40), (40, 47), (47, 54)))
        if ln.startswith(("ATOM  ", "HETATM")):
            xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
            occ.append(float(ln[54:60]))
            els.append(ln[76:78].strip().capitalize())
    assert cell is not None
    return cell, els, np.array(xyz), occ


def _read_xtl(path: Path) -> tuple[list[float], list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cell = [float(v) for v in lines[lines.index("CELL") + 1].split()]
    i0 = lines.index("ATOMS") + 2
    rows = [ln.split() for ln in lines[i0:lines.index("EOF")]]
    return cell, [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def test_parse_formats():
    assert parse_formats(None) == ["xyz", "p1", "xtl", "pdb"]
    assert parse_formats("xyz, .PDB,vrml") == ["xyz", "pdb", "wrl"]
    assert parse_formats(["all"])[-1] == "multicif"
    with pytest.raises(ValueError):
        parse_formats("mol2")


def test_pdb_orthogonal_matrix_matches_lattice():
    m = pdb_orthogonal_matrix(4.0, 5.0, 6.0, 80.0, 95.0, 110.0)
    lat = Lattice(m)
    assert lat.abc == pytest.approx((4.0, 5.0, 6.0))
    assert lat.angles == pytest.approx((80.0, 95.0, 110.0))
    assert m[0, 1] == m[0, 2] == m[1, 2] == 0.0


@pytest.mark.parametrize("make", [_nacl, _mg])
def test_formats_round_trip(make, tmp_path):
    s = make()
    res = export_structures([("x", s)], "xyz,p1,xtl,pdb", tmp_path)
    files = res["structures"][0]["files"]
    want = Counter(site.specie.symbol for site in s)

    els, xyz = _read_xyz(Path(files["xyz"]))
    assert Counter(els) == want
    assert np.allclose(xyz, s.cart_coords, atol=1e-6)

    p1 = CifParser(files["p1"]).parse_structures(primitive=False)[0]
    assert len(p1) == len(s)
    assert p1.composition.almost_equals(s.composition)
    assert "_symmetry_space_group_name_H-M   'P 1'" in Path(files["p1"]).read_text()

    cell, els, frac = _read_xtl(Path(files["xtl"]))
    assert cell == pytest.approx([*s.lattice.abc, *s.lattice.angles], abs=1e-5)
    assert Counter(els) == want
    assert np.allclose(frac, s.frac_coords, atol=1e-5)

    cell, els, pxyz, occ = _read_pdb(Path(files["pdb"]))
    assert cell[:3] == pytest.approx(s.lattice.abc, abs=1e-3)
    assert cell[3:] == pytest.approx(s.lattice.angles, abs=1e-2)
    assert Counter(els) == want
    assert occ == pytest.approx([1.0] * len(s))
    m = pdb_orthogonal_matrix(*s.lattice.abc, *s.lattice.angles)
    assert np.allclose(pxyz, s.frac_coords @ m, atol=2e-3)


def test_disordered_pdb_keeps_occupancy_and_xyz_keeps_composition(tmp_path):
    s = _b2_ni2mnga()
    res = export_structures([("b2", s)], "xyz,pdb,p1", tmp_path)
    files = res["structures"][0]["files"]
    _, els, _, occ = _read_pdb(Path(files["pdb"]))
    assert len(els) == 8 + 16  # 8 Ni + 8 sites x (Mn, Ga)
    assert sum(o for e, o in zip(els, occ) if e == "Mn") == pytest.approx(4.0)
    els, _ = _read_xyz(Path(files["xyz"]))
    assert Counter(els) == {"Ni": 8, "Mn": 4, "Ga": 4}
    assert any("ordered proxy" in w for w in res["warnings"])
    p1 = CifParser(files["p1"]).parse_structures(primitive=False)[0]
    assert p1.composition.almost_equals(s.composition)


def test_multi_cif_round_trip(tmp_path):
    entries = [("NaCl", _nacl()), ("Mg", _mg()), ("NaCl", _nacl())]
    path = write_multi_cif(entries, tmp_path / "all.cif")
    text = path.read_text()
    assert text.count("\ndata_") + text.startswith("data_") == 3
    assert "data_NaCl_2" in text
    parsed = CifParser(str(path)).parse_structures(primitive=False)
    assert [len(x) for x in parsed] == [8, 2, 8]


def test_vrml_has_colours(tmp_path):
    res = export_structures([("nacl", _nacl())], ["wrl"], tmp_path)
    wrl = Path(res["structures"][0]["files"]["wrl"])
    text = wrl.read_text(encoding="utf-8", errors="replace")
    assert text.startswith("#VRML V2.0")
    assert text.count("diffuseColor") >= 2  # Na and Cl (+ bonds / frame)


def test_cli_export(tmp_path):
    cif_a = tmp_path / "nacl.cif"
    cif_b = tmp_path / "mg.cif"
    _nacl().to(filename=str(cif_a))
    _mg().to(filename=str(cif_b))
    out = tmp_path / "out"
    rc = server.main(["export", str(cif_a), str(cif_b), "--formats", "xyz,pdb,multicif",
                      "--out-dir", str(out)])
    assert rc == 0
    assert (out / "nacl.xyz").is_file() and (out / "mg.pdb").is_file()
    parsed = CifParser(str(out / "structures_all.cif")).parse_structures(primitive=False)
    assert len(parsed) == 2
    assert server.main(["export", str(cif_a), "--formats", "mol2"]) == 1


def test_mcp_impl_single_file_default_dir(tmp_path):
    cif = tmp_path / "mg.cif"
    _mg().to(filename=str(cif))
    res = server._export_structure_impl(str(cif), "p1,multicif,wrl", None)
    run = Path(res["output_dir"])
    assert run.parent == (tmp_path / "出力").resolve() and run.name.endswith("_export_mg")
    assert Path(res["structures"][0]["files"]["p1"]).parent == run / "構造"
    assert Path(res["structures"][0]["files"]["wrl"]).parent == run / "3D"
    assert Path(res["index_html"]).is_file()
    assert res["multicif"].endswith("mg_all.cif")

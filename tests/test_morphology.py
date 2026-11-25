"""Batch 20: crystal shape (morphology) from {hkl} and central distances."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from conftest import png_background_ratio
from shido.core.morphology import build_morphology, parse_faces


def _nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _mg() -> Structure:
    return Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(3.21, 5.21), ["Mg"], [[1 / 3, 2 / 3, 0.25]])


def test_parse_faces_forms():
    assert parse_faces("1 0 0:1.0,1 1 1:0.9") == [((1, 0, 0), 1.0), ((1, 1, 1), 0.9)]
    assert parse_faces(["100:2"]) == [((1, 0, 0), 2.0)]
    assert parse_faces([{"hkl": [1, 0, -1, 0], "distance": 1}]) == [((1, 0, 0), 1.0)]
    with pytest.raises(ValueError):
        parse_faces("1 0 0:-1")


def test_cube_from_100():
    m = build_morphology(_nacl(), "1 0 0:1.0")
    areas = [f["area"] for f in m.visible_faces]
    assert len(areas) == 6 and len(m.vertices) == 8
    assert np.allclose(areas, 4.0)
    assert m.volume == pytest.approx(8.0)


def test_octahedron_from_111():
    m = build_morphology(_nacl(), "1 1 1:1.0")
    assert len(m.visible_faces) == 8 and len(m.vertices) == 6
    # regular octahedron with inradius 1: edge a = sqrt(6), volume = sqrt(2)/3 a^3
    a = math.sqrt(6.0)
    assert m.volume == pytest.approx(math.sqrt(2) / 3 * a**3)
    assert np.allclose([f["area"] for f in m.visible_faces], math.sqrt(3) / 4 * a**2)


def test_cuboctahedral_mix_and_hidden_family():
    m = build_morphology(_nacl(), "1 0 0:1.0,1 1 1:0.9")
    rep = m.report()
    assert rep["n_faces"] == 14
    assert sum(f["area_fraction"] for f in rep["families"]) == pytest.approx(1.0)
    far = build_morphology(_nacl(), "1 0 0:1.0,1 1 1:2.0")
    fam111 = far.report()["families"][1]
    assert fam111["n_on_crystal"] == 0 and fam111["area"] == 0.0


def test_hexagonal_prism():
    m = build_morphology(_mg(), "1 0 0:1.0,0 0 1:1.5")
    fams = m.report()["families"]
    assert len(m.visible_faces) == 8
    assert fams[0]["n_on_crystal"] == 6 and fams[1]["n_on_crystal"] == 2
    # hexagon with inradius 1: area 2*sqrt(3); prism height 3
    assert fams[1]["area"] == pytest.approx(2 * 2 * math.sqrt(3))
    assert m.volume == pytest.approx(2 * math.sqrt(3) * 3.0)


def test_unbounded_raises():
    with pytest.raises(ValueError):
        build_morphology(_mg(), "0 0 1:1.0")


def test_cli_writes_png_csv_stl_html(tmp_path: Path):
    from shido import server

    cif = tmp_path / "NaCl.cif"
    _nacl().to(filename=str(cif))
    out = tmp_path / "out"
    rc = server.main(["morphology", str(cif), "--faces", "1 0 0:1.0,1 1 1:0.9", "--out-dir", str(out)])
    assert rc == 0
    png = out / "morphology.png"
    assert png.is_file() and png_background_ratio(png) < 0.97
    with (out / "morphology_faces.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 14
    import pyvista as pv

    stl = pv.read(str(out / "morphology.stl"))
    assert stl.n_cells > 0
    b = np.asarray(stl.bounds)
    assert max(b[1] - b[0], b[3] - b[2], b[5] - b[4]) == pytest.approx(30.0, rel=1e-3)
    html = (out / "morphology.html").read_text(encoding="utf-8")
    assert "{1 1 1}" in html and "http" not in html.split("<script>")[1]


def test_mcp_default_run_folder(tmp_path: Path):
    from shido import server

    cif = tmp_path / "Mg.cif"
    _mg().to(filename=str(cif))
    rep = server._morphology_impl(cif, "1 0 0:1.0,0 0 1:1.5,1 0 1:1.3", export_formats="none")
    run = Path(rep["output_dir"])
    assert (run / "図" / "morphology.png").is_file()
    assert (run / "数値" / "morphology_families.csv").is_file()
    assert not (run / "3D").exists()
    assert rep["index_html"]

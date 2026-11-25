"""Batch 24: VESTA (*.vesta) write / read round trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from shido.analyzer import detect_structure_format, load_structure_robust
from shido.core.exporters import export_structures, parse_formats
from shido.core.vesta_io import read_vesta, spaced_hm, write_vesta

SAMPLE = Path(__file__).parent / "data" / "vesta" / "Ni2MnGa_L21.vesta"


def _nacl() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.640), ["Na", "Cl"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def _ni2mnga() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.82), ["Ni", "Mn", "Ga"],
                                     [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0, 0, 0]])


def _triclinic() -> Structure:
    lat = Lattice.from_parameters(5.1, 6.3, 7.4, 82.0, 97.5, 103.2)
    return Structure(lat, ["Si", "O", "O", "Ca"],
                     [[0.1, 0.2, 0.3], [0.33, 0.12, 0.71], [0.6, 0.55, 0.05], [0.8, 0.9, 0.45]])


def _mg() -> Structure:
    return Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(3.209, 5.211), ["Mg"],
                                     [[1 / 3, 2 / 3, 0.25]])


def _b2_disordered() -> Structure:
    return Structure(Lattice.cubic(2.91), ["Ni", {"Mn": 0.5, "Ga": 0.5}],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def _assert_same(a: Structure, b: Structure, tol: float = 1e-4) -> None:
    assert np.allclose(a.lattice.abc, b.lattice.abc, atol=1e-5)
    assert np.allclose(a.lattice.angles, b.lattice.angles, atol=1e-5)
    assert len(a) == len(b)
    assert a.composition.almost_equals(b.composition)
    used: set[int] = set()
    for s in a:
        hit = None
        for j, t in enumerate(b):
            if j in used:
                continue
            d = s.frac_coords - t.frac_coords
            d -= np.round(d)
            if np.all(np.abs(d) < tol) and s.species.almost_equals(t.species):
                hit = j
                break
        assert hit is not None, f"site {s} not found after read-back"
        used.add(hit)


@pytest.mark.parametrize("make,n_struc,group", [
    (_nacl, 2, "225 1 F m -3 m"),
    (_ni2mnga, 3, "225 1 F m -3 m"),
    (_triclinic, 4, "1 1 P 1"),
    (_mg, 1, "194 1 P 63/m m c"),
])
def test_round_trip(tmp_path, make, n_struc, group):
    s = make()
    path = write_vesta(s, tmp_path / "x.vesta", title=make.__name__)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#VESTA_FORMAT_VERSION")
    lines = text.splitlines()
    assert lines[lines.index("GROUP") + 1] == group
    struc = lines[lines.index("STRUC") + 1: lines.index("THERI 0")]
    assert sum(1 for ln in struc if ln.split() and ln.split()[0].isdigit() and len(ln.split()) >= 7
               and ln.split()[0] != "0") == n_struc
    back, info = read_vesta(path)
    _assert_same(s, back)
    assert info["group"] == group


def test_disordered_occupancy_kept(tmp_path):
    s = _b2_disordered()
    back, _ = read_vesta(write_vesta(s, tmp_path / "b2.vesta"))
    _assert_same(s, back)
    mixed = [site for site in back if not site.is_ordered]
    assert len(mixed) == 1 and mixed[0].species["Mn"] == pytest.approx(0.5)


def test_bonds_and_styles(tmp_path):
    p = write_vesta(_nacl(), tmp_path / "nacl.vesta", bonds=[("Na", "Cl", 0.0, 3.0)])
    _, info = read_vesta(p)
    assert info["bonds"] == [("Na", "Cl", 0.0, 3.0)]
    assert set(info["site_styles"]) == {"Na1", "Cl1"}
    # default bonds: Na-Cl at 2.82 A is found
    _, info2 = read_vesta(write_vesta(_nacl(), tmp_path / "d.vesta"))
    assert any({a, b} == {"Na", "Cl"} and dmax > 2.82 for a, b, _, dmax in info2["bonds"])


def test_spaced_hm():
    assert spaced_hm("Fm-3m") == "F m -3 m"
    assert spaced_hm("P6/mcc") == "P 6/m c c"
    assert spaced_hm("P-1") == "P -1"


def test_load_structure_robust_and_export(tmp_path):
    assert "vesta" in parse_formats("vesta")
    res = export_structures([("NaCl", _nacl())], "vesta", tmp_path)
    path = Path(res["structures"][0]["files"]["vesta"])
    assert path.name == "NaCl.vesta"
    assert detect_structure_format(path) == "vesta"
    noext = tmp_path / "noext"
    noext.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    assert detect_structure_format(noext) == "vesta"
    s, warns = load_structure_robust(path)
    _assert_same(_nacl(), s)


def test_sample_file_for_visual_check():
    """Keep one written file for the owner to open in VESTA (regenerated each run)."""
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    write_vesta(_ni2mnga(), SAMPLE, title="Ni2MnGa_L21")
    back, _ = read_vesta(SAMPLE)
    assert len(back) == 16

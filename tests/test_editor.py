"""Batch 22: edit atoms / sites and build a structure from a space group."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser

from shido import server
from shido.core.editor import (
    apply_operations,
    create_structure,
    lattice_from_params,
    merge_duplicates,
    write_edited,
)
from shido.core.validator import ScientificValidationError


def _nacl() -> Structure:
    return create_structure("Fm-3m", {"a": 5.64}, [["Na", 0, 0, 0], ["Cl", 0.5, 0.5, 0.5]])


def test_create_nacl_matches_literature():
    s = _nacl()
    assert len(s) == 8
    assert s.composition.reduced_formula == "NaCl"
    assert s.lattice.abc == pytest.approx((5.64, 5.64, 5.64))
    assert s.lattice.angles == pytest.approx((90.0, 90.0, 90.0))
    assert s.get_space_group_info()[1] == 225
    # Na-Cl = a/2, octahedral coordination
    na = next(i for i, site in enumerate(s) if site.specie.symbol == "Na")
    neigh = s.get_neighbors(s[na], 5.64 / 2 + 0.01)
    assert len(neigh) == 6
    assert all(n.specie.symbol == "Cl" for n in neigh)
    assert all(n.nn_distance == pytest.approx(2.82, abs=1e-6) for n in neigh)
    assert CrystalNN().get_cn(s, na) == 6


def test_lattice_follows_crystal_system():
    hexa = lattice_from_params("P6_3/mmc", {"a": 3.209, "c": 5.211})
    assert hexa.angles == pytest.approx((90, 90, 120))
    with pytest.raises(ValueError):
        lattice_from_params("P2_1/c", {"a": 5.0})  # monoclinic needs b, c, beta


def test_replace_changes_composition():
    s, log = apply_operations(_nacl(), [{"op": "replace", "from": "Na", "to": "K"}])
    assert s.composition.reduced_formula == "KCl"
    assert len(s) == 8 and len(log) == 1
    # partial replacement on one site -> mixed site
    s2, _ = apply_operations(_nacl(), [{"op": "replace", "site": 0, "to": {"K": 0.5, "Na": 0.5}}])
    assert s2.composition.get_el_amt_dict() == pytest.approx({"Na": 3.5, "K": 0.5, "Cl": 4.0})


def test_remove_reduces_sites_and_occupancy():
    s, _ = apply_operations(_nacl(), [{"op": "remove", "site": [0, 1]}])
    assert len(s) == 6
    s2, _ = apply_operations(_nacl(), [{"op": "remove", "element": "Cl"}])
    assert len(s2) == 4 and s2.composition.reduced_formula == "Na"
    s3, _ = apply_operations(_nacl(), [{"op": "occupancy", "element": "Cl", "value": 0.5}])
    assert s3.composition["Cl"] == pytest.approx(2.0)


def test_move_and_add():
    s, _ = apply_operations(
        _nacl(),
        [{"op": "add", "element": "Li", "coords": [0.25, 0.25, 0.25]},
         {"op": "move", "site": -1, "shift": [0.5, 0.0, 0.0]}],
    )
    assert len(s) == 9
    assert s[-1].frac_coords == pytest.approx([0.75, 0.25, 0.25])


def test_merge_duplicates():
    base = _nacl()
    s, _ = apply_operations(base, [{"op": "add", "element": "Na", "coords": [0, 0, 0.0005]}])
    assert len(s) == 9
    merged, removed = merge_duplicates(s, 0.01)
    assert removed == 1 and len(merged) == 8
    # as an operation and via merge_tol
    s2, log = apply_operations(base, [{"op": "add", "element": "Na", "coords": [0, 0, 0.0005]}], merge_tol=0.01)
    assert len(s2) == 8 and "1 duplicate" in log[-1]
    # different species at the same place -> one mixed site
    b2 = Structure(Lattice.cubic(3.0), [{"Mn": 0.5}, {"Ga": 0.5}], [[0, 0, 0], [0, 0, 0]])
    m, _ = merge_duplicates(b2, 0.01)
    assert len(m) == 1 and m.composition.get_el_amt_dict() == pytest.approx({"Mn": 0.5, "Ga": 0.5})


def test_too_close_atom_is_stopped(tmp_path: Path):
    s, _ = apply_operations(_nacl(), [{"op": "add", "element": "Li", "coords": [0.05, 0, 0]}])  # 0.28 A
    with pytest.raises(ScientificValidationError):
        write_edited(s, tmp_path, stem="bad")
    assert not (tmp_path / "bad.cif").exists()


def test_write_and_reread(tmp_path: Path):
    s, _ = apply_operations(_nacl(), [{"op": "replace", "from": "Na", "to": "K"}])
    files = write_edited(s, tmp_path, stem="kcl", poscar=True)
    back = CifParser(files["cif"]).parse_structures(primitive=False)[0]
    assert back.composition.reduced_formula == "KCl" and len(back) == 8
    assert Path(files["poscar"]).is_file()


def test_bad_operation_message():
    with pytest.raises(ValueError, match="operation #0"):
        apply_operations(_nacl(), [{"op": "remove", "site": 99}])
    with pytest.raises(ValueError, match="unknown op"):
        apply_operations(_nacl(), [{"op": "explode"}])


def test_cli_create_and_edit(tmp_path: Path):
    out1 = tmp_path / "c"
    rc = server.main(["create", "--sg", "Fm-3m", "--lattice", "5.64", "--site", "Na 0 0 0",
                      "--site", "Cl 0.5 0.5 0.5", "--out-dir", str(out1)])
    assert rc == 0
    cif = out1 / "NaCl_created.cif"
    assert cif.is_file()
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps([{"op": "replace", "from": "Na", "to": "K"}]), encoding="utf-8")
    out2 = tmp_path / "e"
    rc = server.main(["edit", str(cif), "--ops", str(ops), "--remove", "0", "--out-dir", str(out2)])
    assert rc == 0
    back = CifParser(str(out2 / "NaCl_created_edited.cif")).parse_structures(primitive=False)[0]
    assert len(back) == 7 and set(back.composition.get_el_amt_dict()) == {"K", "Cl"}
    # an overlap is refused, nothing written
    out3 = tmp_path / "x"
    rc = server.main(["edit", str(cif), "--add", "Li 0.05 0 0", "--out-dir", str(out3)])
    assert rc == 1
    assert not (out3 / "NaCl_created_edited.cif").exists()


def test_impl_default_run_folder():
    res = server._create_structure_impl("Fm-3m", [5.64], [["Na", 0, 0, 0], ["Cl", 0.5, 0.5, 0.5]])
    cif = Path(res["files"]["cif"])
    assert cif.parent.name == "構造" and cif.name == "NaCl_created.cif"
    assert res["index_html"] and Path(res["index_html"]).is_file()
    assert res["output"]["spacegroup_number"] == 225 and res["min_distance"] == pytest.approx(2.82, abs=1e-3)

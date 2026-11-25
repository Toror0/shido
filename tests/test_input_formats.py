"""Batch 9: POSCAR/CONTCAR, XSF and CSSR are read like CIF (lattice, composition, atom count)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.cssr import Cssr
from pymatgen.io.vasp import Poscar
from pymatgen.io.xcrysden import XSF

from shido import server
from shido.analyzer import (
    detect_structure_format,
    is_structure_file,
    load_structure_robust,
    load_structures_all,
    structure_stem,
)
from shido.core.output_paths import new_run_dir


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _write_all(folder: Path) -> dict[str, Path]:
    s = _nacl()
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "cif": folder / "NaCl.cif",
        "poscar": folder / "POSCAR",
        "vasp": folder / "NaCl.vasp",
        "xsf": folder / "NaCl.xsf",
        "cssr": folder / "NaCl.cssr",
    }
    CifWriter(s).write_file(paths["cif"])
    Poscar(s).write_file(paths["poscar"])
    Poscar(s).write_file(paths["vasp"])
    paths["xsf"].write_text(XSF(s).to_str(), encoding="utf-8")
    paths["cssr"].write_text(str(Cssr(s)), encoding="utf-8")
    return paths


def test_same_structure_from_all_formats(tmp_path: Path) -> None:
    paths = _write_all(tmp_path)
    ref, _ = load_structure_robust(paths["cif"])
    for key, path in paths.items():
        struct, warns = load_structure_robust(path)
        assert len(struct) == len(ref) == 8, key
        assert struct.composition.reduced_formula == "NaCl", key
        assert struct.lattice.abc == pytest.approx(ref.lattice.abc, abs=1e-3), key
        assert struct.lattice.angles == pytest.approx(ref.lattice.angles, abs=1e-2), key
        assert not any("fallback" in w for w in warns), key


def test_detect_format_by_extension_name_and_content(tmp_path: Path) -> None:
    paths = _write_all(tmp_path)
    assert detect_structure_format(paths["cif"]) == "cif"
    assert detect_structure_format(paths["poscar"]) == "poscar"
    assert detect_structure_format(tmp_path / "CONTCAR_relax") == "poscar"
    assert detect_structure_format(paths["xsf"]) == "xsf"
    assert detect_structure_format(paths["cssr"]) == "cssr"
    # extensionless, unknown name: decided by content
    odd = tmp_path / "structure_si"
    odd.write_text(paths["poscar"].read_text())
    assert detect_structure_format(odd) == "poscar"
    assert len(load_structure_robust(odd)[0]) == 8
    odd_xsf = tmp_path / "nacl_xcrysden"
    odd_xsf.write_text(paths["xsf"].read_text())
    assert detect_structure_format(odd_xsf) == "xsf"
    # a POSCAR without data_ blocks still works through load_structures_all
    (entry,) = load_structures_all(paths["poscar"])
    assert entry[2] == "POSCAR" and len(entry[0]) == 8


def test_batch_discovers_poscar_and_other_formats(tmp_path: Path) -> None:
    paths = _write_all(tmp_path / "in")
    sub = tmp_path / "in" / "relax"
    sub.mkdir()
    Poscar(_nacl()).write_file(sub / "CONTCAR")
    (tmp_path / "in" / "notes.txt").write_text("not a structure")
    (tmp_path / "in" / "output_old").mkdir()
    Poscar(_nacl()).write_file(tmp_path / "in" / "output_old" / "POSCAR")

    found = server._discover_cifs(tmp_path / "in")
    names = sorted(p.name for p in found)
    assert names == sorted([p.name for p in paths.values()] + ["CONTCAR"])
    assert all(is_structure_file(p) for p in found)


def test_output_names_for_extensionless_files(tmp_path: Path) -> None:
    assert structure_stem(Path("POSCAR")) == "POSCAR"
    assert structure_stem(Path("POSCAR_ca_0.85")) == "POSCAR_ca_0.85"
    assert structure_stem(Path("NaCl.vasp")) == "NaCl"
    assert structure_stem(Path("NaCl.cif")) == "NaCl"
    run = new_run_dir("", structure_stem(Path("POSCAR")), root=tmp_path)
    assert run.name.endswith("_POSCAR")


@pytest.mark.parametrize(
    "name, text, label",
    [
        ("POSCAR", "broken\n1.0\n1 0\n", "POSCAR"),
        ("bad.xsf", "CRYSTAL\nPRIMVEC\n1 0 0\n", "XSF"),
        ("bad.cssr", "5.0 5.0 5.0\n90 90 90 SPGR = 1 P 1\n3 0\n0 x\n1 Na 0 0 0\n", "CSSR"),
    ],
)
def test_broken_file_gives_clear_error(tmp_path: Path, name: str, text: str, label: str) -> None:
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(ValueError) as info:
        load_structure_robust(path)
    msg = str(info.value)
    assert name in msg and f"as {label}" in msg

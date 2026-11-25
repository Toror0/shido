"""Where results go: a new 出力/<date>_<name>/ per run, fixed sub-folders, はじめに読む.html."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from urllib.parse import unquote

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from shido import server
from shido.core.output_paths import INDEX_NAME, new_run_dir, write_index_html


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def _cif(path: Path) -> Path:
    CifWriter(_nacl()).write_file(str(path))
    return path


def _links(index: Path) -> list[Path]:
    text = index.read_text(encoding="utf-8")
    refs = re.findall(r'(?:href|src)="([^"]+)"', text)
    return [(index.parent / unquote(r)).resolve() for r in refs]


def _assert_index_complete(run: Path) -> None:
    index = run / INDEX_NAME
    assert index.is_file()
    text = index.read_text(encoding="utf-8")
    assert "http://" not in text and "https://" not in text  # offline, one file
    links = _links(index)
    assert links, "no links"
    for p in links:
        assert p.is_file(), p
    files = {p.resolve() for p in run.rglob("*") if p.is_file() and p.name != INDEX_NAME}
    assert files <= set(links), sorted(str(p) for p in files - set(links))


def test_new_run_dir_names_and_same_second(tmp_path):
    now = dt.datetime(2026, 11, 24, 15, 30, 12)
    a = new_run_dir("", "NaCl", root=tmp_path, now=now)
    b = new_run_dir("", "NaCl", root=tmp_path, now=now)
    c = new_run_dir("batch", "my cifs", root=tmp_path, now=now)
    assert a.name == "2026-11-24_153012_NaCl"
    assert b.name == "2026-11-24_153012_NaCl_2"
    assert c.name == "2026-11-24_153012_batch_my_cifs"
    assert a.is_dir() and b.is_dir() and not any(a.iterdir())


def test_default_root_is_env_folder(tmp_path, _output_root_in_tmp):
    run = new_run_dir("demo", "Ni2MnGa")
    assert run.parent == _output_root_in_tmp.resolve()
    assert run.name.endswith("_demo_Ni2MnGa")


def test_cli_default_makes_new_folder_each_time(tmp_path, _output_root_in_tmp, capsys):
    cif = _cif(tmp_path / "NaCl.cif")
    args = [str(cif), "--no-polyhedra", "--xrd", "--html"]
    assert server.main(args) == 0
    first = capsys.readouterr().out
    assert server.main(args) == 0
    second = capsys.readouterr().out
    runs = sorted(p for p in _output_root_in_tmp.iterdir() if p.is_dir())
    assert len(runs) == 2 and runs[0] != runs[1]
    for run, out in zip(runs, (first, second)):
        assert run.name.split("_", 2)[-1].startswith("NaCl")
        for sub in ("図", "XRD", "3D", "数値", "構造"):
            assert (run / sub).is_dir(), sub
        assert (run / "図" / "figure_overview.png").is_file()
        assert (run / "XRD" / "xrd_peaks.csv").is_file()
        assert (run / "XRD" / "figure_xrd.png").is_file()
        assert (run / "3D" / "structure.html").is_file()
        assert (run / "数値" / "lattice_summary.csv").is_file()
        assert (run / "構造" / "POSCAR").is_file()
        _assert_index_complete(run)
        lines = out.strip().splitlines()
        assert lines[-2] == f"保存先: {run}"
        assert lines[-1] == f"{INDEX_NAME}: {run / INDEX_NAME}"
    text = (runs[0] / INDEX_NAME).read_text(encoding="utf-8")
    assert "NaCl.cif" in text and "XRD" in text and "格子定数" in text
    assert '<img src="XRD/figure_xrd.png"' in text  # relative img


def test_cli_slab_goes_into_subfolder(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    assert server.main([str(cif), "--no-polyhedra", "--hkl", "1", "0", "0"]) == 0
    (run,) = list(_output_root_in_tmp.iterdir())
    assert (run / "slab_100" / "数値" / "surface_summary.csv").is_file()
    assert list((run / "slab_100" / "図").glob("*.png"))
    _assert_index_complete(run)


def test_cli_out_dir_keeps_old_place(tmp_path, _output_root_in_tmp, capsys):
    cif = _cif(tmp_path / "NaCl.cif")
    base = tmp_path / "base"
    assert server.main([str(cif), "--out-dir", str(base), "--no-polyhedra"]) == 0
    out = base / "output_NaCl"
    assert (out / "数値" / "lattice_summary.csv").is_file()
    assert not _output_root_in_tmp.exists()
    assert f"保存先: {out.resolve()}" in capsys.readouterr().out


def test_subcommand_saves_by_default(tmp_path, _output_root_in_tmp, capsys):
    cif = _cif(tmp_path / "NaCl.cif")
    assert server.main(["symmetry", str(cif)]) == 0
    printed = capsys.readouterr().out
    assert "Fm-3m" in printed
    (run,) = list(_output_root_in_tmp.iterdir())
    assert run.name.endswith("_symmetry_NaCl")
    assert (run / "数値" / "symmetry_sites.csv").is_file()
    assert f"保存先: {run}" in printed
    _assert_index_complete(run)

    assert server.main(["measure", str(cif), "--atoms", "Na1", "Cl1"]) == 0
    capsys.readouterr()
    (mrun,) = [p for p in _output_root_in_tmp.iterdir() if p.name.endswith("_measure_NaCl")]
    assert (mrun / "数値" / "measure.json").is_file()


def test_batch_default_folder(tmp_path, _output_root_in_tmp):
    in_dir = tmp_path / "cifs"
    in_dir.mkdir()
    _cif(in_dir / "NaCl.cif")
    res = server.run_batch(in_dir, show_polyhedra=False)
    run = Path(res["output_dir"])
    assert run.parent == _output_root_in_tmp.resolve() and run.name.endswith("_batch_cifs")
    assert Path(res["batch_summary_csv"]) == run / "数値" / "batch_summary.csv"
    assert (run / "001_NaCl" / "図" / "figure_overview.png").is_file()
    assert Path(res["index_html"]) == run / INDEX_NAME
    _assert_index_complete(run)


def test_export_out_dir_unchanged(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    res = server._export_structure_impl(str(cif), "xyz", tmp_path / "ex")
    assert Path(res["structures"][0]["files"]["xyz"]).parent == (tmp_path / "ex").resolve()
    assert res["index_html"] is None and not _output_root_in_tmp.exists()


def test_index_html_handles_japanese_names(tmp_path):
    run = tmp_path / "r"
    (run / "図").mkdir(parents=True)
    (run / "図" / "見本 1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run / "数値").mkdir()
    (run / "数値" / "bond_lengths.csv").write_text("a,b\n", encoding="utf-8")
    index = write_index_html(run, title="テスト", source="x.cif", settings={"xrd": True, "html": None})
    text = index.read_text(encoding="utf-8")
    assert "結合の長さ" in text and "XRD（粉末回折）" in text and "回せる 3D（HTML）" not in text
    _assert_index_complete(run)

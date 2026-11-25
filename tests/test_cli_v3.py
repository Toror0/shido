"""v3 integration: CLI options, run_bulk_analysis / run_batch outputs, MCP tool registry."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from shido import server
from shido.analyzer import build_ni2mnga_structure
from shido.core import __version__
from shido.renderer import pretty_formula


def _cu() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.615), ["Cu"], [[0, 0, 0]])


def _write_cif(path: Path, structure: Structure) -> Path:
    CifWriter(structure).write_file(str(path))
    return path


@pytest.fixture()
def ni2mnga_cif(tmp_path: Path) -> Path:
    return _write_cif(tmp_path / "Ni2MnGa.cif", build_ni2mnga_structure())


V3_KEYS = ("xrd_peaks_csv", "xrd_figure_png", "stl_file", "html_file")


def test_version_is_v3():
    assert __version__.startswith("3.")
    parser = server._build_arg_parser()
    assert f"v{__version__}" in parser.description


def test_default_run_makes_no_v3_outputs(tmp_path: Path):
    res = server.run_bulk_analysis(build_ni2mnga_structure(), tmp_path / "out", show_polyhedra=False)
    for key in V3_KEYS:
        assert res[key] is None, key
    assert res["stl_split_files"] == []
    assert not (tmp_path / "out" / "3D").exists()
    assert not (tmp_path / "out" / "XRD" / "xrd_peaks.csv").exists()
    # v3 CSVs from analyzer are always there and now reported
    assert Path(res["bond_angles_csv"]).is_file()
    assert Path(res["atomic_positions_csv"]).is_file()
    assert res["poscar_cell"] == "input"


def test_cli_single_cif_with_all_v3_options(ni2mnga_cif: Path, tmp_path: Path, capsys):
    base = tmp_path / "cli"
    rc = server.main([
        str(ni2mnga_cif), "--out-dir", str(base), "--no-polyhedra",
        "--xrd", "--wavelength", "MoKa", "--stl-split", "--stl-max-mm", "60",
        "--html", "--poscar-cell", "primitive",
    ])
    assert rc == 0
    out = base / "output_Ni2MnGa"
    assert (out / "XRD" / "xrd_peaks.csv").is_file()
    assert (out / "XRD" / "figure_xrd.png").is_file()
    assert (out / "3D" / "structure.stl").is_file()
    for el in ("Ni", "Mn", "Ga", "bonds"):
        assert (out / "3D" / f"structure_{el}.stl").is_file(), el
    assert (out / "3D" / "structure.html").is_file()
    poscar = (out / "構造" / "POSCAR").read_text()
    assert sum(int(x) for x in poscar.splitlines()[6].split()) == 4  # primitive L21: 4 atoms
    text = capsys.readouterr().out
    assert f"v{__version__}" in text
    assert "XRD peaks CSV" in text and "3D viewer HTML" in text


def test_cli_default_single_cif_has_no_v3_files(ni2mnga_cif: Path, tmp_path: Path):
    base = tmp_path / "cli_default"
    assert server.main([str(ni2mnga_cif), "--out-dir", str(base), "--no-polyhedra"]) == 0
    out = base / "output_Ni2MnGa"
    assert (out / "数値" / "lattice_summary.csv").is_file()
    assert not (out / "3D").exists()
    assert not (out / "XRD" / "xrd_peaks.csv").exists()
    assert not (out / "XRD" / "figure_xrd.png").exists()


def test_batch_with_v3_options(tmp_path: Path):
    in_dir = tmp_path / "cifs"
    in_dir.mkdir()
    _write_cif(in_dir / "a_Ni2MnGa.cif", build_ni2mnga_structure())
    _write_cif(in_dir / "b_Cu.cif", _cu())
    out_dir = tmp_path / "batch_out"
    rc = server.main(["batch", str(in_dir), "--out-dir", str(out_dir), "--no-polyhedra", "--xrd", "--html"])
    assert rc == 0
    for sub in ("001_a_Ni2MnGa", "002_b_Cu"):
        assert (out_dir / sub / "XRD" / "xrd_peaks.csv").is_file(), sub
        assert (out_dir / sub / "3D" / "structure.html").is_file(), sub
        assert not (out_dir / sub / "3D" / "structure.stl").exists()


def test_run_batch_returns_new_paths(tmp_path: Path):
    in_dir = tmp_path / "cifs"
    in_dir.mkdir()
    _write_cif(in_dir / "Cu.cif", _cu())
    res = server.run_batch(in_dir, tmp_path / "o", show_polyhedra=False, stl=True)
    assert res["succeeded"] == 1
    bulk = res["items"][0]["bulk_analysis"]
    assert Path(bulk["stl_file"]).is_file()
    assert bulk["html_file"] is None


def test_all_blocks_single_cif_and_batch(tmp_path: Path):
    a = CifWriter(build_ni2mnga_structure()).__str__()
    b = CifWriter(_cu()).__str__()
    in_dir = tmp_path / "multi"
    in_dir.mkdir()
    multi = in_dir / "two.cif"
    multi.write_text(a + "\n" + b, encoding="utf-8")

    base = tmp_path / "single"
    assert server.main([str(multi), "--out-dir", str(base), "--no-polyhedra", "--all-blocks"]) == 0
    outs = sorted(p.name for p in base.glob("output_two_*"))
    assert len(outs) == 2, outs

    # without --all-blocks only one output folder
    base2 = tmp_path / "single_default"
    assert server.main([str(multi), "--out-dir", str(base2), "--no-polyhedra"]) == 0
    assert [p.name for p in base2.glob("output_*")] == ["output_two"]

    res = server.run_batch(in_dir, tmp_path / "batch", show_polyhedra=False, all_blocks=True)
    assert res["succeeded"] == 2
    blocks = {it["data_block"] for it in res["items"]}
    assert len(blocks) == 2 and None not in blocks


def test_mcp_tools_registered():
    mcp = server.create_mcp_server()
    tools = asyncio.run(mcp.list_tools())
    by_name = {t.name: t for t in tools}
    for name in ("compute_xrd_pattern", "export_3d_model", "analyze_structure"):
        assert name in by_name, name
    def _schema(tool):  # mcp 1.x: inputSchema / mcp 2.x: input_schema
        return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema")

    assert "poscar_cell" in _schema(by_name["analyze_structure"])["properties"]
    assert "format" in _schema(by_name["export_3d_model"])["properties"]


def test_pretty_formula_x2yz_independent_of_site_order():
    s = build_ni2mnga_structure()
    assert pretty_formula(s) == "Ni2MnGa"
    # reverse the site order (as a slab can) -> still Ni2MnGa
    rev = Structure.from_sites(list(reversed(s.sites)))
    assert pretty_formula(rev) == "Ni2MnGa"
    co2fe_al = Structure(
        Lattice.cubic(5.7), ["Al", "Fe", "Co", "Co"],
        [[0.5, 0.5, 0.5], [0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )
    assert pretty_formula(co2fe_al) == "Co2FeAl"

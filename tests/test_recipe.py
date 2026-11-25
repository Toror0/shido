"""recipe.json in every run folder, and `rerun` (CLI ``server.py rerun`` / MCP ``rerun_recipe``)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from shido import server
from shido.core import recipe as rc
from shido.core.output_paths import INDEX_NAME


def _cif(path: Path) -> Path:
    s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    CifWriter(s).write_file(str(path))
    return path


def _runs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _only_new_run(root: Path, before: list[Path]) -> Path:
    new = [p for p in _runs(root) if p not in before]
    assert len(new) == 1, new
    return new[0]


def _load(run: Path) -> dict:
    return json.loads((run / rc.RECIPE_NAME).read_text(encoding="utf-8"))


def _hashes(run: Path) -> dict[str, str]:
    return {o["path"]: o["sha256"] for o in _load(run)["outputs"]}


def test_cli_run_writes_recipe(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    assert server.main(["export", str(cif), "--formats", "xyz,p1"]) == 0
    run = _only_new_run(_output_root_in_tmp, [])
    r = _load(run)

    assert r["schema"] == rc.SCHEMA
    assert r["tool"]["name"] == "CrystalAnalyzerMCP" and r["tool"]["version"]
    assert {"pymatgen", "pyvista", "vtk", "numpy"} <= set(r["libraries"])
    assert r["python"]
    assert r["call"]["kind"] == "cli" and r["call"]["argv"][0] == "export"
    # options with their defaults filled in
    assert r["call"]["resolved_options"]["formats"] == "xyz,p1"
    assert "multi_name" in r["call"]["resolved_options"]
    assert r["inputs"] == [{"path": str(cif.resolve()), "kind": "file",
                            "sha256": rc.sha256(cif), "bytes": cif.stat().st_size}]
    assert r["settings"]["theme"] and "render" in r["settings"]["effective"]
    assert r["validation"]["status"] == "passed"
    names = sorted(o["path"] for o in r["outputs"])
    assert names == ["構造/NaCl.xyz", "構造/NaCl_P1.cif"]
    for o in r["outputs"]:
        assert rc.sha256(run / o["path"]) == o["sha256"]

    index = (run / INDEX_NAME).read_text(encoding="utf-8")
    assert "この図の作り方（レシピ）" in index
    assert "python server.py rerun" in index and str(run) in index
    assert f"CrystalAnalyzerMCP {r['tool']['version']}" in index  # methods draft, value from the recipe


def test_rerun_same_settings_gives_same_files(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    server.main(["export", str(cif), "--formats", "xyz,p1"])
    first = _only_new_run(_output_root_in_tmp, [])

    before = _runs(_output_root_in_tmp)
    assert server.main(["rerun", str(first)]) == 0
    second = _only_new_run(_output_root_in_tmp, before)
    assert second != first
    assert _hashes(second) == _hashes(first)  # same names, same CIF / XYZ bytes
    r2 = _load(second)
    assert r2["rerun_of"] == str((first / rc.RECIPE_NAME).resolve())
    assert r2["call"]["argv"] == _load(first)["call"]["argv"]


def test_rerun_set_changes_only_the_named_value(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    server.main(["symmetry", str(cif)])
    first = _only_new_run(_output_root_in_tmp, [])
    r1 = _load(first)
    assert r1["call"]["resolved_options"]["symprec"] == 0.01

    before = _runs(_output_root_in_tmp)
    server.main(["rerun", str(first / rc.RECIPE_NAME), "--set", "symprec=0.1", "--set", "render.dpi=600"])
    second = _only_new_run(_output_root_in_tmp, before)
    r2 = _load(second)

    opts1, opts2 = r1["call"]["resolved_options"], r2["call"]["resolved_options"]
    assert opts2["symprec"] == 0.1
    assert {k: v for k, v in opts2.items() if k != "symprec"} == {k: v for k, v in opts1.items() if k != "symprec"}
    eff1, eff2 = r1["settings"]["effective"], r2["settings"]["effective"]
    assert eff2["render"]["dpi"] == 600
    eff2["render"]["dpi"] = eff1["render"]["dpi"]
    assert eff2 == eff1  # nothing else in the settings changed
    assert set(_hashes(second)) == set(_hashes(first))
    # the settings files themselves were not touched
    from shido import settings as cfg

    assert cfg.get("render.dpi") == eff1["render"]["dpi"]


def test_rerun_warns_when_input_changed(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    server.main(["export", str(cif), "--formats", "xyz"])
    first = _only_new_run(_output_root_in_tmp, [])
    with cif.open("a", encoding="utf-8") as fh:
        fh.write("\n# edited later\n")

    res = rc.rerun(first)  # warns, does not stop
    assert res["exit_code"] == 0
    assert any("SHA-256" in w and "NaCl.cif" in w for w in res["rerun"]["warnings"])
    assert Path(res["rerun"]["new_recipe"]).is_file()


def test_rerun_warns_on_other_library_version(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    server.main(["export", str(cif), "--formats", "xyz"])
    first = _only_new_run(_output_root_in_tmp, [])
    path = first / rc.RECIPE_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["libraries"]["numpy"] = "0.0.1"
    path.write_text(json.dumps(data), encoding="utf-8")

    res = rc.rerun(path)
    assert any(w.startswith("numpy の版が違います") for w in res["rerun"]["warnings"])
    assert not any("SHA-256" in w for w in res["rerun"]["warnings"])


def test_mcp_tool_records_and_rerun_recipe(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    mcp = server.create_mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert "rerun_recipe" in tools
    # the recording wrapper keeps the tool schema
    assert {"cif_path", "symprec", "output_dir"} <= set(tools["get_symmetry_info"].input_schema["properties"])

    info = rc.TOOLS["get_symmetry_info"](str(cif))
    run = Path(info["output_dir"])
    r = _load(run)
    assert r["call"] == {"kind": "mcp", "tool": "get_symmetry_info",
                         "arguments": {"cif_path": str(cif.resolve()), "symprec": 0.01,
                                       "output_dir": None, "include_operations": True}}
    assert info["recipe"] == str(run / rc.RECIPE_NAME)

    again = _run_mcp_tool(mcp, "rerun_recipe", {"recipe_path": str(run), "overrides": {"symprec": 0.2}})
    run2 = Path(again["rerun"]["output_dir"])
    assert run2 != run
    r2 = _load(run2)
    assert r2["call"]["tool"] == "get_symmetry_info"
    assert r2["call"]["arguments"]["symprec"] == 0.2
    assert r2["call"]["arguments"]["cif_path"] == str(cif.resolve())
    assert _hashes(run2).keys() == _hashes(run).keys()

    with pytest.raises(ValueError, match="引数"):
        rc.rerun(run, {"no_such_option": 1})


def _run_mcp_tool(mcp, name: str, args: dict) -> dict:
    """Call a tool the way a client does and return its dict result."""
    out = asyncio.run(mcp.call_tool(name, args))
    if isinstance(out, tuple):  # (content, structured)
        out = out[1]
    if isinstance(out, dict) and set(out) == {"result"}:
        out = out["result"]
    if not isinstance(out, dict):  # CallToolResult (mcp 2.x) or a list of TextContent
        structured = getattr(out, "structured_content", None) or getattr(out, "structuredContent", None)
        if structured is not None:
            out = structured.get("result", structured)
        else:
            content = getattr(out, "content", out)
            out = json.loads(content[0].text)
    return out


def test_bulk_analysis_cli_writes_recipe_and_reruns(tmp_path, _output_root_in_tmp):
    cif = _cif(tmp_path / "NaCl.cif")
    assert server.main([str(cif)]) == 0
    first = _only_new_run(_output_root_in_tmp, [])
    r = _load(first)
    assert r["call"]["argv"] == [str(cif.resolve())]  # made absolute: rerun works from any folder
    assert r["call"]["resolved_options"]["target"] == str(cif)
    assert any(o["path"].startswith("図/") and o["path"].endswith(".png") for o in r["outputs"])
    assert "この図の作り方（レシピ）" in (first / INDEX_NAME).read_text(encoding="utf-8")

    before = _runs(_output_root_in_tmp)
    server.main(["rerun", str(first)])
    second = _only_new_run(_output_root_in_tmp, before)
    h1, h2 = _hashes(first), _hashes(second)
    assert h1.keys() == h2.keys()  # same file names
    for name in h1:
        if name.endswith((".csv", ".cif")):
            assert h1[name] == h2[name], name


def test_parse_overrides():
    assert rc.parse_overrides(["theme=vesta", "render.dpi = 600"]) == {"theme": "vesta", "render.dpi": "600"}
    with pytest.raises(ValueError):
        rc.parse_overrides(["theme"])


def test_no_recipe_outside_a_recorded_call(tmp_path):
    assert rc.write_recipe(tmp_path) is None
    assert not (tmp_path / rc.RECIPE_NAME).exists()

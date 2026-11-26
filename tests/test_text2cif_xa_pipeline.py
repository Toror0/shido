"""End-to-end: text2cif "Mn2CoGa の逆ホイスラー、a=5.80" -> XA CIF -> MCP analyze_structure.

text2cif is the sibling repository (../text2cif). Skipped when it cannot be imported.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

T2C_ROOT = Path(__file__).resolve().parents[2] / "text2cif"


def _describe_to_cif():
    if T2C_ROOT.is_dir() and str(T2C_ROOT) not in sys.path:
        sys.path.append(str(T2C_ROOT))  # appended: never shadows Shido modules
    try:
        from text2cif.core.describe import describe_to_cif
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"text2cif is not importable: {exc}")
    return describe_to_cif


def test_text2cif_xa_to_analyze_structure(tmp_path: Path) -> None:
    describe_to_cif = _describe_to_cif()
    cif = tmp_path / "Mn2CoGa_XA.cif"
    r = describe_to_cif("Mn2CoGa の逆ホイスラー、a=5.80", str(cif))
    assert r["ok"] and r["prototype"] == "heusler_xa"
    assert cif.is_file()

    from shido import server

    mcp = server.create_mcp_server()
    out = tmp_path / "analyze"
    call = asyncio.run(mcp.call_tool("analyze_structure", {"cif_path": str(cif), "output_dir": str(out)}))
    content = call.content if hasattr(call, "content") else call[0]
    res = json.loads(content[0].text)

    summ = res["summary"]
    assert summ["spacegroup_number"] == 216
    assert summ["formula"].split() and sorted(summ["formula"].split()) == ["Co4", "Ga4", "Mn8"]
    assert summ["num_sites"] == 16
    for key in ("figure_overview", "lattice_summary_csv", "atomic_positions_csv", "cif_file"):
        assert Path(res[key]).is_file(), key

    pos = pd.read_csv(res["atomic_positions_csv"])
    by_wyckoff = {w: set(g["element"]) for w, g in pos.groupby("wyckoff")}
    assert by_wyckoff == {"4a": {"Mn"}, "4d": {"Mn"}, "4c": {"Co"}, "4b": {"Ga"}}

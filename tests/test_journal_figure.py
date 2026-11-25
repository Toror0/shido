"""Journal figures: journals/*.yaml (extends, validation), layout to mm x dpi, panel
labels, safe defaults for unverified rules, the check, CLI and MCP."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from shido import server
from shido.core.figure_layout import (
    ImagePanel,
    XrdPanel,
    check_image_file,
    make_journal_figure,
    mm_to_px,
    panel_label_text,
)
from shido.core.journal_spec import JournalSpecError, list_journals, load_journal

COD = Path(__file__).parent / "data" / "cod" / "9001735.cif"
MAIN_JOURNALS = ["acta_materialia", "prb", "nature", "chem_mater", "j_appl_cryst", "appl_phys_lett"]


def _png(path: Path, size=(1600, 1200), dpi=300, color=(40, 90, 160)) -> Path:
    im = Image.new("RGB", size, "white")
    im.paste(Image.new("RGB", (size[0] // 2, size[1] // 2), color), (size[0] // 4, size[1] // 4))
    im.save(path, dpi=(dpi, dpi))
    return path


def _write(d: Path, name: str, text: str) -> None:
    (d / f"{name}.yaml").write_text(text, encoding="utf-8")


BASE = """name: Base
publisher: Pub
width_mm: {single: 80, one_half: 120, double: 170}
max_height_mm: 200
fonts: [Arial]
text_pt: {body: 7, min: 6}
min_line_pt: 0.5
dpi: {line: 600, photo: 300, combination: 500}
formats: [tiff, pdf]
color_mode: RGB
sources: [{url: "https://example.org/base", checked: 2026-11-25}]
unverified: [panel_label.style, panel_label.bold, panel_label.size_pt]
"""


# --- YAML: loading, extends, validation --------------------------------------------

def test_all_journal_files_load_and_bases_are_hidden():
    ids = {j["id"] for j in list_journals()}
    assert set(MAIN_JOURNALS) <= ids
    assert {"adv_mater", "materials_transactions", "nat_commun", "prl"} <= ids
    assert not any(i.startswith("_") for i in ids)
    for j in list_journals():  # every journal says where its rules come from
        assert load_journal(j["id"]).sources


def test_extends_inherits_publisher_and_merges_sources():
    acta = load_journal("acta_materialia")
    assert acta.publisher == "Elsevier" and acta.name == "Acta Materialia"
    assert acta.get("width_mm.single") == 90
    assert "panel_label.style" in acta.unverified
    nc = load_journal("Nat. Commun.")  # alias
    assert nc.id == "nat_commun"
    assert len(nc.sources) == 3 and all(isinstance(s["checked"], str) for s in nc.sources)


def test_child_value_clears_parent_unverified(tmp_path):
    _write(tmp_path, "_base", BASE)
    _write(tmp_path, "child", 'name: Child\nextends: _base\npanel_label: {style: "a", bold: true}\n')
    spec = load_journal("child", tmp_path)
    assert spec.unverified == ["panel_label.size_pt"]
    assert spec.get("panel_label.style") == "a"
    assert spec.get("panel_label.size_pt") == 7  # falls back to the body text size
    assert "panel_label.size_pt" in spec.defaults_used


@pytest.mark.parametrize("text, msg", [
    ("name: X\nextends: _base\ncolour: red\n", "知らない項目"),
    ("name: X\nextends: _base\nwidth_mm: {single: -3}\n", "正の数"),
    ("name: X\nextends: _base\npanel_label: {style: '[a]'}\n", "panel_label.style"),
    ("name: X\nextends: _base\nmin_line_pt: 0.3\nunverified: [min_line_pt]\n", "unverified にも"),
    ("name: X\nextends: _missing\n", "ありません"),
])
def test_validation_errors(tmp_path, text, msg):
    _write(tmp_path, "_base", BASE)
    _write(tmp_path, "bad", text)
    with pytest.raises(JournalSpecError, match=msg):
        load_journal("bad", tmp_path)


def test_rule_missing_and_not_unverified_is_an_error(tmp_path):
    _write(tmp_path, "nofonts", BASE.replace("fonts: [Arial]\n", ""))
    with pytest.raises(JournalSpecError, match="fonts"):
        load_journal("nofonts", tmp_path)


def test_extends_cycle(tmp_path):
    _write(tmp_path, "a", "name: A\nextends: b\n")
    _write(tmp_path, "b", "name: B\nextends: a\n")
    with pytest.raises(JournalSpecError, match="ぐるぐる"):
        load_journal("a", tmp_path)


def test_unknown_journal_lists_the_known_ones():
    with pytest.raises(JournalSpecError, match="acta_materialia"):
        load_journal("no_such_journal")


# --- layout: pixels = mm x dpi -------------------------------------------------------

@pytest.mark.parametrize("journal", [j["id"] for j in list_journals()])
def test_single_width_pixels_follow_mm_and_dpi(tmp_path, journal):
    img = _png(tmp_path / "p.png")
    res = make_journal_figure([img, img], journal, tmp_path / "out", width="single", fmt="png")
    spec = load_journal(journal)
    mm = float(spec.get("width_mm.single"))
    dpi = float(spec.get("dpi.combination"))
    with Image.open(res["files"]["png"]) as im:
        assert im.width == mm_to_px(mm, dpi)
        assert round(im.info["dpi"][0]) == round(dpi)
    assert abs(res["width_mm"] - mm) <= 0.1
    width_item = next(i for i in res["check"]["items"] if i["item"] == "幅 (mm)")
    assert width_item["status"] == "ok"


def test_double_width_and_custom_mm(tmp_path):
    img = _png(tmp_path / "p.png")
    res = make_journal_figure([img] * 3, "nature", tmp_path / "o1", width="double", fmt="tiff")
    assert res["size_px"][0] == mm_to_px(183, 500)
    with Image.open(res["files"]["tiff"]) as im:
        assert im.size == tuple(res["size_px"]) and round(im.info["dpi"][0]) == 500
    res2 = make_journal_figure([img], "nature", tmp_path / "o2", width=120.5, fmt="png")
    assert res2["width"] == "custom" and abs(res2["width_mm"] - 120.5) < 0.1


def test_tall_stack_is_shrunk_to_max_height(tmp_path):
    tall = _png(tmp_path / "tall.png", size=(400, 1600))
    res = make_journal_figure([tall, tall], "acta_materialia", tmp_path / "o", columns=1, fmt="png")
    assert res["height_mm"] <= 248 + 0.1
    assert any("縮め" in w for w in res["warnings"])
    assert res["size_px"][0] == mm_to_px(90, 500)  # still the column width


# --- panel labels ------------------------------------------------------------------

def test_panel_label_text_styles():
    assert [panel_label_text(i, "a") for i in range(3)] == ["a", "b", "c"]
    assert panel_label_text(1, "(a)") == "(b)"
    assert panel_label_text(0, "A") == "A"


def test_nature_bold_lowercase_and_aps_parentheses(tmp_path):
    img = _png(tmp_path / "p.png")
    nat = make_journal_figure([img, img], "nature", tmp_path / "n", fmt="png")
    assert [p["label"] for p in nat["panels"]] == ["a", "b"]
    assert nat["panel_label"] == {"style": "a", "bold": True, "size_pt": 8}
    prb = make_journal_figure([img, img], "prb", tmp_path / "p", fmt="png")
    assert [p["label"] for p in prb["panels"]] == ["(a)", "(b)"]
    assert prb["panel_label"]["style"] == "(a)"
    # the label is really drawn: dark pixels in the label strip above panel (a)
    with Image.open(prb["files"]["png"]) as im:
        y1 = int(round(prb["panels"][0]["y_mm"] / 25.4 * prb["dpi"]))
        strip = im.convert("L").crop((0, 0, im.width // 4, max(1, y1)))
        assert strip.getextrema()[0] < 80


# --- unverified rules -> safe defaults + warning -----------------------------------

def test_unverified_rules_use_safe_defaults_and_say_so(tmp_path):
    img = _png(tmp_path / "p.png")
    res = make_journal_figure([img, img], "acta_materialia", tmp_path / "a", fmt="tiff")
    assert res["panel_label"]["style"] == "(a)" and res["panel_label"]["bold"] is False
    assert {"panel_label.style", "panel_label.bold"} <= set(res["defaults_used"])
    assert any("panel_label.style" in w and "未確認" in w for w in res["warnings"])

    wiley = make_journal_figure([img], "adv_mater", tmp_path / "w", fmt="tiff")
    assert abs(wiley["width_mm"] - 80) < 0.1 and wiley["dpi"] == 600
    assert "width_mm.single" in wiley["defaults_used"]
    width_item = next(i for i in wiley["check"]["items"] if i["item"] == "幅 (mm)")
    assert width_item.get("limit_is_default") is True
    text_item = next(i for i in wiley["check"]["items"] if i["item"].startswith("最小の文字"))
    assert text_item["limit"] == ">= 7"  # safe default minimum


# --- the check -----------------------------------------------------------------------

def test_check_fails_for_a_too_small_low_dpi_image(tmp_path):
    small = _png(tmp_path / "small.png", size=(200, 150), dpi=72)
    res = check_image_file(small, "acta_materialia", "single")
    assert res["ok"] is False
    status = {i["item"]: i["status"] for i in res["items"]}
    assert status["幅 (mm)"] == "fail" and status["解像度 (dpi)"] == "fail"


def test_check_passes_for_own_output(tmp_path):
    img = _png(tmp_path / "p.png", size=(3000, 2000))
    res = make_journal_figure([img], "nature", tmp_path / "o", fmt="tiff")
    chk = check_image_file(res["files"]["tiff"], "nature", "single")
    assert chk["ok"] is True


def test_estimated_small_text_in_structure_view_is_a_warning(tmp_path):
    img = _png(tmp_path / "view.png", size=(1200, 1200))
    panels = [ImagePanel(img, text_frac=0.021), ImagePanel(img, text_frac=0.021), ImagePanel(img, text_frac=0.021)]
    res = make_journal_figure(panels, "prb", tmp_path / "o", fmt="png")
    est = [i for i in res["check"]["items"] if "view.png" in i["item"] and i["item"].startswith("最小の文字")]
    assert est and all(i["status"] == "warn" for i in est)
    assert res["check"]["ok"] is True  # estimates warn, they do not fail


def test_xrd_panel_text_and_lines_are_exact(tmp_path):
    peaks = [{"two_theta_deg": 30.0 + 7 * i, "intensity_rel": 100 - 15 * i, "hkl": f"(1 {-i} 0)"} for i in range(5)]
    res = make_journal_figure([XrdPanel(peaks, (10, 90), "Cu K$\\alpha$")], "appl_phys_lett", tmp_path / "o", fmt="png")
    texts = [i for i in res["check"]["items"] if i["item"].startswith("最小の文字") and "XRD" in i["item"]]
    assert texts and all(i["status"] == "ok" and i["value"] >= 8 for i in texts)  # AIP min 8 pt
    lines = [i for i in res["check"]["items"] if i["item"].startswith("最小の線") and "XRD" in i["item"]]
    assert lines and all(i["status"] == "ok" and i["value"] >= 0.5 for i in lines)


@pytest.mark.parametrize("fmt", ["pdf", "eps"])
def test_pdf_and_eps_print_size(tmp_path, fmt):
    img = _png(tmp_path / "p.png")
    res = make_journal_figure([img], "acta_materialia", tmp_path / "o", fmt=fmt)
    data = Path(res["files"][fmt]).read_bytes()
    w_pt = res["size_px"][0] * 72 / res["dpi"]
    if fmt == "pdf":
        box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)", data)
    else:
        box = re.search(rb"%%BoundingBox: 0 0 (\d+) (\d+)", data)
    assert box and abs(float(box.group(1)) - w_pt) <= 1.0
    assert Path(res["files"]["preview_png"]).is_file()


# --- CLI / MCP -------------------------------------------------------------------------

def test_cli_figure_from_cif_writes_figure_json_and_index(tmp_path, capsys):
    rc = server.main(["figure", "--journal", "acta_materialia", "--width", "single",
                      "--panels", "bulk:iso,xrd", "--cif", str(COD), "--out-dir", str(tmp_path / "run")])
    assert rc == 0
    run = tmp_path / "run"
    figs = list((run / "図").glob("journal_figure_acta_materialia_single.tif"))
    assert figs
    data = json.loads((run / "数値" / "figure_check.json").read_text(encoding="utf-8"))
    assert data["journal"] == "acta_materialia" and len(data["panels"]) == 2
    assert [p["label"] for p in data["panels"]] == ["(a)", "(b)"]
    html = (run / "はじめに読む.html").read_text(encoding="utf-8")
    assert "検査" in html and "Acta Materialia" in html


def test_cli_list_and_check(tmp_path, capsys):
    assert server.main(["figure", "--list"]) == 0
    assert "acta_materialia" in capsys.readouterr().out
    small = _png(tmp_path / "s.png", size=(100, 80), dpi=72)
    assert server.main(["figure", "--journal", "prb", "--check", str(small)]) == 2


def test_mcp_tools_registered(tmp_path):
    mcp = server.create_mcp_server()
    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"make_journal_figure", "list_journals"} <= tools

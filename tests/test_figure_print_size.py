"""Structure views drawn at the finished size (render_single_view print mode), the shared
element legend, the exact check of their text / lines, the hkl labels kept off the sticks,
and small fixes of `figure` (image names, PDF check, wavelength text, relative paths)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from matplotlib.transforms import Bbox
from PIL import Image

from shido import server
from shido.core.figure_layout import (
    XrdPanel,
    _hits_stick,
    _label_offsets,
    check_image_file,
    make_journal_figure,
    mm_to_px,
)

COD = Path(__file__).parent / "data" / "cod" / "9001735.cif"


@pytest.fixture(scope="module")
def structure_data(tmp_path_factory):
    from shido.analyzer import analyze_structure_data, load_structure_robust

    s = load_structure_robust(COD)
    s = s[0] if isinstance(s, tuple) else s
    return s, analyze_structure_data(s, tmp_path_factory.mktemp("data"))


def _render(structure_data, path, **kw):
    from shido.renderer import render_single_view

    s, d = structure_data
    a, b, c = s.lattice.matrix
    return render_single_view(s, path, a + 0.36 * b + 0.24 * c, c, render_bonds=d["render_bonds"],
                              polyhedra_envs=d["polyhedra_envs"], show_polyhedra=True, **kw)


def test_print_mode_size_and_text_pt(structure_data, tmp_path):
    rep: dict = {}
    out = _render(structure_data, tmp_path / "v.png", show_legend=True, print_width_mm=40,
                  print_height_mm=50, print_dpi=300, print_fonts=["Arial"], print_text_pt=5,
                  print_line_pt=0.5, print_report=rep)
    with Image.open(out) as im:
        assert im.size == (mm_to_px(40, 300), mm_to_px(50, 300))
        assert round(im.info["dpi"][0]) == 300
    whats = {t["what"] for t in rep["texts"]}
    assert {"縮尺の数字", "方位の a・b・c", "元素の凡例"} <= whats
    assert all(t["pt"] >= 5 for t in rep["texts"])  # rounded up, never below the asked pt
    assert {ln["what"] for ln in rep["lines"]} >= {"縮尺の棒", "単位胞の線", "方位の矢印"}
    assert all(ln["pt"] >= 0.5 for ln in rep["lines"])
    assert rep["elements"] and all(c.startswith("#") for _, c in rep["elements"])


def test_default_render_unchanged(structure_data, tmp_path, monkeypatch):
    import shido.renderer as r

    def boom(*a, **k):
        raise AssertionError("print mode used without print_width_mm")

    monkeypatch.setattr(r, "_annotate_print_image", boom)
    calls = []
    real_axes = r.add_orientation_axes
    monkeypatch.setattr(r, "add_orientation_axes", lambda *a, **k: calls.append(1) or real_axes(*a, **k))
    out = _render(structure_data, tmp_path / "d.png")
    side = r._px(1200, r._px_scale())
    with Image.open(out) as im:
        assert im.size == (side, side)
    assert calls == [1]  # the VTK triad of the normal output is still there


def test_figure_structure_panels_exact_and_shared_legend(tmp_path):
    run = tmp_path / "run"
    rc = server.main(["figure", "--journal", "nature", "--width", "double", "--panels", "bulk:iso,bulk:a,xrd",
                      "--cif", str(COD), "--format", "png", "--out-dir", str(run)])
    assert rc == 0
    data = json.loads((run / "数値" / "figure_check.json").read_text(encoding="utf-8"))
    items = data["check"]["items"]
    text_items = [i for i in items if i["item"].startswith("最小の文字")]
    assert not [i for i in items if i["status"] == "warn"]
    struct_texts = [i for i in text_items if "斜めから" in i["item"] or "a 軸" in i["item"]]
    assert struct_texts and all(i["status"] == "ok" and i["value"] >= 5 for i in struct_texts)
    assert any("元素の凡例" in i["item"] for i in text_items)
    assert not any("方向の名前" in i["item"] or "説明の帯" in i["item"] for i in text_items)
    assert [p["kind"] for p in data["panels"]] == ["structure", "structure", "plot"]
    # paths inside the run folder are relative (a rerun writes the same file)
    assert data["files"]["png"] == "図/journal_figure_nature_double.png"
    assert data["check_json"] == "数値/figure_check.json"


def test_figure_view_labels_option(tmp_path):
    from shido.journal_figure import make_journal_figure_impl

    res = make_journal_figure_impl("bulk:c", "prb", "single", COD, "png", None, None, True,
                                   tmp_path / "r", view_labels=True, caption_bar=True, legend=False)
    names = [i["item"] for i in res["check"]["items"]]
    assert any("方向の名前" in n for n in names) and any("説明の帯" in n for n in names)
    assert not any("元素の凡例" in n for n in names)
    assert res["check"]["ok"]


def test_hkl_label_moves_off_a_neighbouring_stick():
    box = Bbox.from_extents(100, 200, 140, 212)  # label over its own peak at x=120
    sticks = [(120.0, 199.0), (138.0, 260.0)]  # a taller neighbour runs through its right edge
    assert _hits_stick(box, sticks, own=0, clear=2.0)
    offsets = _label_offsets(box, sticks, own=0, clear=2.0, dpi=72.0)
    dx, dy = offsets[1]
    assert dx < 0 and dy == 0  # slid left, away from the neighbour
    moved = Bbox.from_extents(100 + dx, 200, 140 + dx, 212)
    assert not _hits_stick(moved, sticks, own=0, clear=2.0)


def test_xrd_wavelength_text_checked_without_hkl(tmp_path):
    peaks = [{"two_theta_deg": 30.0, "intensity_rel": 100.0, "hkl": ""}]
    res = make_journal_figure([XrdPanel(peaks, (10, 90), "Cu K$\\alpha$")], "prb", tmp_path / "o", fmt="png")
    assert any("波長" in i["item"] for i in res["check"]["items"])
    assert not any("パネル記号" in i["item"] for i in res["check"]["items"])  # one panel: no label


def test_image_named_xrd_needs_no_cif(tmp_path, monkeypatch):
    from shido.journal_figure import _build_panels

    monkeypatch.chdir(tmp_path)
    Image.new("RGB", (300, 200), "white").save("xrd_plot.png", dpi=(300, 300))
    objs = _build_panels(["xrd_plot.png"], None, tmp_path / "m")
    assert len(objs) == 1 and objs[0].kind == "image"


@pytest.mark.parametrize("fmt", ["PDF", "EPS"])
def test_check_vector_file_gives_clear_error(tmp_path, fmt):
    p = tmp_path / f"f.{fmt.lower()}"
    Image.new("RGB", (400, 300), "white").save(p, format=fmt)
    with pytest.raises(ValueError, match="測れません"):
        check_image_file(p, "prb")
    assert server.main(["figure", "--journal", "prb", "--check", str(p)]) == 1

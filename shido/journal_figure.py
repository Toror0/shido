"""Journal figures from a CIF: ``server.py figure`` and the MCP tools ``make_journal_figure`` /
``list_journals``. The layout / labels / check live in `crystal.core.figure_layout`, the
journal rules in ``journals/*.yaml`` (`crystal.core.journal_spec`).

Panels are given as a comma-separated list (``"bulk:iso,xrd"``):

* ``bulk:iso`` / ``bulk:a`` / ``bulk:c`` — the structure views of the CIF, rendered at the
  finished size of their cell (mm x dpi) in the journal's fonts: scale bar and a/b/c
  orientation arrows in the journal's pt (exact sizes in the check), one shared element
  legend under the panels. The view name badge ("Isometric", "[100]") and the formula /
  lattice line belong in the caption, so they are off unless ``view_labels`` /
  ``caption_bar``.
* ``bulk:overview`` — the old 3-view overview image (scaled; its text is only estimated)
* ``xrd`` or ``xrd:MoKa`` — the powder pattern, drawn at the finished size in the journal's
  fonts (exact text / line sizes)
* anything else — an existing image file (PNG / TIFF / JPEG, e.g. a slice or a graph)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from shido.core.figure_layout import (
    ImagePanel,
    StructurePanel,
    XrdPanel,
    check_image_file,
    make_journal_figure,
)
from shido.core import recipe as _recipe
from shido.core.journal_spec import JournalSpecError, list_journals, load_journal
from shido.core.output_paths import DATA_DIR, FIG_DIR, INDEX_NAME, new_run_dir, write_index_html

BULK_VIEWS: dict[str, tuple[str, str]] = {
    "iso": ("view_isometric.png", "斜めから見た図"),
    "isometric": ("view_isometric.png", "斜めから見た図"),
    "a": ("view_a_axis.png", "a 軸の方向から見た図"),
    "c": ("view_c_axis.png", "c 軸の方向から見た図"),
    "overview": ("figure_overview.png", "3 方向の図"),
}
#: camera direction / up (in a, b, c) and badge of the structure views drawn at print size
VIEW_GEOMETRY: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], str]] = {
    "iso": ((1.0, 0.36, 0.24), (0.0, 0.0, 1.0), "Isometric"),
    "isometric": ((1.0, 0.36, 0.24), (0.0, 0.0, 1.0), "Isometric"),
    "a": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "[100]"),
    "c": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "[001]"),
}
#: extra height of a structure panel for the strip with the scale bar and a/b/c arrows
STRIP_FRACTION = 0.14
MATERIAL_DIR = "素材"


def parse_panels(panels: str | list[str]) -> list[str]:
    items = panels.split(",") if isinstance(panels, str) else list(panels)
    out = [str(p).strip() for p in items if str(p).strip()]
    if not out:
        raise ValueError("パネルを 1 つ以上指定します（例 \"bulk:iso,xrd\"）")
    return out


def _structure_panel(struct: Any, key: str, data: dict[str, Any], material_dir: Path,
                     view_labels: bool, caption_bar: bool) -> StructurePanel:
    import numpy as np

    from shido.renderer import (
        _format_bulk_caption,
        _structure_elements_sorted,
        render_single_view,
        resolve_show_polyhedra,
        structure_view_aspect,
    )

    mat = np.asarray(struct.lattice.matrix, dtype=float)
    cam_abc, up_abc, badge = VIEW_GEOMETRY[key]
    cam = np.asarray(cam_abc) @ mat
    up = np.asarray(up_abc) @ mat
    caption = _format_bulk_caption(struct, summary=data.get("summary")) if caption_bar else None

    def render(out_path: Path, w_mm: float, h_mm: float, dpi: float, fonts: list[str], text_pt: float,
               line_pt: float, report: dict[str, Any]) -> Path:
        return render_single_view(
            struct, out_path, cam, up, render_bonds=data["render_bonds"],
            polyhedra_envs=data["polyhedra_envs"], show_polyhedra=resolve_show_polyhedra(None, struct),
            orientation_label=badge if view_labels else None, caption_text=caption,
            show_legend=False, print_width_mm=w_mm, print_height_mm=h_mm, print_dpi=dpi,
            print_fonts=fonts, print_text_pt=text_pt, print_line_pt=line_pt, print_report=report)

    aspect = structure_view_aspect(struct, cam, up) / (1.0 + STRIP_FRACTION + (0.06 if caption_bar else 0.0))
    fname, title = BULK_VIEWS[key]
    return StructurePanel(render, material_dir / FIG_DIR / fname.replace(".png", "_print.png"),
                          title=f"{title}（{key}）", aspect=min(1.8, max(0.55, aspect)),
                          elements=_structure_elements_sorted(struct))


def _is_cif_panel(token: str) -> bool:
    """bulk:… / xrd / xrd:… (an image file such as xrd_plot.png is not)."""
    low = token.lower()
    return low.startswith(("bulk:", "xrd:")) or low == "xrd"


def _build_panels(tokens: list[str], cif: Path | None, material_dir: Path,
                  view_labels: bool = False, caption_bar: bool = False) -> list[Any]:
    needs_cif = [t for t in tokens if _is_cif_panel(t)]
    if needs_cif and cif is None:
        raise ValueError(f"{needs_cif} には CIF が要ります（--cif）")
    struct = None
    if cif is not None and needs_cif:
        from shido.analyzer import load_structure_robust

        loaded = load_structure_robust(cif)
        struct = loaded[0] if isinstance(loaded, tuple) else loaded
    views_dir: Path | None = None
    data: dict[str, Any] | None = None
    objs: list[Any] = []
    for tok in tokens:
        low = tok.lower()
        if low.startswith("bulk:"):
            key = low.split(":", 1)[1]
            if key not in BULK_VIEWS:
                raise ValueError(f"知らない構造の図 {tok!r}（使える: {', '.join('bulk:' + k for k in BULK_VIEWS)}）")
            if key in VIEW_GEOMETRY:
                if data is None:
                    from shido.analyzer import analyze_structure_data

                    data = analyze_structure_data(struct, material_dir)
                objs.append(_structure_panel(struct, key, data, material_dir, view_labels, caption_bar))
                continue
            if views_dir is None:
                from shido.server import run_bulk_analysis

                run_bulk_analysis(struct, material_dir)
                views_dir = material_dir / FIG_DIR
            fname, title = BULK_VIEWS[key]
            objs.append(ImagePanel(views_dir / fname, title=f"{title}（{fname}）", text_frac=None))
        elif low == "xrd" or low.startswith("xrd:"):
            from shido.xrd import compute_xrd

            wl = tok.split(":", 1)[1] if ":" in tok else None
            res = compute_xrd(struct, material_dir / "xrd", wavelength=wl, make_figure=False)
            label = str(res["wavelength_label"]).replace("Ka", " K$\\alpha$")
            objs.append(XrdPanel(res["peaks"], res["two_theta_range"], wavelength_label=label,
                                 title="XRD"))
        else:
            p = Path(tok)
            if not p.is_file():
                raise ValueError(f"パネル {tok!r} は bulk:… / xrd でも、ある画像ファイルでもありません")
            objs.append(ImagePanel(p))
    return objs


def _index_notes(res: dict[str, Any]) -> list[str]:
    chk = res["check"]
    notes = [f"雑誌 {res['journal_name']}（{res['publisher']}）、幅 {res['width']} = {res['width_mm']} mm、"
             f"高さ {res['height_mm']} mm、{res['dpi']:g} dpi、{res['format'].upper()}。"
             + ("検査：合格（決まりを破っている所はありません）。" if chk["ok"] else "検査：直す所があります（下の fail）。")]
    for it in chk["items"]:
        if it["status"] != "ok":
            notes.append(f"[{it['status']}] {it['item']}：{it['value']}（決まり {it['limit']}）{it['note']}")
    notes += res["warnings"]
    notes.append("検査の全部は 数値/figure_check.json にあります。")
    return notes


def make_journal_figure_impl(
    panels: str | list[str],
    journal: str,
    width: str | float = "single",
    cif: str | Path | None = None,
    fmt: str | None = None,
    dpi: float | None = None,
    columns: int | None = None,
    labels: bool = True,
    output_dir: str | Path | None = None,
    view_labels: bool = False,
    caption_bar: bool = False,
    legend: bool = True,
) -> dict[str, Any]:
    """Shared by the MCP tool and the CLI. New run folder unless ``output_dir`` is given.

    ``view_labels`` / ``caption_bar`` put the view name badge / the formula and lattice line
    into the structure panels (off: they belong in the caption); ``legend`` adds the shared
    element legend under the panels.
    """
    spec = load_journal(journal)
    tokens = parse_panels(panels)
    cif_path = Path(cif) if cif else None
    if cif_path is not None and not cif_path.is_file():
        raise ValueError(f"file not found: {cif_path}")
    if output_dir:
        run = Path(output_dir)
        run.mkdir(parents=True, exist_ok=True)
    else:
        run = new_run_dir("figure", f"{spec.id}_{cif_path.stem if cif_path else 'images'}")
    objs = _build_panels(tokens, cif_path, run / MATERIAL_DIR, view_labels, caption_bar)
    res = make_journal_figure(objs, spec, run / FIG_DIR, width=width, fmt=fmt, dpi=dpi,
                              columns=columns, labels=labels, legend=legend)
    data_dir = run / DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    check_path = data_dir / "figure_check.json"
    root = run.resolve()
    # paths inside the run folder are written relative, so a rerun elsewhere gives the same file
    saved = dict(res, files={k: _rel(v, root) for k, v in res["files"].items()},
                 check_json=_rel(check_path, root))
    check_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    res["check_json"] = str(check_path.resolve())
    settings = {"journal": f"{spec.name} ({spec.id})", "width": res["width"], "panels": ", ".join(tokens),
                "format": res["format"], "dpi": res["dpi"]}
    res["output_dir"] = str(root)
    res["index_html"] = str(write_index_html(
        run, title=f"雑誌の図：{spec.name}", source=str(cif_path.resolve()) if cif_path else None,
        settings=settings, notes=_index_notes(res), result=res))
    if _recipe.current() is not None:
        res["recipe"] = str(root / _recipe.RECIPE_NAME)
    return res


def _rel(path: str | Path, root: Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return str(p)


def check_file_impl(path: str | Path, journal: str, width: str | float = "single",
                    dpi: float | None = None) -> dict[str, Any]:
    return check_image_file(path, journal, width=width, dpi=dpi)


def _print_check(chk: dict[str, Any]) -> None:
    print("検査: " + ("合格" if chk["ok"] else "直す所あり"))
    for it in chk["items"]:
        print(f"  [{it['status']}] {it['item']}: {it['value']} (limit {it['limit']}) {it['note']}")


def run_figure_cli(argv: list[str]) -> int:
    """`server.py figure --journal acta_materialia --width single --panels "bulk:iso,xrd" --cif X.cif`"""
    p = argparse.ArgumentParser(
        prog="Shido figure",
        description="Journal-ready figure: panels at a journal column width, panel labels, format, check.",
    )
    p.add_argument("--journal", help="journal id or abbreviation (see --list), e.g. acta_materialia, prb, nature")
    p.add_argument("--width", default="single", help="single / one_half / double or mm (default single)")
    p.add_argument("--panels", help='comma list: bulk:iso, bulk:a, bulk:c, bulk:overview, xrd, xrd:MoKa, or image files')
    p.add_argument("--cif", default=None, help="CIF for bulk:* / xrd panels")
    p.add_argument("--format", dest="fmt", default=None, help="tiff / pdf / eps / png / jpeg (default: what the journal takes)")
    p.add_argument("--dpi", type=float, default=None, help="override the journal's dpi")
    p.add_argument("--columns", type=int, default=None, help="panels per row (default: up to 3 in one row)")
    p.add_argument("--no-labels", action="store_true", help="no panel labels")
    p.add_argument("--view-labels", action="store_true",
                   help='write the view name ("Isometric", "[100]") in the structure panels')
    p.add_argument("--caption-bar", action="store_true",
                   help="write the formula / lattice line under the structure panels")
    p.add_argument("--no-legend", action="store_true", help="no shared element legend")
    p.add_argument("--out-dir", default=None, help="write here instead of a new 出力/ folder")
    p.add_argument("--list", action="store_true", help="list the journals and exit")
    p.add_argument("--check", default=None, metavar="IMAGE", help="only check an existing image against the journal")
    args = p.parse_args(argv)
    try:
        if args.list:
            for j in list_journals():
                w = j["width_mm"]
                print(f"  {j['id']:<24} {j['publisher']:<34} single={w['single']} double={w['double']} "
                      f"(unverified {len(j['unverified'])})")
            return 0
        if not args.journal:
            print("Error: --journal is required (see --list)", file=sys.stderr)
            return 1
        if args.check:
            res = check_file_impl(args.check, args.journal, args.width, args.dpi)
            _print_check(res)
            return 0 if res["ok"] else 2
        if not args.panels:
            print("Error: --panels is required", file=sys.stderr)
            return 1
        res = make_journal_figure_impl(args.panels, args.journal, args.width, args.cif, args.fmt, args.dpi,
                                       args.columns, not args.no_labels, args.out_dir,
                                       args.view_labels, args.caption_bar, not args.no_legend)
    except (JournalSpecError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{res['journal_name']}: {res['width']} {res['width_mm']} x {res['height_mm']} mm, "
          f"{res['dpi']:g} dpi, {res['format']}")
    for kind, f in res["files"].items():
        print(f"  {kind}: {f}")
    _print_check(res["check"])
    for w in res["warnings"]:
        print(f"  note: {w}")
    print(f"保存先: {res['output_dir']}")
    print(f"{INDEX_NAME}: {res['index_html']}")
    return 0


__all__ = ["check_file_impl", "make_journal_figure_impl", "parse_panels", "run_figure_cli"]

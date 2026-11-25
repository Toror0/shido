"""Where results are saved.

* Every run gets a new folder ``crystal-mcp/出力/<YYYY-MM-DD_HHMMSS>_<what>/``
  (`new_run_dir`). Giving ``output_dir`` / ``--out-dir`` keeps the old behaviour.
* Inside, the same sub-folders everywhere: 図/ (figures), XRD/, 3D/ (HTML, STL,
  VRML), 数値/ (CSV, JSON), 構造/ (CIF, POSCAR, XYZ, ...).
* `write_index_html` puts ``はじめに読む.html`` at the top: one offline file with
  what was run, thumbnails and links to every file, for beginners.

``CRYSTAL_MCP_OUTPUT_ROOT`` (environment variable) moves the ``出力`` folder (tests).
"""

from __future__ import annotations

import datetime as _dt
import html
import itertools
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

FIG_DIR = "図"
XRD_DIR = "XRD"
MODEL_DIR = "3D"
DATA_DIR = "数値"
STRUCT_DIR = "構造"
SUBDIRS: tuple[str, ...] = (FIG_DIR, XRD_DIR, MODEL_DIR, DATA_DIR, STRUCT_DIR)

OUTPUT_ROOT_NAME = "出力"
INDEX_NAME = "はじめに読む.html"
ENV_OUTPUT_ROOT = "CRYSTAL_MCP_OUTPUT_ROOT"


def output_root() -> Path:
    """``crystal-mcp/出力`` (next to the exe when frozen; env CRYSTAL_MCP_OUTPUT_ROOT wins)."""
    env = os.environ.get(ENV_OUTPUT_ROOT)
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parents[2]  # crystal/core/ -> repo root
    return base / OUTPUT_ROOT_NAME


def _safe_part(text: Any) -> str:
    """Folder-name-safe text (Japanese kept; Windows-forbidden characters -> '_')."""
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text)).strip("._ ")
    return s[:60]


def run_label(kind: str = "", name: str = "") -> str:
    parts = [p for p in (_safe_part(kind), _safe_part(name)) if p]
    return "_".join(parts) or "run"


def new_run_dir(
    kind: str = "",
    name: str = "",
    root: str | Path | None = None,
    now: _dt.datetime | None = None,
) -> Path:
    """Create and return a new, empty ``<root>/<YYYY-MM-DD_HHMMSS>_<kind>_<name>`` folder.

    ``kind`` "" is a plain CIF analysis (``..._<CIF name>``); e.g. kind "batch" gives
    ``..._batch_<folder>``. A folder made in the same second gets ``_2``, ``_3``, ...
    """
    base_root = Path(root) if root is not None else output_root()
    base_root.mkdir(parents=True, exist_ok=True)
    stamp = (now or _dt.datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    base = f"{stamp}_{run_label(kind, name)}"
    for i in itertools.count(1):
        cand = base_root / (base if i == 1 else f"{base}_{i}")
        try:
            cand.mkdir()
        except FileExistsError:
            continue
        return cand.resolve()
    raise RuntimeError("unreachable")  # pragma: no cover


def run_dir_or_new(output_dir: str | Path | None, kind: str = "", name: str = "") -> tuple[Path, bool]:
    """(folder, is_new): the given folder as-is, else a new run folder."""
    if output_dir:
        return Path(output_dir), False
    return new_run_dir(kind, name), True


# ---------------------------------------------------------------------------
# はじめに読む.html
# ---------------------------------------------------------------------------

SUBDIR_NOTES: dict[str, str] = {
    FIG_DIR: "結晶構造の図（PNG など）。論文やスライドにそのまま使えます。",
    XRD_DIR: "粉末回折（XRD・中性子）の図と、ピークの一覧（CSV）。",
    MODEL_DIR: "3D のファイル。HTML はブラウザで開くとマウスで回せます。STL は 3D プリンタ用です。",
    DATA_DIR: "数値の表（CSV は Excel で開けます）と JSON。",
    STRUCT_DIR: "構造のファイル（CIF・POSCAR など）。VESTA や計算ソフトで開けます。",
}

SETTING_LABELS: dict[str, str] = {
    "input": "入力",
    "miller_index": "切り出し面 (h k l)",
    "show_polyhedra": "配位多面体を描く",
    "transparent": "背景を透過",
    "theme": "カラーテーマ",
    "xrd": "XRD（粉末回折）",
    "wavelength": "X 線の波長",
    "xrd_options": "XRD の詳しい設定",
    "stl": "STL（3D プリント用）",
    "stl_split": "STL を元素ごとに分ける",
    "stl_max_mm": "STL の最大サイズ (mm)",
    "html": "回せる 3D（HTML）",
    "style": "描き方",
    "poscar_cell": "POSCAR の格子",
    "view_dir": "追加の視線 [u v w]",
    "view_normal": "追加の視線 (h k l) の法線",
    "perspective": "遠近法で描く",
    "labels": "原子のラベル",
    "bonds": "結合の指定",
    "polyhedra_center": "多面体の中心の元素",
    "bond_color": "結合の色",
    "frac_range": "描く範囲（分率座標）",
    "image_formats": "追加の画像形式",
    "show_planes": "描いた格子面",
    "vectors": "矢印（磁気モーメントなど）",
    "volume": "等値面のファイル",
    "isovalue": "等値面の値",
    "all_blocks": "CIF の全データブロック",
    "formula": "組成",
    "operations": "編集の操作の数",
    "ordering": "規則度",
    "formats": "書き出した形式",
    "animate": "回転アニメーションの形式",
    "animate_frames": "アニメーションのコマ数",
    "animate_axis": "アニメーションの回転軸",
    "supercell": "スーパーセル",
    "export_formats": "追加の書き出し形式",
    "probability": "熱振動楕円体の確率 (%)",
    "journal": "雑誌",
    "width": "段の幅",
    "panels": "パネル",
    "format": "形式",
    "dpi": "解像度 (dpi)",
}

_EXACT_NOTES: dict[str, str] = {
    "lattice_summary.csv": "格子定数（a, b, c と角度）・体積・密度・空間群などのまとめ",
    "bond_lengths.csv": "原子どうしの結合の長さ（Å）の一覧",
    "bond_angles.csv": "結合どうしの角度（°）の一覧",
    "atomic_positions.csv": "各原子の位置（分率座標と Å 単位の座標）",
    "surface_summary.csv": "切り出したスラブ（表面）ごとの厚さ・原子数などのまとめ",
    "batch_summary.csv": "まとめて解析した CIF ごとの結果の一覧（成功・失敗も）",
    "phase_comparison.csv": "変態前後の 2 つの相の格子・体積などの比較",
    "sublattice_summary.csv": "副格子ごとの元素の入り方のまとめ",
    "pair_distances.csv": "強調した元素ペアの距離の一覧",
    "ca_sweep_table.csv": "c/a を変えた構造の一覧（格子定数・体積）",
    "demo_ca_energy.csv": "c/a ごとのエネルギー（デモ用の模型の値。DFT ではありません）",
    "energy_minima.csv": "エネルギーがいちばん低くなる c/a の候補",
    "symmetry_info.json": "空間群・対称操作・サイト対称のまとめ（JSON）",
    "symmetry_sites.csv": "サイトごとの Wyckoff 記号・多重度・サイト対称",
    "symmetry_operations.csv": "対称操作（x, y, z の形）の一覧",
    "polyhedra.csv": "配位多面体ごとの体積・ゆがみ・結合原子価和",
    "polyhedra.json": "配位多面体の解析のまとめ（JSON）",
    "structure_factors.csv": "構造因子 F(hkl) の一覧（大きさ・位相・消滅則）",
    "structure_factors.json": "構造因子の計算条件とまとめ（JSON）",
    "displacement.csv": "原子の熱振動（U_eq・楕円体の軸）の一覧",
    "displacement.json": "熱振動の解析のまとめ（JSON）",
    "measure.json": "距離・角度などの測定結果（JSON）",
    "xrd_peaks.csv": "XRD のピーク一覧（2θ・強度・hkl）",
    "neutron_peaks.csv": "中性子回折のピーク一覧（2θ・強度・hkl）",
    "xrd_profile.csv": "XRD を曲線にしたデータ（2θ と強度）",
    "neutron_profile.csv": "中性子回折を曲線にしたデータ（2θ と強度）",
    "xrd_observed_compare.csv": "測定データと計算の比べ（差も）",
    "neutron_observed_compare.csv": "測定データと計算の比べ（差も）",
    "structure.cif": "解析した構造の CIF",
    "recipe.json": "この結果の作り方（レシピ）：ツールと部品の版・設定・入力と結果のファイルの指紋。rerun で同じものをもう一度作れます",
    "POSCAR": "VASP 用の構造ファイル（POSCAR）",
    "structure.html": "ブラウザで開くとマウスで回せる 3D の構造",
    "structure.stl": "3D プリンタ用のモデル（mm 単位）",
    "figure_overview.png": "いろいろな向きから見た図をまとめた 1 枚",
    "figure_check.json": "雑誌の図の検査の結果（幅・高さ・dpi・形式・最小の文字と線、未確認で既定を使った項目）",
    "batch_gallery.png": "まとめて解析した構造の図の一覧",
}

_PATTERN_NOTES: list[tuple[str, str]] = [
    (r"^view_a_axis\.", "a 軸の方向から見た図"),
    (r"^view_c_axis\.", "c 軸の方向から見た図"),
    (r"^view_isometric\.", "斜めから見た図"),
    (r"rotation\.gif$", "構造がひと回りするアニメーション（GIF。スライドにそのまま貼れます）"),
    (r"rotation\.mp4$", "構造がひと回りするアニメーション（MP4 動画）"),
    (r"^view_custom\.", "指定した方向から見た図"),
    (r"^journal_figure_.*_見本\.png$", "雑誌の図の見本（画面で見る用の小さい PNG。投稿には同じ名前の TIFF などを使う）"),
    (r"^journal_figure_", "雑誌の決まりどおりに仕上げた図（段の幅・パネル記号・形式）"),
    (r"^figure_xrd_observed\.", "測定データと計算の XRD を重ねた図"),
    (r"^figure_neutron_observed\.", "測定データと計算の中性子回折を重ねた図"),
    (r"^figure_xrd\.", "計算した粉末 XRD パターンの図"),
    (r"^figure_neutron\.", "計算した中性子回折パターンの図"),
    (r"^slab_.*_side\.", "スラブを横から見た図"),
    (r"^slab_.*_top\.", "スラブを上から見た図"),
    (r"^slab_.*\.cif$", "切り出したスラブ（表面）の CIF"),
    (r"^POSCAR_slab_", "切り出したスラブの POSCAR（VASP 用）"),
    (r"^POSCAR_ca_", "c/a を変えた構造の POSCAR（VASP 用）"),
    (r"^ca_.*\.cif$", "c/a を変えた構造の CIF"),
    (r"^POSCAR", "VASP 用の構造ファイル（POSCAR）"),
    (r"^heusler_.*\.cif$", "ホイスラー合金の置換モデルの CIF"),
    (r"^figure_phase_compare\.", "変態前後の 2 相を同じ縮尺で並べた図"),
    (r"^panel_", "相ごとの図"),
    (r"^figure_heusler_model", "ホイスラー合金の副格子モデルの図"),
    (r"^figure_ca_energy\.", "c/a とエネルギーの関係の図"),
    (r"^structure_[A-Za-z]+\.stl$", "元素ごとに分けた STL（色分け印刷用）"),
    (r"^structure_bonds\.stl$", "結合と単位胞の STL（色分け印刷用）"),
    (r"^slice_.*\.png$", "体積データの断面図（色の地図＋等高線、軸は Å）"),
    (r"^slice_.*\.csv$", "体積データの断面の数値（面内の座標と値）"),
    (r"^line_profile_.*\.png$", "体積データのラインプロファイル（線に沿った値のグラフ）"),
    (r"^line_profile_.*\.csv$", "ラインプロファイルの数値（距離と値）"),
    (r"^model_density_xray\.", "構造から計算したモデル電子密度（X 線、電子/Å³。VESTA・slice・--volume で開けます）"),
    (r"^model_density_neutron\.", "構造から計算した核密度（中性子の散乱長密度、fm/Å³）"),
    (r"^patterson\.", "パターソン関数（原子間ベクトルの地図。原点がいちばん高い）"),
    (r"^diff\.(cube|grd)$", "2 つの体積データの差（入力 − 比較相手）"),
    (r"^morphology\.png$", "結晶外形（モルフォロジー）の図。面の色は {hkl} ごと、凡例は面積の割合"),
    (r"^morphology_faces\.csv$", "結晶外形の面ごとの面積の表（等価な (hkl) 1 枚ずつ）"),
    (r"^morphology_families\.csv$", "結晶外形の {hkl} ごとの面積の合計と割合・体積"),
    (r"^overlay\.png$", "複数の構造を重ねた図（土台の構造に、ずらし・回転した別の構造を重ねたもの。表面の分子・粒界など）"),
    (r"^molecules\.csv$", "分子の一覧（結合でつながった原子のまとまりごとの組成・重心。extended=yes はつながりが無限に続く網目）"),
    (r"^hbonds\.csv$", "水素結合の一覧（供与体-H…受容体の距離と角度）"),
    (r"^morphology\.stl$", "結晶外形の 3D プリンタ用モデル（mm）"),
    (r"^morphology\.html$", "結晶外形をブラウザでマウスで回して見る 3D"),
    (r"\.cube?$", "体積データ（Gaussian Cube 形式）"),
    (r"\.grd$", "体積データ（VESTA の grd 形式）"),
    (r"_transformed\.cif$", "格子を変換した構造の CIF"),
    (r"_transformed\.vasp$", "格子を変換した構造の POSCAR（VASP 用）"),
    (r"_edited\.cif$", "原子を編集した構造の CIF（追加・削除・置き換えなど。検算ずみ）"),
    (r"_edited\.vasp$", "原子を編集した構造の POSCAR（VASP 用）"),
    (r"_created\.cif$", "空間群と非対称単位から作った構造の CIF（検算ずみ）"),
    (r"_created\.vasp$", "空間群と非対称単位から作った構造の POSCAR（VASP 用）"),
    (r"_P1\.cif$", "対称性を P1 にした CIF（全原子を書き出し）"),
    (r"_all\.cif$", "すべての構造を 1 つにまとめた CIF"),
    (r"\.xyz$", "XYZ 形式（原子の座標だけ）"),
    (r"\.xtl$", "XTL 形式（格子と原子）"),
    (r"\.pdb$", "PDB 形式"),
    (r"\.wrl$", "VRML 形式の 3D モデル"),
    (r"\.cif$", "CIF（結晶構造のファイル）"),
    (r"\.html?$", "ブラウザで開くとマウスで回せる 3D"),
    (r"\.stl$", "3D プリンタ用のモデル"),
    (r"\.csv$", "数値の表（Excel で開けます）"),
    (r"\.json$", "数値のまとめ（JSON）"),
    (r"\.svg$", "SVG 形式（拡大してもきれい）"),
    (r"\.(tiff?|jpe?g|pdf|eps)$", "別の形式の画像"),
]

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif"}


def describe_file(name: str) -> str:
    """One-line Japanese note for a result file name."""
    if name in _EXACT_NOTES:
        return _EXACT_NOTES[name]
    for pat, note in _PATTERN_NOTES:
        if re.search(pat, name, flags=re.I):
            return note
    return ""


def _href(path: Path, base: Path) -> str:
    rel = os.path.relpath(path, base).replace(os.sep, "/")
    return quote(rel, safe="/._-")


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "する" if value else "しない"
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={_fmt_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _category(rel_parts: tuple[str, ...]) -> tuple[str, str]:
    """(group = sub-folder path before the category, category name or '')."""
    for i, part in enumerate(rel_parts[:-1]):
        if part in SUBDIRS:
            return "/".join(rel_parts[:i]), part
    return "/".join(rel_parts[:-1]), ""


MAX_LISTED_FILES = 3000  # a huge folder given by mistake must not hang the run


def collect_files(run_dir: str | Path) -> dict[str, dict[str, list[Path]]]:
    """{group: {category: [files]}}; category '' = other files of that group."""
    root = Path(run_dir)
    groups: dict[str, dict[str, list[Path]]] = {}
    found = itertools.islice((p for p in root.rglob("*") if p.is_file()), MAX_LISTED_FILES)
    for p in sorted(found, key=lambda q: str(q.relative_to(root)).lower()):
        if p.name == INDEX_NAME:
            continue
        group, cat = _category(p.relative_to(root).parts)
        groups.setdefault(group, {}).setdefault(cat, []).append(p)
    return groups


_CSS = """
body{font-family:"Yu Gothic UI","Meiryo",sans-serif;margin:0;background:#f6f7f9;color:#222;line-height:1.6}
main{max-width:1100px;margin:0 auto;padding:16px 20px 48px}
h1{font-size:1.5em;margin:.4em 0}h2{font-size:1.25em;border-left:6px solid #2f6fb3;padding-left:8px;margin-top:1.6em}
h3{font-size:1.05em;margin:1.2em 0 .4em}
.box{background:#fff;border:1px solid #dde1e6;border-radius:8px;padding:12px 16px;margin:10px 0}
table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #dde1e6;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#eef2f6;white-space:nowrap}
.grid{display:flex;flex-wrap:wrap;gap:12px}
.card{background:#fff;border:1px solid #dde1e6;border-radius:8px;padding:8px;width:240px}
.card img{width:100%;height:180px;object-fit:contain;background:#fafafa}
.card.wide{width:100%;max-width:720px}.card.wide img{height:auto;max-height:480px}
.small{font-size:.88em;color:#555}a{color:#1f5fa8}code{background:#eef2f6;padding:0 4px;border-radius:3px}
"""

_CAT_TITLES = {
    FIG_DIR: "図（図/）",
    XRD_DIR: "XRD・中性子回折（XRD/）",
    MODEL_DIR: "3D（3D/）",
    DATA_DIR: "数値（数値/）",
    STRUCT_DIR: "構造ファイル（構造/）",
    "": "そのほかのファイル",
}


def _file_table(files: list[Path], base: Path, open_note: str = "") -> str:
    rows = []
    for f in files:
        note = describe_file(f.name) or open_note
        rows.append(
            f'<tr><td><a href="{_href(f, base)}">{html.escape(f.name)}</a></td>'
            f"<td>{html.escape(note)}</td></tr>"
        )
    return "<table><tr><th>ファイル</th><th>中身</th></tr>" + "".join(rows) + "</table>"


def _image_cards(files: list[Path], base: Path, wide: bool = False) -> str:
    cards = []
    for f in files:
        href = _href(f, base)
        note = describe_file(f.name)
        cards.append(
            f'<div class="card{" wide" if wide else ""}"><a href="{href}"><img src="{href}" '
            f'alt="{html.escape(f.name)}"></a><div><a href="{href}">{html.escape(f.name)}</a></div>'
            f'<div class="small">{html.escape(note)}</div></div>'
        )
    return '<div class="grid">' + "".join(cards) + "</div>"


def _section(cat: str, files: list[Path], base: Path) -> str:
    out = [f"<h3>{html.escape(_CAT_TITLES.get(cat, cat))}</h3>"]
    if cat in SUBDIR_NOTES:
        out.append(f'<p class="small">{html.escape(SUBDIR_NOTES[cat])}</p>')
    images = [f for f in files if f.suffix.lower() in _IMAGE_EXT]
    others = [f for f in files if f.suffix.lower() not in _IMAGE_EXT]
    if cat == MODEL_DIR:
        htmls = [f for f in files if f.suffix.lower() in {".html", ".htm"}]
        for f in htmls:
            out.append(
                f'<p><a href="{_href(f, base)}"><b>{html.escape(f.name)}</b> を開く</a>'
                "（ブラウザでマウスを使って回せます。インターネットは使いません）</p>"
            )
    if images:
        out.append(_image_cards(images, base, wide=(cat == XRD_DIR)))
    if others:
        if images:
            out.append('<p class="small">同じ図のほかの形式・そのほかのファイル：</p>')
        out.append(_file_table(others, base))
    return "".join(out)


def write_index_html(
    run_dir: str | Path,
    *,
    title: str,
    source: str | None = None,
    settings: Mapping[str, Any] | None = None,
    notes: list[str] | None = None,
    when: _dt.datetime | None = None,
    result: Any = None,
) -> Path:
    """Write ``<run_dir>/はじめに読む.html`` listing every file under run_dir; returns its path.

    Inside a recorded CLI command / MCP tool (`crystal.core.recipe`), ``recipe.json`` is
    written first (``result`` adds its warnings) and the page gets 「この図の作り方（レシピ）」.
    """
    from shido.core.recipe import index_section, write_recipe

    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    recipe = write_recipe(root, result, title)
    stamp = (when or _dt.datetime.now()).strftime("%Y年%m月%d日 %H:%M:%S")
    groups = collect_files(root)

    head = [
        f"<h1>{html.escape(title)}</h1>",
        '<div class="box"><table>',
        f"<tr><th>した日時</th><td>{html.escape(stamp)}</td></tr>",
    ]
    if source:
        head.append(f"<tr><th>入力</th><td>{html.escape(str(source))}</td></tr>")
    head.append(f"<tr><th>この結果の場所</th><td><code>{html.escape(str(root))}</code></td></tr>")
    head.append("</table></div>")

    shown = {k: v for k, v in (settings or {}).items() if v not in (None, "", [], {}, ())}
    if shown:
        head.append("<h2>使った主な設定</h2><div class='box'><table>")
        for k, v in shown.items():
            label = SETTING_LABELS.get(k, k)
            head.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(_fmt_value(v))}</td></tr>")
        head.append("</table></div>")
    for n in notes or []:
        head.append(f'<p class="box small">{html.escape(n)}</p>')
    if recipe:
        head.append(index_section(recipe, root))

    present = sorted({c for g in groups.values() for c in g if c}, key=SUBDIRS.index)
    if present:
        head.append("<h2>フォルダの中身</h2><div class='box'><table>")
        for c in present:
            head.append(f"<tr><th>{html.escape(c)}/</th><td>{html.escape(SUBDIR_NOTES[c])}</td></tr>")
        head.append("</table><p class='small'>下の図をクリックすると大きく表示されます。"
                    "ファイル名をクリックすると、そのファイルを開けます。</p></div>")

    body: list[str] = []
    order = sorted(groups, key=lambda g: (g != "", g.lower()))
    for g in order:
        cats = groups[g]
        body.append(f"<h2>{'結果' if g == '' else 'フォルダ ' + html.escape(g) + '/ の結果'}</h2>")
        for c in [*SUBDIRS, ""]:
            if cats.get(c):
                body.append(_section(c, cats[c], root))
    if not groups:
        body.append("<p>ファイルはありません。</p>")

    doc = (
        "<!DOCTYPE html>\n<html lang=\"ja\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        "<body><main>" + "".join(head) + "".join(body)
        + "<p class='small'>CrystalAnalyzerMCP が作りました。このファイルはインターネットを使いません。</p>"
        "</main></body></html>\n"
    )
    path = root / INDEX_NAME
    path.write_text(doc, encoding="utf-8")
    return path


__all__ = [
    "DATA_DIR",
    "FIG_DIR",
    "INDEX_NAME",
    "MODEL_DIR",
    "OUTPUT_ROOT_NAME",
    "STRUCT_DIR",
    "SUBDIRS",
    "XRD_DIR",
    "collect_files",
    "describe_file",
    "new_run_dir",
    "output_root",
    "run_dir_or_new",
    "run_label",
    "write_index_html",
]

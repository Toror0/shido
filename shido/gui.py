"""Simple desktop UI (tkinter) for Shido.

Allows researchers to select CIF files, major crystallographic planes (Miller indices),
Heusler alloy composition/elements, and color themes, then run analysis and open the
output folder directly. Uses only Python's standard library tkinter.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Sequence

_MODULE_DIR = str(Path(__file__).resolve().parents[1])  # repo root (holds the crystal package)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from shido.analyzer import (  # noqa: E402
    build_ni2mnga_structure,
    load_structure_robust,
    sanitize_name,
    structure_stem,
)
from shido.core import __version__  # noqa: E402
from shido.ca_sweep import generate_ca_sweep, generate_demo_ca_energy_csv, plot_ca_energy_curve  # noqa: E402
from shido.heusler_builder import generate_heusler_models  # noqa: E402
from shido.phase_compare import build_ni2mnga_martensite_l10, compare_phases  # noqa: E402
from shido.core.output_paths import DATA_DIR, INDEX_NAME, new_run_dir, output_root  # noqa: E402
from shido.server import (  # noqa: E402
    _ellipsoid_kwargs,
    _finish_run,
    _parse_wavelength,
    _shown_options,
    _volume_kwargs,
    _xrd_extension_options,
    run_batch,
    run_bulk_analysis,
    run_demo,
)
from shido.slab_cutter import MAJOR_PLANES, cut_slabs, resolve_miller_index  # noqa: E402
from shido.theme_manager import list_themes, set_active_theme  # noqa: E402
from shido import settings as cfg  # noqa: E402
from shido.settings_tab import SettingsTab  # noqa: E402

ELEMENT_CHOICES = ["Ni", "Mn", "Ga", "In", "Fe", "Co", "Sn", "Sb", "Ti", "Al", "Cu"]

# POSCAR の格子の選び方（画面の表示 → server の値）
POSCAR_CELL_LABELS = {
    "入力のまま": "input",
    "規約格子": "conventional",
    "基本格子": "primitive",
}

STYLE_LABELS: dict[str, str] = {
    "球棒 (ball-stick)": "ball-stick",
    "空間充填 (space-filling)": "space-filling",
    "スティック (stick)": "stick",
    "ワイヤフレーム (wireframe)": "wireframe",
    "熱振動楕円体 (ellipsoid)": "ellipsoid",
}

LABEL_MODE_LABELS: dict[str, str | None] = {
    "なし": None,
    "サイト名 (site)": "site",
    "元素記号 (element)": "element",
}

RADIATION_LABELS: dict[str, str] = {
    "X線 (xray)": "xray",
    "中性子 (neutron)": "neutron",
}

ANIMATE_LABELS: dict[str, str | None] = {
    "なし": None,
    "GIF": "gif",
    "MP4 (要ffmpeg)": "mp4",
}

ANIMATE_AXIS_CHOICES: tuple[str, ...] = ("up", "a", "b", "c", "x", "y", "z")
IMAGE_FORMAT_CHOICES: tuple[str, ...] = ("tiff", "jpeg", "pdf", "eps", "svg")
EXPORT_FORMAT_CHOICES: tuple[str, ...] = ("xyz", "p1", "xtl", "pdb", "wrl", "vesta")
WAVELENGTH_CHOICES: tuple[str, ...] = ("", "CuKa", "MoKa", "CoKa", "FeKa", "CrKa", "AgKa", "CuKa1")

DISORDER_CELL_NOTE = (
    "部分占有（不規則）の構造で規約格子・基本格子を選ぶと、POSCAR の組成がずれることがあります"
    "（例: B2 の Ni2MnGa で Mn が消える）。『入力のまま』をおすすめします。"
)


def _parse_optional_floats(values: Sequence[Any], expected: int, label: str) -> list[float] | None:
    texts = [str(v).strip() for v in values]
    if all(not t for t in texts):
        return None
    if any(not t for t in texts) or len(texts) != expected:
        raise ValueError(f"{label}は {expected} 個の数をすべて入れてください（全部空欄で既定）")
    try:
        return [float(t) for t in texts]
    except ValueError:
        raise ValueError(f"{label}には数を入れてください") from None


def _parse_optional_supercell(values: Sequence[Any]) -> list[int] | None:
    texts = [str(v).strip() for v in values]
    if all(not t for t in texts):
        return None
    if any(not t for t in texts) or len(texts) != 3:
        raise ValueError("スーパーセルは a b c の 3 つの正の整数を入れてください（例: 2 2 1）")
    try:
        nums = [int(t) for t in texts]
    except ValueError:
        raise ValueError("スーパーセルは整数で入れてください（例: 2 2 1）") from None
    if any(n < 1 for n in nums):
        raise ValueError("スーパーセルは 1 以上の整数にしてください")
    return None if nums == [1, 1, 1] else nums


def build_bulk_options(
    *,
    xrd: bool = False,
    stl: bool = False,
    html: bool = False,
    poscar_cell: str = "input",
    style: str = "ball-stick",
    probability: Any = "",
    view_dir: Sequence[Any] = ("", "", ""),
    perspective: bool = False,
    labels: str | None = None,
    frac_range: Sequence[Any] = ("", "", "", "", "", ""),
    supercell: Sequence[Any] = ("", "", ""),
    image_formats: Sequence[str] | str | None = None,
    export_formats: Sequence[str] | str | None = None,
    wavelength: Any = "",
    radiation: str = "xray",
    volume: Any = "",
    isovalue: Any = "",
    iso_both_signs: bool = False,
    animate: str | None = None,
    animate_frames: Any = 72,
    animate_axis: str = "up",
) -> dict[str, Any]:
    """GUI の入力値から `server.run_bulk_analysis` 用の辞書を作る（CLI と同じ名前・形）。"""
    cell_val = POSCAR_CELL_LABELS.get(poscar_cell, poscar_cell) or "input"
    style_val = STYLE_LABELS.get(style, style) or "ball-stick"

    prob_val: float | None = None
    prob_text = "" if probability is None else str(probability).strip()
    if style_val == "ellipsoid" and prob_text:
        try:
            prob_val = float(prob_text)
        except ValueError:
            raise ValueError("楕円体の確率 (%) は数で入れてください（例: 50）") from None
        if not (0.0 < prob_val < 100.0):
            raise ValueError("楕円体の確率 (%) は 0 より大きく 100 未満にしてください")

    vd = _parse_optional_floats(view_dir, 3, "視線方向 [u v w]")
    if vd is not None and all(abs(v) < 1e-12 for v in vd):
        raise ValueError("視線方向 [0 0 0] は向きになりません")

    lbl_raw = LABEL_MODE_LABELS.get(labels, labels) if isinstance(labels, str) else labels
    lbl_val = None if lbl_raw in (None, "", "none") else str(lbl_raw)

    fr = _parse_optional_floats(frac_range, 6, "描く範囲 (x0 x1 y0 y1 z0 z1)")
    if fr is not None and (fr[1] < fr[0] or fr[3] < fr[2] or fr[5] < fr[4]):
        raise ValueError("描く範囲は上限を下限以上にしてください")

    sc = _parse_optional_supercell(supercell)

    if isinstance(image_formats, str):
        img_list = [f.strip().lower() for f in image_formats.split(",") if f.strip()]
    else:
        img_list = [str(f).strip().lower() for f in (image_formats or []) if str(f).strip()]
    img_extra = [f for f in dict.fromkeys(img_list) if f != "png"]
    img_val: str | None = ",".join(img_extra) if img_extra else None

    if isinstance(export_formats, str):
        exp_list = [f.strip().lower() for f in export_formats.split(",") if f.strip()]
    else:
        exp_list = [str(f).strip().lower() for f in (export_formats or []) if str(f).strip()]
    exp_clean = list(dict.fromkeys(exp_list))
    exp_val: str | None = ",".join(exp_clean) if exp_clean else None

    wl_text = "" if wavelength is None else str(wavelength).strip()
    wl_val = _parse_wavelength(wl_text) if wl_text else None
    rad_val = RADIATION_LABELS.get(radiation, radiation) or "xray"
    xrd_opts = _xrd_extension_options(radiation=rad_val)
    xrd_on = bool(xrd or wl_val is not None or xrd_opts)

    vol_text = "" if volume is None else str(volume).strip()
    iso_text = "" if isovalue is None else str(isovalue).strip()
    vol_kw = _volume_kwargs(vol_text or None, iso_text or None, both_signs=bool(iso_both_signs))

    anim_raw = ANIMATE_LABELS.get(animate, animate) if isinstance(animate, str) else animate
    anim_val = None if anim_raw in (None, "", "none") else str(anim_raw).strip().lower()
    anim_kw: dict[str, Any] = {}
    if anim_val:
        frames_text = "" if animate_frames is None else str(animate_frames).strip()
        try:
            frames_val = 72 if not frames_text else int(frames_text)
        except ValueError:
            raise ValueError("回転アニメのコマ数は整数で入れてください（例: 72）") from None
        if not (2 <= frames_val <= 720):
            raise ValueError("回転アニメのコマ数は 2〜720 の整数にしてください")
        axis_val = str(animate_axis or "up").strip().lower()
        anim_kw = {"animate": anim_val, "animate_frames": frames_val, "animate_axis": axis_val}

    opts: dict[str, Any] = {
        "poscar_cell": cell_val,
        "xrd": xrd_on,
        "xrd_options": xrd_opts,
        "wavelength": wl_val,
        "stl": bool(stl),
        "html": bool(html),
        "style": style_val,
        "view_dir": vd,
        "perspective": bool(perspective),
        "labels": lbl_val,
        "frac_range": fr,
        "image_formats": img_val,
        **vol_kw,
        **_ellipsoid_kwargs(style_val, None, prob_val),
        **anim_kw,
    }
    if sc is not None:
        opts["supercell"] = sc
    if exp_val is not None:
        opts["export_formats"] = exp_val
    return opts


class CollapsibleFrame(ttk.Frame):
    """クリックで開け閉めできる枠（ふだんは閉じて画面をすっきり保つ）。"""

    def __init__(self, master: tk.Misc, title: str, expanded: bool = False) -> None:
        super().__init__(master)
        self._title = title
        self._expanded = bool(expanded)
        self._btn_var = tk.StringVar()
        self.toggle_btn = ttk.Button(
            self, textvariable=self._btn_var, command=self.toggle, style="Toolbutton"
        )
        self.toggle_btn.pack(fill="x", anchor="w")
        self.body = ttk.Frame(self, padding=(12, 4, 4, 2))
        self._sync()

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self._sync()

    def _sync(self) -> None:
        arrow = "▼" if self._expanded else "▶"
        self._btn_var.set(f"{arrow} {self._title}")
        if self._expanded:
            self.body.pack(fill="x", expand=True)
        else:
            self.body.pack_forget()


def warning_to_japanese(text: str) -> str:
    """読み込みの警告（英語）を、実行ログ用の日本語にする。知らない形は原文のまま返す。"""
    w = str(text)
    m = re.search(r"Space group '(.+?)' has two origin choices.*?origin choice (\d) was selected automatically", w)
    if m:
        return (f"空間群 {m.group(1)} には原点の選び方が2通りあり、CIF に書かれていなかったので、"
                f"原子の数が合う「原点 {m.group(2)}」を自動で選びました。")
    m = re.search(r"Space group '(.+?)' has two origin choices.*?origin choice 1 was assumed", w)
    if m:
        return (f"空間群 {m.group(1)} には原点の選び方が2通りあり、どちらか確かめられなかったので"
                "「原点 1」として読みました。構造が正しいか確かめてください"
                "（空間群の記号に :1 か :2 を足すと確実です）。")
    m = re.search(r"Disordered/partial-occupancy structure \((\d+) disordered sites", w)
    if m:
        return f"部分占有（不規則）のサイトが {m.group(1)} か所あります。図や POSCAR では占有率の大きい元素で代表させます。"
    if w.startswith("Recovered malformed CIF"):
        return "CIF に壊れた所があったので、直してから読み込みました。"
    if w.startswith("Loaded via Structure.from_file fallback"):
        return "通常の CIF 読み込みに失敗したので、別の方法で読み込みました。中身を確かめてください。"
    return w


class CrystalAnalyzerGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Shido v{__version__} — 結晶構造解析・論文図表生成")
        self.geometry("860x760")
        self.minsize(740, 620)
        self.last_output_dir: Path | None = None
        self.last_index_html: Path | None = None

        self._build_ui()
        # Bring window cleanly to the foreground on startup without staying pinned forever
        self.lift()
        self.attributes("-topmost", True)
        self.focus_force()
        self.after(250, lambda: self.attributes("-topmost", False))

    def _on_run_wheel(self, event: tk.Event) -> None:
        bbox = self._run_canvas.bbox("all")
        if bbox and (bbox[3] - bbox[1]) > self._run_canvas.winfo_height():
            self._run_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # タブ：「実行」（今までの画面）と「前提数値」
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        run_tab = ttk.Frame(self.notebook)
        self.notebook.add(run_tab, text="  実行  ")
        self.settings_tab = SettingsTab(self.notebook, on_saved=self._apply_settings_to_run_tab)
        self.notebook.add(self.settings_tab, text="  前提数値  ")

        # 下側（実行ボタン・保存先・実行ログ）は常に画面内に残す
        bottom = ttk.Frame(run_tab)
        bottom.pack(side="bottom", fill="x")

        btn_frame = ttk.Frame(bottom)
        btn_frame.pack(fill="x", padx=10, pady=6)
        self.run_btn = ttk.Button(btn_frame, text="▶ 解析・図表を自動出力する", command=self._start_run)
        self.run_btn.pack(side="left", ipadx=12, ipady=4)
        self.open_btn = ttk.Button(btn_frame, text="フォルダを開く", command=self._open_output, state="disabled")
        self.open_btn.pack(side="left", padx=10, ipadx=8, ipady=4)
        self.index_btn = ttk.Button(
            btn_frame, text=f"{INDEX_NAME} を開く", command=self._open_index, state="disabled"
        )
        self.index_btn.pack(side="left", padx=4, ipadx=8, ipady=4)
        self.status_var = tk.StringVar(value="待機中")
        ttk.Label(btn_frame, textvariable=self.status_var).pack(side="right")

        saved_row = ttk.Frame(bottom)
        saved_row.pack(fill="x", padx=10)
        ttk.Label(saved_row, text="保存先:").pack(side="left")
        self.saved_var = tk.StringVar(value="（まだ実行していません）")
        ttk.Entry(saved_row, textvariable=self.saved_var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=4
        )

        log_frame = ttk.LabelFrame(bottom, text="実行ログ", padding=6)
        log_frame.pack(fill="x", padx=10, pady=(4, 8))
        self.log_text = tk.Text(log_frame, height=8, font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # 上側（設定欄）は折りたたみを開いたときにもはみ出さないようスクロール可能に
        scroll_outer = ttk.Frame(run_tab)
        scroll_outer.pack(fill="both", expand=True)
        self._run_canvas = tk.Canvas(scroll_outer, highlightthickness=0)
        sb = ttk.Scrollbar(scroll_outer, orient="vertical", command=self._run_canvas.yview)
        run = ttk.Frame(self._run_canvas)
        run.bind("<Configure>", lambda _e: self._run_canvas.configure(scrollregion=self._run_canvas.bbox("all")))
        win = self._run_canvas.create_window((0, 0), window=run, anchor="nw")
        self._run_canvas.bind("<Configure>", lambda e: self._run_canvas.itemconfigure(win, width=e.width))
        self._run_canvas.configure(yscrollcommand=sb.set)
        self._run_canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._run_canvas.bind("<Enter>", lambda _e: self._run_canvas.bind_all("<MouseWheel>", self._on_run_wheel))
        self._run_canvas.bind("<Leave>", lambda _e: self._run_canvas.unbind_all("<MouseWheel>"))

        # Mode selector
        mode_frame = ttk.LabelFrame(run, text="1. 実行モード", padding=8)
        mode_frame.pack(fill="x", **pad)

        self.mode_var = tk.StringVar(value="bulk_slab")
        modes = [
            ("基本解析＋スラブ切り出し", "bulk_slab"),
            ("変態前後 2相比較", "phase_compare"),
            ("ホイスラー副格子モデル", "heusler"),
            ("DFT c/a スイープ", "ca_sweep"),
            ("フォルダ一括バッチ", "batch"),
        ]
        for i, (label, val) in enumerate(modes):
            ttk.Radiobutton(
                mode_frame, text=label, value=val, variable=self.mode_var, command=self._sync_mode_ui
            ).grid(row=i // 3, column=i % 3, sticky="w", padx=6, pady=1)

        # Input file / folder
        io_frame = ttk.LabelFrame(run, text="2. 入力データ（空欄のまま実行すると Ni2MnGa デモ構造を使用）", padding=8)
        io_frame.pack(fill="x", **pad)

        row1 = ttk.Frame(io_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="CIF / フォルダ:", width=16).pack(side="left")
        self.cif_var = tk.StringVar(value="")
        ttk.Entry(row1, textvariable=self.cif_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row1, text="CIF選択...", command=self._pick_cif).pack(side="left", padx=2)
        ttk.Button(row1, text="フォルダ...", command=self._pick_folder).pack(side="left", padx=2)

        row2 = ttk.Frame(io_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="出力先フォルダ:", width=16).pack(side="left")
        self.out_var = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row2, text="変更...", command=self._pick_out_dir).pack(side="left", padx=2)
        ttk.Label(
            io_frame,
            text=f"出力先が空欄なら、実行ごとに {output_root()} の中へ新しいフォルダ（日時_名前）を作ります。",
            foreground="#555555", wraplength=760,
        ).pack(fill="x", pady=(2, 0))

        # Dynamic container so mode-specific frames appear/disappear cleanly in place
        self.dynamic_container = ttk.Frame(run)
        self.dynamic_container.pack(fill="x", padx=10, pady=2)

        # Crystallographic plane & slab options
        self.plane_frame = ttk.LabelFrame(self.dynamic_container, text="3. 切り出し面（ミラー指数）とスラブ設定", padding=8)

        p_row1 = ttk.Frame(self.plane_frame)
        p_row1.pack(fill="x", pady=2)
        ttk.Label(p_row1, text="主要な結晶面:", width=16).pack(side="left")
        plane_labels = [p["label"] for p in MAJOR_PLANES] + ["カスタム入力（下の h k l を使用）"]
        self.plane_combo = ttk.Combobox(p_row1, values=plane_labels, state="readonly", width=44)
        self.plane_combo.current(0)
        self.plane_combo.pack(side="left", padx=4)
        self.plane_combo.bind("<<ComboboxSelected>>", self._on_plane_select)

        p_row2 = ttk.Frame(self.plane_frame)
        p_row2.pack(fill="x", pady=2)
        ttk.Label(p_row2, text="ミラー指数 (h k l):", width=16).pack(side="left")
        self.h_var = tk.StringVar(value="1")
        self.k_var = tk.StringVar(value="1")
        self.l_var = tk.StringVar(value="1")
        ttk.Entry(p_row2, textvariable=self.h_var, width=5).pack(side="left", padx=3)
        ttk.Entry(p_row2, textvariable=self.k_var, width=5).pack(side="left", padx=3)
        ttk.Entry(p_row2, textvariable=self.l_var, width=5).pack(side="left", padx=3)

        self.filter_polar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(p_row2, text="極性面を自動除外", variable=self.filter_polar_var).pack(side="left", padx=12)
        self.dedup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(p_row2, text="等価スラブ重複削除", variable=self.dedup_var).pack(side="left", padx=4)

        # Heusler options (shown only when mode == "heusler")
        self.heusler_frame = ttk.LabelFrame(self.dynamic_container, text="3. ホイスラー合金パラメータ", padding=8)

        h_row = ttk.Frame(self.heusler_frame)
        h_row.pack(fill="x", pady=2)
        ttk.Label(h_row, text="組成式:", width=8).pack(side="left")
        self.formula_var = tk.StringVar()
        ttk.Entry(h_row, textvariable=self.formula_var, width=16).pack(side="left", padx=4)
        ttk.Label(h_row, text="規則度:").pack(side="left", padx=(8, 2))
        self.ordering_var = tk.StringVar()
        ttk.Combobox(h_row, textvariable=self.ordering_var, values=["L21", "B2", "XA"], state="readonly", width=6).pack(side="left", padx=2)
        ttk.Label(h_row, text="強調ペア元素:").pack(side="left", padx=(10, 2))
        self.el1_var = tk.StringVar()
        self.el2_var = tk.StringVar()
        ttk.Combobox(h_row, textvariable=self.el1_var, values=ELEMENT_CHOICES, width=5).pack(side="left", padx=2)
        ttk.Label(h_row, text="-").pack(side="left")
        ttk.Combobox(h_row, textvariable=self.el2_var, values=ELEMENT_CHOICES, width=5).pack(side="left", padx=2)

        # Theme & output appearance（「見た目」）
        style_frame = ttk.LabelFrame(run, text="見た目・描画設定", padding=8)
        style_frame.pack(fill="x", **pad)

        s_row = ttk.Frame(style_frame)
        s_row.pack(fill="x", pady=2)
        ttk.Label(s_row, text="カラーテーマ:", width=16).pack(side="left")
        self.theme_var = tk.StringVar()
        ttk.Combobox(s_row, textvariable=self.theme_var, values=list_themes(), state="readonly", width=20).pack(side="left", padx=4)
        self.transparent_var = tk.BooleanVar()
        ttk.Checkbutton(s_row, text="背景を透過PNGにする", variable=self.transparent_var).pack(side="left", padx=12)

        # 折りたたみの「詳しい見た目」（既定は閉じている）
        self.appearance_adv = CollapsibleFrame(
            style_frame, "詳しい見た目（表示スタイル・視線・範囲・スーパーセル）", expanded=False
        )
        self.appearance_adv.pack(fill="x", pady=(4, 0))
        ab = self.appearance_adv.body

        a_row1 = ttk.Frame(ab)
        a_row1.pack(fill="x", pady=2)
        ttk.Label(a_row1, text="表示スタイル:", width=16).pack(side="left")
        self.style_var = tk.StringVar(value="球棒 (ball-stick)")
        style_combo = ttk.Combobox(
            a_row1, textvariable=self.style_var, values=list(STYLE_LABELS), state="readonly", width=24
        )
        style_combo.pack(side="left", padx=4)
        style_combo.bind("<<ComboboxSelected>>", self._sync_style_ui)
        ttk.Label(a_row1, text="楕円体の確率 (%):").pack(side="left", padx=(10, 2))
        self.probability_var = tk.StringVar(value="")
        self.probability_entry = ttk.Entry(a_row1, textvariable=self.probability_var, width=6, state="disabled")
        self.probability_entry.pack(side="left", padx=2)
        ttk.Label(a_row1, text="（空欄=50%）", foreground="#666666").pack(side="left", padx=4)

        a_row2 = ttk.Frame(ab)
        a_row2.pack(fill="x", pady=2)
        ttk.Label(a_row2, text="視線方向 [u v w]:", width=16).pack(side="left")
        self.view_u_var = tk.StringVar(value="")
        self.view_v_var = tk.StringVar(value="")
        self.view_w_var = tk.StringVar(value="")
        for var in (self.view_u_var, self.view_v_var, self.view_w_var):
            ttk.Entry(a_row2, textvariable=var, width=5).pack(side="left", padx=2)
        self.perspective_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(a_row2, text="透視投影", variable=self.perspective_var).pack(side="left", padx=12)
        ttk.Label(a_row2, text="ラベル:").pack(side="left", padx=(6, 2))
        self.labels_var = tk.StringVar(value="なし")
        ttk.Combobox(
            a_row2, textvariable=self.labels_var, values=list(LABEL_MODE_LABELS), state="readonly", width=16
        ).pack(side="left", padx=2)

        a_row3 = ttk.Frame(ab)
        a_row3.pack(fill="x", pady=2)
        ttk.Label(a_row3, text="描く範囲 (分率):", width=16).pack(side="left")
        self.range_x0_var = tk.StringVar(value="")
        self.range_x1_var = tk.StringVar(value="")
        self.range_y0_var = tk.StringVar(value="")
        self.range_y1_var = tk.StringVar(value="")
        self.range_z0_var = tk.StringVar(value="")
        self.range_z1_var = tk.StringVar(value="")
        for axis_lbl, v0, v1 in (
            ("x:", self.range_x0_var, self.range_x1_var),
            ("y:", self.range_y0_var, self.range_y1_var),
            ("z:", self.range_z0_var, self.range_z1_var),
        ):
            ttk.Label(a_row3, text=axis_lbl).pack(side="left", padx=(6, 1))
            ttk.Entry(a_row3, textvariable=v0, width=5).pack(side="left", padx=1)
            ttk.Label(a_row3, text="〜").pack(side="left")
            ttk.Entry(a_row3, textvariable=v1, width=5).pack(side="left", padx=1)
        ttk.Label(a_row3, text="（空欄=単位胞）", foreground="#666666").pack(side="left", padx=6)

        a_row4 = ttk.Frame(ab)
        a_row4.pack(fill="x", pady=2)
        ttk.Label(a_row4, text="スーパーセル (a b c):", width=16).pack(side="left")
        self.sc_a_var = tk.StringVar(value="")
        self.sc_b_var = tk.StringVar(value="")
        self.sc_c_var = tk.StringVar(value="")
        for var in (self.sc_a_var, self.sc_b_var, self.sc_c_var):
            ttk.Entry(a_row4, textvariable=var, width=4).pack(side="left", padx=2)
        ttk.Label(a_row4, text="（空欄または 1 1 1 でそのまま。例: 2 2 1）", foreground="#666666").pack(
            side="left", padx=6
        )

        # v3: 追加で出すもの（基本解析・バッチのとき）
        extra_frame = ttk.LabelFrame(run, text="追加の出力（基本解析・バッチのとき）", padding=8)
        extra_frame.pack(fill="x", **pad)
        e_row1 = ttk.Frame(extra_frame)
        e_row1.pack(fill="x", pady=2)
        self.xrd_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(e_row1, text="XRD（粉末回折パターン）", variable=self.xrd_var).pack(side="left", padx=4)
        self.stl_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(e_row1, text="STL（3D プリント用）", variable=self.stl_var).pack(side="left", padx=12)
        self.html_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(e_row1, text="HTML（マウスで回せる 3D）", variable=self.html_var).pack(side="left", padx=12)
        e_row2 = ttk.Frame(extra_frame)
        e_row2.pack(fill="x", pady=2)
        ttk.Label(e_row2, text="POSCAR の格子:", width=16).pack(side="left")
        self.poscar_cell_var = tk.StringVar(value="入力のまま")
        cell_combo = ttk.Combobox(
            e_row2, textvariable=self.poscar_cell_var, values=list(POSCAR_CELL_LABELS),
            state="readonly", width=12,
        )
        cell_combo.pack(side="left", padx=4)
        cell_combo.bind("<<ComboboxSelected>>", self._on_poscar_cell_select)
        self.cell_note_var = tk.StringVar(value="")
        ttk.Label(e_row2, textvariable=self.cell_note_var, foreground="#b35900", wraplength=520).pack(
            side="left", padx=8
        )

        # 折りたたみの「詳しい出力」（既定は閉じている）
        self.extra_adv = CollapsibleFrame(
            extra_frame,
            "詳しい出力（画像形式・書き出し形式・XRD/中性子・体積データ・回転アニメ）",
            expanded=False,
        )
        self.extra_adv.pack(fill="x", pady=(4, 0))
        eb = self.extra_adv.body

        x_row1 = ttk.Frame(eb)
        x_row1.pack(fill="x", pady=2)
        ttk.Label(x_row1, text="画像形式:", width=16).pack(side="left")
        self._png_always_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(x_row1, text="PNG (常時)", variable=self._png_always_var, state="disabled").pack(
            side="left", padx=4
        )
        self.image_format_vars: dict[str, tk.BooleanVar] = {}
        for fmt in IMAGE_FORMAT_CHOICES:
            var = tk.BooleanVar(value=False)
            self.image_format_vars[fmt] = var
            ttk.Checkbutton(x_row1, text=fmt.upper(), variable=var).pack(side="left", padx=4)

        x_row2 = ttk.Frame(eb)
        x_row2.pack(fill="x", pady=2)
        ttk.Label(x_row2, text="構造書き出し:", width=16).pack(side="left")
        self.export_format_vars: dict[str, tk.BooleanVar] = {}
        export_labels = {
            "xyz": "XYZ",
            "p1": "P1 CIF",
            "xtl": "XTL",
            "pdb": "PDB",
            "wrl": "VRML (wrl)",
            "vesta": "VESTA",
        }
        for fmt in EXPORT_FORMAT_CHOICES:
            var = tk.BooleanVar(value=False)
            self.export_format_vars[fmt] = var
            ttk.Checkbutton(x_row2, text=export_labels[fmt], variable=var).pack(side="left", padx=4)

        x_row3 = ttk.Frame(eb)
        x_row3.pack(fill="x", pady=2)
        ttk.Label(x_row3, text="XRD 波長 / 線種:", width=16).pack(side="left")
        self.xrd_wavelength_var = tk.StringVar(value="")
        ttk.Combobox(
            x_row3, textvariable=self.xrd_wavelength_var, values=list(WAVELENGTH_CHOICES), width=10
        ).pack(side="left", padx=4)
        self.xrd_radiation_var = tk.StringVar(value="X線 (xray)")
        ttk.Combobox(
            x_row3, textvariable=self.xrd_radiation_var, values=list(RADIATION_LABELS),
            state="readonly", width=15,
        ).pack(side="left", padx=4)
        ttk.Label(x_row3, text="（空欄=前提数値。Åの数値入力も可。指定するとXRDを自動ON）", foreground="#666666").pack(
            side="left", padx=4
        )

        x_row4 = ttk.Frame(eb)
        x_row4.pack(fill="x", pady=2)
        ttk.Label(x_row4, text="体積データ:", width=16).pack(side="left")
        self.volume_var = tk.StringVar(value="")
        ttk.Entry(x_row4, textvariable=self.volume_var, width=26).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(x_row4, text="参照...", command=self._pick_volume).pack(side="left", padx=2)
        ttk.Button(x_row4, text="クリア", command=lambda: self.volume_var.set("")).pack(side="left", padx=2)
        ttk.Label(x_row4, text="等値面:").pack(side="left", padx=(6, 2))
        self.isovalue_var = tk.StringVar(value="")
        ttk.Entry(x_row4, textvariable=self.isovalue_var, width=10).pack(side="left", padx=2)
        self.iso_both_signs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(x_row4, text="±両符号", variable=self.iso_both_signs_var).pack(side="left", padx=4)

        x_row5 = ttk.Frame(eb)
        x_row5.pack(fill="x", pady=2)
        ttk.Label(x_row5, text="回転アニメ:", width=16).pack(side="left")
        self.animate_var = tk.StringVar(value="なし")
        anim_combo = ttk.Combobox(
            x_row5, textvariable=self.animate_var, values=list(ANIMATE_LABELS), state="readonly", width=15
        )
        anim_combo.pack(side="left", padx=4)
        anim_combo.bind("<<ComboboxSelected>>", self._sync_animate_ui)
        ttk.Label(x_row5, text="コマ数:").pack(side="left", padx=(8, 2))
        self.animate_frames_var = tk.StringVar(value="72")
        self.animate_frames_entry = ttk.Entry(
            x_row5, textvariable=self.animate_frames_var, width=6, state="disabled"
        )
        self.animate_frames_entry.pack(side="left", padx=2)
        ttk.Label(x_row5, text="回転軸:").pack(side="left", padx=(8, 2))
        self.animate_axis_var = tk.StringVar(value="up")
        self.animate_axis_combo = ttk.Combobox(
            x_row5, textvariable=self.animate_axis_var, values=list(ANIMATE_AXIS_CHOICES),
            state="disabled", width=6,
        )
        self.animate_axis_combo.pack(side="left", padx=2)

        self._apply_settings_to_run_tab(cfg.load_settings())
        self._sync_mode_ui()
        self._sync_style_ui()
        self._sync_animate_ui()

    def _apply_settings_to_run_tab(self, data: dict) -> None:
        """前提数値の値を、実行タブの初期値に入れる（保存したときにも呼ばれる）。"""
        slab = data.get("slab", {})
        heus = data.get("heusler", {})
        rend = data.get("render", {})
        self.filter_polar_var.set(bool(slab.get("filter_polar", True)))
        self.dedup_var.set(bool(slab.get("deduplicate", True)))
        self.formula_var.set(str(heus.get("formula", "Ni50Mn35In15")))
        self.ordering_var.set(str(heus.get("ordering", "L21")))
        pair = list(heus.get("highlight_pair", ["Mn", "Mn"])) + ["Mn", "Mn"]
        self.el1_var.set(str(pair[0]))
        self.el2_var.set(str(pair[1]))
        self.theme_var.set(str(rend.get("theme", "acta_modern")))
        self.transparent_var.set(bool(rend.get("transparent_background", False)))
        mat = data.get("lattice", {})
        self.status_var.set(f"待機中（前提: {mat.get('material', '')} a = {mat.get('a_austenite')} Å）")

    def _on_plane_select(self, _event: object = None) -> None:
        idx = self.plane_combo.current()
        if 0 <= idx < len(MAJOR_PLANES):
            h, k, l = MAJOR_PLANES[idx]["hkl"]
            self.h_var.set(str(h))
            self.k_var.set(str(k))
            self.l_var.set(str(l))

    def _sync_mode_ui(self) -> None:
        mode = self.mode_var.get()
        self.plane_frame.pack_forget()
        self.heusler_frame.pack_forget()
        if mode in {"bulk_slab", "batch"}:
            self.plane_frame.pack(fill="x", pady=3)
        elif mode == "heusler":
            self.heusler_frame.pack(fill="x", pady=3)

    def _on_poscar_cell_select(self, _event: object = None) -> None:
        if POSCAR_CELL_LABELS.get(self.poscar_cell_var.get(), "input") == "input":
            self.cell_note_var.set("")
        else:
            self.cell_note_var.set("部分占有の構造では組成がずれることがあります（『入力のまま』がおすすめ）")

    def _sync_style_ui(self, _event: object = None) -> None:
        style_val = STYLE_LABELS.get(self.style_var.get(), self.style_var.get())
        self.probability_entry.configure(state="normal" if style_val == "ellipsoid" else "disabled")

    def _sync_animate_ui(self, _event: object = None) -> None:
        anim_val = ANIMATE_LABELS.get(self.animate_var.get(), self.animate_var.get())
        if anim_val:
            self.animate_frames_entry.configure(state="normal")
            self.animate_axis_combo.configure(state="readonly")
        else:
            self.animate_frames_entry.configure(state="disabled")
            self.animate_axis_combo.configure(state="disabled")

    def _bulk_options(self) -> dict[str, Any]:
        """追加の出力・見た目の設定（server.run_bulk_analysis に渡す辞書。CLI と同じ名前・形）。"""
        return build_bulk_options(
            xrd=bool(self.xrd_var.get()),
            stl=bool(self.stl_var.get()),
            html=bool(self.html_var.get()),
            poscar_cell=self.poscar_cell_var.get(),
            style=self.style_var.get(),
            probability=self.probability_var.get(),
            view_dir=(self.view_u_var.get(), self.view_v_var.get(), self.view_w_var.get()),
            perspective=bool(self.perspective_var.get()),
            labels=self.labels_var.get(),
            frac_range=(
                self.range_x0_var.get(),
                self.range_x1_var.get(),
                self.range_y0_var.get(),
                self.range_y1_var.get(),
                self.range_z0_var.get(),
                self.range_z1_var.get(),
            ),
            supercell=(self.sc_a_var.get(), self.sc_b_var.get(), self.sc_c_var.get()),
            image_formats=[fmt for fmt, var in self.image_format_vars.items() if var.get()],
            export_formats=[fmt for fmt, var in self.export_format_vars.items() if var.get()],
            wavelength=self.xrd_wavelength_var.get(),
            radiation=self.xrd_radiation_var.get(),
            volume=self.volume_var.get(),
            isovalue=self.isovalue_var.get(),
            iso_both_signs=bool(self.iso_both_signs_var.get()),
            animate=self.animate_var.get(),
            animate_frames=self.animate_frames_var.get(),
            animate_axis=self.animate_axis_var.get(),
        )

    def _log_load_warnings(self, warns: list, name: str = "") -> None:
        for w in warns or []:
            head = f"[読み込みの注意{(' ' + name) if name else ''}] "
            self._log(head + warning_to_japanese(w))

    def _warn_cell_if_disordered(self, struct, poscar_cell: str, name: str = "") -> None:
        if poscar_cell != "input" and not getattr(struct, "is_ordered", True):
            self._log(f"[注意{(' ' + name) if name else ''}] {DISORDER_CELL_NOTE}")
            self.after(0, lambda: messagebox.showwarning("POSCAR の格子", DISORDER_CELL_NOTE))

    def _log_extra_outputs(self, bulk_res: dict) -> None:
        for key, label in (
            ("view_custom", "指定方向の図"),
            ("xrd_figure_png", "XRD の図"),
            ("stl_file", "STL"),
            ("html_file", "回せる HTML"),
        ):
            if bulk_res.get(key):
                self._log(f"{label}: {bulk_res[key]}")
        for fmt, path in (bulk_res.get("exported_files") or {}).items():
            self._log(f"書き出し ({fmt.upper()}): {path}")
        if bulk_res.get("animation"):
            self._log(f"回転アニメ: {bulk_res['animation'].get('path')}")

    def _pick_cif(self) -> None:
        path = filedialog.askopenfilename(filetypes=[
                ("Structure files", "*.cif *.mcif *.vasp *.poscar *.contcar *.xsf *.cssr *.vesta POSCAR* CONTCAR*"),
                ("CIF files", "*.cif *.mcif"),
                ("POSCAR / CONTCAR", "*.vasp *.poscar *.contcar POSCAR* CONTCAR*"),
                ("XSF / CSSR / VESTA", "*.xsf *.cssr *.vesta"),
                ("All files", "*.*"),
            ])
        if path:
            self.cif_var.set(path)

    def _pick_volume(self) -> None:
        path = filedialog.askopenfilename(filetypes=[
                ("Volumetric files", "*.cube *.cub *.xsf *.grd CHGCAR* CHG* PARCHG* LOCPOT* ELFCAR*"),
                ("Gaussian Cube", "*.cube *.cub"),
                ("VASP CHGCAR / LOCPOT / ELFCAR", "CHGCAR* CHG* PARCHG* LOCPOT* ELFCAR*"),
                ("XSF / VESTA grd", "*.xsf *.grd"),
                ("All files", "*.*"),
            ])
        if path:
            self.volume_var.set(path)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.cif_var.set(path)

    def _pick_out_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def _log(self, msg: str) -> None:
        def _append() -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(0, _append)

    @staticmethod
    def _open_path(path: Path) -> None:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    def _open_output(self) -> None:
        if self.last_output_dir and self.last_output_dir.exists():
            self._open_path(self.last_output_dir)

    def _open_index(self) -> None:
        if self.last_index_html and self.last_index_html.exists():
            self._open_path(self.last_index_html)

    def _start_run(self) -> None:
        self.run_btn.configure(state="disabled")
        self.status_var.set("実行中...")
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self) -> None:
        try:
            theme = set_active_theme(self.theme_var.get())
            mode = self.mode_var.get()
            inp = self.cif_var.get().strip()
            # 空欄なら 出力/<日時>_<名前>/ を毎回新しく作る。指定したら今までどおりその中に作る
            out_text = self.out_var.get().strip()
            out_base = Path(out_text) if out_text else None
            try:
                hkl = (int(self.h_var.get()), int(self.k_var.get()), int(self.l_var.get()))
            except ValueError:
                raise ValueError("ミラー指数 h k l には整数を入れてください（例: 1 1 0）") from None
            if hkl == (0, 0, 0):
                raise ValueError("ミラー指数 (0 0 0) は面になりません")
            a0 = float(cfg.get("lattice.a_austenite"))
            ca_mart = float(cfg.get("lattice.martensite_c_over_a"))
            transparent = bool(self.transparent_var.get())

            bulk_opts = self._bulk_options()
            self._log(f"--- 実行開始 (v{__version__}): mode={mode}, theme={theme} ---")

            if mode == "bulk_slab":
                if not inp:
                    res = run_demo(
                        miller_index=hkl,
                        base_dir=out_base,
                        filter_polar=self.filter_polar_var.get(),
                        deduplicate=self.dedup_var.get(),
                        transparent=transparent,
                        **bulk_opts,
                    )
                    self._log_extra_outputs(res["bulk_analysis"])
                    finished = res
                else:
                    cif_p = Path(inp)
                    struct, warns = load_structure_robust(cif_p)
                    self._log_load_warnings(warns)
                    self._warn_cell_if_disordered(struct, bulk_opts["poscar_cell"])
                    stem = sanitize_name(structure_stem(cif_p))
                    hkl_tag = f"{hkl[0]}{hkl[1]}{hkl[2]}"
                    if out_base is None:
                        bulk_dir = new_run_dir("", structure_stem(cif_p))
                        slab_dir = bulk_dir / f"slab_{hkl_tag}"
                    else:
                        bulk_dir = out_base / f"output_{stem}"
                        slab_dir = out_base / f"output_slab_{stem}_{hkl_tag}"
                    if "adp" in bulk_opts and bulk_opts["adp"] is None:
                        bulk_opts["adp"] = str(cif_p)
                    bulk_res = run_bulk_analysis(
                        struct, bulk_dir, parse_warnings=warns, transparent=transparent, **bulk_opts
                    )
                    self._log_extra_outputs(bulk_res)
                    cut_slabs(
                        struct, hkl, slab_dir,
                        filter_polar=self.filter_polar_var.get(),
                        deduplicate=self.dedup_var.get(),
                        transparent_background=transparent,
                    )
                    finished = _finish_run(
                        {}, bulk_dir, f"CIF の解析：{cif_p.name}", source=cif_p.resolve(),
                        settings={"miller_index": list(hkl), "transparent": transparent, **_shown_options(bulk_opts)},
                    )

            elif mode == "phase_compare":
                parent = load_structure_robust(inp)[0] if inp else build_ni2mnga_structure(a0)
                mart = build_ni2mnga_martensite_l10(a0, c_over_a=ca_mart)
                out = (
                    out_base / "output_phase_compare" if out_base
                    else new_run_dir("phase_compare", Path(inp).stem if inp else "Ni2MnGa")
                )
                compare_phases(parent, mart, out, transparent_background=transparent)
                finished = _finish_run({}, out, "変態前後の 2 相の比較", source=inp or "Ni2MnGa（デモ）",
                                       settings={"transparent": transparent})

            elif mode == "heusler":
                formula = self.formula_var.get().strip() or "Ni50Mn35In15"
                ordering = self.ordering_var.get()
                el1, el2 = self.el1_var.get().strip() or "Mn", self.el2_var.get().strip() or "Mn"
                res = generate_heusler_models(
                    formula=formula,
                    ordering=ordering,
                    output_dir=(
                        out_base / f"output_heusler_{sanitize_name(formula)}_{ordering}" if out_base
                        else new_run_dir("heusler", f"{sanitize_name(formula)}_{ordering}")
                    ),
                    highlight_pair=(el1, el2),
                    a=float(cfg.get("lattice.heusler_a")),
                    supercell=tuple(cfg.get("heusler.supercell")),
                    num_patterns=int(cfg.get("heusler.num_patterns")),
                    seed=int(cfg.get("heusler.seed")),
                    transparent_background=transparent,
                )
                finished = _finish_run({}, res["output_dir"], f"ホイスラー合金の副格子モデル：{formula}",
                                       settings={"formula": formula, "ordering": ordering})

            elif mode == "ca_sweep":
                base_s = load_structure_robust(inp)[0] if inp else build_ni2mnga_structure(a0)
                ca_dir = (
                    out_base / "output_ca_sweep_gui" if out_base
                    else new_run_dir("ca_sweep", Path(inp).stem if inp else "Ni2MnGa")
                )
                sw = generate_ca_sweep(
                    base_s, ca_dir,
                    ca_min=float(cfg.get("ca_sweep.ca_min")),
                    ca_max=float(cfg.get("ca_sweep.ca_max")),
                    ca_step=float(cfg.get("ca_sweep.ca_step")),
                )
                demo_e = generate_demo_ca_energy_csv(sw["sweep_table_csv"], ca_dir / DATA_DIR / "demo_ca_energy.csv")
                plot_ca_energy_curve(demo_e, base_s, ca_dir, points_label="Model energies (demo, not DFT)")
                finished = _finish_run({}, ca_dir, "c/a スイープとエネルギー曲線", source=inp or "Ni2MnGa（デモ）")

            elif mode == "batch":
                if not inp or not Path(inp).is_dir():
                    raise ValueError("バッチモードではCIFファイルが入ったフォルダを選択してください。")
                res = run_batch(
                    input_dir=inp,
                    output_dir=out_base / f"output_batch_{sanitize_name(Path(inp).name)}" if out_base else None,
                    miller_index=hkl,
                    filter_polar=self.filter_polar_var.get(),
                    deduplicate=self.dedup_var.get(),
                    transparent=transparent,
                    **bulk_opts,
                )
                for item in res.get("items", []):
                    name = Path(item["cif_file"]).name
                    self._log_load_warnings(item.get("parse_warnings") or [], name)
                    bulk = item.get("bulk_analysis") or {}
                    if bulk_opts["poscar_cell"] != "input" and not bulk.get("summary", {}).get("is_ordered", True):
                        self._log(f"[注意 {name}] {DISORDER_CELL_NOTE}")
                    if item.get("status") == "error":
                        self._log(f"[エラー {name}] {item.get('error')}")
                finished = res
            else:
                raise ValueError(f"Unknown mode: {mode}")

            self.last_output_dir = Path(finished["output_dir"]).resolve()
            index = finished.get("index_html")
            self.last_index_html = Path(index) if index else None
            self._log(f"完了しました。保存先: {self.last_output_dir}")
            if index:
                self._log(f"はじめに {index} を開くと、できたファイルの一覧が見られます。")
            self.after(0, lambda: self.status_var.set("完了"))
            self.after(0, lambda: self.saved_var.set(str(self.last_output_dir)))
            self.after(0, lambda: self.open_btn.configure(state="normal"))
            self.after(0, lambda: self.index_btn.configure(state="normal" if index else "disabled"))
        except Exception as exc:
            self._log(f"[エラー] {type(exc).__name__}: {exc}")
            self.after(0, lambda: self.status_var.set("エラー"))
            self.after(0, lambda e=exc: messagebox.showerror("実行エラー", str(e)))
        finally:
            self.after(0, lambda: self.run_btn.configure(state="normal"))


def main() -> None:
    app = CrystalAnalyzerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

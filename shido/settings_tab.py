"""GUI の「前提数値」タブ。

settings_defaults.yaml（既定値）に、利用者の変更（settings_user.yaml）を重ねた値を
表示・編集する。保存すると次の実行からすぐ使われる（再起動はいらない）。
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from shido import settings as cfg
from shido.theme_manager import list_themes

# (設定キー, ラベル, 単位, 種類, 説明)  種類: float / int / str / bool / choice:<a,b>
FIELD_GROUPS: list[tuple[str, list[tuple[str, str, str, str, str]]]] = [
    ("材料・格子定数", [
        ("lattice.a_austenite", "母相の格子定数 a", "Å", "float", "L2_1 立方晶。デモ構造・2相比較・c/a スイープの元"),
        ("lattice.martensite_c_over_a", "マルテンサイトの c/a", "", "float", "2相比較で使う L1_0 正方晶の c/a"),
        ("lattice.heusler_a", "ホイスラーモデルの a", "Å", "float", "副格子モデルの格子定数"),
    ]),
    ("c/a スイープ（DFT 入力）", [
        ("ca_sweep.ca_min", "c/a の最小", "", "float", ""),
        ("ca_sweep.ca_max", "c/a の最大", "", "float", ""),
        ("ca_sweep.ca_step", "刻み", "", "float", "c/a = 1.00 を必ず含むように刻む"),
    ]),
    ("スラブ（表面の切り出し）", [
        ("slab.min_slab_size", "スラブの厚さ（最小）", "Å", "float", ""),
        ("slab.min_vacuum_size", "真空層の厚さ", "Å", "float", ""),
        ("slab.layer_z_tol", "同じ層とみなす高さの差", "Å", "float", "原子層を数えるときの許容差"),
        ("slab.filter_polar", "極性面を自動で除く", "", "bool", ""),
        ("slab.deduplicate", "同じ形のスラブをまとめる", "", "bool", ""),
        ("slab.polar_dipole_tol", "極性面とみなす双極子", "e/Å", "float", "面積あたり。金属は判定しない"),
    ]),
    ("変態前後の2相比較（格子の対応づけ）", [
        ("phase_compare.match_ltol", "長さの許容差", "", "float", "対応が見つからず n/a になるときは少し大きく"),
        ("phase_compare.match_stol", "原子位置の許容差", "", "float", ""),
        ("phase_compare.match_angle_tol", "角度の許容差", "度", "float", ""),
    ]),
    ("ホイスラー副格子モデル", [
        ("heusler.formula", "組成式", "", "str", "例: Ni50Mn35In15、Ni2MnGa"),
        ("heusler.ordering", "規則度", "", "choice:L21,B2,XA", ""),
        ("heusler.supercell", "スーパーセル", "倍", "int3", "a b c 方向の倍数（例: 2 2 2）"),
        ("heusler.num_patterns", "配置パターン数", "個", "int", ""),
        ("heusler.highlight_pair", "強調するペア", "", "pair", "例: Mn Mn"),
        ("heusler.pair_cutoff_factor", "ペア距離を数える範囲", "× a", "float", ""),
        ("heusler.seed", "乱数の種", "", "int", "同じ値なら同じ配置になる"),
    ]),
    ("結合", [
        ("bonds.max_bond_length", "結合線を描く最大距離", "Å", "float", ""),
    ]),
    ("図の見た目", [
        ("render.theme", "カラーテーマ", "", "theme", ""),
        ("render.dpi", "解像度", "dpi", "int", "論文用は 300 以上"),
        ("render.ball_radius_scale", "球の大きさ（原子半径の何倍）", "倍", "float", ""),
        ("render.ball_radius_min", "球の半径の下限", "Å", "float", ""),
        ("render.ball_radius_max", "球の半径の上限", "Å", "float", ""),
        ("render.bond_radius", "結合の棒の太さ（半径）", "Å", "float", ""),
        ("render.unit_cell_line_radius", "単位格子の線の太さ（半径）", "Å", "float", ""),
        ("render.transparent_background", "背景を透過にする", "", "bool", ""),
    ]),
]


def _get_path(d: dict[str, Any], path: str) -> Any:
    node: Any = d
    for part in path.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = d
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


class SettingsTab(ttk.Frame):
    """前提数値の編集タブ。on_saved は保存後に呼ばれる（実行タブの表示を合わせるため）。"""

    RADII_COLUMNS = 6

    def __init__(self, master: tk.Misc, on_saved: Callable[[dict[str, Any]], None] | None = None) -> None:
        super().__init__(master)
        self.on_saved = on_saved
        self.vars: dict[str, tuple[str, tk.Variable]] = {}
        self.radius_vars: dict[str, tk.StringVar] = {}
        self.material_var = tk.StringVar()
        self.radius_source_var = tk.StringVar()

        self._build_scroll_area()
        self._build_buttons()
        self._build_fields()
        self.load_into_widgets(cfg.reload())

    # ---- 画面づくり -----------------------------------------------------
    def _build_scroll_area(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas)
        self.body.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        # マウスホイールでスクロール（このタブの上にいるときだけ）
        self.canvas.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _build_buttons(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=6, side="bottom")
        ttk.Button(bar, text="保存（次の実行から使う）", command=self.save).pack(side="left", ipadx=8, ipady=3)
        ttk.Button(bar, text="変更を取り消す", command=lambda: self.load_into_widgets(cfg.reload())).pack(side="left", padx=6)
        ttk.Button(bar, text="すべて既定値に戻す", command=self.reset_defaults).pack(side="left", padx=6)
        self.msg_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.msg_var, foreground="#1a7f37").pack(side="right")

    def _build_fields(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # 材料のプリセット
        mat = ttk.LabelFrame(self.body, text="よく使う材料（選ぶと格子定数 a を入れる）", padding=8)
        mat.pack(fill="x", **pad)
        presets = cfg.get("material_presets", []) or []
        labels = [f"{p['name']}  (a = {p['a']} Å, {p.get('structure', '')})" for p in presets]
        self._preset_names = [p["name"] for p in presets]
        self.material_combo = ttk.Combobox(mat, values=labels, state="readonly", width=40)
        self.material_combo.pack(side="left")
        self.material_combo.bind("<<ComboboxSelected>>", self._on_material)
        self.material_note = ttk.Label(mat, text="", foreground="#555", wraplength=250, justify="left")
        self.material_note.pack(side="left", padx=10)

        for title, fields in FIELD_GROUPS:
            box = ttk.LabelFrame(self.body, text=title, padding=8)
            box.pack(fill="x", **pad)
            box.columnconfigure(3, weight=1)
            for row, (key, label, unit, kind, hint) in enumerate(fields):
                self._add_field(box, row, key, label, unit, kind, hint)

        self._build_radii()

    def _add_field(self, box: ttk.Frame, row: int, key: str, label: str, unit: str, kind: str, hint: str) -> None:
        ttk.Label(box, text=label, width=26).grid(row=row, column=0, sticky="w", pady=1)
        if kind == "bool":
            var: tk.Variable = tk.BooleanVar()
            ttk.Checkbutton(box, variable=var).grid(row=row, column=1, sticky="w")
        elif kind.startswith("choice:") or kind == "theme":
            var = tk.StringVar()
            values = list_themes() if kind == "theme" else kind.split(":", 1)[1].split(",")
            ttk.Combobox(box, textvariable=var, values=values, state="readonly", width=14).grid(row=row, column=1, sticky="w")
        else:
            var = tk.StringVar()
            ttk.Entry(box, textvariable=var, width=16).grid(row=row, column=1, sticky="w")
        ttk.Label(box, text=unit, width=5).grid(row=row, column=2, sticky="w", padx=(4, 8))
        if hint:
            ttk.Label(box, text=hint, foreground="#666", wraplength=250, justify="left").grid(row=row, column=3, sticky="w")
        self.vars[key] = (kind, var)

    def _build_radii(self) -> None:
        box = ttk.LabelFrame(self.body, text="表示用の原子半径 [Å]（球の大きさ = この値 × 球の大きさの倍率）", padding=8)
        box.pack(fill="x", padx=10, pady=4)

        top = ttk.Frame(box)
        top.pack(fill="x", pady=(0, 6))
        ttk.Button(top, text="共有結合半径で埋める", command=lambda: self._fill_radii("covalent")).pack(side="left")
        ttk.Button(top, text="金属結合半径で埋める", command=lambda: self._fill_radii("metallic")).pack(side="left", padx=6)
        ttk.Label(top, text="今の表:").pack(side="left", padx=(12, 2))
        ttk.Label(top, textvariable=self.radius_source_var, foreground="#1f4e9c").pack(side="left")

        self.radii_grid = ttk.Frame(box)
        self.radii_grid.pack(fill="x")

        add = ttk.Frame(box)
        add.pack(fill="x", pady=(6, 0))
        ttk.Label(add, text="元素を追加:").pack(side="left")
        self.new_sym = tk.StringVar()
        self.new_val = tk.StringVar()
        ttk.Entry(add, textvariable=self.new_sym, width=5).pack(side="left", padx=3)
        ttk.Entry(add, textvariable=self.new_val, width=7).pack(side="left", padx=3)
        ttk.Label(add, text="Å").pack(side="left")
        ttk.Button(add, text="追加", command=self._add_radius).pack(side="left", padx=6)
        ttk.Label(box, text="表に無い元素は pymatgen の値を使う。欄を空にすると、その元素は表から外れる。", foreground="#666").pack(anchor="w", pady=(4, 0))

    def _render_radii(self, table: dict[str, Any]) -> None:
        for child in self.radii_grid.winfo_children():
            child.destroy()
        self.radius_vars = {}
        for i, (sym, val) in enumerate(table.items()):
            r, c = divmod(i, self.RADII_COLUMNS)
            cell = ttk.Frame(self.radii_grid)
            cell.grid(row=r, column=c, sticky="w", padx=4, pady=1)
            ttk.Label(cell, text=sym, width=3, anchor="e").pack(side="left")
            try:
                text = "" if val in (None, "") else f"{float(val):g}"
            except (TypeError, ValueError):
                text = str(val)  # 入力途中の文字はそのまま残す（保存のときに知らせる）
            var = tk.StringVar(value=text)
            var.trace_add("write", lambda *_: self.radius_source_var.set("custom（手で直した）"))
            ttk.Entry(cell, textvariable=var, width=6).pack(side="left", padx=2)
            self.radius_vars[sym] = var

    # ---- 動き -------------------------------------------------------------
    def _on_material(self, _e: object = None) -> None:
        idx = self.material_combo.current()
        if idx < 0:
            return
        preset = (cfg.get("material_presets", []) or [])[idx]
        self.material_var.set(preset["name"])
        self.vars["lattice.a_austenite"][1].set(f"{float(preset['a']):g}")
        self.material_note.configure(text=preset.get("note", "") or "")

    def _fill_radii(self, source: str) -> None:
        table = (cfg.get("radius_tables", {}) or {}).get(source) or {}
        if not table:
            messagebox.showinfo("原子半径", f"{source} の表がまだありません（settings_defaults.yaml の radius_tables）。")
            return
        current = {s: v.get() for s, v in self.radius_vars.items()}
        merged = {**{s: current.get(s) for s in current}, **table}
        self._render_radii(merged)
        self.radius_source_var.set("covalent（共有結合半径）" if source == "covalent" else "metallic（金属結合半径・12配位）")

    def _add_radius(self) -> None:
        sym = self.new_sym.get().strip().capitalize()
        try:
            from pymatgen.core import Element
            Element(sym)
            val = float(self.new_val.get())
            if val <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("元素を追加", "元素記号（例: Fe）と 0 より大きい半径を入れてください。")
            return
        table = {s: v.get() for s, v in self.radius_vars.items()}
        table[sym] = val
        self._render_radii(table)
        self.radius_source_var.set("custom（手で直した）")
        self.new_sym.set("")
        self.new_val.set("")

    def load_into_widgets(self, data: dict[str, Any]) -> None:
        for key, (kind, var) in self.vars.items():
            val = _get_path(data, key)
            if kind == "bool":
                var.set(bool(val))
            elif kind in ("int3", "pair"):
                var.set(" ".join(str(x) for x in (val or [])))
            else:
                var.set("" if val is None else str(val))
        name = _get_path(data, "lattice.material") or ""
        self.material_var.set(name)
        if name in self._preset_names:
            self.material_combo.current(self._preset_names.index(name))
            self.material_note.configure(text=(cfg.material_preset(name) or {}).get("note", "") or "")
        else:
            self.material_combo.set("")
            self.material_note.configure(text="")
        self._render_radii(dict(data.get("atomic_radii") or {}))
        src = data.get("radius_source", "covalent")
        self.radius_source_var.set({"covalent": "covalent（共有結合半径）",
                                    "metallic": "metallic（金属結合半径・12配位）"}.get(src, "custom（手で直した）"))
        self.msg_var.set("")

    def collect(self) -> dict[str, Any]:
        """画面の値を設定一式にまとめる。おかしい値があれば ValueError（全部まとめて知らせる）。"""
        data = cfg.load_settings()
        errors: list[str] = []
        labels = {k: lbl for _, fs in FIELD_GROUPS for (k, lbl, *_r) in fs}
        for key, (kind, var) in self.vars.items():
            raw = var.get()
            try:
                if kind == "bool":
                    val: Any = bool(raw)
                elif kind == "float":
                    val = float(str(raw).strip())
                    if not math.isfinite(val) or val <= 0:
                        raise ValueError
                elif kind == "int":
                    val = int(str(raw).strip())
                    if val < 0 or (val == 0 and key != "heusler.seed"):
                        raise ValueError
                elif kind == "int3":
                    val = [int(x) for x in str(raw).replace(",", " ").split()]
                    if len(val) != 3 or min(val) < 1:
                        raise ValueError
                elif kind == "pair":
                    val = [x.strip().capitalize() for x in str(raw).replace(",", " ").replace("-", " ").split()]
                    if len(val) != 2:
                        raise ValueError
                    from pymatgen.core import Element
                    for sym in val:
                        Element(sym)  # 元素記号でなければ ValueError
                else:
                    val = str(raw).strip()
                    if not val:
                        raise ValueError
                    if key == "heusler.formula":
                        from shido.heusler_builder import parse_alloy_composition
                        from pymatgen.core import Element
                        for sym in parse_alloy_composition(val):  # 読めない組成式なら ValueError
                            Element(sym)  # 実在しない元素なら ValueError
            except (ValueError, KeyError):
                errors.append(f"・{labels.get(key, key)}: 「{raw}」")
                continue
            _set_path(data, key, val)

        ca_min, ca_max = _get_path(data, "ca_sweep.ca_min"), _get_path(data, "ca_sweep.ca_max")
        if isinstance(ca_min, float) and isinstance(ca_max, float) and ca_min >= ca_max:
            errors.append("・c/a の最小は最大より小さくしてください")
        rmin, rmax = _get_path(data, "render.ball_radius_min"), _get_path(data, "render.ball_radius_max")
        if isinstance(rmin, float) and isinstance(rmax, float) and rmin >= rmax:
            errors.append("・球の半径の下限は上限より小さくしてください")

        radii: dict[str, float] = {}
        for sym, var in self.radius_vars.items():
            raw = var.get().strip()
            if not raw:
                continue  # 空欄 = 表から外す
            try:
                v = float(raw)
                if not math.isfinite(v) or v <= 0:
                    raise ValueError
                radii[sym] = v
            except ValueError:
                errors.append(f"・原子半径 {sym}: 「{raw}」")
        data["atomic_radii"] = radii
        src = self.radius_source_var.get().split("（")[0]
        data["radius_source"] = src if src in ("covalent", "metallic", "custom") else "custom"
        if self.material_var.get():
            data.setdefault("lattice", {})["material"] = self.material_var.get()

        if errors:
            raise ValueError("次の値を直してください（数は 0 より大きく）:\n" + "\n".join(errors))
        return data

    def save(self) -> None:
        try:
            data = self.collect()
        except ValueError as exc:
            messagebox.showerror("前提数値", str(exc))
            return
        cfg.save_user_settings(data)
        merged = cfg.reload()
        self.load_into_widgets(merged)  # 空欄にした元素などを消して、保存された中身を見せる
        self.msg_var.set("保存しました")
        self.after(4000, lambda: self.msg_var.set(""))
        if self.on_saved:
            self.on_saved(merged)

    def reset_defaults(self) -> None:
        if not messagebox.askyesno("前提数値", "変えた値をすべて消して、既定値に戻しますか？"):
            return
        cfg.reset_user_settings()
        merged = cfg.reload()
        self.load_into_widgets(merged)
        self.msg_var.set("既定値に戻しました")
        if self.on_saved:
            self.on_saved(merged)

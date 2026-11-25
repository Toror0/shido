"""Journal-ready multi-panel figures: lay out panels at a journal column width, add panel
labels in the journal's style, write the journal's format, then check the result.

Sizes are decided in the finished (printed) size: a text of ``p`` pt is ``p / 72 * dpi``
pixels. Panels we draw ourselves (`XrdPanel`) are drawn at their finished size with the
journal's fonts, so their text and line sizes are exact; so are the 3D structure views
(`StructurePanel`, rendered at the cell's mm x dpi), which share one element legend under
the panels. Bitmap panels (`ImagePanel`,
e.g. the 3D structure views) are only scaled; their text size can at best be estimated
(``text_frac`` = text height / image width) and is reported as an estimate.

All rules come from `crystal.core.journal_spec` (``journals/*.yaml``); nothing here is
journal-specific.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from shido.core.journal_spec import JournalSpec, load_journal

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0
GAP_MM = 2.0                 # space between panels
WIDTH_TOLERANCE_MM = 0.5     # finished width may differ this much from the column width
#: Text height of the PIL annotations of the renderer's single views (legend, scale bar,
#: caption ~2.1-2.5 % of the image width). Used only for the size estimate.
STRUCTURE_VIEW_TEXT_FRAC = 0.021
FORMAT_PREFERENCE = ("tiff", "pdf", "eps", "png")
WRITABLE_FORMATS = ("tiff", "pdf", "eps", "png", "jpeg")
_EXT = {"tiff": ".tif", "pdf": ".pdf", "eps": ".eps", "png": ".png", "jpeg": ".jpg"}


def mm_to_px(mm: float, dpi: float) -> int:
    return int(round(mm / MM_PER_INCH * dpi))


def px_to_mm(px: float, dpi: float) -> float:
    return px / dpi * MM_PER_INCH


def pt_to_px(pt: float, dpi: float) -> float:
    return pt / PT_PER_INCH * dpi


# ---------------------------------------------------------------------------
# fonts
# ---------------------------------------------------------------------------

_FONT_FILES: dict[str, tuple[str, str]] = {
    "arial": ("arial.ttf", "arialbd.ttf"),
    "helvetica": ("arial.ttf", "arialbd.ttf"),  # metric-compatible stand-in on Windows
    "times new roman": ("times.ttf", "timesbd.ttf"),
    "times": ("times.ttf", "timesbd.ttf"),
    "courier": ("cour.ttf", "courbd.ttf"),
}


def find_font(families: Sequence[str], bold: bool = False) -> tuple[str, str]:
    """(font file, family name used) for the first family found; DejaVu Sans as last resort."""
    import matplotlib

    for fam in families:
        files = _FONT_FILES.get(str(fam).strip().lower())
        if not files:
            continue
        fname = files[1 if bold else 0]
        for cand in (fname, f"C:/Windows/Fonts/{fname}"):
            try:
                ImageFont.truetype(cand, size=10)
            except OSError:
                continue
            used = "Arial" if fname.startswith("arial") else str(fam)
            return cand, used
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    return str(ttf), "DejaVu Sans"


def _legend_rows(items: Sequence[tuple[str, str]], font: Any, text_px: int,
                 max_width: int) -> tuple[list[list[tuple[str, str, int]]], int, int]:
    """Wrap legend items (symbol, colour) into rows no wider than ``max_width``."""
    marker = int(round(text_px * 0.8))
    gap = max(1, int(round(text_px * 0.35)))
    space = int(round(text_px * 1.1))
    rows: list[list[tuple[str, str, int]]] = [[]]
    row_w = [0]
    for sym, col in items:
        bb = font.getbbox(sym)
        w = marker + gap + (bb[2] - bb[0])
        add = w if not rows[-1] else w + space
        if rows[-1] and row_w[-1] + add > max_width:
            rows.append([])
            row_w.append(0)
            add = w
        rows[-1].append((sym, col, w))
        row_w[-1] += add
    row_h = int(round(text_px * 1.35))
    return rows, max(row_w), row_h * len(rows)


def legend_size(items: Sequence[tuple[str, str]], font: Any, text_px: int, max_width: int) -> tuple[int, int]:
    """(width, height) px of `draw_legend` for these items."""
    if not items:
        return 0, 0
    _, w, h = _legend_rows(items, font, text_px, max_width)
    return w, h


def draw_legend(draw: ImageDraw.ImageDraw, items: Sequence[tuple[str, str]], x0: float, y0: float,
                font: Any, text_px: int, line_px: int, max_width: int, align: str = "left") -> tuple[int, int]:
    """Element legend: a coloured circle + symbol per item, in rows (wrapped at ``max_width``).

    ``align`` 'left' / 'center' / 'right' places each row in [x0, x0 + max_width]. Text is
    ``text_px`` (the font's em size), the circle outline ``line_px``. Returns (width, height).
    """
    if not items:
        return 0, 0
    rows, width, height = _legend_rows(items, font, text_px, max_width)
    marker = int(round(text_px * 0.8))
    gap = max(1, int(round(text_px * 0.35)))
    space = int(round(text_px * 1.1))
    row_h = int(round(text_px * 1.35))
    for r, row in enumerate(rows):
        rw = sum(w for _, _, w in row) + space * (len(row) - 1)
        x = x0 + {"left": 0, "center": (max_width - rw) / 2, "right": max_width - rw}[align]
        cy = y0 + r * row_h + row_h / 2.0
        for sym, col, w in row:
            draw.ellipse([(x, cy - marker / 2), (x + marker, cy + marker / 2)], fill=col,
                         outline=(26, 26, 26), width=max(1, int(line_px)))
            draw.text((x + marker + gap, cy), sym, fill=(17, 17, 17), font=font, anchor="lm")
            x += w + space
    return width, height


def panel_label_text(index: int, style: str) -> str:
    """'(a)', 'a', 'A', '(A)' or 'a)' for panel number ``index`` (0 = a)."""
    letter = chr(ord("a") + index % 26)
    return {"(a)": f"({letter})", "a": letter, "A": letter.upper(), "(A)": f"({letter.upper()})",
            "a)": f"{letter})"}[style]


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

@dataclass
class Drawn:
    """A panel drawn at its finished size, with what is known about its text / lines."""

    image: Image.Image
    texts: list[dict[str, Any]] = field(default_factory=list)   # {"what", "pt", "estimate"}
    lines: list[dict[str, Any]] = field(default_factory=list)   # {"what", "pt", "estimate"}
    effective_dpi: float | None = None
    fonts: list[str] = field(default_factory=list)


class ImagePanel:
    """An existing bitmap (PNG, TIFF, ...) scaled to its cell. Text size only estimated."""

    kind = "image"

    def __init__(self, path: str | Path, title: str | None = None, text_frac: float | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"パネルの画像がありません: {self.path}")
        with Image.open(self.path) as im:
            im.load()
            self._image = _flatten(im)
        self.title = title or self.path.name
        self.text_frac = text_frac

    @property
    def aspect(self) -> float:
        w, h = self._image.size
        return w / h

    def draw(self, w_px: int, h_px: int, dpi: float, spec: JournalSpec) -> Drawn:
        src_w = self._image.width
        img = self._image.resize((w_px, h_px), Image.LANCZOS)
        texts = []
        if self.text_frac:
            pt = self.text_frac * w_px / dpi * PT_PER_INCH
            texts.append({"what": f"{self.title} の中の文字（凡例・目盛り・説明）", "pt": round(pt, 2),
                          "estimate": True})
        return Drawn(img, texts=texts, effective_dpi=src_w / (w_px / dpi))


class StructurePanel:
    """A 3D structure view rendered at its finished size (mm x dpi) in the journal's fonts.

    ``render(out_path, width_mm, height_mm, dpi, fonts, text_pt, line_pt, report)`` draws
    the PNG (e.g. `renderer.render_single_view` in print mode) and fills ``report`` with
    "texts" / "lines" ({"what", "pt"}), "fonts" and "elements" ([symbol, colour]); those
    sizes are what was drawn, so the check judges them exactly. ``elements`` (symbols,
    known before drawing) go into the figure's shared element legend.
    """

    kind = "structure"

    def __init__(self, render: Any, out_path: str | Path, title: str, aspect: float = 1.0,
                 elements: Sequence[str] = ()):
        self.render = render
        self.out_path = Path(out_path)
        self.title = title
        self.aspect = float(aspect)
        self.elements = list(elements)
        self.colors: dict[str, str] = {}

    def draw(self, w_px: int, h_px: int, dpi: float, spec: JournalSpec) -> Drawn:
        text_pt = annotation_pt(spec)
        line_pt = max(0.5, float(spec.get("min_line_pt")))
        fonts = list(spec.get("fonts"))
        report: dict[str, Any] = {}
        self.render(self.out_path, px_to_mm(w_px, dpi), px_to_mm(h_px, dpi), dpi, fonts, text_pt,
                    line_pt, report)
        with Image.open(self.out_path) as im:
            im.load()
            img = _flatten(im)
        if img.size != (w_px, h_px):  # mm round trip off by a pixel
            canvas = Image.new("RGB", (w_px, h_px), "white")
            canvas.paste(img.crop((0, 0, min(w_px, img.width), min(h_px, img.height))), (0, 0))
            img = canvas
        self.colors = {str(e): str(c) for e, c in report.get("elements", [])}
        texts = [{"what": f"{self.title} の{t['what']}", "pt": t["pt"], "estimate": False}
                 for t in report.get("texts", [])]
        lines = [{"what": f"{self.title} の{ln['what']}", "pt": ln["pt"], "estimate": False}
                 for ln in report.get("lines", [])]
        return Drawn(img, texts=texts, lines=lines, effective_dpi=float(dpi),
                     fonts=list(report.get("fonts", [])))


def annotation_pt(spec: JournalSpec) -> float:
    """Size of small in-figure text (hkl, scale bar, a/b/c): 85 % of body, not below min."""
    return max(float(spec.get("text_pt.min")), round(float(spec.get("text_pt.body")) * 0.85, 1))


class XrdPanel:
    """Powder pattern (sticks) drawn at the finished size with the journal's fonts."""

    kind = "plot"

    def __init__(self, peaks: list[dict[str, Any]], two_theta_range: Sequence[float],
                 wavelength_label: str = "", title: str = "XRD", label_top_n: int = 6,
                 aspect: float = 1.35):
        self.peaks = list(peaks)
        self.two_theta_range = (float(two_theta_range[0]), float(two_theta_range[1]))
        self.wavelength_label = wavelength_label
        self.title = title
        self.label_top_n = label_top_n
        self.aspect = aspect

    def draw(self, w_px: int, h_px: int, dpi: float, spec: JournalSpec) -> Drawn:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        body = float(spec.get("text_pt.body"))
        tmin = float(spec.get("text_pt.min"))
        small = max(tmin, round(body * 0.85, 1))
        min_line = float(spec.get("min_line_pt"))
        stick_lw = max(0.75, min_line)
        axis_lw = max(0.5, min_line)
        families = [f for f in spec.get("fonts") if _mpl_has_font(f)] or ["DejaVu Sans"]
        used_font = font_manager.FontProperties(family=families).get_name()

        with matplotlib.rc_context({
            "font.family": "sans-serif", "font.sans-serif": [*families, "DejaVu Sans"],
            "font.size": body, "axes.linewidth": axis_lw, "xtick.major.width": axis_lw,
            "ytick.major.width": axis_lw, "xtick.minor.width": axis_lw, "ytick.minor.width": axis_lw,
            "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
            "mathtext.fontset": "dejavusans",
        }):
            fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, layout="constrained")
            fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72)
            FigureCanvasAgg(fig)  # a real renderer, to measure the hkl labels
            ax = fig.add_subplot()
            lo, hi = self.two_theta_range
            xs = [p["two_theta_deg"] for p in self.peaks]
            ys = [p["intensity_rel"] for p in self.peaks]
            ax.vlines(xs, 0.0, ys, colors="#1f3a93", linewidth=stick_lw)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 118)
            ax.set_yticks([])
            ax.set_xlabel(r"2$\theta$ (deg)", fontsize=body, labelpad=1.5)
            ax.set_ylabel("Intensity (a.u.)", fontsize=body, labelpad=2)
            ax.tick_params(labelsize=body, length=2.5, pad=1.5)
            ax.minorticks_on()
            ax.tick_params(which="minor", length=1.3)
            ax.tick_params(axis="y", which="both", left=False, right=False)
            wl = None
            if self.wavelength_label:
                wl = ax.text(0.98, 0.96, self.wavelength_label, transform=ax.transAxes, ha="right",
                             va="top", fontsize=small)
            top = _label_peaks(self.peaks, self.label_top_n, min_sep=0.02 * (hi - lo))
            fig.canvas.draw()  # final layout, so the label boxes below are where they will be
            kept_boxes = [wl.get_window_extent()] if wl is not None else []
            ax_box = ax.get_window_extent()
            shown = []
            renderer = fig.canvas.get_renderer()
            sticks = [tuple(ax.transData.transform((x, y))) for x, y in zip(xs, ys)]
            clear = 1.0 / 72 * dpi + stick_lw / 72 * dpi  # gap kept to a neighbouring stick (px)
            for p in top:  # strongest first; drop a label that overlaps or leaves the axes
                ann = ax.annotate(_hkl_label(p.get("hkl", "")), (p["two_theta_deg"], p["intensity_rel"]),
                                  xytext=(0, 1.5), textcoords="offset points", ha="center",
                                  va="bottom", fontsize=small)
                ann.set_in_layout(False)
                own = self.peaks.index(p)
                placed = None
                for dx_pt, dy_pt in _label_offsets(ann.get_window_extent(renderer), sticks, own, clear, dpi):
                    ann.xyann = (dx_pt, 1.5 + dy_pt)
                    box = ann.get_window_extent(renderer)
                    inside = box.x0 >= ax_box.x0 + 1 and box.x1 <= ax_box.x1 - 1 and box.y1 <= ax_box.y1 - 1
                    if (inside and not any(box.overlaps(b) for b in kept_boxes)
                            and not _hits_stick(box, sticks, own, clear)):
                        placed = box
                        break
                if placed is None:
                    ann.remove()
                    continue
                kept_boxes.append(placed)
                shown.append(p)
            top = shown
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if img.size != (w_px, h_px):  # rounding of figsize * dpi
            canvas = Image.new("RGB", (w_px, h_px), "white")
            canvas.paste(img.crop((0, 0, min(w_px, img.width), min(h_px, img.height))), (0, 0))
            img = canvas
        texts = [{"what": f"{self.title} の軸の文字・目盛り", "pt": body, "estimate": False}]
        if top:
            texts.append({"what": f"{self.title} の hkl", "pt": small, "estimate": False})
        if wl is not None:
            texts.append({"what": f"{self.title} の波長", "pt": small, "estimate": False})
        lines = [{"what": f"{self.title} の軸の線", "pt": axis_lw, "estimate": False},
                 {"what": f"{self.title} のピークの線", "pt": stick_lw, "estimate": False}]
        return Drawn(img, texts=texts, lines=lines, effective_dpi=float(dpi), fonts=[used_font])


def _label_peaks(peaks: list[dict[str, Any]], n: int, min_sep: float) -> list[dict[str, Any]]:
    """The ``n`` strongest peaks with an hkl, at least ``min_sep`` deg apart (labels must not overlap)."""
    chosen: list[dict[str, Any]] = []
    for p in sorted(peaks, key=lambda q: -q["intensity_rel"]):
        if len(chosen) >= n:
            break
        if str(p.get("hkl", "")).strip() and all(abs(p["two_theta_deg"] - c["two_theta_deg"]) >= min_sep
                                                  for c in chosen):
            chosen.append(p)
    return chosen


def _hits_stick(box: Any, sticks: Sequence[tuple[float, float]], own: int, clear: float) -> bool:
    """True if a stick other than ``own`` (display x, top y) runs into the label box."""
    return any(i != own and box.x0 - clear < x < box.x1 + clear and y > box.y0 - clear
               for i, (x, y) in enumerate(sticks))


def _label_offsets(box: Any, sticks: Sequence[tuple[float, float]], own: int, clear: float,
                   dpi: float) -> list[tuple[float, float]]:
    """Offsets (points) to try for an hkl label: as is, slid sideways away from the sticks
    it would touch (at most 60 % of its width, so it stays over its peak), then lifted
    above them (at most 2 label heights)."""
    to_pt = 72.0 / dpi
    hits = [(x, y) for i, (x, y) in enumerate(sticks)
            if i != own and box.x0 - clear < x < box.x1 + clear and y > box.y0 - clear]
    out = [(0.0, 0.0)]
    if not hits:
        return out
    cx = (box.x0 + box.x1) / 2
    limit = 0.6 * box.width
    right = [x for x, _ in hits if x >= cx]
    left = [x for x, _ in hits if x < cx]
    if right and not left:
        out.append((-(box.x1 + clear - min(right)) * to_pt, 0.0))
    if left and not right:
        out.append(((max(left) + clear - box.x0) * to_pt, 0.0))
    out = [o for o in out if abs(o[0]) / to_pt <= limit]
    lift = max(y for _, y in hits) + clear - box.y0
    if lift <= 2 * box.height:
        out.append((0.0, lift * to_pt))
    return out


def _hkl_label(text: str) -> str:
    """'(1 -1 0), (1 0 -1)' -> the first family, negative indices with an overbar."""
    from shido.xrd import _hkl_mathtext

    first = str(text).split(",")[0]
    nums = [int(v) for v in re.findall(r"-?\d+", first)]
    return _hkl_mathtext(tuple(nums)) if nums else first.strip()


def _mpl_has_font(name: str) -> bool:
    from matplotlib import font_manager

    try:
        font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
        return True
    except ValueError:
        return False


def _flatten(im: Image.Image) -> Image.Image:
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, "white")
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return im.convert("RGB")


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    x: int
    y: int
    w: int
    h: int
    label_y: int


def default_columns(n: int) -> int:
    return n if n <= 3 else (2 if n == 4 else 3)


def layout_cells(aspects: Sequence[float], width_px: int, columns: int, gap_px: int,
                 label_px: int, max_height_px: int | None) -> tuple[list[Cell], int, float]:
    """Cells filling ``width_px``; each row shares one height. Returns (cells, height, shrink).

    A short last row keeps the height of the row above (left-aligned) instead of growing.
    If the figure is taller than ``max_height_px`` everything shrinks (centred) and
    shrink < 1 is returned; the canvas keeps the column width.
    """
    rows = [list(range(i, min(i + columns, len(aspects)))) for i in range(0, len(aspects), columns)]
    heights: list[float] = []
    for r in rows:
        h = (width_px - gap_px * (len(r) - 1)) / sum(aspects[i] for i in r)
        if heights and len(r) < columns:
            h = min(h, heights[-1])
        heights.append(h)
    total = sum(heights) + len(rows) * label_px + gap_px * (len(rows) - 1)
    shrink = 1.0
    if max_height_px is not None and total > max_height_px:
        shrink = (max_height_px - len(rows) * label_px - gap_px * (len(rows) - 1)) / sum(heights)
        shrink = max(shrink, 0.05)
    cells: list[Cell] = [None] * len(aspects)  # type: ignore[list-item]
    y = 0
    for r, h in zip(rows, heights):
        h_s = h * shrink
        row_w = sum(aspects[i] * h_s for i in r) + gap_px * (len(r) - 1)
        x = (width_px - row_w) / 2 if shrink < 1 else 0.0
        for i in r:
            w = aspects[i] * h_s
            cells[i] = Cell(int(round(x)), y + label_px, max(1, int(round(w))), max(1, int(round(h_s))), y)
            x += w + gap_px
        y += label_px + int(round(h_s)) + gap_px
    height = y - gap_px
    return cells, height, shrink


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def choose_format(spec: JournalSpec, fmt: str | None) -> str:
    if fmt:
        f = fmt.strip().lower().lstrip(".")
        f = {"tif": "tiff", "jpg": "jpeg"}.get(f, f)
        if f not in WRITABLE_FORMATS:
            raise ValueError(f"書き出せない形式 {fmt!r}（使える: {', '.join(WRITABLE_FORMATS)}）")
        return f
    accepted = spec.formats()
    for f in FORMAT_PREFERENCE:
        if f in accepted:
            return f
    return "tiff"


def save_image(img: Image.Image, path: Path, fmt: str, dpi: float, color_mode: str = "RGB") -> Path:
    """Write ``img`` so that its print size is px / dpi (PDF / EPS embed the bitmap)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = img
    if color_mode == "CMYK" and fmt in ("tiff", "jpeg", "eps", "pdf"):
        out = img.convert("CMYK")
    elif color_mode == "grayscale":
        out = img.convert("L")
    d = (dpi, dpi)
    if fmt == "tiff":
        out.save(path, format="TIFF", dpi=d, compression="tiff_lzw")
    elif fmt == "png":
        out.save(path, format="PNG", dpi=d)
    elif fmt == "jpeg":
        out.save(path, format="JPEG", dpi=d, quality=95, subsampling=0)
    elif fmt == "pdf":
        out.save(path, format="PDF", resolution=float(dpi))
    elif fmt == "eps":
        out.save(path, format="EPS")
        w, h = out.size  # PIL writes 1 px = 1 pt; rescale to the print size
        pw, ph = w * PT_PER_INCH / dpi, h * PT_PER_INCH / dpi
        data = path.read_bytes()
        data = data.replace(f"%%BoundingBox: 0 0 {w} {h}".encode(),
                            f"%%BoundingBox: 0 0 {math.ceil(pw)} {math.ceil(ph)}".encode(), 1)
        data = data.replace(f"\n{w} {h} scale".encode(), f"\n{pw:.3f} {ph:.3f} scale".encode(), 1)
        path.write_bytes(data)
    else:  # pragma: no cover - choose_format guards this
        raise ValueError(fmt)
    return path


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def _item(name: str, status: str, value: Any, limit: Any, note: str = "", default: bool = False) -> dict[str, Any]:
    out = {"item": name, "status": status, "value": value, "limit": limit, "note": note}
    if default:
        out["limit_is_default"] = True
        out["note"] = (note + " " if note else "") + "決まりが未確認なので安全側の既定で確かめた。"
    return out


def check_figure(
    spec: JournalSpec,
    *,
    width_px: int,
    height_px: int,
    dpi: float | None,
    width: str | float = "single",
    fmt: str | None = None,
    color_mode: str | None = None,
    dpi_kind: str = "combination",
    texts: Sequence[dict[str, Any]] = (),
    lines: Sequence[dict[str, Any]] = (),
    panels: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Check a finished figure against the journal. Returns {"ok", "items": [...]}.

    status: "ok", "fail" (breaks a rule), "warn" (probably breaks it / estimate / blurry),
    "unknown" (cannot be measured from a bitmap). ``ok`` is True when nothing failed.
    """
    items: list[dict[str, Any]] = []
    width_key, target_mm = spec.width_mm(width)
    width_default = width_key != "custom" and not spec.has(f"width_mm.{width_key}")
    if not dpi or dpi <= 0:
        items.append(_item("解像度 (dpi)", "fail", None, None, "画像に dpi が書かれていないので印刷の大きさが決まらない。"))
        return {"ok": False, "items": items}
    w_mm = round(px_to_mm(width_px, dpi), 2)
    h_mm = round(px_to_mm(height_px, dpi), 2)
    if width_key == "custom":
        widest = float(spec.get("width_mm.double"))
        status = "ok" if w_mm <= widest + WIDTH_TOLERANCE_MM else "fail"
        items.append(_item("幅 (mm)", status, w_mm, f"<= {widest}", "幅を mm で指定した。", not spec.has("width_mm.double")))
    else:
        status = "ok" if abs(w_mm - target_mm) <= WIDTH_TOLERANCE_MM else "fail"
        note = f"{width_key} の段の幅 {target_mm} mm に合わせる。"
        items.append(_item("幅 (mm)", status, w_mm, target_mm, note, width_default))
    max_h = spec.max_height_mm()
    if max_h is None:
        items.append(_item("高さ (mm)", "ok", h_mm, None, "この雑誌は高さの決まりが無い。"))
    else:
        items.append(_item("高さ (mm)", "ok" if h_mm <= max_h + 0.01 else "fail", h_mm, f"<= {max_h}",
                           default=not spec.has("max_height_mm")))
    need = float(spec.get(f"dpi.{dpi_kind}"))
    items.append(_item("解像度 (dpi)", "ok" if dpi + 0.5 >= need else "fail", round(dpi, 1), f">= {need:g}",
                       f"図の種類 {dpi_kind} の決まり。", not spec.has(f"dpi.{dpi_kind}")))
    if fmt:
        accepted = spec.formats()
        items.append(_item("形式", "ok" if fmt in accepted else "fail", fmt, accepted,
                           default=not spec.has("formats")))
    if color_mode:
        want = spec.get("color_mode")
        items.append(_item("色の形式", "ok" if color_mode == want else "warn", color_mode, want,
                           default=not spec.has("color_mode")))

    tmin = float(spec.get("text_pt.min"))
    tdef = not spec.has("text_pt.min")
    if texts:
        for t in texts:
            small = float(t["pt"]) + 1e-6 < tmin
            status = ("warn" if t.get("estimate") else "fail") if small else "ok"
            note = "画像の中の文字なので大きさは見積もり。描き直すか、パネルを大きくする。" if t.get("estimate") else ""
            items.append(_item(f"最小の文字 (pt)：{t['what']}", status, t["pt"], f">= {tmin:g}", note, tdef))
    else:
        items.append(_item("最小の文字 (pt)", "unknown", None, f">= {tmin:g}", "画像からは文字の大きさを測れない。", tdef))
    lmin = float(spec.get("min_line_pt"))
    ldef = not spec.has("min_line_pt")
    for ln in lines:
        small = float(ln["pt"]) + 1e-6 < lmin
        status = ("warn" if ln.get("estimate") else "fail") if small else "ok"
        items.append(_item(f"最小の線 (pt)：{ln['what']}", status, ln["pt"], f">= {lmin:g}", default=ldef))
    if not lines:
        items.append(_item("最小の線 (pt)", "unknown", None, f">= {lmin:g}", "画像からは線の太さを測れない。", ldef))
    elif any(p.get("kind") == "image" for p in panels):
        items.append(_item("最小の線 (pt)：画像のパネル", "unknown", None, f">= {lmin:g}",
                           "画像として貼ったパネルの線の太さは測れない。", ldef))
    for p in panels:
        eff = p.get("effective_dpi")
        if eff is not None and eff + 0.5 < need:
            items.append(_item(f"元の画像の細かさ：{p['label'] or ''} {p['source']}", "warn", round(eff), f">= {need:g}",
                               "元の画像より拡大して貼った（ぼやける）。元の図を高い dpi で作り直すとよい。"))
    return {"ok": not any(i["status"] == "fail" for i in items), "items": items}


def check_image_file(path: str | Path, journal: str | JournalSpec, width: str | float = "single",
                     dpi_kind: str = "combination", dpi: float | None = None) -> dict[str, Any]:
    """Check an existing bitmap (PNG / TIFF / JPEG) against a journal (size, dpi, format)."""
    spec = load_journal(journal) if isinstance(journal, str) else journal
    p = Path(path)
    if p.suffix.lower() in (".pdf", ".eps", ".ps", ".ai", ".svg"):
        raise ValueError(f"{p.name}: PDF / EPS / SVG は画素の画像ではないので、ここでは測れません。"
                         "PNG / TIFF / JPEG で確かめるか、figure で作ったときの 数値/figure_check.json を見てください。")
    try:
        with Image.open(p) as im:
            w, h = im.size
            file_dpi = im.info.get("dpi")
            mode = im.mode
    except UnidentifiedImageError as exc:
        raise ValueError(f"{p.name}: 画像として読めません（PNG / TIFF / JPEG を渡してください）") from exc
    d = float(dpi) if dpi else (float(round(file_dpi[0], 1)) if file_dpi else None)
    fmt = {"tif": "tiff", "jpg": "jpeg"}.get(p.suffix.lower().lstrip("."), p.suffix.lower().lstrip("."))
    color = "CMYK" if mode == "CMYK" else ("grayscale" if mode in ("L", "1") else "RGB")
    res = check_figure(spec, width_px=w, height_px=h, dpi=d, width=width, fmt=fmt, color_mode=color,
                       dpi_kind=dpi_kind)
    res.update(file=str(p.resolve()), journal=spec.id, defaults_used=dict(spec.defaults_used))
    return res


# ---------------------------------------------------------------------------
# the whole figure
# ---------------------------------------------------------------------------

def make_journal_figure(
    panels: Sequence[ImagePanel | XrdPanel | str | Path],
    journal: str | JournalSpec,
    out_dir: str | Path,
    *,
    width: str | float = "single",
    fmt: str | None = None,
    dpi: float | None = None,
    dpi_kind: str = "combination",
    columns: int | None = None,
    labels: bool = True,
    stem: str | None = None,
    preview: bool = True,
    legend: bool = True,
) -> dict[str, Any]:
    """Lay out ``panels`` at the journal column width, label, write, check.

    ``panels``: `ImagePanel` / `XrdPanel` objects or image paths. ``width``: 'single' /
    'one_half' / 'double' or mm. ``dpi`` None = the journal's dpi for ``dpi_kind``
    ('combination' suits 3D views + text). ``fmt`` None = the first of TIFF / PDF / EPS /
    PNG the journal takes. A PNG preview (``<stem>_見本.png``, 1200 px wide) is also
    written unless the figure itself is PNG. Returns a JSON-ready dict with ``check``.

    ``legend``: structure panels (`StructurePanel`) share one element legend under the
    panels, in the journal's body text size (none if there is no structure panel).
    """
    spec = load_journal(journal) if isinstance(journal, str) else journal
    spec.defaults_used.clear()
    objs = [p if isinstance(p, (ImagePanel, XrdPanel, StructurePanel)) else ImagePanel(p) for p in panels]
    if not objs:
        raise ValueError("パネルが 1 つもありません")
    width_key, width_mm = spec.width_mm(width)
    use_dpi = float(dpi) if dpi else float(spec.get(f"dpi.{dpi_kind}"))
    fmt_used = choose_format(spec, fmt)
    color_mode = str(spec.get("color_mode"))
    width_px = mm_to_px(width_mm, use_dpi)
    max_h = spec.max_height_mm()
    max_h_px = None if max_h is None else mm_to_px(max_h, use_dpi)

    label_style = str(spec.get("panel_label.style"))
    label_bold = bool(spec.get("panel_label.bold"))
    label_pt = float(spec.get("panel_label.size_pt"))
    label_px = int(round(pt_to_px(label_pt, use_dpi)))
    font_path, label_font_name = find_font(spec.get("fonts"), bold=label_bold)
    font = ImageFont.truetype(font_path, size=max(1, label_px))
    strip_px = int(round(label_px * 1.3)) if labels and len(objs) > 1 else 0
    gap_px = mm_to_px(GAP_MM, use_dpi)
    ncol = max(1, min(columns or default_columns(len(objs)), len(objs)))
    legend_syms: list[str] = []
    for o in objs:
        for e in getattr(o, "elements", []) or []:
            if e not in legend_syms:
                legend_syms.append(e)
    legend_h = 0
    if legend and legend_syms:
        leg_pt = float(spec.get("text_pt.body"))
        leg_px = max(4, math.ceil(pt_to_px(leg_pt, use_dpi) - 1e-6))
        leg_font_path, leg_font_name = find_font(spec.get("fonts"))
        leg_font = ImageFont.truetype(leg_font_path, size=leg_px)
        leg_line_px = max(1, math.ceil(pt_to_px(max(0.5, float(spec.get("min_line_pt"))), use_dpi) - 1e-6))
        _, lh = legend_size([(e, "#000000") for e in legend_syms], leg_font, leg_px, width_px)
        legend_h = lh + gap_px
    layout_max = None if max_h_px is None else max_h_px - legend_h
    cells, height_px, shrink = layout_cells([o.aspect for o in objs], width_px, ncol, gap_px, strip_px, layout_max)
    panels_h = height_px
    height_px += legend_h

    canvas = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(canvas)
    texts: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    fonts_used: set[str] = set()
    panel_rows: list[dict[str, Any]] = []
    for i, (obj, cell) in enumerate(zip(objs, cells)):
        drawn = obj.draw(cell.w, cell.h, use_dpi, spec)
        canvas.paste(drawn.image, (cell.x, cell.y))
        label = panel_label_text(i, label_style) if strip_px else ""
        if label:
            draw.text((cell.x, cell.label_y), label, fill="black", font=font, anchor="la")
        texts.extend(drawn.texts)
        lines.extend(drawn.lines)
        fonts_used.update(drawn.fonts)
        panel_rows.append({
            "label": label, "source": obj.title if hasattr(obj, "title") else "", "kind": obj.kind,
            "x_mm": round(px_to_mm(cell.x, use_dpi), 2), "y_mm": round(px_to_mm(cell.y, use_dpi), 2),
            "w_mm": round(px_to_mm(cell.w, use_dpi), 2), "h_mm": round(px_to_mm(cell.h, use_dpi), 2),
            "effective_dpi": None if drawn.effective_dpi is None else round(drawn.effective_dpi, 1),
        })
    if legend_h:
        colors: dict[str, str] = {}
        for o in objs:
            colors.update(getattr(o, "colors", {}) or {})
        items = [(e, colors.get(e, "#888888")) for e in legend_syms]
        draw_legend(draw, items, 0, panels_h + gap_px, leg_font, leg_px, leg_line_px, width_px, align="center")
        texts.append({"what": "元素の凡例（図の下）", "pt": round(leg_px / use_dpi * PT_PER_INCH, 2),
                      "estimate": False})
        lines.append({"what": "元素の凡例の丸の縁", "pt": round(leg_line_px / use_dpi * PT_PER_INCH, 2),
                      "estimate": False})
        fonts_used.add(leg_font_name)
    if strip_px:
        fonts_used.add(label_font_name)
        texts.append({"what": f"パネル記号 {label_style}{'（太字）' if label_bold else ''}", "pt": label_pt,
                      "estimate": False})

    out = Path(out_dir)
    stem = stem or f"journal_figure_{spec.id}_{width_key}"
    fig_path = save_image(canvas, out / f"{stem}{_EXT[fmt_used]}", fmt_used, use_dpi, color_mode)
    files = {fmt_used: str(fig_path.resolve())}
    if preview and fmt_used != "png":
        pw = min(1200, width_px)
        prev = canvas.resize((pw, max(1, int(round(height_px * pw / width_px)))), Image.LANCZOS)
        prev_path = out / f"{stem}_見本.png"
        prev.save(prev_path, format="PNG", dpi=(use_dpi * pw / width_px,) * 2)
        files["preview_png"] = str(prev_path.resolve())

    warnings: list[str] = []
    if shrink < 1:
        warnings.append(f"高さの上限 {max_h} mm に入らないので、パネルを {shrink:.0%} に縮めて中央に置いた"
                        "（列を増やすとよい）。")
    if strip_px and label_font_name not in spec.get("fonts"):
        warnings.append(f"パネル記号のフォントは {label_font_name}（雑誌の候補 {spec.get('fonts')} の代わり）。")
    check = check_figure(spec, width_px=width_px, height_px=height_px, dpi=use_dpi, width=width,
                         fmt=fmt_used, color_mode=color_mode, dpi_kind=dpi_kind, texts=texts,
                         lines=lines, panels=panel_rows)
    defaults = dict(spec.defaults_used)
    for key, value in defaults.items():
        warnings.append(f"{key} は未確認なので安全側の既定 {value} を使った。")
    return {
        "journal": spec.id,
        "journal_name": spec.name,
        "publisher": spec.publisher,
        "width": width_key,
        "width_mm": round(px_to_mm(width_px, use_dpi), 2),
        "height_mm": round(px_to_mm(height_px, use_dpi), 2),
        "size_px": [width_px, height_px],
        "dpi": use_dpi,
        "dpi_kind": dpi_kind,
        "format": fmt_used,
        "color_mode": color_mode,
        "panel_label": {"style": label_style, "bold": label_bold, "size_pt": label_pt},
        "fonts_used": sorted(fonts_used),
        "panels": panel_rows,
        "files": files,
        "defaults_used": defaults,
        "unverified": list(spec.unverified),
        "sources": spec.sources,
        "warnings": warnings,
        "check": check,
    }


__all__ = [
    "ImagePanel",
    "STRUCTURE_VIEW_TEXT_FRAC",
    "StructurePanel",
    "annotation_pt",
    "draw_legend",
    "legend_size",
    "XrdPanel",
    "check_figure",
    "check_image_file",
    "choose_format",
    "find_font",
    "layout_cells",
    "make_journal_figure",
    "panel_label_text",
]

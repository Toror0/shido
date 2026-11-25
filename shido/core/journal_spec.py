"""Journal figure rules, read from ``journals/*.yaml`` (one journal or publisher per file).

Nothing journal-specific is written in code: every number comes from a YAML file.

* ``extends: _elsevier`` inherits a publisher file (files starting with ``_`` are
  publisher bases, not listed as journals). The child wins key by key (deep merge).
* A value that was not confirmed on the publisher's page is **not filled in**; its
  dotted key goes in ``unverified`` (e.g. ``panel_label.bold``). When such a value is
  needed, the safe default in ``SAFE_DEFAULTS`` is used and reported in
  ``JournalSpec.defaults_used`` so the result can say so.
* ``max_height_mm: null`` means "the journal has no height rule" (not unverified).

Use `load_journal(id)` / `list_journals()`; `validate_journal_file` checks one file.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def journals_dir() -> Path:
    """``crystal-mcp/journals`` (bundled next to the code when frozen)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "journals"
    return Path(__file__).resolve().parents[2] / "journals"


#: Values used when a journal does not give (or we did not confirm) a rule.
#: Chosen on the safe side: narrow widths, large minimum text, thick lines, 600 dpi.
SAFE_DEFAULTS: dict[str, Any] = {
    "width_mm.single": 80.0,
    "width_mm.one_half": 120.0,
    "width_mm.double": 170.0,
    "max_height_mm": 225.0,
    "fonts": ["Arial", "Helvetica"],
    "text_pt.body": 8.0,
    "text_pt.min": 7.0,
    "min_line_pt": 0.5,
    "dpi.line": 600,
    "dpi.photo": 600,
    "dpi.combination": 600,
    "formats": ["tiff", "pdf", "eps", "png"],
    "color_mode": "RGB",
    "panel_label.style": "(a)",
    "panel_label.bold": False,
    # panel_label.size_pt falls back to text_pt.body (see JournalSpec.get)
}

WIDTH_KEYS = ("single", "one_half", "double")
WIDTH_ALIASES = {
    "single": "single", "1": "single", "one": "single", "column": "single", "1col": "single",
    "one_half": "one_half", "1.5": "one_half", "onehalf": "one_half", "one-half": "one_half",
    "1.5col": "one_half",
    "double": "double", "2": "double", "two": "double", "full": "double", "2col": "double",
    "page": "double",
}
DPI_KINDS = ("line", "photo", "combination")
LABEL_STYLES = ("(a)", "a", "A", "(A)", "a)")
FORMATS = ("tiff", "pdf", "eps", "png", "jpeg", "ps", "ai", "svg", "office", "hpgl", "gif")
COLOR_MODES = ("RGB", "CMYK", "grayscale")

_TOP_KEYS = {
    "id", "name", "publisher", "extends", "aliases", "journals", "width_mm", "max_height_mm",
    "fonts", "text_pt", "min_line_pt", "dpi", "formats", "color_mode", "panel_label",
    "sources", "unverified", "notes",
}
_SUB_KEYS = {
    "width_mm": set(WIDTH_KEYS),
    "text_pt": {"body", "min"},
    "dpi": set(DPI_KINDS),
    "panel_label": {"style", "bold", "size_pt"},
}
#: Every rule a journal must give or list in ``unverified`` (after extends).
REQUIRED_RULES: tuple[str, ...] = (
    "width_mm.single", "width_mm.one_half", "width_mm.double", "max_height_mm", "fonts",
    "text_pt.body", "text_pt.min", "min_line_pt", "dpi.line", "dpi.photo", "dpi.combination",
    "formats", "color_mode", "panel_label.style", "panel_label.bold", "panel_label.size_pt",
)


class JournalSpecError(ValueError):
    """A journal YAML is missing, malformed or inconsistent."""


def _get_dotted(data: dict[str, Any], key: str) -> tuple[bool, Any]:
    cur: Any = data
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise JournalSpecError(f"{path.name}: YAML の書き方が正しくありません: {exc}") from exc
    if not isinstance(data, dict):
        raise JournalSpecError(f"{path.name}: 中身が「キー: 値」の形ではありません")
    return data


def _check_shape(data: dict[str, Any], where: str) -> None:
    """Types and allowed keys of one (unmerged) file."""
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise JournalSpecError(f"{where}: 知らない項目 {sorted(unknown)}（使える項目は journals/README.md）")
    for top, subs in _SUB_KEYS.items():
        if top in data:
            if not isinstance(data[top], dict):
                raise JournalSpecError(f"{where}: {top} は「キー: 値」の形で書きます")
            bad = set(data[top]) - subs
            if bad:
                raise JournalSpecError(f"{where}: {top} に知らない項目 {sorted(bad)}（使える: {sorted(subs)}）")

    def _pos(key: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise JournalSpecError(f"{where}: {key} は正の数にします（いま {value!r}）")

    for k, v in (data.get("width_mm") or {}).items():
        _pos(f"width_mm.{k}", v)
    if data.get("max_height_mm") is not None:
        _pos("max_height_mm", data["max_height_mm"])
    for k, v in (data.get("text_pt") or {}).items():
        _pos(f"text_pt.{k}", v)
    if "min_line_pt" in data:
        _pos("min_line_pt", data["min_line_pt"])
    for k, v in (data.get("dpi") or {}).items():
        _pos(f"dpi.{k}", v)
    if "fonts" in data and (not isinstance(data["fonts"], list) or not data["fonts"]):
        raise JournalSpecError(f"{where}: fonts はフォント名のリストにします")
    if "formats" in data:
        fmts = data["formats"]
        if not isinstance(fmts, list) or not fmts:
            raise JournalSpecError(f"{where}: formats は形式のリストにします（例 [tiff, pdf, eps]）")
        bad = [f for f in fmts if str(f).lower() not in FORMATS]
        if bad:
            raise JournalSpecError(f"{where}: formats に知らない形式 {bad}（使える: {list(FORMATS)}）")
    if "color_mode" in data and data["color_mode"] not in COLOR_MODES:
        raise JournalSpecError(f"{where}: color_mode は {list(COLOR_MODES)} のどれか")
    pl = data.get("panel_label") or {}
    if "style" in pl and pl["style"] not in LABEL_STYLES:
        raise JournalSpecError(f"{where}: panel_label.style は {list(LABEL_STYLES)} のどれか（いま {pl['style']!r}）")
    if "bold" in pl and not isinstance(pl["bold"], bool):
        raise JournalSpecError(f"{where}: panel_label.bold は true か false")
    if "size_pt" in pl:
        _pos("panel_label.size_pt", pl["size_pt"])
    for i, src in enumerate(data.get("sources") or []):
        if not isinstance(src, dict) or not src.get("url") or not src.get("checked"):
            raise JournalSpecError(f"{where}: sources[{i}] には url と checked（確かめた日付）が要ります")
    unv = data.get("unverified", [])
    if not isinstance(unv, list):
        raise JournalSpecError(f"{where}: unverified は項目名のリストにします")
    bad_unv = [u for u in unv if u not in REQUIRED_RULES]
    if bad_unv:
        raise JournalSpecError(f"{where}: unverified に知らない項目 {bad_unv}（使える: {list(REQUIRED_RULES)}）")
    both = [u for u in unv if _get_dotted(data, u)[0]]
    if both:
        raise JournalSpecError(f"{where}: 値を書いたのに unverified にも入っている: {both}（どちらかにします）")


def _resolve(name: str, directory: Path, chain: tuple[str, ...] = ()) -> dict[str, Any]:
    """Merged data of ``name`` (file stem) with its extends chain applied."""
    if name in chain:
        raise JournalSpecError(f"extends がぐるぐる回っています: {' -> '.join((*chain, name))}")
    path = directory / f"{name}.yaml"
    if not path.is_file():
        where = f"（{chain[-1]}.yaml の extends）" if chain else ""
        raise JournalSpecError(f"雑誌の設定ファイルがありません: {path.name}{where}")
    data = _read_yaml(path)
    _check_shape(data, path.name)
    parent_name = data.get("extends")
    if not parent_name:
        merged = copy.deepcopy(data)
        merged["unverified"] = list(data.get("unverified", []))
        merged["sources"] = list(data.get("sources", []))
        return merged
    parent = _resolve(str(parent_name), directory, (*chain, name))
    merged = _deep_merge(parent, {k: v for k, v in data.items() if k not in {"unverified", "sources"}})
    # a rule the child gives is no longer unverified; the child's own unverified are added
    unv = [u for u in parent.get("unverified", []) if not _get_dotted(data, u)[0]]
    unv += [u for u in data.get("unverified", []) if u not in unv]
    merged["unverified"] = unv
    merged["sources"] = [*parent.get("sources", []), *data.get("sources", [])]
    return merged


@dataclass
class JournalSpec:
    """Resolved rules of one journal. Use `get` so unverified rules fall back safely."""

    id: str
    name: str
    publisher: str
    data: dict[str, Any]
    unverified: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    defaults_used: dict[str, Any] = field(default_factory=dict)

    def has(self, key: str) -> bool:
        found, _ = _get_dotted(self.data, key)
        return found and key not in self.unverified

    def get(self, key: str) -> Any:
        """The journal's value, or the safe default (then recorded in ``defaults_used``)."""
        found, value = _get_dotted(self.data, key)
        if found and key not in self.unverified:
            return value
        if key == "panel_label.size_pt":
            value = self.get("text_pt.body")
        elif key == "text_pt.body":
            value = max(SAFE_DEFAULTS["text_pt.body"], float(self.get("text_pt.min")))
        else:
            value = copy.deepcopy(SAFE_DEFAULTS[key])
        self.defaults_used[key] = value
        return value

    def width_mm(self, width: str | float | int) -> tuple[str, float]:
        """('single' | 'one_half' | 'double' | 'custom', mm) for a width name or a number in mm."""
        if isinstance(width, (int, float)) and not isinstance(width, bool):
            mm = float(width)
        else:
            text = str(width).strip().lower()
            if text in WIDTH_ALIASES:
                key = WIDTH_ALIASES[text]
                return key, float(self.get(f"width_mm.{key}"))
            try:
                mm = float(text.removesuffix("mm"))
            except ValueError:
                raise JournalSpecError(
                    f"段の幅 {width!r} が分かりません（single / one_half / double か mm の数）"
                ) from None
        if mm <= 0:
            raise JournalSpecError("段の幅は正の数にします")
        return "custom", mm

    def max_height_mm(self) -> float | None:
        """None = the journal has no height rule."""
        found, value = _get_dotted(self.data, "max_height_mm")
        if found and "max_height_mm" not in self.unverified:
            return None if value is None else float(value)
        return float(self.get("max_height_mm"))

    def formats(self) -> list[str]:
        return [str(f).lower() for f in self.get("formats")]

    def summary(self) -> dict[str, Any]:
        """Short JSON for list_journals (journal values only; unverified shown as such)."""
        def val(key: str) -> Any:
            found, value = _get_dotted(self.data, key)
            return value if found and key not in self.unverified else "未確認"
        return {
            "id": self.id,
            "name": self.name,
            "publisher": self.publisher,
            "width_mm": {k: val(f"width_mm.{k}") for k in WIDTH_KEYS},
            "max_height_mm": val("max_height_mm"),
            "text_pt_min": val("text_pt.min"),
            "min_line_pt": val("min_line_pt"),
            "dpi": {k: val(f"dpi.{k}") for k in DPI_KINDS},
            "formats": val("formats"),
            "panel_label": {k: val(f"panel_label.{k}") for k in ("style", "bold", "size_pt")},
            "unverified": list(self.unverified),
        }


def _journal_ids(directory: Path) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.yaml") if not p.stem.startswith("_"))


def _alias_map(directory: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for jid in _journal_ids(directory):
        data = _read_yaml(directory / f"{jid}.yaml")
        for alias in data.get("aliases") or []:
            out[str(alias).strip().lower()] = jid
    return out


def _normalize_id(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


def load_journal(journal: str, directory: str | Path | None = None) -> JournalSpec:
    """Resolved `JournalSpec` for a file stem (``acta_materialia``) or an alias (``Acta Mater.``)."""
    d = Path(directory) if directory is not None else journals_dir()
    ids = _journal_ids(d)
    key = _normalize_id(journal)
    if key not in ids:
        aliases = {_normalize_id(a): j for a, j in _alias_map(d).items()}
        if key not in aliases:
            raise JournalSpecError(
                f"雑誌 {journal!r} の設定がありません。使える雑誌: {', '.join(ids)}"
                "（足し方は journals/README.md）"
            )
        key = aliases[key]
    data = _resolve(key, d)
    validate_resolved(data, f"{key}.yaml")
    return JournalSpec(
        id=key,
        name=str(data["name"]),
        publisher=str(data["publisher"]),
        data=data,
        unverified=list(data.get("unverified", [])),
        sources=[{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in src.items()}
                 for src in data.get("sources", [])],
    )


def validate_resolved(data: dict[str, Any], where: str) -> None:
    """Checks after extends: name / publisher / sources, every rule given or unverified."""
    for key in ("name", "publisher"):
        if not data.get(key):
            raise JournalSpecError(f"{where}: {key} がありません")
    if not data.get("sources"):
        raise JournalSpecError(f"{where}: sources（調べた URL と日付）がありません")
    missing = [r for r in REQUIRED_RULES if not _get_dotted(data, r)[0] and r not in data["unverified"]
               and r != "panel_label.size_pt"]
    if missing:
        raise JournalSpecError(
            f"{where}: 値も unverified も無い項目 {missing}。値を書くか、確かめていなければ unverified に入れます"
        )


def validate_journal_file(path: str | Path) -> JournalSpec:
    """Load and fully validate one journal file (also its extends chain)."""
    p = Path(path)
    return load_journal(p.stem, p.parent)


def list_journals(directory: str | Path | None = None) -> list[dict[str, Any]]:
    """`JournalSpec.summary` of every journal file (publisher bases ``_*.yaml`` skipped)."""
    d = Path(directory) if directory is not None else journals_dir()
    return [load_journal(j, d).summary() for j in _journal_ids(d)]


__all__ = [
    "JournalSpec",
    "JournalSpecError",
    "REQUIRED_RULES",
    "SAFE_DEFAULTS",
    "journals_dir",
    "list_journals",
    "load_journal",
    "validate_journal_file",
]

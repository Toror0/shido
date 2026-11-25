"""Color theme manager (ColorManager + PlaneStyle) for CrystalAnalyzerMCP.

Supports 4 themes from ``themes.yaml`` (acta_modern, cpk_classic, okabe_ito_safe,
mono_contrast), 4-tier element color resolution, and (hkl) crystal plane styling.
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymatgen.core import Structure

_DEFAULT_YAML_PATH = Path(__file__).resolve().parents[1] / "themes.yaml"  # repo root

_HEX_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")


def _is_valid_hex(color: Any) -> bool:
    """True for '#RRGGBB' (or 'RRGGBB') strings."""
    return isinstance(color, str) and bool(_HEX_RE.match(color.strip()))


def _normalize_hex(color: str) -> str:
    """'aabbcc' / '#aabbcc' -> '#AABBCC'."""
    return "#" + color.strip().lstrip("#").upper()


def _generate_distinct_color(used: set[str]) -> str:
    """Deterministic extra color (golden-angle hue walk) not in ``used``.

    Used when the theme palette and the Jmol table are exhausted (e.g. a 10-element
    high-entropy alloy), so every element still gets a valid, distinct '#RRGGBB'.
    """
    for k in range(1, 4096):
        hue = (k * 0.618033988749895) % 1.0
        sat = 0.55 + 0.25 * ((k // 7) % 2)
        val = 0.80 - 0.20 * ((k // 3) % 2)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        cand = "#{:02X}{:02X}{:02X}".format(round(r * 255), round(g * 255), round(b * 255))
        if cand not in used:
            return cand
    return "#808080"

# 【優先度4】Jmol / CPK 国際標準色（最終フォールバック）
JMOL_FALLBACK_COLORS: Dict[str, str] = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
    "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7",
    "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050",
    "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#668F8F",
    "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1",
    "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
    "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F",
    "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F",
    "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00",
    "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F", "Ba": "#00C900",
    "La": "#70D4FF", "Ce": "#FFFFC7", "Nd": "#C7FFC7", "Sm": "#8FFFC7",
    "Gd": "#45FFC7", "Tb": "#30FFC7", "Dy": "#1FFFC7", "Hf": "#4DC2FF",
    "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB", "Os": "#266696",
    "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123", "Hg": "#B8B8D0",
    "Pb": "#575961", "Bi": "#9E4FB5",
}


@dataclass(frozen=True)
class PlaneStyle:
    """結晶面(hkl)の描画スタイル定義"""

    fill_color: str
    edge_color: str
    fill_opacity: float
    edge_opacity: float
    line_width: float


def _load_yaml_safe(path: Path) -> Dict[str, Any]:
    """Load YAML using ruamel.yaml (bundled with pymatgen) or PyYAML."""
    if not path.exists():
        raise FileNotFoundError(f"テーマ設定ファイルが見つかりません: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        from ruamel.yaml import YAML

        yaml_loader = YAML(typ="safe")
        return yaml_loader.load(text) or {}
    except Exception:
        pass
    import yaml  # type: ignore[import-untyped]

    return yaml.safe_load(text) or {}


class ColorManager:
    """元素カラー（4段階優先順位）および結晶面カラーをテーマ設定に基づいて管理するクラス。"""

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_YAML_PATH,
        theme_name: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        self._raw_config = _load_yaml_safe(self.config_path)
        self.themes: Dict[str, Any] = self._raw_config.get("themes", {})
        default_name = self._raw_config.get("default_theme", "acta_modern")
        self.set_theme(theme_name or default_name)

    def list_themes(self) -> List[str]:
        """利用可能なテーマ名の一覧を返す"""
        return list(self.themes.keys())

    def set_theme(self, theme_name: str) -> None:
        """アクティブなカラーテーマを切り替える"""
        if theme_name not in self.themes:
            available = ", ".join(self.list_themes())
            raise ValueError(f"未定義のテーマ '{theme_name}' です。利用可能: [{available}]")
        self.theme_name = theme_name
        self.theme = self.themes[theme_name]

    @property
    def scene_colors(self) -> Dict[str, Any]:
        """背景・単位格子・結合棒・輪郭線・3軸インジケーターの色設定を返す"""
        return {
            "background": self.theme.get("background", "#FFFFFF"),
            "unit_cell": self.theme.get("unit_cell", "#2B2D42"),
            "subcell_highlight": self.theme.get("subcell_highlight", "#D90429"),
            "bond": self.theme.get("bond", "#B8C0CC"),
            "silhouette": self.theme.get("silhouette", "#1F2421"),
            "axes": self.theme.get("axes", {"a": "#E63946", "b": "#2A9D8F", "c": "#264653"}),
        }

    def resolve_element_colors(
        self,
        species_counts: Dict[str, float] | Sequence[str],
        overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """4段階優先順位（overrides > theme elements > fallback_order > Jmol）で元素カラーを解決する。"""
        overrides = overrides or {}
        theme_elements: Dict[str, str] = self.theme.get("elements", {})
        fallback_palette: List[str] = self.theme.get("fallback_order", [])

        if isinstance(species_counts, dict):
            ordered_elements = [
                el for el, _ in sorted(species_counts.items(), key=lambda x: (-x[1], x[0]))
            ]
        else:
            ordered_elements = list(dict.fromkeys(species_counts))

        resolved: Dict[str, str] = {}
        used_colors: set[str] = set()
        unresolved_elements: List[str] = []

        for el in ordered_elements:
            # A malformed color in overrides / themes.yaml falls through to the next tier.
            if el in overrides and _is_valid_hex(overrides[el]):
                color = _normalize_hex(overrides[el])
                resolved[el] = color
                used_colors.add(color)
            elif el in theme_elements and _is_valid_hex(theme_elements[el]):
                color = _normalize_hex(theme_elements[el])
                resolved[el] = color
                used_colors.add(color)
            else:
                unresolved_elements.append(el)

        # Reserve palette: valid, not yet used, no duplicates. Never indexed past its end.
        available_fallbacks: List[str] = []
        for c in fallback_palette:
            if _is_valid_hex(c):
                cn = _normalize_hex(c)
                if cn not in used_colors and cn not in available_fallbacks:
                    available_fallbacks.append(cn)
        fallback_iter = iter(available_fallbacks)
        for el in unresolved_elements:
            color = next(fallback_iter, None)
            if color is None:
                jmol = JMOL_FALLBACK_COLORS.get(el)
                if jmol is not None and _is_valid_hex(jmol) and _normalize_hex(jmol) not in used_colors:
                    color = _normalize_hex(jmol)
                else:
                    color = _generate_distinct_color(used_colors)
            resolved[el] = color
            used_colors.add(color)

        return resolved

    def resolve_plane_styles(
        self,
        num_planes: int,
        custom_planes: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[PlaneStyle]:
        """複数枚の結晶面(hkl)に対して、塗りつぶし色・透過度・輪郭線色を自動割り当てする。"""
        planes_cfg: Dict[str, Any] = self.theme.get("planes", {})
        default_opacity = float(planes_cfg.get("default_opacity", 0.32))
        edge_opacity = float(planes_cfg.get("edge_opacity", 0.85))
        line_width = float(planes_cfg.get("line_width", 2.0))
        palette: List[Dict[str, str]] = planes_cfg.get(
            "palette", [{"fill": "#3A86FF", "edge": "#1D4ED8"}]
        )
        custom_planes = custom_planes or []

        styles: List[PlaneStyle] = []
        for idx in range(num_planes):
            base = palette[idx % len(palette)] if palette else {"fill": "#3A86FF", "edge": "#1D4ED8"}
            override = custom_planes[idx] if idx < len(custom_planes) else {}
            styles.append(
                PlaneStyle(
                    fill_color=override.get("fill", base["fill"]),
                    edge_color=override.get("edge", base["edge"]),
                    fill_opacity=float(override.get("opacity", default_opacity)),
                    edge_opacity=float(override.get("edge_opacity", edge_opacity)),
                    line_width=float(override.get("line_width", line_width)),
                )
            )
        return styles

    @staticmethod
    def hex_to_rgb_float(hex_color: str) -> Tuple[float, float, float]:
        """PyVista / VTK 向けに '#52B788' を (0.321, 0.718, 0.533) に変換する"""
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Process-wide singleton helpers (shared across renderer / phase_compare / server)
# ---------------------------------------------------------------------------
_ACTIVE_MANAGER: Optional[ColorManager] = None
# Last ``render.theme`` value seen in settings. The active theme follows that setting
# whenever it CHANGES (e.g. picked in the GUI), so no restart is needed; an explicit
# set_active_theme() (CLI --theme, MCP) still wins until the setting changes again.
_UNSEEN = object()
_LAST_SETTINGS_THEME: Any = _UNSEEN


def _settings_theme_name() -> Optional[str]:
    try:
        from shido import settings as _settings

        name = _settings.get("render.theme", None)
    except Exception:
        return None
    return str(name) if name else None


def get_color_manager() -> ColorManager:
    global _ACTIVE_MANAGER, _LAST_SETTINGS_THEME
    if _ACTIVE_MANAGER is None:
        _ACTIVE_MANAGER = ColorManager(_DEFAULT_YAML_PATH)
    wanted = _settings_theme_name()
    if wanted != _LAST_SETTINGS_THEME:
        _LAST_SETTINGS_THEME = wanted
        if wanted and wanted in _ACTIVE_MANAGER.themes:
            _ACTIVE_MANAGER.set_theme(wanted)
    return _ACTIVE_MANAGER


def list_themes() -> List[str]:
    return get_color_manager().list_themes()


def get_active_theme_name() -> str:
    return get_color_manager().theme_name


def set_active_theme(theme_name: Optional[str]) -> str:
    cm = get_color_manager()
    if theme_name is not None:
        cm.set_theme(theme_name)
    return cm.theme_name


def get_theme(theme_name: Optional[str] = None) -> Dict[str, Any]:
    cm = get_color_manager()
    if theme_name and theme_name in cm.themes:
        theme = dict(cm.themes[theme_name])
        theme["name"] = theme_name
        return theme
    theme = dict(cm.theme)
    theme["name"] = cm.theme_name
    return theme


def resolve_structure_colors(
    structure: Structure,
    theme_name: Optional[str] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    cm = get_color_manager()
    prev = cm.theme_name
    if theme_name and theme_name != prev:
        cm.set_theme(theme_name)
    try:
        # element_composition merges oxidation states (Fe2+ and Fe3+ -> Fe) instead of overwriting.
        species_counts = {
            el.symbol: float(amt) for el, amt in structure.composition.element_composition.items()
        }
        return cm.resolve_element_colors(species_counts, overrides=overrides)
    finally:
        if theme_name and theme_name != prev:
            cm.set_theme(prev)

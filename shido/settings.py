"""前提数値（Assumed parameters）の読み書き。

- 既定値は ``settings_defaults.yaml``（Git で管理。よく使う数値を入れてある）
- 利用者が GUI などで変えた値は ``settings_user.yaml``（このPCだけ。Git には入れない）
- 読むときは「既定値 ← 利用者の値」の順に深く上書きしてまとめた辞書を返す

各モジュールは import 時ではなく **呼ばれたときに** ``get()`` で読むこと。
そうすれば GUI で値を変えたあと、アプリを再起動しなくても反映される。

    from shido.settings import get, atomic_radius
    a = get("lattice.a_austenite")          # 5.825
    r = atomic_radius("Ni")                  # 表示用の原子半径 [Å]
"""

from __future__ import annotations

import contextlib
import copy
import threading
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml

# yaml はリポジトリのいちばん上（crystal/ の1つ上）に置く
_DIR = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = _DIR / "settings_defaults.yaml"
# GUI で変えた値は .exe（PyInstaller）のときは .exe の横に書く（中身のフォルダには書かない）
_USER_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else _DIR
USER_PATH = _USER_DIR / "settings_user.yaml"

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_CACHE_STAMP: tuple[float, float] | None = None
_MISSING = object()


def _stamp() -> tuple[float, float]:
    """2つのファイルの更新時刻。GUI と MCP サーバーが別々に動いていても変更に気づくため。"""
    def mt(p: Path) -> float:
        return p.stat().st_mtime if p.exists() else 0.0
    return (mt(DEFAULTS_PATH), mt(USER_PATH))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} の中身が辞書（キー: 値）になっていません")
    return data


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """over で base を上書き。over の値が None なら「その項目を消す」。
    ただし既定値が辞書（項目のまとまり）のところを None や数値で壊すことはできない。"""
    out = copy.deepcopy(base)
    for key, val in over.items():
        if isinstance(out.get(key), dict):
            if isinstance(val, dict):
                out[key] = _deep_merge(out[key], val)
            # 辞書でない値（None など）は無視して既定値を守る
        elif val is None:
            out.pop(key, None)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_defaults() -> dict[str, Any]:
    """既定値だけを返す（利用者の上書きは含まない）。"""
    return _read_yaml(DEFAULTS_PATH)


def load_settings(force: bool = False) -> dict[str, Any]:
    """既定値に利用者の値を重ねた設定一式を返す（コピーを返すので書き換えても安全）。"""
    global _CACHE, _CACHE_STAMP
    if _OVERRIDE is not None:  # a recipe rerun: the recorded settings, not the files
        return copy.deepcopy(_OVERRIDE)
    with _LOCK:
        stamp = _stamp()
        if _CACHE is None or force or stamp != _CACHE_STAMP:
            _CACHE = _deep_merge(load_defaults(), _read_yaml(USER_PATH))
            _CACHE_STAMP = stamp
        return copy.deepcopy(_CACHE)


_OVERRIDE: dict[str, Any] | None = None


@contextlib.contextmanager
def overridden(full_settings: dict[str, Any]) -> Iterator[None]:
    """この中だけ、ファイルの代わりに ``full_settings``（設定一式）を使う。
    レシピの再実行（recipe.rerun）で、記録した時の前提数値をそのまま使うため。ファイルは書き換えない。"""
    global _OVERRIDE
    previous = _OVERRIDE
    _OVERRIDE = copy.deepcopy(full_settings)
    try:
        yield
    finally:
        _OVERRIDE = previous


def reload() -> dict[str, Any]:
    return load_settings(force=True)


def get(path: str, default: Any = _MISSING) -> Any:
    """``"render.dpi"`` のようなドット区切りのキーで値を読む。"""
    node: Any = load_settings()
    for part in path.split("."):
        if isinstance(node, dict) and node.get(part) is not None:
            node = node[part]
        else:
            if default is _MISSING:
                raise KeyError(f"設定 '{path}' がありません（settings_defaults.yaml を確認）")
            return default
    return node


def _diff_from_defaults(full: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """既定値と違う部分だけを取り出す（user ファイルを小さく保つため）。"""
    diff: dict[str, Any] = {}
    # 既定値にあって新しい値に無い項目（例：原子半径の表から外した元素）は None で「消した」と記録する
    for key in defaults:
        if key not in full:
            diff[key] = None
    for key, val in full.items():
        dval = defaults.get(key, _MISSING)
        if isinstance(val, dict) and isinstance(dval, dict):
            sub = _diff_from_defaults(val, dval)
            if sub:
                diff[key] = sub
        elif dval is _MISSING or dval != val:
            diff[key] = copy.deepcopy(val)
    return diff


def save_user_settings(full_settings: dict[str, Any]) -> Path:
    """設定一式を受け取り、既定値と違う部分だけを settings_user.yaml に保存する。"""
    diff = _diff_from_defaults(full_settings, load_defaults())
    with USER_PATH.open("w", encoding="utf-8") as fh:
        fh.write("# GUI などで変えた前提数値（既定値と違うものだけ）。消すと既定値に戻る。\n")
        yaml.safe_dump(diff, fh, allow_unicode=True, sort_keys=False)
    reload()
    return USER_PATH


def update(values: dict[str, Any]) -> dict[str, Any]:
    """``{"render.dpi": 600, "lattice.a_austenite": 5.8}`` のように一部だけ変えて保存する。"""
    full = load_settings()
    for path, val in values.items():
        validate_value(path, val)
        parts = path.split(".")
        node = full
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = val
    save_user_settings(full)
    return load_settings()


def validate_value(path: str, val: Any) -> None:
    """MCP などから1項目を変えるときの確認。おかしければ ValueError。
    - 既定値に無いキーは不可（原子半径 atomic_radii.<元素> は追加してよい）
    - 既定値と同じ種類（数・真偽・文字・リスト）であること。数は有限で、0 より大きい（seed は 0 以上）
    """
    import math

    defaults = load_defaults()
    parts = path.split(".")
    if parts[0] == "atomic_radii" and len(parts) == 2:
        from pymatgen.core import Element

        Element(parts[1])  # 元素記号でなければ ValueError
        ref: Any = 1.0
    else:
        ref = defaults
        for part in parts:
            if not isinstance(ref, dict) or part not in ref:
                raise ValueError(f"'{path}' という前提数値はありません")
            ref = ref[part]
        if isinstance(ref, dict):
            raise ValueError(f"'{path}' は項目のまとまりなので、中の項目を1つずつ指定してください")
    if isinstance(ref, bool):
        if not isinstance(val, bool):
            raise ValueError(f"'{path}' は true / false で指定してください")
    elif isinstance(ref, (int, float)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f"'{path}' は数で指定してください")
        if val < 0 or (val == 0 and not path.endswith("seed")):
            raise ValueError(f"'{path}' は 0 より大きい数にしてください")
        if isinstance(ref, int) and not isinstance(ref, bool) and int(val) != val:
            raise ValueError(f"'{path}' は整数にしてください")
    elif isinstance(ref, str):
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"'{path}' は文字で指定してください")
    elif isinstance(ref, list):
        if not isinstance(val, list) or len(val) != len(ref):
            raise ValueError(f"'{path}' は {len(ref)} 個のリストで指定してください")


def reset_user_settings() -> None:
    """利用者の上書きを消して既定値に戻す。"""
    if USER_PATH.exists():
        USER_PATH.unlink()
    reload()


def atomic_radius(symbol: str) -> float | None:
    """表示用の原子半径 [Å]（``atomic_radii`` 表）。表に無い元素は None。"""
    table = get("atomic_radii", {}) or {}
    val = table.get(symbol)
    return float(val) if val is not None else None


def material_preset(name: str) -> dict[str, Any] | None:
    for preset in get("material_presets", []) or []:
        if preset.get("name") == name:
            return preset
    return None

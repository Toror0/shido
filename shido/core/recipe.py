"""Recipe: how to make the same result again (``recipe.json`` in every run folder).

One run = one call of a CLI command (``python server.py ...``) or one MCP tool.
The entry point opens a *recording* (`recording_cli` / `recorded_tool`); the code
that finishes a run folder (``server._finish_run``) calls `write_recipe`, which
writes ``recipe.json`` with

* the tool version (and git short hash), Python and the main library versions,
* the command (argv, plus the parsed options with defaults filled in) or the MCP
  tool name and all its arguments (defaults filled in),
* each input file with its SHA-256,
* the settings in effect (settings_defaults.yaml + settings_user.yaml) and the colour theme,
* the guardrail (validator) result and every warning,
* every result file with its SHA-256.

`rerun` reads a recipe and runs the same call again into a NEW folder, with the
recorded settings and theme; ``overrides`` change a few things (``theme=...`` is a
command option / tool argument, ``render.dpi=600`` (with a dot) is a setting).
A changed input file or library version gives a warning, never an error.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import copy
import datetime as _dt
import functools
import hashlib
import html
import importlib.metadata as _md
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

RECIPE_NAME = "recipe.json"
SCHEMA = "crystal-mcp/recipe/1"
TOOL_NAME = "Shido"
LIBRARIES: tuple[str, ...] = ("pymatgen", "pyvista", "vtk", "numpy", "spglib", "matplotlib", "scipy")
# Arguments / options that only say WHERE to save: a rerun always makes a new folder.
OUTPUT_ARGS: frozenset[str] = frozenset({"output_dir", "output_path", "out_dir", "base_dir"})
OUTPUT_FLAGS: frozenset[str] = frozenset({"--out-dir", "--output-dir", "--output", "-o"})

# The call being recorded (None outside a CLI command / MCP tool).
_CURRENT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("crystal_recipe", default=None)
# MCP tools by name, filled by `recorded_tool` (used by `rerun`).
TOOLS: dict[str, Callable[..., Any]] = {}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def git_revision() -> dict[str, Any] | None:
    """{"commit": short hash, "dirty": bool} of the Shido checkout, or None (no git / exe)."""
    if getattr(sys, "frozen", False):
        return None
    root = str(_repo_root())
    try:
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": bool(dirty)} if commit else None


def library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in LIBRARIES:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = None
    return out


def environment() -> dict[str, Any]:
    from shido.core import __version__

    return {
        "tool": {"name": TOOL_NAME, "version": __version__, "git": git_revision()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": library_versions(),
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    """Plain JSON value (Paths -> str, tuples -> lists, numpy scalars -> numbers)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)  # numpy scalar
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)  # numpy array
    if callable(tolist):
        return _jsonable(tolist())
    return str(value)


#: At most this many structure files of an input folder get a SHA-256.
MAX_FOLDER_FILES = 500


def _folder_files(folder: Path) -> list[dict[str, Any]]:
    """Structure files in an input folder (what ``batch`` reads) with their SHA-256."""
    from shido.analyzer import is_structure_file
    from shido.core.output_paths import OUTPUT_ROOT_NAME

    out: list[dict[str, Any]] = []
    for f in sorted(folder.rglob("*"), key=lambda q: q.relative_to(folder).as_posix().lower()):
        rel = f.relative_to(folder)
        if any(part.lower().startswith("output_") or part == OUTPUT_ROOT_NAME for part in rel.parts[:-1]):
            continue
        if f.is_file() and is_structure_file(f):
            out.append({"path": rel.as_posix(), "sha256": sha256(f), "bytes": f.stat().st_size})
            if len(out) >= MAX_FOLDER_FILES:
                break
    return out


def _input_entry(path: Path) -> dict[str, Any]:
    p = path.resolve()
    if p.is_dir():
        return {"path": str(p), "kind": "folder", "files": _folder_files(p)}
    return {"path": str(p), "kind": "file", "sha256": sha256(p), "bytes": p.stat().st_size}


def _existing_paths(s: str) -> list[Path]:
    """Existing files / folders named by ``s``: the text itself, else the value of
    ``--key=value``, else the parts of a comma list (``bulk:iso,a.png``)."""
    if s and len(s) < 1024 and "\n" not in s and not s.startswith("-"):
        try:
            p = Path(s)
            if p.exists():
                return [p]
        except (OSError, ValueError):
            pass
    if s.startswith("-") and "=" in s:
        return _existing_paths(s.partition("=")[2])
    if "," in s:
        return [p for part in s.split(",") if part.strip() for p in _existing_paths(part.strip())]
    return []


def _find_inputs(values: Any) -> list[dict[str, Any]]:
    """Every str that names an existing file or folder (inside lists / dicts too)."""
    found: dict[str, dict[str, Any]] = {}

    def walk(v: Any) -> None:
        if isinstance(v, (str, Path)):
            for p in _existing_paths(str(v)):
                try:
                    if str(p.resolve()) not in found:
                        e = _input_entry(p)
                        found[e["path"]] = e
                except OSError:
                    pass
        elif isinstance(v, Mapping):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(values)
    return list(found.values())


def _collect_warnings(result: Any, out: list[str]) -> None:
    """Every list under a key ending in 'warnings' (any depth)."""
    if isinstance(result, Mapping):
        for k, v in result.items():
            if str(k).endswith("warnings") and isinstance(v, (list, tuple)):
                for w in v:
                    if str(w) not in out:
                        out.append(str(w))
            else:
                _collect_warnings(v, out)
    elif isinstance(result, (list, tuple)):
        for v in result:
            _collect_warnings(v, out)


# ---------------------------------------------------------------------------
# Recording (entry points)
# ---------------------------------------------------------------------------

def current() -> dict[str, Any] | None:
    return _CURRENT.get()


@contextlib.contextmanager
def _recording(call: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Open a recording unless one is already open (the outermost call wins)."""
    if _CURRENT.get() is not None:
        yield _CURRENT.get()  # type: ignore[misc]
        return
    ctx = {"call": call, "warnings": [], "started": _dt.datetime.now().isoformat(timespec="seconds")}
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)


#: First words of ``server.py`` that pick a command (never made into a path, even when a
#: folder of the same name is in the current folder).
CLI_SUBCOMMANDS: frozenset[str] = frozenset({
    "rerun", "interactive", "menu", "measure", "symmetry", "transform", "export", "polyhedra",
    "structure-factors", "sf", "displacement", "adp", "slice", "volume-slice", "edit", "create",
    "volume-model", "density", "morphology", "crystal-shape", "figure", "journal-figure",
})


def _existing_absolute(s: str) -> str | None:
    if not s or len(s) >= 1024 or "\n" in s:
        return None
    try:
        p = Path(s)
        return str(p.resolve()) if p.exists() else None
    except (OSError, ValueError):
        return None


def _absolute_if_path(token: Any, *, first: bool = False) -> str:
    """An argv token naming an existing file / folder becomes absolute (rerun works from any folder).

    Also the value of ``--key=value`` and the parts of a comma list (``bulk:iso,a.png``).
    ``first``: argv[0], a command name such as ``figure`` stays as it is."""
    s = str(token)
    if first and s.lower() in CLI_SUBCOMMANDS:
        return s
    if s.startswith("-"):
        name, eq, val = s.partition("=")
        if eq and val and not name[1:2].isdigit():
            return f"{name}={_absolute_if_path(val)}"
        return s
    if "=" in s:
        return s
    whole = _existing_absolute(s)
    if whole is not None:
        return whole
    if "," in s:
        parts = s.split(",")
        new = [_existing_absolute(p.strip()) or p for p in parts]
        if new != parts:
            return ",".join(new)
    return s


def _absolute_tree(value: Any) -> Any:
    """`_absolute_if_path` on every str inside lists / dicts (MCP tool arguments)."""
    if isinstance(value, str):
        return _absolute_if_path(value)
    if isinstance(value, list):
        return [_absolute_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _absolute_tree(v) for k, v in value.items()}
    return value


@contextlib.contextmanager
def recording_cli(argv: list[str]) -> Iterator[dict[str, Any]]:
    """Record one CLI command. The options argparse resolved (defaults filled in) are
    picked up from the command's own parser while the recording is open."""
    call: dict[str, Any] = {"kind": "cli", "argv": [_absolute_if_path(a, first=i == 0) for i, a in enumerate(argv)],
                            "resolved_options": None}
    with _recording(call) as ctx:
        if ctx["call"] is not call:  # nested: the outer call is the recipe
            yield ctx
            return
        original = argparse.ArgumentParser.parse_known_args

        def spy(self: argparse.ArgumentParser, *a: Any, **kw: Any) -> Any:
            ns, rest = original(self, *a, **kw)
            call["resolved_options"] = _jsonable(vars(ns))
            return ns, rest

        argparse.ArgumentParser.parse_known_args = spy  # type: ignore[method-assign]
        try:
            yield ctx
        finally:
            argparse.ArgumentParser.parse_known_args = original  # type: ignore[method-assign]


def recorded_tool(fn: Callable[..., Any], name: str | None = None) -> Callable[..., Any]:
    """Wrap an MCP tool so each call records its name and every argument (defaults filled in).
    The signature is kept (``functools.wraps``), so the tool schema does not change."""
    tool_name = name or fn.__name__
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = _absolute_tree(_jsonable(dict(bound.arguments)))
        except TypeError:
            arguments = _jsonable(kwargs)
        with _recording({"kind": "mcp", "tool": tool_name, "arguments": arguments}):
            return fn(*args, **kwargs)

    TOOLS[tool_name] = wrapper
    return wrapper


def recording_tool_decorator(tool: Callable[..., Any]) -> Callable[..., Any]:
    """``mcp.tool = recording_tool_decorator(mcp.tool)``: every ``@mcp.tool()`` records its calls."""

    @functools.wraps(tool)
    def tool_with_recipe(*a: Any, **kw: Any) -> Callable[[Callable[..., Any]], Any]:
        register = tool(*a, **kw)
        name = kw.get("name") or (a[0] if a and isinstance(a[0], str) else None)

        def deco(fn: Callable[..., Any]) -> Any:
            return register(recorded_tool(fn, name))

        return deco

    return tool_with_recipe


def note_result(result: Any) -> None:
    """Keep the warnings of a result that does not reach `write_recipe` itself."""
    ctx = _CURRENT.get()
    if ctx is not None:
        _collect_warnings(result, ctx["warnings"])


# ---------------------------------------------------------------------------
# Writing recipe.json
# ---------------------------------------------------------------------------

def _settings_snapshot() -> dict[str, Any]:
    from shido import settings as cfg
    from shido.theme_manager import get_active_theme_name

    effective = cfg.load_settings()
    return {
        "effective": _jsonable(effective),
        "changed_from_defaults": _jsonable(cfg._diff_from_defaults(effective, cfg.load_defaults())),
        "theme": get_active_theme_name(),
    }


def _output_files(run: Path) -> list[dict[str, Any]]:
    from shido.core.output_paths import INDEX_NAME, MAX_LISTED_FILES

    files = []
    for p in sorted(run.rglob("*"), key=lambda q: q.relative_to(run).as_posix().lower()):
        if not p.is_file() or (p.parent == run and p.name in {RECIPE_NAME, INDEX_NAME}):
            continue
        files.append({"path": p.relative_to(run).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size})
        if len(files) >= MAX_LISTED_FILES:
            break
    return files


def build_recipe(run_dir: str | Path, result: Any = None, title: str = "") -> dict[str, Any] | None:
    ctx = _CURRENT.get()
    if ctx is None:
        return None
    run = Path(run_dir).resolve()
    call = copy.deepcopy(ctx["call"])
    if call["kind"] == "cli":
        values: Any = [a for a, prev in zip(call["argv"], [None, *call["argv"]]) if prev not in OUTPUT_FLAGS]
    else:
        values = {k: v for k, v in call["arguments"].items() if k not in OUTPUT_ARGS}
    warnings_: list[str] = list(ctx["warnings"])
    _collect_warnings(result, warnings_)
    recipe: dict[str, Any] = {
        "schema": SCHEMA,
        "title": title,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        **environment(),
        "call": call,
        "inputs": _find_inputs(values),
        "settings": _settings_snapshot(),
        "validation": {
            "status": "passed",
            "note": "Guardrails (lattice, NaN/Inf, overlapping atoms, composition, blank image) "
                    "raise an error when they fail; this run finished, so they passed.",
            "warnings": warnings_,
        },
        "outputs": _output_files(run),
    }
    if ctx.get("rerun_of"):
        recipe["rerun_of"] = ctx["rerun_of"]
    return recipe


def write_recipe(run_dir: str | Path, result: Any = None, title: str = "") -> dict[str, Any] | None:
    """Write ``<run_dir>/recipe.json`` when a CLI command / MCP tool is being recorded.
    Returns the recipe (None when nothing is recorded, e.g. the GUI or a plain library call)."""
    recipe = build_recipe(run_dir, result, title)
    if recipe is None:
        return None
    remember_run_dir(run_dir)
    path = Path(run_dir).resolve() / RECIPE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return recipe


def load_recipe(path: str | Path) -> tuple[dict[str, Any], Path]:
    """(recipe, recipe.json path) from a run folder or a recipe.json."""
    p = Path(path)
    if p.is_dir():
        p = p / RECIPE_NAME
    if not p.is_file():
        raise FileNotFoundError(f"{RECIPE_NAME} がありません: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not str(data.get("schema", "")).startswith("crystal-mcp/recipe/"):
        raise ValueError(f"Shido のレシピではありません: {p}")
    return data, p.resolve()


# ---------------------------------------------------------------------------
# Rerun
# ---------------------------------------------------------------------------

def parse_overrides(pairs: list[str]) -> dict[str, str]:
    """["theme=dark", "render.dpi=600"] -> {"theme": "dark", "render.dpi": "600"}."""
    out: dict[str, str] = {}
    for item in pairs:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set は key=value の形で書きます: {item!r}")
        out[key.strip()] = value.strip()
    return out


def _typed(value: Any) -> Any:
    """'600' -> 600, 'true' -> True, '[1, 0, 0]' -> [1, 0, 0]; other text stays text."""
    if not isinstance(value, str):
        return value
    import yaml

    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return str(value).replace(",", " ").split()


def _apply_cli_override(argv: list[str], key: str, value: Any, recipe: dict[str, Any]) -> list[str]:
    """Replace / add / remove ``--key`` in argv. ``input`` replaces the recorded input file."""
    if key == "input":
        old = {e["path"] for e in recipe.get("inputs", [])}
        for i, a in enumerate(argv):
            if not a.startswith("-") and str(Path(a).resolve()) in old:
                return [*argv[:i], str(value), *argv[i + 1 :]]
        raise ValueError("このレシピには置きかえる入力ファイルがありません")
    flags = {"--" + key.replace("_", "-"), "--" + key}
    typed = _typed(value)
    out: list[str] = []
    i, found = 0, False
    while i < len(argv):
        a = argv[i]
        name, eq, _ = a.partition("=")
        if name in flags:
            found = True
            if eq:  # --key=value
                i += 1
            else:
                i += 1
                while i < len(argv) and not (argv[i].startswith("--") or (argv[i].startswith("-") and argv[i][1:2].isalpha())):
                    i += 1
            if typed is True:
                out.append(name)
            elif typed is not False and typed is not None:
                out.extend([name, *_tokens(value)])
            continue
        out.append(a)
        i += 1
    if not found:
        flag = "--" + key.replace("_", "-")
        if typed is True:
            out.append(flag)
        elif typed is not False and typed is not None:
            out.extend([flag, *_tokens(value)])
    return out


def _strip_output_flags(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        name = a.partition("=")[0]
        if name in OUTPUT_FLAGS:
            skip = "=" not in a
            continue
        out.append(a)
    return out


def check_recipe(recipe: dict[str, Any]) -> list[str]:
    """Warnings for a rerun: changed / missing inputs, other tool or library versions."""
    warns: list[str] = []
    for e in recipe.get("inputs", []):
        p = Path(e["path"])
        if not p.exists():
            warns.append(f"入力が見つかりません: {p}")
        elif e.get("kind") == "file" and e.get("sha256") and sha256(p) != e["sha256"]:
            warns.append(f"入力ファイルの中身がレシピの時と違います（SHA-256 が不一致）: {p}")
        elif e.get("kind") == "folder" and "files" in e:
            now_files = {f["path"]: f["sha256"] for f in _folder_files(p)}
            old_files = {f["path"]: f["sha256"] for f in e["files"]}
            changed = sorted(k for k in old_files if k in now_files and now_files[k] != old_files[k])
            gone = sorted(k for k in old_files if k not in now_files)
            added = sorted(k for k in now_files if k not in old_files)
            for label, names in (("中身が違う（SHA-256 が不一致）", changed), ("無くなった", gone), ("増えた", added)):
                if names:
                    more = f" ほか {len(names) - 5} 個" if len(names) > 5 else ""
                    warns.append(f"入力フォルダ {p} のファイルが{label}: {', '.join(names[:5])}{more}")
    now = environment()
    old_tool, new_tool = recipe.get("tool") or {}, now["tool"]
    if old_tool.get("version") != new_tool["version"]:
        warns.append(f"{TOOL_NAME} の版が違います: レシピ {old_tool.get('version')} / 今 {new_tool['version']}")
    old_git, new_git = (old_tool.get("git") or {}).get("commit"), (new_tool.get("git") or {}).get("commit")
    if old_git and new_git and old_git != new_git:
        warns.append(f"{TOOL_NAME} のコミットが違います: レシピ {old_git} / 今 {new_git}")
    if recipe.get("python") and recipe["python"] != now["python"]:
        warns.append(f"Python の版が違います: レシピ {recipe['python']} / 今 {now['python']}")
    for lib, ver in (recipe.get("libraries") or {}).items():
        cur = now["libraries"].get(lib) if lib in now["libraries"] else _safe_version(lib)
        if ver != cur:
            warns.append(f"{lib} の版が違います: レシピ {ver} / 今 {cur}")
    return warns


def _safe_version(name: str) -> str | None:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _set_dotted(tree: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def rerun(source: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run a recipe again into a new folder; returns the run's result plus ``rerun``
    (warnings, the overrides used, the new recipe path)."""
    from shido import settings as cfg
    from shido.theme_manager import get_active_theme_name, set_active_theme

    recipe, recipe_path = load_recipe(source)
    overrides = dict(overrides or {})
    warns = check_recipe(recipe)
    snap = recipe.get("settings") or {}
    settings_now = copy.deepcopy(snap.get("effective") or cfg.load_settings())
    call = recipe["call"]
    theme = snap.get("theme")

    for key, value in overrides.items():
        if "." in key:  # a setting such as render.dpi
            typed = _typed(value)
            cfg.validate_value(key, typed)
            _set_dotted(settings_now, key, typed)
            if key == "render.theme":
                theme = typed

    if call["kind"] == "cli":
        argv = _strip_output_flags(list(call["argv"]))
        for key, value in overrides.items():
            if "." not in key:
                argv = _apply_cli_override(argv, key, value, recipe)
                if key == "theme":
                    theme = _typed(value)
    elif call["kind"] == "mcp":
        arguments = dict(call["arguments"])
        for key in OUTPUT_ARGS & arguments.keys():
            arguments[key] = None
        for key, value in overrides.items():
            if "." in key:
                continue
            if key not in arguments:
                raise ValueError(f"ツール {call['tool']} に引数 {key!r} はありません（使える: {', '.join(arguments)}）")
            arguments[key] = _typed(value)
            if key == "theme":
                theme = arguments[key]
    else:
        raise ValueError(f"知らない呼び方です: {call.get('kind')}")

    previous_theme = get_active_theme_name()
    ctx_holder: dict[str, Any] = {}
    try:
        with cfg.overridden(settings_now):
            if theme:
                try:
                    set_active_theme(str(theme))
                except Exception as exc:  # noqa: BLE001 - an unknown theme is only a warning
                    warns.append(f"テーマ {theme!r} を使えませんでした: {exc}")
            if call["kind"] == "cli":
                from shido import server

                with recording_cli(argv) as ctx:
                    ctx["rerun_of"] = str(recipe_path)
                    ctx_holder = ctx
                    code = server.main(argv)
                result: dict[str, Any] = {"exit_code": code, "argv": argv}
            else:
                if call["tool"] not in TOOLS:
                    from shido import server

                    server.create_mcp_server()  # fills TOOLS
                fn = TOOLS[call["tool"]]
                with _recording({"kind": "mcp", "tool": call["tool"], "arguments": _jsonable(arguments)}) as ctx:
                    ctx["rerun_of"] = str(recipe_path)
                    ctx_holder = ctx
                    out = fn(**arguments)
                result = dict(out) if isinstance(out, Mapping) else {"result": out}
    finally:
        set_active_theme(previous_theme)
    new_dirs = ctx_holder.get("run_dirs", [])
    result["rerun"] = {
        "recipe": str(recipe_path),
        "overrides": _jsonable(overrides),
        "warnings": warns,
        "output_dir": new_dirs[-1] if new_dirs else result.get("output_dir"),
        "new_recipe": str(Path(new_dirs[-1]) / RECIPE_NAME) if new_dirs else None,
    }
    return result


def remember_run_dir(run_dir: str | Path) -> None:
    ctx = _CURRENT.get()
    if ctx is not None:
        dirs = ctx.setdefault("run_dirs", [])
        s = str(Path(run_dir).resolve())
        if s not in dirs:
            dirs.append(s)


# ---------------------------------------------------------------------------
# はじめに読む.html: 「この図の作り方（レシピ）」
# ---------------------------------------------------------------------------

def rerun_command(run_dir: str | Path) -> str:
    return f'python server.py rerun "{Path(run_dir).resolve()}"'


def methods_sentences(recipe: Mapping[str, Any]) -> tuple[str, str]:
    """(Japanese, English) draft for a Methods section. Every value comes from the recipe."""
    tool = recipe.get("tool") or {}
    libs = recipe.get("libraries") or {}
    git_info = tool.get("git") or {}
    git = git_info.get("commit")
    if git and git_info.get("dirty"):
        git += "-dirty"  # uncommitted changes: say so
    ver = tool.get("version", "?")
    ver_txt = f"{ver}（git {git}）" if git else f"{ver}"
    ver_en = f"{ver} (git {git})" if git else f"{ver}"
    snap = recipe.get("settings") or {}
    eff = snap.get("effective") or {}
    render = eff.get("render") or {}
    theme = snap.get("theme")
    args = (recipe.get("call") or {}).get("arguments") or (recipe.get("call") or {}).get("resolved_options") or {}
    style = args.get("style")

    jp_lib = []
    en_lib = []
    if libs.get("pymatgen"):
        jp_lib.append(f"pymatgen {libs['pymatgen']}")
        en_lib.append(f"pymatgen {libs['pymatgen']}")
    if libs.get("spglib"):
        jp_lib.append(f"spglib {libs['spglib']}")
        en_lib.append(f"spglib {libs['spglib']}")
    draw = []
    if libs.get("pyvista"):
        draw.append(f"PyVista {libs['pyvista']}")
    if libs.get("vtk"):
        draw.append(f"VTK {libs['vtk']}")

    cond_jp, cond_en = [], []
    if style:
        cond_jp.append(f"描き方 {style}")
        cond_en.append(f"{style} style")
    if theme:
        cond_jp.append(f"配色テーマ {theme}")
        cond_en.append(f"colour theme {theme}")
    if render.get("dpi"):
        cond_jp.append(f"{render['dpi']} dpi")
        cond_en.append(f"{render['dpi']} dpi")
    if (eff.get("bonds") or {}).get("max_bond_length") is not None:
        cond_jp.append(f"結合の最大長 {eff['bonds']['max_bond_length']} Å")
        cond_en.append(f"maximum bond length {eff['bonds']['max_bond_length']} Å")

    jp = f"結晶構造の解析と図の作成には {TOOL_NAME} {ver_txt}を用いた"
    en = f"Crystal structures were analysed and drawn with {TOOL_NAME} {ver_en}"
    if jp_lib:
        jp += f"（構造の処理に {'・'.join(jp_lib)}"
        en += f" (structure handling: {', '.join(en_lib)}"
        if draw:
            jp += f"、描画に {'・'.join(draw)}"
            en += f"; rendering: {', '.join(draw)}"
        jp += "）"
        en += ")"
    jp += "。"
    en += "."
    if cond_jp:
        jp += f"図の主な条件は {'、'.join(cond_jp)} である。"
        en += f" Figures used {', '.join(cond_en)}."
    return jp, en


def index_section(recipe: Mapping[str, Any], run_dir: str | Path) -> str:
    """HTML for the 「この図の作り方（レシピ）」 section of はじめに読む.html."""
    run = Path(run_dir).resolve()
    jp, en = methods_sentences(recipe)
    call = recipe.get("call") or {}
    if call.get("kind") == "mcp":
        how = f"MCP ツール <code>{html.escape(str(call.get('tool')))}</code> で作りました。"
    else:
        how = "コマンド <code>python server.py " + html.escape(" ".join(call.get("argv") or [])) + "</code> で作りました。"
    n_in = len(recipe.get("inputs") or [])
    n_out = len(recipe.get("outputs") or [])
    warn = (recipe.get("validation") or {}).get("warnings") or []
    rows = [
        "<h2>この図の作り方（レシピ）</h2><div class='box'>",
        f"<p>{how} 使ったツールと部品の版、設定のすべて、入力 {n_in} 個と結果 {n_out} 個のファイルの"
        f"指紋（SHA-256）を <a href=\"{RECIPE_NAME}\">{RECIPE_NAME}</a> に残しました。</p>",
        "<p><b>同じ図をもう一度作る</b>（新しいフォルダにできます）：</p>",
        f"<p><code>{html.escape(rerun_command(run))}</code></p>",
        "<p class='small'>一部だけ変えるときは <code>--set</code> を付けます。例："
        "<code>--set render.dpi=600</code>（点の入った名前は前提数値）、"
        "<code>--set style=space-filling</code>（点の無い名前はコマンドの設定）。"
        "AI には「この recipe.json を rerun_recipe でもう一度作って」と頼めます。"
        "入力ファイルや部品の版がレシピの時と違うときは、止めずに警告を出します。</p>",
        f"<p class='small'>検算（格子・原子の重なり・組成など）：通過。警告 {len(warn)} 件"
        + ("（" + html.escape(" / ".join(warn[:5])) + ("…" if len(warn) > 5 else "") + "）" if warn else "")
        + "</p>",
        "<p><b>論文の方法の節に書ける文（下書き）</b>：</p>",
        f"<p>{html.escape(jp)}</p><p>{html.escape(en)}</p>",
        "<p class='small'>版と設定の値は recipe.json から入れています。書き足すときも値は recipe.json を見てください。</p>",
        "</div>",
    ]
    return "".join(rows)


__all__ = [
    "RECIPE_NAME",
    "TOOLS",
    "build_recipe",
    "check_recipe",
    "environment",
    "index_section",
    "load_recipe",
    "methods_sentences",
    "note_result",
    "parse_overrides",
    "recorded_tool",
    "recording_cli",
    "recording_tool_decorator",
    "remember_run_dir",
    "rerun",
    "rerun_command",
    "sha256",
    "write_recipe",
]

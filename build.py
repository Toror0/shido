"""配布用バイナリのビルド前に必ずテストを通すビルドゲート。"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# exe（GUI）に丸ごと入れる部品
COLLECT_ALL = ("pymatgen", "pyvista", "vtkmodules", "spglib", "monty", "matplotlib", "scipy", "pandas")
# exe に入れない部品（GUI は MCP サーバーを使わない）
EXCLUDES = (
    "mcp", "typer", "pyvista.trame", "trame",
    "pytest", "IPython", "jupyter", "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6",
)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        sep = argv.index("--")
        own_argv = argv[:sep]
        pytest_extra = argv[sep + 1 :]
    else:
        own_argv = argv
        pytest_extra = []

    parser = argparse.ArgumentParser(
        description="テストを通してから配布用バイナリをビルドします。"
    )
    parser.add_argument(
        "--skip-binary",
        action="store_true",
        help="テストだけ実行し、バイナリのビルドは行いません。",
    )
    return parser.parse_args(own_argv), pytest_extra


def main(argv: list[str] | None = None) -> int:
    args, pytest_extra = parse_args(sys.argv[1:] if argv is None else argv)

    pytest_cmd = [sys.executable, "-m", "pytest", "-v", "tests", *pytest_extra]
    print("テストを実行します...", flush=True)
    test_result = subprocess.run(pytest_cmd, cwd=ROOT_DIR)
    if test_result.returncode != 0:
        print("テストが失敗したのでビルドを中止しました")
        return test_result.returncode

    if args.skip_binary:
        print("テストが通りました（--skip-binary のためビルドは省略しました）")
        return 0

    if importlib.util.find_spec("PyInstaller") is None:
        print(
            "PyInstaller が入っていません: "
            r".venv\Scripts\pip install -r requirements-dev.txt"
        )
        return 2

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        "Shido",
        "--windowed",
        "--add-data",
        "themes.yaml;.",
        "--add-data",
        "settings_defaults.yaml;.",
        "--add-data",
        "journals;journals",
    ]
    # 部品のデータ（pymatgen の periodic_table.json.gz など）・DLL・中で読むモジュールを全部入れる
    for pkg in COLLECT_ALL:
        pyinstaller_cmd += ["--collect-all", pkg]
    # GUI では使わない部分（MCP・pyvista の jupyter/trame など。読み込むと止まる）は入れない
    for mod in EXCLUDES:
        pyinstaller_cmd += ["--exclude-module", mod]
    pyinstaller_cmd.append("gui.py")
    print("テストが通りました。バイナリをビルドします...", flush=True)
    build_result = subprocess.run(pyinstaller_cmd, cwd=ROOT_DIR)
    return build_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

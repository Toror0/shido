@echo off
rem ===========================================================
rem  crystal-mcp の準備（Windows 用）
rem  - このフォルダに .venv（専用の Python の箱）を作り、部品を入れる
rem  - 何回実行しても大丈夫（あるものはそのまま使い、足りない部品だけ入れる）
rem  - ダブルクリックで実行できる。最後に止まるので、読んだら閉じる
rem  - 止まらずに終わらせたいときは「setup_windows.bat /nopause」
rem ===========================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VPY=.venv\Scripts\python.exe"
set "CHECK=import sys, sysconfig; ok = (3, 11) <= sys.version_info[:2] <= (3, 13) and sysconfig.get_platform().startswith('win') and 'msys' not in sys.executable.lower(); sys.exit(0 if ok else 1)"

echo.
echo [1/4] Python の箱（.venv）を確かめます
if exist "%VPY%" goto :have_venv

where py >nul 2>nul
if errorlevel 1 (
  echo   「py」コマンドが見つかりません。
  echo   python.org から Windows 用の Python 3.13 を入れてから、もう一度実行してください。
  goto :fail
)

set "PYVER="
for %%v in (3.13 3.12 3.11) do (
  if not defined PYVER (
    py -%%v -c "%CHECK%" >nul 2>nul && set "PYVER=%%v"
  )
)
if not defined PYVER (
  echo   Windows 用の Python 3.11〜3.13 が見つかりません。
  echo   python.org から Python 3.13 を入れてから、もう一度実行してください。
  goto :fail
)
echo   Python %PYVER% で .venv を作ります（少し待ちます）
py -%PYVER% -m venv .venv
if errorlevel 1 (
  echo   .venv を作れませんでした。
  goto :fail
)
goto :venv_ok

:have_venv
"%VPY%" -c "%CHECK%" >nul 2>nul
if errorlevel 1 (
  echo   今ある .venv の Python が 3.11〜3.13 の Windows 版ではありません。
  "%VPY%" -V
  echo   .venv フォルダを消してから、もう一度実行してください。
  goto :fail
)
echo   今ある .venv をそのまま使います

:venv_ok
"%VPY%" -V

echo.
echo [2/4] pip（部品を入れる道具）を新しくします
"%VPY%" -m pip install --upgrade pip --disable-pip-version-check -q
if errorlevel 1 goto :pipfail

echo.
echo [3/4] 部品を入れます（初回は数分かかります）
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto :pipfail

echo.
echo [4/4] 部品が読み込めるか確かめます
"%VPY%" -c "import pymatgen, pyvista, vtk, numpy, pandas, matplotlib, PIL, yaml, mcp, tkinter; print('  OK: すべて読み込めました')"
if errorlevel 1 (
  echo   読み込めない部品があります。上のエラーを見てください。
  goto :fail
)

echo.
echo ===========================================================
echo  準備ができました。
echo  画面（GUI）を開く : launch_gui.vbs をダブルクリック
echo  コマンドで試す   : .venv\Scripts\python.exe server.py demo
echo  Claude Code に登録 : README.md の「Claude Code に登録する」を見る
echo ===========================================================
if /i not "%~1"=="/nopause" pause
exit /b 0

:pipfail
echo   部品を入れられませんでした。ネットにつながっているか確かめて、もう一度実行してください。

:fail
echo.
echo  準備は終わっていません。
if /i not "%~1"=="/nopause" pause
exit /b 1

"""Batch 27: GUI options (_bulk_options / build_bulk_options) match CLI and keep defaults unchanged."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from shido import server
from shido.gui import CrystalAnalyzerGUI, build_bulk_options


def _cli_opts(argv: list[str]) -> dict:
    parser = server._build_arg_parser()
    return server._bulk_options_from_args(parser.parse_args(argv))


def _nacl() -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def test_default_bulk_options_match_cli() -> None:
    gui_opts = build_bulk_options()
    cli_opts = _cli_opts(["demo"])
    assert set(gui_opts) <= set(cli_opts)
    for key, val in gui_opts.items():
        assert cli_opts[key] == val, f"mismatch on default key {key!r}: {val!r} != {cli_opts[key]!r}"
    # Conditional keys must be absent under defaults so outputs stay unchanged.
    for absent in (
        "adp",
        "probability",
        "volume",
        "isovalue",
        "iso_color",
        "iso_color_negative",
        "iso_opacity",
        "iso_both_signs",
        "animate",
        "animate_frames",
        "animate_axis",
        "supercell",
        "export_formats",
    ):
        assert absent not in gui_opts
        assert absent not in cli_opts
    assert server._shown_options(gui_opts) == {}


def test_all_gui_options_match_cli_dict() -> None:
    gui_opts = build_bulk_options(
        xrd=False,  # wavelength / neutron implies xrd=True just like CLI
        stl=True,
        html=True,
        poscar_cell="規約格子",
        style="熱振動楕円体 (ellipsoid)",
        probability="90",
        view_dir=("1", "1", "0"),
        perspective=True,
        labels="サイト名 (site)",
        frac_range=("-0.5", "1.5", "0", "1", "0", "1"),
        supercell=("2", "2", "1"),
        image_formats=["tiff", "pdf", "svg"],
        export_formats=["xyz", "p1", "xtl", "pdb", "wrl", "vesta"],
        wavelength="1.5406",
        radiation="中性子 (neutron)",
        volume="CHGCAR",
        isovalue="0.05,0.1",
        iso_both_signs=True,
        animate="GIF",
        animate_frames="36",
        animate_axis="c",
    )
    cli_opts = _cli_opts([
        "demo",
        "--stl",
        "--html",
        "--poscar-cell", "conventional",
        "--style", "ellipsoid",
        "--probability", "90",
        "--view-dir", "1", "1", "0",
        "--perspective",
        "--labels", "site",
        "--range", "-0.5", "1.5", "0", "1", "0", "1",
        "--supercell", "2", "2", "1",
        "--image-format", "tiff,pdf,svg",
        "--export", "xyz,p1,xtl,pdb,wrl,vesta",
        "--wavelength", "1.5406",
        "--radiation", "neutron",
        "--volume", "CHGCAR",
        "--isovalue", "0.05,0.1",
        "--iso-both-signs",
        "--animate", "gif",
        "--frames", "36",
        "--animate-axis", "c",
    ])
    assert set(gui_opts) <= set(cli_opts)
    for key, val in gui_opts.items():
        assert cli_opts[key] == val, f"mismatch on key {key!r}: {val!r} != {cli_opts[key]!r}"
    assert gui_opts["xrd"] is True
    assert gui_opts["wavelength"] == pytest.approx(1.5406)
    assert gui_opts["xrd_options"] == {"radiation": "neutron"}
    assert gui_opts["supercell"] == [2, 2, 1]
    assert gui_opts["export_formats"] == "xyz,p1,xtl,pdb,wrl,vesta"


@pytest.mark.parametrize("style", ["ball-stick", "space-filling", "stick", "wireframe", "ellipsoid"])
def test_styles_match_cli(style: str) -> None:
    gui_opts = build_bulk_options(style=style, probability="50")
    cli_opts = _cli_opts(["demo", "--style", style, "--probability", "50"])
    for key, val in gui_opts.items():
        assert cli_opts[key] == val, (style, key)
    if style == "ellipsoid":
        assert gui_opts["adp"] is None and gui_opts["probability"] == 50.0
    else:
        assert "adp" not in gui_opts and "probability" not in gui_opts


def test_noop_inputs_stay_default() -> None:
    # PNG alone, 1 1 1 supercell, stray isovalue without volume, frames without animate -> all no-ops
    gui_opts = build_bulk_options(
        image_formats=["png"],
        supercell=("1", "1", "1"),
        isovalue="0.05",
        animate="なし",
        animate_frames="24",
    )
    assert gui_opts == build_bulk_options()

    # Source key wavelength (e.g. MoKa) stays a string and enables XRD like CLI
    wl_opts = build_bulk_options(wavelength="MoKa")
    cli_wl = _cli_opts(["demo", "--wavelength", "MoKa"])
    assert wl_opts["wavelength"] == cli_wl["wavelength"] == "MoKa"
    assert wl_opts["xrd"] is True and cli_wl["xrd"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"view_dir": ("1", "0", "")},
        {"view_dir": ("0", "0", "0")},
        {"frac_range": ("0", "1", "0", "1", "0", "")},
        {"frac_range": ("1", "0", "0", "1", "0", "1")},
        {"supercell": ("2", "2", "")},
        {"supercell": ("2", "0", "1")},
        {"style": "ellipsoid", "probability": "100"},
        {"style": "ellipsoid", "probability": "0"},
        {"animate": "gif", "animate_frames": "1"},
    ],
)
def test_invalid_gui_inputs_raise_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        build_bulk_options(**kwargs)


def test_default_gui_options_do_not_change_bulk_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_render(structure, output_dir, **kw):
        calls.append({"n_sites": len(structure), **kw})
        return {}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    s = _nacl()
    plain_dir = tmp_path / "plain"
    gui_dir = tmp_path / "gui"
    res_plain = server.run_bulk_analysis(s, plain_dir)
    res_gui = server.run_bulk_analysis(s, gui_dir, **build_bulk_options())

    plain_files = sorted(str(p.relative_to(plain_dir)) for p in plain_dir.rglob("*") if p.is_file())
    gui_files = sorted(str(p.relative_to(gui_dir)) for p in gui_dir.rglob("*") if p.is_file())
    assert gui_files == plain_files
    assert set(res_gui) == set(res_plain)
    assert res_gui["xrd_figure_png"] is None and res_gui["stl_file"] is None and res_gui["html_file"] is None
    assert "exported_files" not in res_gui and "animation" not in res_gui and "supercell" not in res_gui

    # Only difference in render kwargs is style: None vs "ball-stick", which normalize_style treats identically.
    c_plain, c_gui = calls
    assert c_plain.pop("style") is None and c_gui.pop("style") == "ball-stick"
    assert c_plain.pop("summary") == c_gui.pop("summary")
    assert len(c_plain.pop("render_bonds")) == len(c_gui.pop("render_bonds"))
    assert len(c_plain.pop("polyhedra_envs")) == len(c_gui.pop("polyhedra_envs"))
    assert c_gui == c_plain


def test_run_bulk_analysis_supercell_and_export_formats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_render(structure, output_dir, **kw):
        seen["n_sites"] = len(structure)
        return {}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    res = server.run_bulk_analysis(_nacl(), tmp_path, supercell=[2, 1, 1], export_formats="xyz,p1,vesta")
    assert seen["n_sites"] == 16
    assert res["supercell"] == [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    exported = res["exported_files"]
    assert set(exported) == {"xyz", "p1", "vesta"}
    for path in exported.values():
        assert Path(path).is_file()


def test_gui_widget_bulk_options_and_collapsible() -> None:
    try:
        app = CrystalAnalyzerGUI()
    except tk.TclError as exc:
        pytest.skip(f"tkinter display unavailable: {exc}")
    try:
        app.withdraw()
        # Collapsible sections start closed and toggle cleanly.
        assert app.appearance_adv.is_expanded is False
        assert app.extra_adv.is_expanded is False
        app.appearance_adv.toggle()
        assert app.appearance_adv.is_expanded is True
        app.appearance_adv.toggle()
        assert app.appearance_adv.is_expanded is False

        # Default widget values match build_bulk_options() and CLI defaults.
        default_opts = app._bulk_options()
        cli_default = _cli_opts(["demo"])
        assert default_opts == build_bulk_options()
        for key, val in default_opts.items():
            assert cli_default[key] == val, key

        # Populated widget values match CLI options with the same flags.
        app.style_var.set("熱振動楕円体 (ellipsoid)")
        app._sync_style_ui()
        app.probability_var.set("75")
        app.view_u_var.set("1")
        app.view_v_var.set("1")
        app.view_w_var.set("0")
        app.perspective_var.set(True)
        app.labels_var.set("元素記号 (element)")
        app.range_x0_var.set("-0.5")
        app.range_x1_var.set("1.5")
        app.range_y0_var.set("0")
        app.range_y1_var.set("1")
        app.range_z0_var.set("0")
        app.range_z1_var.set("1")
        app.sc_a_var.set("2")
        app.sc_b_var.set("2")
        app.sc_c_var.set("1")
        app.image_format_vars["tiff"].set(True)
        app.image_format_vars["jpeg"].set(True)
        app.export_format_vars["xyz"].set(True)
        app.export_format_vars["vesta"].set(True)
        app.xrd_wavelength_var.set("MoKa")
        app.xrd_radiation_var.set("中性子 (neutron)")
        app.volume_var.set("CHGCAR")
        app.isovalue_var.set("0.05")
        app.iso_both_signs_var.set(True)
        app.animate_var.set("GIF")
        app._sync_animate_ui()
        app.animate_frames_var.set("48")
        app.animate_axis_var.set("a")

        widget_opts = app._bulk_options()
        cli_full = _cli_opts([
            "demo",
            "--style", "ellipsoid",
            "--probability", "75",
            "--view-dir", "1", "1", "0",
            "--perspective",
            "--labels", "element",
            "--range", "-0.5", "1.5", "0", "1", "0", "1",
            "--supercell", "2", "2", "1",
            "--image-format", "tiff,jpeg",
            "--export", "xyz,vesta",
            "--wavelength", "MoKa",
            "--radiation", "neutron",
            "--volume", "CHGCAR",
            "--isovalue", "0.05",
            "--iso-both-signs",
            "--animate", "gif",
            "--frames", "48",
            "--animate-axis", "a",
        ])
        for key, val in widget_opts.items():
            assert cli_full[key] == val, f"widget mismatch on {key!r}: {val!r} != {cli_full[key]!r}"
    finally:
        app.destroy()

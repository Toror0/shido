"""Coordination polyhedra default: auto (only with an anion-forming element); explicit wins."""

from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from shido import server
from shido.renderer import resolve_show_polyhedra, structure_has_anion


def _mn2coga() -> Structure:
    """Ordered inverse Heusler XA Mn2CoGa (F-43m), an intermetallic."""
    return Structure.from_spacegroup(
        "F-43m", Lattice.cubic(5.80), ["Mn", "Mn", "Co", "Ga"],
        [[0, 0, 0], [0.75, 0.75, 0.75], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )


def _srtio3() -> Structure:
    return Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    )


def test_has_anion() -> None:
    assert not structure_has_anion(_mn2coga())
    assert structure_has_anion(_srtio3())
    nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert structure_has_anion(nacl)
    # partial occupancy counts too
    mixed = Structure(Lattice.cubic(3.0), [{"Fe": 0.9, "N": 0.1}], [[0, 0, 0]])
    assert structure_has_anion(mixed)


@pytest.mark.parametrize(
    ("show", "maker", "center", "expected"),
    [
        (None, _mn2coga, None, False),   # metal: auto -> no polyhedra
        (None, _srtio3, None, True),     # oxide: auto -> polyhedra as before
        (True, _mn2coga, None, True),    # explicit wins
        (False, _srtio3, None, False),   # explicit wins
        (None, _mn2coga, ["Mn"], True),  # choosing a centre asks for polyhedra
    ],
)
def test_resolve(show, maker, center, expected) -> None:
    assert resolve_show_polyhedra(show, maker(), center) is expected


@pytest.fixture()
def seen(monkeypatch: pytest.MonkeyPatch) -> dict:
    got: dict = {}

    def fake_render(structure, output_dir, **kw):
        got.update(kw)
        return {"figure_overview": None}

    monkeypatch.setattr(server, "render_bulk_views", fake_render)
    return got


def test_run_bulk_analysis_default_is_auto(tmp_path: Path, seen: dict) -> None:
    server.run_bulk_analysis(_mn2coga(), tmp_path / "metal")
    assert seen["show_polyhedra"] is False
    seen.clear()
    server.run_bulk_analysis(_srtio3(), tmp_path / "oxide")
    assert seen["show_polyhedra"] is True
    seen.clear()
    server.run_bulk_analysis(_mn2coga(), tmp_path / "forced", show_polyhedra=True)
    assert seen["show_polyhedra"] is True
    seen.clear()
    server.run_bulk_analysis(_srtio3(), tmp_path / "off", show_polyhedra=False)
    assert seen["show_polyhedra"] is False


@pytest.mark.parametrize(("flags", "expected"), [([], False), (["--polyhedra"], True), (["--no-polyhedra"], False)])
def test_cli_flags_metal(tmp_path: Path, seen: dict, flags: list[str], expected: bool) -> None:
    cif = tmp_path / "Mn2CoGa.cif"
    CifWriter(_mn2coga()).write_file(str(cif))
    assert server.main([str(cif), "--out-dir", str(tmp_path / "out"), *flags]) == 0
    assert seen["show_polyhedra"] is expected


@pytest.mark.parametrize(("flags", "expected"), [([], True), (["--no-polyhedra"], False)])
def test_cli_flags_oxide(tmp_path: Path, seen: dict, flags: list[str], expected: bool) -> None:
    cif = tmp_path / "SrTiO3.cif"
    CifWriter(_srtio3()).write_file(str(cif))
    assert server.main([str(cif), "--out-dir", str(tmp_path / "out"), *flags]) == 0
    assert seen["show_polyhedra"] is expected


def test_render_bulk_views_default_is_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from shido import renderer

    calls: list[bool] = []

    def fake_single(*args, **kw):
        calls.append(kw.get("show_polyhedra"))
        raise RuntimeError("stop")  # only the resolved flag matters here

    monkeypatch.setattr(renderer, "render_single_view", fake_single)
    for maker, expected in ((_mn2coga, False), (_srtio3, True)):
        calls.clear()
        with pytest.raises(RuntimeError, match="stop"):
            renderer.render_bulk_views(maker(), tmp_path, render_bonds=[], polyhedra_envs=[])
        assert calls and calls[0] is expected

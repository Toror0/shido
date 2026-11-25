"""Integration tests on real-world CIF files from the Crystallography Open Database (COD)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from shido import analyzer
from shido import renderer
from conftest import SMALL_WINDOW, png_background_ratio

COD_DIR = Path(__file__).resolve().parent / "data" / "cod"

COD_CASES = [
    ("1000030", "partial occupancy + uncertainties (YBCO)"),
    ("1000068", "mixed site Nb/W + uncertainties (LiNbWO6)"),
    ("1530416", "split H site 0.5/0.5 at 0.358 A (KDP)"),
    ("1008785", "hydrogen-bearing molecular crystal + uncertainties (Urea)"),
    ("1100710", "triclinic P-1 + hydrogens + uncertainties (Pt/Au complex)"),
    ("9001735", "triclinic P-1 + split Bi 0.5/0.5 + mixed Cu/Fe (Medenbachite)"),
]

_CELL_VALUE_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\(\d+\))?$")


def _parse_cif_cell_lengths(cif_path: Path) -> tuple[float, float, float]:
    """Read _cell_length_{a,b,c} directly from CIF text, stripping any '(esd)' uncertainty."""
    text = cif_path.read_text(encoding="utf-8", errors="replace")
    values: list[float] = []
    for tag in ("_cell_length_a", "_cell_length_b", "_cell_length_c"):
        match = re.search(rf"^{tag}\s+(\S+)", text, flags=re.MULTILINE)
        assert match is not None, f"Missing {tag} in {cif_path.name}"
        num_match = _CELL_VALUE_RE.match(match.group(1).strip())
        assert num_match is not None, f"Unparseable {tag} value {match.group(1)!r} in {cif_path.name}"
        values.append(float(num_match.group(1)))
    return values[0], values[1], values[2]


@pytest.mark.parametrize(
    "cod_id,desc",
    COD_CASES,
    ids=[cid for cid, _ in COD_CASES],
)
def test_cod_cif_load_and_analyze(cod_id: str, desc: str, out_dir: Path):
    cif_path = COD_DIR / f"{cod_id}.cif"
    assert cif_path.exists(), f"Missing fixture: {cif_path}"
    assert cif_path.stat().st_size <= 200_000

    cif_a, cif_b, cif_c = _parse_cif_cell_lengths(cif_path)

    structure, parse_warnings = analyzer.load_structure_robust(cif_path)
    result = analyzer.analyze_structure_data(structure, out_dir, parse_warnings=parse_warnings)

    summary = result["summary"]
    assert summary["a_angstrom"] == pytest.approx(cif_a, abs=1e-3), f"{cod_id} ({desc}) a mismatch"
    assert summary["b_angstrom"] == pytest.approx(cif_b, abs=1e-3), f"{cod_id} ({desc}) b mismatch"
    assert summary["c_angstrom"] == pytest.approx(cif_c, abs=1e-3), f"{cod_id} ({desc}) c mismatch"
    assert Path(result["lattice_summary_csv"]).exists()
    assert Path(result["bond_lengths_csv"]).exists()


def test_cod_cif_single_view_render(out_dir: Path):
    """Verify off-screen rendering on one real COD CIF (partial-occupancy YBCO 1000030)."""
    cif_path = COD_DIR / "1000030.cif"
    structure, _ = analyzer.load_structure_robust(cif_path)
    _, render_bonds, _ = analyzer.compute_bonds(structure)

    a_vec = structure.lattice.matrix[0]
    c_vec = structure.lattice.matrix[2]
    out_png = out_dir / "cod_1000030_a.png"

    rendered = renderer.render_single_view(
        structure=structure,
        output_path=out_png,
        camera_dir=a_vec,
        view_up=c_vec,
        render_bonds=render_bonds,
        window_size=SMALL_WINDOW,
    )
    assert rendered.exists() and rendered.stat().st_size > 0
    assert png_background_ratio(rendered) < 0.995

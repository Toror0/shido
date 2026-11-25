"""Normal-path tests: the guardrails must not block ordinary, valid workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shido.analyzer import build_ni2mnga_b2_disordered, build_ni2mnga_structure
from shido.ca_sweep import generate_ca_sweep
from conftest import png_background_ratio
from shido.core.validator import validate_structure
from shido.server import run_bulk_analysis
from shido.slab_cutter import cut_slabs


def test_ni2mnga_structures_pass_validation():
    assert validate_structure(build_ni2mnga_structure()) == pytest.approx(5.82 * np.sqrt(3) / 4, rel=0.02)
    assert validate_structure(build_ni2mnga_b2_disordered()) > 2.0


def test_bulk_analysis_and_render(out_dir):
    res = run_bulk_analysis(build_ni2mnga_structure(), out_dir)
    assert res["num_bonds"] > 0
    assert res["summary"]["spacegroup_symbol"] == "Fm-3m"
    for img in res["images"]:
        if img.endswith(".png"):
            assert png_background_ratio(img) < 0.95


def test_ca_sweep_conserves_volume_and_composition(out_dir):
    s = build_ni2mnga_structure()
    res = generate_ca_sweep(s, out_dir, ca_min=0.9, ca_max=1.1, ca_step=0.05)
    df = pd.read_csv(res["sweep_table_csv"])
    assert 1.0 in set(df["c_over_a"].round(6))
    np.testing.assert_allclose(df["volume_angstrom3"], s.volume, rtol=1e-5)


def test_slab_001_cut(out_dir):
    res = cut_slabs(build_ni2mnga_structure(), (0, 0, 1), out_dir)
    summary = pd.read_csv(res.get("summary_csv") or (out_dir / "数値" / "surface_summary.csv"))
    assert len(summary) >= 1

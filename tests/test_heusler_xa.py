"""Inverse Heusler XA (F-43m, Hg2CuTi type) in the Heusler sublattice model.

XA: X -> 4a (0,0,0) + 4d (3/4,3/4,3/4), Y -> 4c (1/4,1/4,1/4), Z -> 4b (1/2,1/2,1/2)
(text2cif heusler_xa). L21 / B2 keep 8c / 4a / 4b (regression).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from shido.core.validator import validate_composition_conserved, validate_structure
from shido.heusler_builder import (
    _assign_species_for_pattern,
    _build_supercell_template,
    generate_heusler_models,
    heusler_sublattice_keys,
)

# Wyckoff of a 16-atom cell site from its fractional coordinates (F-43m, origin at 4a)
_FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float)
_REPS = {"4a": (0.0, 0.0, 0.0), "4b": (0.5, 0.5, 0.5), "4c": (0.25, 0.25, 0.25), "4d": (0.75, 0.75, 0.75)}


def _wyckoff_of(frac: np.ndarray) -> str:
    for label, rep in _REPS.items():
        for t in _FCC:
            d = (np.asarray(frac) - np.asarray(rep) - t) % 1.0
            d = np.minimum(d, 1.0 - d)
            if np.all(d < 1e-3):
                return label
    raise AssertionError(f"{frac} is not on a 16-atom Heusler site")


@pytest.fixture(scope="module")
def xa_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("xa")
    return generate_heusler_models(
        formula="Mn2CoGa", ordering="XA", supercell=(1, 1, 1), num_patterns=1, a=5.80,
        output_dir=out, highlight_pair=("Mn", "Mn"), seed=0, pair_cutoff_factor=0.75,
        transparent_background=False,
    )


def test_sublattice_keys() -> None:
    assert heusler_sublattice_keys("L21") == ("8c", "4a", "4b")
    assert heusler_sublattice_keys("b2") == ("8c", "4a", "4b")
    assert heusler_sublattice_keys("XA") == ("4a", "4d", "4c", "4b")
    with pytest.raises(ValueError):
        heusler_sublattice_keys("L10")


def test_xa_outputs_exist(xa_result: dict) -> None:
    assert xa_result["ordering"] == "XA"
    assert xa_result["num_atoms"] == 16
    files = [*xa_result["structure_files"]["cif"], *xa_result["structure_files"]["poscar"],
             *xa_result["image_files"].values(), *xa_result["data_files"].values()]
    for f in files:
        assert Path(f).is_file() and Path(f).stat().st_size > 0, f
    assert xa_result["site_roles"] == {"X_4a_4d": ["Mn"], "Y_4c": ["Co"], "Z_4b": ["Ga"]}
    assert set(xa_result["sublattice_summary"]) == {"4a", "4d", "4c", "4b"}


def test_xa_space_group_composition_and_validator(xa_result: dict) -> None:
    s = Structure.from_file(xa_result["structure_files"]["cif"][0])
    assert SpacegroupAnalyzer(s, symprec=1e-3).get_space_group_number() == 216
    assert s.composition.element_composition.as_dict() == {"Mn": 8.0, "Co": 4.0, "Ga": 4.0}
    assert s.composition.reduced_composition == Composition("Mn2CoGa")
    validate_structure(s)
    validate_composition_conserved(Composition("Mn2CoGa"), s, context="XA Mn2CoGa")


def test_xa_site_assignment(xa_result: dict) -> None:
    s = Structure.from_file(xa_result["structure_files"]["cif"][0])
    by_site: dict[str, set[str]] = {}
    for site in s:
        by_site.setdefault(_wyckoff_of(site.frac_coords), set()).add(site.specie.symbol)
    assert by_site == {"4a": {"Mn"}, "4d": {"Mn"}, "4c": {"Co"}, "4b": {"Ga"}}
    summ = xa_result["sublattice_summary"]
    assert summ["4a"]["counts"]["Mn"] == 4 and summ["4d"]["counts"]["Mn"] == 4
    assert summ["4c"]["counts"]["Co"] == 4 and summ["4b"]["counts"]["Ga"] == 4


def test_xa_supercell_assignment() -> None:
    _, frac, labels = _build_supercell_template((2, 2, 2), 5.8, ordering="XA")
    assert len(labels) == 128 and set(labels) == {"4a", "4d", "4c", "4b"}
    species = _assign_species_for_pattern(
        labels, {"Mn": 64, "Co": 32, "Ga": 32}, ["Mn", "Co", "Ga"], "XA", np.random.default_rng(1)
    )
    for lab, el in zip(labels, species):
        assert el == {"4a": "Mn", "4d": "Mn", "4c": "Co", "4b": "Ga"}[lab]


@pytest.mark.parametrize("ordering", ["L21", "B2"])
def test_l21_b2_template_unchanged(ordering: str) -> None:
    """Regression: L21 / B2 keep the 8c / 4a / 4b template in the old order."""
    _, frac, labels = _build_supercell_template((1, 1, 1), 5.8)
    _, frac2, labels2 = _build_supercell_template((1, 1, 1), 5.8, ordering=ordering)
    assert labels == labels2 == ["8c"] * 8 + ["4a"] * 4 + ["4b"] * 4
    assert np.allclose(np.array(frac), np.array(frac2))
    species = _assign_species_for_pattern(
        labels, {"Ni": 8, "Mn": 4, "Ga": 4}, ["Ni", "Mn", "Ga"], ordering, np.random.default_rng(0)
    )
    assert species[:8] == ["Ni"] * 8
    if ordering == "L21":
        assert species[8:12] == ["Mn"] * 4 and species[12:] == ["Ga"] * 4


def test_l21_still_fm3m(tmp_path: Path) -> None:
    res = generate_heusler_models(
        formula="Ni2MnGa", ordering="L21", supercell=(1, 1, 1), num_patterns=1, a=5.82,
        output_dir=tmp_path, highlight_pair=("Mn", "Mn"), seed=0,
    )
    s = Structure.from_file(res["structure_files"]["cif"][0])
    assert SpacegroupAnalyzer(s, symprec=1e-3).get_space_group_number() == 225
    assert res["site_roles"] == {"X_8c": ["Ni"], "Y_4a": ["Mn"], "Z_4b": ["Ga"]}
    assert set(res["sublattice_summary"]) == {"8c", "4a", "4b"}

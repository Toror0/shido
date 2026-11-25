"""Reference-value verification tests against analytical formulas and literature data."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from shido.analyzer import (
    _analyze_symmetry,
    analyze_structure_data,
    build_ni2mnga_structure,
    compute_bonds,
)
from shido.ca_sweep import apply_constant_volume_ca
from shido.slab_cutter import cut_slabs


def _coordination_numbers(bonds_df: pd.DataFrame, num_sites: int) -> dict[int, int]:
    """Count per-site coordination numbers from deduplicated undirected bonds_df."""
    counts: Counter[int] = Counter({i: 0 for i in range(num_sites)})
    for row in bonds_df.itertuples(index=False):
        counts[int(row.site_i)] += 1
        counts[int(row.site_j)] += 1
    return dict(counts)


# 1. NaCl (rock-salt, Fm-3m)
# Literature:
#   - Room-temperature lattice parameter a = 5.640 A (halite / rock-salt structure, Fm-3m, No. 225).
#   - Sources:
#       * Crystallography Open Database (COD), Halite (NaCl, Fm-3m): https://www.crystallography.net/cod/
#       * R. W. G. Wyckoff, "Crystal Structures", Vol. 1 (rock-salt prototype).
def test_nacl_interplanar_spacings_and_nearest_neighbor():
    a = 5.640  # A (literature room-temperature value)
    struct = Structure.from_spacegroup(
        sg="Fm-3m",
        lattice=Lattice.cubic(a),
        species=["Na", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    assert len(struct) == 8

    # Theoretical cubic interplanar spacings d_hkl = a / sqrt(h^2 + k^2 + l^2)
    assert struct.lattice.d_hkl((1, 1, 1)) == pytest.approx(a / math.sqrt(3.0), abs=1e-9)
    assert struct.lattice.d_hkl((2, 0, 0)) == pytest.approx(a / 2.0, abs=1e-9)
    assert struct.lattice.d_hkl((2, 2, 0)) == pytest.approx(a / math.sqrt(8.0), abs=1e-9)

    # Nearest-neighbor Na-Cl bond length: theoretical a/2 = 2.820 A
    bonds_df, _, _ = compute_bonds(struct)
    assert not bonds_df.empty
    min_dist = float(bonds_df["distance_angstrom"].min())
    assert min_dist == pytest.approx(a / 2.0, abs=1e-3)
    assert min_dist == pytest.approx(2.820, abs=1e-3)


# 2. Si (diamond cubic, Fd-3m)
# Literature:
#   - Lattice parameter a = 5.431020511(89) A at 22.5 C in vacuum (rounded here to 5.431 A),
#     yielding Si-Si bond length a*sqrt(3)/4 = 2.3517 A (~2.352 A) and tetrahedral CN = 4.
#   - Source:
#       * NIST Reference on Constants, Units, and Uncertainty (CODATA recommended value,
#         "lattice parameter of silicon"): https://physics.nist.gov/cgi-bin/cuu/Value?asil
def test_si_diamond_bond_length_and_coordination():
    a = 5.431  # A (NIST CODATA recommended value, rounded to 3 decimals)
    # Fd-3m origin choice 1 (pymatgen default): 8a Wyckoff site at (0, 0, 0)
    struct = Structure.from_spacegroup(
        sg="Fd-3m",
        lattice=Lattice.cubic(a),
        species=["Si"],
        coords=[[0.0, 0.0, 0.0]],
    )
    assert len(struct) == 8

    bonds_df, _, _ = compute_bonds(struct)
    assert not bonds_df.empty

    expected_bond = a * math.sqrt(3.0) / 4.0  # ~2.35169 A
    assert bonds_df["distance_angstrom"].min() == pytest.approx(expected_bond, abs=1e-3)
    assert bonds_df["distance_angstrom"].max() == pytest.approx(expected_bond, abs=1e-3)
    assert float(bonds_df["distance_angstrom"].mean()) == pytest.approx(2.352, abs=1e-3)

    cn_by_site = _coordination_numbers(bonds_df, len(struct))
    assert set(cn_by_site.values()) == {4}


# 3. Cu (fcc, Fm-3m)
# Literature:
#   - Lattice parameter a = 3.615 A (361.5 pm), experimental bulk density rho = 8.96 g/cm^3
#     (WebElements lists 8920 kg/m^3; standard CRC/textbook room-temperature value 8.96 g/cm^3),
#     nearest-neighbor distance a/sqrt(2) = 2.5562 A (~2.556 A), CN = 12.
#   - Sources:
#       * WebElements, Copper crystal structure: https://www.webelements.com/copper/crystal_structure.html
#       * WebElements, Copper physical properties: https://www.webelements.com/copper/physics.html
def test_cu_fcc_nearest_neighbor_coordination_and_density(out_dir: Path):
    a = 3.615  # A
    lit_density_g_cm3 = 8.96  # g/cm^3 (room-temperature bulk Cu)
    struct = Structure.from_spacegroup(
        sg="Fm-3m",
        lattice=Lattice.cubic(a),
        species=["Cu"],
        coords=[[0.0, 0.0, 0.0]],
    )
    assert len(struct) == 4

    bonds_df, _, _ = compute_bonds(struct)
    assert not bonds_df.empty

    expected_nn = a / math.sqrt(2.0)  # ~2.55619 A
    assert float(bonds_df["distance_angstrom"].min()) == pytest.approx(expected_nn, abs=1e-3)
    assert float(bonds_df["distance_angstrom"].min()) == pytest.approx(2.556, abs=1e-3)

    cn_by_site = _coordination_numbers(bonds_df, len(struct))
    assert set(cn_by_site.values()) == {12}

    res = analyze_structure_data(struct, out_dir)
    density = float(res["summary"]["density_g_cm3"])
    assert density == pytest.approx(lit_density_g_cm3, rel=0.01)


# 4. Fe (bcc / alpha-Fe, Im-3m)
# Literature:
#   - Lattice parameter a = 2.866 A (286.6 pm), experimental bulk density rho = 7.874 g/cm^3 (~7.87 g/cm^3),
#     nearest-neighbor distance a*sqrt(3)/2 = 2.4820 A (~2.482 A).
#   - Sources:
#       * WebElements, Iron crystal structure: https://www.webelements.com/iron/crystal_structure.html
#       * WebElements, Iron physical properties: https://www.webelements.com/iron/physics.html
def test_fe_bcc_nearest_neighbor_and_density(out_dir: Path):
    a = 2.866  # A
    lit_density_g_cm3 = 7.87  # g/cm^3 (literature value 7.874 g/cm^3)
    struct = Structure.from_spacegroup(
        sg="Im-3m",
        lattice=Lattice.cubic(a),
        species=["Fe"],
        coords=[[0.0, 0.0, 0.0]],
    )
    assert len(struct) == 2

    bonds_df, _, _ = compute_bonds(struct)
    assert not bonds_df.empty

    expected_nn = a * math.sqrt(3.0) / 2.0  # ~2.48203 A
    assert float(bonds_df["distance_angstrom"].min()) == pytest.approx(expected_nn, abs=1e-3)
    assert float(bonds_df["distance_angstrom"].min()) == pytest.approx(2.482, abs=1e-3)

    res = analyze_structure_data(struct, out_dir)
    density = float(res["summary"]["density_g_cm3"])
    assert density == pytest.approx(lit_density_g_cm3, rel=0.01)


# 5. Ni2MnGa (L2_1 full-Heusler, Fm-3m)
# Literature:
#   - Room-temperature austenitic L2_1 cubic phase: space group Fm-3m (No. 225), a = 5.825 A,
#     Ni (8c) - Mn (4a) nearest-neighbor distance a*sqrt(3)/4 = 2.5223 A.
#   - Source:
#       * P. J. Webster, K. R. A. Ziebeck, S. L. Town, and M. S. Peak,
#         "Magnetic order and phase transformation in Ni2MnGa",
#         Philosophical Magazine Part B, 49(3), 295-310 (1984):
#         https://doi.org/10.1080/13642818408227646
def test_ni2mnga_l21_symmetry_and_ni_mn_distance():
    struct = build_ni2mnga_structure()
    a = float(struct.lattice.a)
    assert a == pytest.approx(5.825, abs=1e-2)

    sg_symbol, sg_number, crystal_system = _analyze_symmetry(struct)
    assert sg_symbol == "Fm-3m"
    assert sg_number == 225
    assert crystal_system == "cubic"

    bonds_df, _, _ = compute_bonds(struct)
    ni_mn = bonds_df[bonds_df["element_pair"] == "Mn-Ni"]
    assert not ni_mn.empty

    expected_ni_mn = a * math.sqrt(3.0) / 4.0
    assert float(ni_mn["distance_angstrom"].min()) == pytest.approx(expected_ni_mn, abs=1e-3)


# 6. Constant-volume c/a sweep (Bain-path tetragonal distortion)
# Theory:
#   - Volume-conserving tetragonal deformation: a' = b' = (V / (c/a))^(1/3), c' = (c/a) * a',
#     so V' = a'^2 * c' = V exactly and c'/a' = target_ca.
def test_ca_sweep_constant_volume_and_exact_ratio():
    base = build_ni2mnga_structure()
    target_ca = 1.2
    distorted = apply_constant_volume_ca(base, target_ca)

    assert distorted.volume == pytest.approx(base.volume, rel=1e-9)
    actual_ca = float(distorted.lattice.c) / float(distorted.lattice.a)
    assert actual_ca == pytest.approx(target_ca, rel=1e-12, abs=1e-12)
    assert float(distorted.lattice.a) == pytest.approx(float(distorted.lattice.b), rel=1e-12)


# 7. Cu (111) slab interplanar spacing
# Literature & theory:
#   - Cu fcc lattice parameter a = 3.615 A (WebElements: https://www.webelements.com/copper/crystal_structure.html).
#   - Theoretical (111) interplanar spacing d(111) = a / sqrt(3) = 2.08712 A.
def test_cu_111_slab_d_hkl(out_dir: Path):
    a = 3.615  # A
    cu_bulk = Structure.from_spacegroup(
        sg="Fm-3m",
        lattice=Lattice.cubic(a),
        species=["Cu"],
        coords=[[0.0, 0.0, 0.0]],
    )

    res = cut_slabs(cu_bulk, (1, 1, 1), out_dir)
    csv_path = Path(res["surface_summary_csv"])
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert not df.empty

    expected_d111 = a / math.sqrt(3.0)  # ~2.08712 A
    for d_val in df["d_hkl_angstrom"]:
        assert float(d_val) == pytest.approx(expected_d111, abs=1e-4)

"""Safe element extraction for ordered *and* partially occupied (disordered) sites.

``site.specie`` raises ``AttributeError`` on a disordered site (e.g. Mn0.5Ga0.5 in a
B2 alloy or a CIF with occupancy < 1). All code paths go through these helpers,
which read ``site.species`` instead.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("crystal_mcp")

_OCC_FULL_TOL = 1e-3


def _symbol(sp: Any) -> str:
    """Element symbol of a pymatgen Element / Species / DummySpecies (oxidation state dropped)."""
    el = getattr(sp, "element", None)  # Species("Mn2+").element -> Element Mn
    if el is not None and hasattr(el, "symbol"):
        return str(el.symbol)
    return str(getattr(sp, "symbol", sp))


def site_occupancies(site: Any) -> list[tuple[str, float]]:
    """Return ``[(element, occupancy), ...]`` sorted by occupancy (desc), then symbol.

    Works for pymatgen ``Site``/``PeriodicSite`` (ordered or disordered), and falls back to
    plain objects with ``.symbol``. Never touches ``site.specie``.
    """
    species = getattr(site, "species", None)
    if species is not None:
        agg: dict[str, float] = {}
        try:
            items = species.items()
        except AttributeError:
            items = []
        for sp, occ in items:
            sym = _symbol(sp)
            agg[sym] = agg.get(sym, 0.0) + float(occ)
        if agg:
            return sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
    if hasattr(site, "symbol"):
        return [(str(site.symbol), 1.0)]
    return [(str(site), 1.0)]


def dominant_element(site: Any) -> str:
    """Element with the largest occupancy on ``site`` (ties broken alphabetically)."""
    occs = site_occupancies(site)
    if not occs:
        raise ValueError(f"Site has no species: {site!r}")
    return occs[0][0]


def is_partially_occupied(site: Any) -> bool:
    """True when the site is mixed (>1 element) or its total occupancy is below 1."""
    occs = site_occupancies(site)
    return len(occs) > 1 or sum(o for _, o in occs) < 1.0 - _OCC_FULL_TOL


def warn_partial_occupancy(structure: Any, context: str = "render") -> int:
    """Log one warning if ``structure`` holds partially occupied sites; return their count."""
    n = sum(1 for site in structure if is_partially_occupied(site))
    if n:
        logger.warning(
            "%s: %d site(s) are partially occupied (occupancy < 1.0 or mixed). "
            "Using the dominant element for labels/bonds and occupancy-banded spheres for drawing.",
            context,
            n,
        )
    return n

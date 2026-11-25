"""Runtime safety layer (validators and defensive helpers) for crystal-mcp."""

# The one place that holds the version number (server.py / gui.py read it).
__version__ = "3.0.0"

from shido.core.species import dominant_element, site_occupancies
from shido.core.validator import (
    ScientificValidationError,
    validate_blank_image,
    validate_composition_conserved,
    validate_lattice,
    validate_miller_index,
    validate_min_distance,
    validate_structure,
)

__all__ = [
    "__version__",
    "ScientificValidationError",
    "dominant_element",
    "site_occupancies",
    "validate_blank_image",
    "validate_composition_conserved",
    "validate_lattice",
    "validate_miller_index",
    "validate_min_distance",
    "validate_structure",
]

"""Powder X-ray diffraction (XRD) pattern: peak table (CSV) and stick figure (PNG/SVG).

Uses pymatgen's kinematic ``XRDCalculator`` (atomic scattering factors, Lorentz-polarization
factor, multiplicity; no Debye-Waller factor unless given). Partially occupied sites are
handled directly: each site scatters with the occupancy-weighted sum of its species'
form factors (average / virtual-crystal approximation), so a B2-disordered Heusler loses
its (111)-type superlattice peaks as it should. No ordered proxy is needed.

Settings (``settings_defaults.yaml`` ``xrd:``): wavelength, two_theta_min, two_theta_max,
label_top_n. Graph styling follows ``ca_sweep.plot_ca_energy_curve`` (no grid, four-sided
frame, inward ticks on all sides, 1/5 minor ticks, dpi from ``render.dpi``).

Extensions (batch 12, all off by default so the default outputs are unchanged):
``profile`` broadens every stick into a Gaussian / Lorentzian / pseudo-Voigt peak of unit
area times the stick intensity (FWHM constant or Caglioti U, V, W), ``radiation="neutron"``
uses pymatgen's ``NDCalculator`` (coherent scattering lengths), and ``observed`` reads a
measured 2-column pattern (.xy / .csv) and overlays it with the scaled calculation and the
difference curve.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402
from pymatgen.analysis.diffraction.neutron import NDCalculator  # noqa: E402
from pymatgen.analysis.diffraction.xrd import WAVELENGTHS, XRDCalculator  # noqa: E402
from pymatgen.core import Structure  # noqa: E402

from shido.core.output_paths import XRD_DIR  # noqa: E402
from shido.core.validator import ScientificValidationError, validate_structure  # noqa: E402

XRD_PEAK_COLUMNS: list[str] = ["two_theta_deg", "intensity_rel", "hkl", "d_angstrom", "multiplicity"]

_DEFAULT_WAVELENGTH = "CuKa"
_DEFAULT_RANGE = (10.0, 90.0)
_DEFAULT_LABEL_TOP_N = 8
_DEFAULT_DPI = 300
_DEFAULT_FWHM = 0.1  # deg 2θ
_DEFAULT_STEP = 0.01  # deg 2θ
_DEFAULT_ETA = 0.5
PROFILE_SHAPES = ("gaussian", "lorentzian", "pseudo_voigt")
RADIATIONS = ("xray", "neutron")


def _setting(path: str, default: Any) -> Any:
    try:
        from shido.settings import get

        return get(path, default)
    except Exception:  # noqa: BLE001 - settings are optional
        return default


def _render_dpi() -> int:
    try:
        dpi = int(round(float(_setting("render.dpi", _DEFAULT_DPI))))
    except (TypeError, ValueError):
        dpi = _DEFAULT_DPI
    return dpi if dpi > 0 else _DEFAULT_DPI


def _resolve_wavelength(wavelength: str | float) -> tuple[float, str]:
    """Return (lambda in Å, label). Accepts a pymatgen key ('CuKa', 'MoKa1', ...) or Å."""
    if isinstance(wavelength, str):
        key = wavelength.strip()
        match = next((k for k in WAVELENGTHS if k.lower() == key.lower()), None)
        if match is None:
            try:
                value = float(key)
            except ValueError:
                raise ScientificValidationError(
                    f"Unknown X-ray source {wavelength!r}. Use one of {sorted(WAVELENGTHS)} or a wavelength in Å."
                ) from None
            return _resolve_wavelength(value)
        return float(WAVELENGTHS[match]), match
    value = float(wavelength)
    if not math.isfinite(value) or value <= 0.0:
        raise ScientificValidationError(f"Wavelength must be a positive number in Å, got {wavelength!r}.")
    return value, f"{value:g} Å"


def _validate_range(two_theta_range: tuple[float, float]) -> tuple[float, float]:
    try:
        lo, hi = (float(two_theta_range[0]), float(two_theta_range[1]))
    except (TypeError, ValueError, IndexError):
        raise ScientificValidationError(f"two_theta_range must be (min, max), got {two_theta_range!r}.") from None
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo < 0.0 or hi > 180.0 or lo >= hi:
        raise ScientificValidationError(
            f"two_theta_range must satisfy 0 <= min < max <= 180 deg, got ({lo}, {hi})."
        )
    return lo, hi


def _hkl_text(hkl: tuple[int, ...]) -> str:
    """CSV form: '(1 1 1)', '(1 -1 0)', hexagonal '(1 0 -1 1)'."""
    return "(" + " ".join(str(int(i)) for i in hkl) + ")"


def _hkl_mathtext(hkl: tuple[int, ...]) -> str:
    """Figure label: (1\\bar{1}0) with overbars for negative indices."""
    if all(0 <= int(i) < 10 for i in hkl):
        return "(" + "".join(str(int(i)) for i in hkl) + ")"  # plain text keeps the Arial font
    parts = [rf"\bar{{{abs(int(i))}}}" if int(i) < 0 else str(int(i)) for i in hkl]
    sep = r"\,\," if any(abs(int(i)) >= 10 for i in hkl) else ""
    return r"$\mathregular{(" + sep.join(parts) + ")}$"


def _source_mathtext(label: str) -> str:
    """'CuKa1' -> 'Cu K$\\alpha_1$' for the figure title; numeric labels unchanged."""
    import re

    m = re.fullmatch(r"([A-Z][a-z]?)K([ab])([12]?)", label)
    if not m:
        return label
    greek = r"\alpha" if m.group(2) == "a" else r"\beta"
    sub = f"_{m.group(3)}" if m.group(3) else ""
    return f"{m.group(1)} K${greek}{sub}$"


def _has_partial_occupancy(structure: Structure) -> bool:
    return any(not site.is_ordered for site in structure)


def _formula_mathtext(structure: Structure) -> str:
    try:
        from shido.renderer import formula_parts

        return "".join(rf"{sym}$_{{{cnt}}}$" if cnt else sym for sym, cnt in formula_parts(structure))
    except Exception:  # noqa: BLE001 - title only
        return structure.composition.reduced_formula


# ---- peak profiles -------------------------------------------------------------------

def normalize_profile_shape(shape: str | None) -> str | None:
    """'gauss' / 'Gaussian' -> 'gaussian', 'pseudo-voigt' / 'pv' -> 'pseudo_voigt'; None stays None."""
    if shape is None:
        return None
    key = str(shape).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gauss": "gaussian", "g": "gaussian", "gaussian": "gaussian",
        "lorentz": "lorentzian", "l": "lorentzian", "lorentzian": "lorentzian", "cauchy": "lorentzian",
        "pv": "pseudo_voigt", "pseudo_voigt": "pseudo_voigt", "pseudovoigt": "pseudo_voigt", "voigt": "pseudo_voigt",
    }
    if key not in aliases:
        raise ScientificValidationError(
            f"Unknown peak profile {shape!r}. Use one of {list(PROFILE_SHAPES)}."
        )
    return aliases[key]


def caglioti_fwhm(two_theta_deg: Any, u: float, v: float, w: float) -> np.ndarray:
    """Caglioti: FWHM^2 = U tan^2(θ) + V tan(θ) + W (FWHM in deg 2θ, U/V/W in deg^2)."""
    t = np.tan(np.radians(np.asarray(two_theta_deg, dtype=float) / 2.0))
    h2 = u * t * t + v * t + w
    if np.any(~np.isfinite(h2)) or np.any(h2 <= 0.0):
        raise ScientificValidationError(
            f"Caglioti U={u}, V={v}, W={w} gives FWHM^2 <= 0 inside the 2θ range; check the signs."
        )
    return np.sqrt(h2)


def profile_curve(
    x: Any,
    centers: Any,
    intensities: Any,
    fwhm: Any,
    shape: str = "pseudo_voigt",
    eta: float = _DEFAULT_ETA,
) -> np.ndarray:
    """Sum of unit-area peaks: y(x) = Σ I_k · P(x - c_k; H_k).

    Each peak's integral over 2θ equals its stick intensity, so changing the shape or the
    FWHM keeps the peak areas. ``fwhm`` is a scalar or one value per peak (deg 2θ);
    ``eta`` is the Lorentzian fraction of the pseudo-Voigt (0 = Gaussian, 1 = Lorentzian).
    """
    shape = normalize_profile_shape(shape) or "pseudo_voigt"
    x = np.asarray(x, dtype=float)
    c = np.asarray(centers, dtype=float).reshape(-1)
    a = np.asarray(intensities, dtype=float).reshape(-1)
    h = np.broadcast_to(np.asarray(fwhm, dtype=float), c.shape).astype(float)
    if np.any(h <= 0.0) or np.any(~np.isfinite(h)):
        raise ScientificValidationError(f"FWHM must be positive, got {fwhm!r}.")
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ScientificValidationError(f"eta (Lorentzian fraction) must be within 0..1, got {eta}.")
    y = np.zeros_like(x)
    for ck, ak, hk in zip(c, a, h):
        dx = x - ck
        gauss = math.sqrt(4.0 * math.log(2.0) / math.pi) / hk * np.exp(-4.0 * math.log(2.0) * dx * dx / (hk * hk))
        lorentz = (2.0 / (math.pi * hk)) / (1.0 + 4.0 * dx * dx / (hk * hk))
        if shape == "gaussian":
            p = gauss
        elif shape == "lorentzian":
            p = lorentz
        else:
            p = eta * lorentz + (1.0 - eta) * gauss
        y += ak * p
    return y


# ---- measured patterns -----------------------------------------------------------------

def load_observed_pattern(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a measured powder pattern: first two numeric columns (2θ [deg], intensity).

    Accepts .xy / .dat / .txt (whitespace) and .csv (comma / semicolon / tab). Lines that
    start with '#', '!', ';' or '%' and non-numeric header lines are skipped. Returns the
    arrays sorted by 2θ.
    """
    p = Path(path)
    if not p.is_file():
        raise ScientificValidationError(f"Observed pattern file not found: {p}")
    xs: list[float] = []
    ys: list[float] = []
    text = p.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#!;%":
            continue
        for sep in (",", ";", "\t"):
            s = s.replace(sep, " ")
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            xv, yv = float(parts[0]), float(parts[1])
        except ValueError:
            continue  # header
        if math.isfinite(xv) and math.isfinite(yv):
            xs.append(xv)
            ys.append(yv)
    if len(xs) < 3:
        raise ScientificValidationError(
            f"Could not read a 2-column pattern (2θ, intensity) from {p.name}: found {len(xs)} numeric rows."
        )
    x = np.asarray(xs)
    y = np.asarray(ys)
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def _fit_scale_background(obs: np.ndarray, calc: np.ndarray) -> tuple[float, float]:
    """Least squares obs ≈ scale · calc + background (constant)."""
    a = np.column_stack([calc, np.ones_like(calc)])
    (scale, background), *_ = np.linalg.lstsq(a, obs, rcond=None)
    return float(scale), float(background)


def compute_xrd(
    structure: Structure,
    output_dir: str | Path,
    wavelength: str | float | None = None,
    two_theta_range: tuple[float, float] | None = None,
    label_top_n: int | None = None,
    symprec: float = 0.0,
    debye_waller_factors: dict[str, float] | None = None,
    make_figure: bool = True,
    radiation: str = "xray",
    profile: str | None = None,
    fwhm: float | None = None,
    caglioti: tuple[float, float, float] | list[float] | None = None,
    eta: float | None = None,
    step: float | None = None,
    observed: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the powder XRD stick pattern of ``structure`` and write CSV + figures.

    Args:
        structure: pymatgen Structure (partial occupancies allowed).
        output_dir: results go to ``<output_dir>/data/xrd_peaks.csv`` and
            ``<output_dir>/images/figure_xrd.png|.svg``.
        wavelength: pymatgen source key ('CuKa', 'MoKa', 'CuKa1', ...) or Å as a number.
            ``None`` -> settings ``xrd.wavelength`` (default 'CuKa', 1.54184 Å).
        two_theta_range: (min, max) in degrees. ``None`` -> settings
            ``xrd.two_theta_min`` / ``xrd.two_theta_max`` (default (10, 90)).
        label_top_n: number of strongest peaks labelled with hkl in the figure.
            ``None`` -> settings ``xrd.label_top_n`` (default 8).
        symprec: passed to XRDCalculator (0 = use the cell as given).
        debye_waller_factors: optional {element: B [Å^2]}.
        make_figure: False skips the PNG/SVG (CSV and peaks only).
        radiation: 'xray' (default, XRDCalculator) or 'neutron' (NDCalculator, nuclear
            scattering lengths). Neutron files are ``neutron_peaks.csv`` /
            ``figure_neutron.png|.svg`` (and ``neutron_profile.csv`` ...).
        profile: None (sticks, default) or 'gaussian' / 'lorentzian' / 'pseudo_voigt'.
            Giving ``fwhm``, ``caglioti`` or ``observed`` alone implies 'pseudo_voigt'.
            Writes ``data/xrd_profile.csv`` (two_theta_deg, intensity = Σ unit-area peaks
            × stick intensity, intensity_rel = scaled to max 100) and draws the curve.
        fwhm: constant FWHM in deg 2θ (default 0.1). Ignored when ``caglioti`` is given.
        caglioti: (U, V, W) in deg^2, FWHM^2 = U tan^2θ + V tanθ + W.
        eta: Lorentzian fraction of the pseudo-Voigt (0..1, default 0.5).
        step: 2θ step of the curve (default 0.01 deg).
        observed: measured pattern (.xy / .csv, 2θ and intensity columns). The curve is
            evaluated on its 2θ points, fitted as scale × calc + background, and
            ``data/xrd_observed_compare.csv`` (two_theta_deg, observed, calculated,
            difference) and ``images/figure_xrd_observed.png|.svg`` are written. When
            ``two_theta_range`` is None the observed 2θ range is used.

    Returns:
        dict with output_dir, xrd_peaks_csv, figure_png, figure_svg, peaks (list of row
        dicts in CSV column order), wavelength_angstrom, wavelength_label,
        two_theta_range, n_peaks, formula, partial_occupancy, occupancy_treatment,
        warnings. With the extensions also radiation, profile (shape, fwhm / caglioti,
        eta, step, profile_csv) and observed (file, compare_csv, figure_png/svg, scale,
        background, rwp_percent); these keys are absent when the options are not used.

    Raises:
        ScientificValidationError (a ValueError) for a bad structure, wavelength or range.
    """
    validate_structure(structure)

    rad = str(radiation or "xray").strip().lower().replace("-", "")
    rad = {"x": "xray", "xrays": "xray", "n": "neutron", "neutrons": "neutron"}.get(rad, rad)
    if rad not in RADIATIONS:
        raise ScientificValidationError(f"radiation must be 'xray' or 'neutron', got {radiation!r}.")
    shape = normalize_profile_shape(profile)
    if shape is None and (fwhm is not None or caglioti is not None or observed is not None):
        shape = "pseudo_voigt"
    obs_xy: tuple[np.ndarray, np.ndarray] | None = None
    if observed is not None:
        obs_xy = load_observed_pattern(observed)

    if wavelength is None:
        wavelength = _setting("xrd.wavelength", _DEFAULT_WAVELENGTH)
    lam, lam_label = _resolve_wavelength(wavelength)
    if two_theta_range is None and obs_xy is not None:
        two_theta_range = (max(0.0, float(obs_xy[0][0])), min(180.0, float(obs_xy[0][-1])))
    if two_theta_range is None:
        two_theta_range = (
            _setting("xrd.two_theta_min", _DEFAULT_RANGE[0]),
            _setting("xrd.two_theta_max", _DEFAULT_RANGE[1]),
        )
    lo, hi = _validate_range(two_theta_range)
    if label_top_n is None:
        label_top_n = _setting("xrd.label_top_n", _DEFAULT_LABEL_TOP_N)
    try:
        label_top_n = max(0, int(label_top_n))
    except (TypeError, ValueError):
        label_top_n = _DEFAULT_LABEL_TOP_N

    partial = _has_partial_occupancy(structure)
    occupancy_treatment = (
        "Partial occupancies used directly: occupancy-weighted atomic scattering factors per site "
        "(average / virtual-crystal approximation, no short-range order)."
        if partial
        else "Fully ordered structure."
    )

    calculator_cls = NDCalculator if rad == "neutron" else XRDCalculator
    calc = calculator_cls(wavelength=lam, symprec=symprec, debye_waller_factors=debye_waller_factors)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pattern = calc.get_pattern(structure, scaled=True, two_theta_range=(lo, hi))

    peaks: list[dict[str, Any]] = []
    families_per_peak: list[list[tuple[int, ...]]] = []
    for two_theta, intensity, fams, d in zip(pattern.x, pattern.y, pattern.hkls, pattern.d_hkls):
        hkls = [tuple(int(i) for i in f["hkl"]) for f in fams]
        mult = int(sum(int(f["multiplicity"]) for f in fams))
        families_per_peak.append(hkls)
        peaks.append(
            {
                "two_theta_deg": round(float(two_theta), 4),
                "intensity_rel": round(float(intensity), 3),
                "hkl": ", ".join(_hkl_text(h) for h in hkls),
                "d_angstrom": round(float(d), 5),
                "multiplicity": mult,
            }
        )

    warnings_out: list[str] = []
    if not peaks:
        warnings_out.append(f"No reflections between 2θ = {lo:g}° and {hi:g}° for λ = {lam:.5f} Å.")
    if any(len(h) > 1 for h in families_per_peak):
        warnings_out.append(
            "Some peaks contain several hkl families at the same 2θ (e.g. (3 3 3) and (5 1 1) in cubic); "
            "they are listed together and multiplicity is their sum."
        )

    stem = "neutron" if rad == "neutron" else "xrd"
    out_root = Path(output_dir)
    data_dir = out_root / XRD_DIR
    images_dir = out_root / XRD_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{stem}_peaks.csv"
    pd.DataFrame(peaks, columns=XRD_PEAK_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8")

    # ---- optional profile curve -------------------------------------------------------
    profile_info: dict[str, Any] | None = None
    curve: tuple[np.ndarray, np.ndarray] | None = None
    centers = np.asarray(pattern.x, dtype=float)
    heights = np.asarray(pattern.y, dtype=float)
    peak_fwhm: np.ndarray | None = None
    if shape is not None:
        if eta is None:
            eta = _DEFAULT_ETA
        if step is None:
            step = _DEFAULT_STEP
        step = float(step)
        if not math.isfinite(step) or step <= 0.0:
            raise ScientificValidationError(f"step must be a positive 2θ increment in degrees, got {step}.")
        if caglioti is not None:
            try:
                u, v, w = (float(t) for t in caglioti)
            except (TypeError, ValueError):
                raise ScientificValidationError(f"caglioti must be (U, V, W), got {caglioti!r}.") from None
            caglioti_fwhm(np.linspace(lo, hi, 181), u, v, w)  # validates FWHM^2 > 0 over the range
            peak_fwhm = caglioti_fwhm(centers, u, v, w) if len(centers) else np.zeros(0)
            fwhm_used: Any = None
        else:
            fwhm_used = float(_DEFAULT_FWHM if fwhm is None else fwhm)
            if not math.isfinite(fwhm_used) or fwhm_used <= 0.0:
                raise ScientificValidationError(f"FWHM must be a positive number in deg 2θ, got {fwhm!r}.")
            peak_fwhm = np.full(len(centers), fwhm_used)
        x_curve = np.arange(lo, hi + 0.5 * step, step)
        y_curve = profile_curve(x_curve, centers, heights, peak_fwhm, shape, eta) if len(centers) else np.zeros_like(x_curve)
        y_max = float(y_curve.max()) if y_curve.size else 0.0
        y_rel = y_curve * (100.0 / y_max) if y_max > 0.0 else y_curve
        curve = (x_curve, y_rel)
        profile_csv = data_dir / f"{stem}_profile.csv"
        pd.DataFrame(
            {"two_theta_deg": np.round(x_curve, 5), "intensity": y_curve, "intensity_rel": np.round(y_rel, 4)}
        ).to_csv(profile_csv, index=False, encoding="utf-8")
        profile_info = {
            "shape": shape,
            "fwhm_deg": fwhm_used,
            "caglioti_uvw": [u, v, w] if caglioti is not None else None,
            "eta": float(eta) if shape == "pseudo_voigt" else None,
            "step_deg": step,
            "n_points": int(x_curve.size),
            "profile_csv": str(profile_csv.resolve()),
            "peak_fwhm_deg": [round(float(h), 5) for h in peak_fwhm],
        }

    # ---- optional overlay with a measured pattern ---------------------------------------
    observed_info: dict[str, Any] | None = None
    if obs_xy is not None:
        ox, oy = obs_xy
        mask = (ox >= lo) & (ox <= hi)
        ox, oy = ox[mask], oy[mask]
        if ox.size < 3:
            raise ScientificValidationError(
                f"The observed pattern has fewer than 3 points between 2θ = {lo:g}° and {hi:g}°."
            )
        c_obs = (
            profile_curve(ox, centers, heights, peak_fwhm, shape, eta) if len(centers) else np.zeros_like(ox)
        )
        scale, background = _fit_scale_background(oy, c_obs)
        c_fit = scale * c_obs + background
        diff = oy - c_fit
        denom = float(np.sum(oy * oy))
        r_pct = 100.0 * math.sqrt(float(np.sum(diff * diff)) / denom) if denom > 0.0 else float("nan")
        compare_csv = data_dir / f"{stem}_observed_compare.csv"
        pd.DataFrame(
            {"two_theta_deg": ox, "observed": oy, "calculated": c_fit, "difference": diff}
        ).to_csv(compare_csv, index=False, encoding="utf-8")
        observed_info = {
            "file": str(Path(observed).resolve()),
            "n_points": int(ox.size),
            "compare_csv": str(compare_csv.resolve()),
            "scale": scale,
            "background": background,
            "r_factor_percent": r_pct,
            "figure_png": None,
            "figure_svg": None,
        }

    png_path: Path | None = None
    svg_path: Path | None = None
    if make_figure:
        images_dir.mkdir(parents=True, exist_ok=True)
        png_path = images_dir / f"figure_{stem}.png"
        svg_path = images_dir / f"figure_{stem}.svg"
        _plot_xrd(
            peaks,
            families_per_peak,
            structure,
            (lo, hi),
            lam,
            lam_label,
            label_top_n,
            png_path,
            svg_path,
            curve=curve,
            radiation=rad,
        )
        if observed_info is not None:
            opng = images_dir / f"figure_{stem}_observed.png"
            osvg = images_dir / f"figure_{stem}_observed.svg"
            cmp_df = pd.read_csv(observed_info["compare_csv"])
            _plot_observed(
                cmp_df, [p["two_theta_deg"] for p in peaks], structure, (lo, hi), lam, lam_label,
                rad, Path(observed).name, observed_info["r_factor_percent"], opng, osvg,
            )
            observed_info["figure_png"] = str(opng.resolve())
            observed_info["figure_svg"] = str(osvg.resolve())

    result = {
        "output_dir": str(out_root.resolve()),
        "xrd_peaks_csv": str(csv_path.resolve()),
        "figure_png": str(png_path.resolve()) if png_path else None,
        "figure_svg": str(svg_path.resolve()) if svg_path else None,
        "peaks": peaks,
        "n_peaks": len(peaks),
        "wavelength_angstrom": lam,
        "wavelength_label": lam_label,
        "two_theta_range": [lo, hi],
        "formula": structure.composition.reduced_formula,
        "partial_occupancy": partial,
        "occupancy_treatment": occupancy_treatment,
        "warnings": warnings_out,
    }
    if rad != "xray":
        result["radiation"] = rad
    if profile_info is not None:
        result["profile"] = profile_info
        result["profile_csv"] = profile_info["profile_csv"]
    if observed_info is not None:
        result["observed"] = observed_info
    return result


def _plot_xrd(
    peaks: list[dict[str, Any]],
    families_per_peak: list[list[tuple[int, ...]]],
    structure: Structure,
    two_theta_range: tuple[float, float],
    lam: float,
    lam_label: str,
    label_top_n: int,
    png_path: Path,
    svg_path: Path,
    curve: tuple[np.ndarray, np.ndarray] | None = None,
    radiation: str = "xray",
) -> None:
    _apply_style()

    fig_dpi = _render_dpi()
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=fig_dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    lo, hi = two_theta_range
    xs = [p["two_theta_deg"] for p in peaks]
    ys = [p["intensity_rel"] for p in peaks]
    if curve is None:
        if peaks:
            ax.vlines(xs, 0.0, ys, colors="#1f3a93", linewidth=1.6, zorder=3)
    else:
        ax.plot(curve[0], curve[1], color="#1f3a93", linewidth=1.2, zorder=3)
        if peaks:  # Bragg positions as short ticks under the baseline
            ax.vlines(xs, -5.0, -1.5, colors="#c0392b", linewidth=0.9, zorder=3)
    ax.axhline(0.0, color="#111111", linewidth=0.8, zorder=2)

    def _height_near(x0: float, window: float) -> float:
        """Top of what is drawn around x0 (the sticks, or the curve)."""
        if curve is None:
            return max(ys[j] for j in range(len(peaks)) if abs(xs[j] - x0) < window)
        m = np.abs(curve[0] - x0) < window
        return float(curve[1][m].max()) if np.any(m) else 0.0

    # hkl labels on the strongest peaks (>= 1 % only). "*" = several families at one 2θ.
    # Labels that would collide with an already placed one are lifted above it.
    order = [i for i in sorted(range(len(peaks)), key=lambda i: -ys[i]) if ys[i] >= 1.0][:label_top_n]
    span = hi - lo
    placed: list[tuple[float, float]] = []  # (x, label baseline y) in data units
    for i in sorted(order, key=lambda i: -ys[i]):
        fams = families_per_peak[i]
        text = _hkl_mathtext(fams[0]) + ("*" if len(fams) > 1 else "")
        frac = (xs[i] - lo) / span
        ha = "right" if frac > 0.97 else ("left" if frac < 0.03 else "center")
        # clear the sticks of neighbouring peaks under the label, then other labels
        y_peak = ys[i] if curve is None else _height_near(xs[i], 0.004 * span)
        y_text = _height_near(xs[i], 0.03 * span) + 1.5
        for px, py in sorted(placed, key=lambda t: t[1]):
            if abs(px - xs[i]) < 0.045 * span and abs(py - y_text) < 5.0:
                y_text = py + 5.0
        placed.append((xs[i], y_text))
        ax.annotate(
            text,
            xy=(xs[i], y_peak),
            xytext=(xs[i], y_text),
            textcoords="data",
            ha=ha,
            va="bottom",
            fontsize=9,
            zorder=4,
            arrowprops=(
                dict(arrowstyle="-", color="#888888", linewidth=0.6, shrinkA=0, shrinkB=1)
                if y_text - y_peak > 3.0
                else None
            ),
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(0.0 if curve is None else -7.0, 118.0)
    ax.set_xlabel(r"2$\theta$ (deg)", fontsize=11.5, labelpad=8)
    ax.set_ylabel("Intensity (a.u.)", fontsize=11.5, labelpad=8)
    ax.set_title(_title(structure, lam, lam_label, radiation), fontsize=12.0, pad=10)
    _style_axes(ax)

    if any(len(f) > 1 for f in (families_per_peak[i] for i in order)):
        ax.text(0.99, 0.97, r"* several hkl families at the same 2$\theta$ (see CSV)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(png_path, dpi=fig_dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = True
    plt.rcParams["svg.fonttype"] = "none"


def _style_axes(ax: Any) -> None:
    ax.grid(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="major", direction="in", top=True, right=True, bottom=True, left=True,
                   length=6.0, width=1.1, labelsize=10)
    ax.tick_params(which="minor", direction="in", top=True, right=True, bottom=True, left=True,
                   length=3.0, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#111111")


def _title(structure: Structure, lam: float, lam_label: str, radiation: str, what: str = "simulated") -> str:
    if radiation == "neutron":
        return rf"{_formula_mathtext(structure)}: {what} powder neutron diffraction ($\lambda$ = {lam:.5f} Å)"
    return rf"{_formula_mathtext(structure)}: {what} powder XRD ({_source_mathtext(lam_label)}, $\lambda$ = {lam:.5f} Å)"


def _plot_observed(
    cmp_df: pd.DataFrame,
    bragg_positions: list[float],
    structure: Structure,
    two_theta_range: tuple[float, float],
    lam: float,
    lam_label: str,
    radiation: str,
    observed_name: str,
    r_factor_percent: float,
    png_path: Path,
    svg_path: Path,
) -> None:
    """Rietveld-style overlay: observed (points), calculated (line), Bragg ticks, difference."""
    _apply_style()
    fig_dpi = _render_dpi()
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=fig_dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = cmp_df["two_theta_deg"].to_numpy()
    obs = cmp_df["observed"].to_numpy()
    calc = cmp_df["calculated"].to_numpy()
    diff = cmp_df["difference"].to_numpy()
    top = float(max(obs.max(), calc.max()))
    bottom = float(min(obs.min(), calc.min()))
    span_y = top - bottom if top > bottom else 1.0
    tick_hi = bottom - 0.04 * span_y
    tick_lo = bottom - 0.10 * span_y
    diff_offset = bottom - 0.16 * span_y - float(max(diff.max(), 0.0))

    ax.plot(x, obs, linestyle="none", marker="o", markersize=2.2, markerfacecolor="none",
            markeredgecolor="#111111", markeredgewidth=0.6, label="Observed", zorder=3)
    ax.plot(x, calc, color="#c0392b", linewidth=1.1, label="Calculated", zorder=4)
    lo, hi = two_theta_range
    if bragg_positions:
        ax.vlines(bragg_positions, tick_lo, tick_hi, colors="#2e7d32", linewidth=0.9, zorder=3,
                  label="Bragg positions")
    ax.plot(x, diff + diff_offset, color="#1f3a93", linewidth=0.9, label="Obs. − Calc.", zorder=3)
    ax.axhline(diff_offset, color="#888888", linewidth=0.6, zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(diff_offset + float(min(diff.min(), 0.0)) - 0.05 * span_y, top + 0.12 * span_y)
    ax.set_xlabel(r"2$\theta$ (deg)", fontsize=11.5, labelpad=8)
    ax.set_ylabel("Intensity (a.u.)", fontsize=11.5, labelpad=8)
    ax.set_title(_title(structure, lam, lam_label, radiation, what="observed vs calculated"), fontsize=12.0, pad=10)
    _style_axes(ax)
    ax.tick_params(axis="y", which="both", labelleft=False)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    r_txt = f"R = {r_factor_percent:.2f} %" if math.isfinite(r_factor_percent) else "R = n/a"
    ax.text(0.01, 0.97, f"{observed_name}\n{r_txt} (fit: scale + constant background)",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(png_path, dpi=fig_dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

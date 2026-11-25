"""CrystalAnalyzerMCP: Dual-mode CLI and FastMCP server for crystal structure analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure the `crystal` package (repo root) can be imported regardless of the current working directory.
_MODULE_DIR = str(Path(__file__).resolve().parents[1])
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from shido.analyzer import (  # noqa: E402
    analyze_structure_data,
    build_ni2mnga_b2_disordered,
    build_ni2mnga_structure,
    generate_demo_cif,
    load_structure_robust,
    load_structures_all,
    sanitize_name,
    structure_stem,
)
from shido.core import __version__  # noqa: E402
from shido.core import recipe as _recipe  # noqa: E402
from shido.core.geometry import measure_geometry as _measure_geometry  # noqa: E402
from shido.core.output_paths import (  # noqa: E402
    DATA_DIR,
    FIG_DIR,
    INDEX_NAME,
    MODEL_DIR,
    OUTPUT_ROOT_NAME,
    STRUCT_DIR,
    new_run_dir,
    write_index_html,
)
from shido.core.symmetry_info import compute_symmetry_info, write_symmetry_report  # noqa: E402
from shido.core.cell_transform import transform_cell as _transform_cell, write_transformed  # noqa: E402
from shido.export3d import export_html, export_stl, split_stl_paths  # noqa: E402
from shido.xrd import compute_xrd  # noqa: E402
from shido.ca_sweep import (  # noqa: E402
    generate_ca_sweep,
    generate_demo_ca_energy_csv,
    plot_ca_energy_curve,
)
from shido.heusler_builder import generate_heusler_models  # noqa: E402
from shido.phase_compare import (  # noqa: E402
    build_ni2mnga_martensite_l10,
    compare_phases,
)
from shido.renderer import RENDER_STYLES, render_batch_gallery, render_bulk_views  # noqa: E402
from shido.renderer import (  # noqa: E402
    POLYHEDRA_AUTO_LABEL,
    normalize_polyhedra_center,
    resolve_show_polyhedra,
)
from shido.slab_cutter import MAJOR_PLANES, cut_slabs, resolve_miller_index  # noqa: E402
from shido.theme_manager import list_themes, set_active_theme  # noqa: E402
from shido import settings as cfg  # noqa: E402


def _or_setting(value: Any, key: str) -> Any:
    """引数が None なら前提数値（settings）の値を使う。"""
    return cfg.get(key) if value is None else value


def _finish_run(
    result: dict[str, Any],
    run_dir: str | Path,
    title: str,
    source: Any = None,
    settings: dict[str, Any] | None = None,
    write_index: bool = True,
) -> dict[str, Any]:
    """Write recipe.json and はじめに読む.html in run_dir; add ``output_dir`` / ``index_html``
    (and ``recipe`` when the call is recorded) to result."""
    run = Path(run_dir).resolve()
    result["output_dir"] = str(run)
    result["index_html"] = (
        str(write_index_html(run, title=title, source=str(source) if source else None, settings=settings,
                             result=result))
        if write_index
        else None
    )
    if write_index and _recipe.current() is not None:
        result["recipe"] = str(run / _recipe.RECIPE_NAME)
    return result


def _print_saved(result: dict[str, Any]) -> None:
    """CLI の最後に、保存先と はじめに読む.html の場所を出す。"""
    print(f"保存先: {result['output_dir']}")
    if result.get("index_html"):
        print(f"{INDEX_NAME}: {result['index_html']}")


def _shown_options(options: dict[str, Any]) -> dict[str, Any]:
    """はじめに読む.html に書く設定（付けたものだけ）。"""
    keep: dict[str, Any] = {}
    for k, v in options.items():
        if k in {"volume", "isovalue"} and v is not None:
            keep[k] = str(v) if k == "volume" else v
        elif k in {"iso_color", "iso_color_negative", "iso_opacity", "iso_both_signs"}:
            continue
        elif k == "poscar_cell" and v == "input":
            continue
        elif k == "style" and v in (None, "ball-stick"):
            continue
        elif v not in (None, False, "", [], {}):
            keep[k] = v
    return keep

BATCH_SUMMARY_COLUMNS: list[str] = [
    "index",
    "cif_file",
    "status",
    "formula",
    "reduced_formula",
    "spacegroup_symbol",
    "spacegroup_number",
    "a_angstrom",
    "b_angstrom",
    "c_angstrom",
    "volume_angstrom3",
    "density_g_cm3",
    "is_ordered",
    "disordered_sites",
    "parse_warnings",
    "error",
    "data_block",
]


def _unpack_robust_load(loaded: Any) -> tuple[Any, list[str]]:
    """Normalize load_structure_robust return value to (structure, parse_warnings)."""
    if isinstance(loaded, tuple) and len(loaded) == 2:
        struct, parse_warnings = loaded
        return struct, list(parse_warnings or [])
    return loaded, []


def _select_gallery_image(bulk_res: dict[str, Any]) -> str | None:
    """Pick the best preview image (figure_overview, else iso view, else first image)."""
    overview = bulk_res.get("figure_overview")
    if overview:
        return str(overview)
    images = [str(p) for p in bulk_res.get("images", []) if p]
    for img in images:
        if "iso" in Path(img).name.lower():
            return img
    return images[0] if images else None


def _discover_cifs(input_dir: Path, exclude_dir: Path | None = None) -> list[Path]:
    """Recursively find structure files under input_dir, skipping any output_* dirs.

    Picks up *.cif plus (batch 9) *.mcif, *.vasp, *.poscar, *.contcar, *.xsf, *.cssr and
    extensionless POSCAR*/CONTCAR* names (`analyzer.is_structure_file`)."""
    from shido.analyzer import is_structure_file

    in_resolved = input_dir.resolve()
    excl_resolved = exclude_dir.resolve() if exclude_dir is not None else None
    found: list[Path] = []
    for candidate in in_resolved.rglob("*"):
        if not candidate.is_file() or not is_structure_file(candidate):
            continue
        try:
            rel_parts = candidate.relative_to(in_resolved).parts[:-1]
        except ValueError:
            rel_parts = candidate.parts[:-1]
        if any(part.lower().startswith("output_") or part == OUTPUT_ROOT_NAME for part in rel_parts):
            continue
        if excl_resolved is not None:
            try:
                candidate.resolve().relative_to(excl_resolved)
                continue
            except ValueError:
                pass
        found.append(candidate)
    found.sort(key=lambda p: str(p.relative_to(in_resolved)).lower())
    return found


POSCAR_CELL_CHOICES: tuple[str, ...] = ("input", "conventional", "primitive")


def _vector_kwargs(
    vectors: Any = None,
    basis: str | None = None,
    scale: float | None = None,
    color: str | None = None,
    radius: float | None = None,
) -> dict[str, Any]:
    """Batch-14 arrow options for render_* / export_html; {} without vectors (defaults unchanged)."""
    if not vectors:
        return {}
    return {
        "vectors": vectors,
        "vector_basis": basis,
        "vector_scale": scale,
        "vector_color": color,
        "vector_radius": radius,
    }


def _volume_kwargs(
    volume: Any = None,
    isovalue: Any = None,
    color: Any = None,
    color_negative: Any = None,
    opacity: float | None = None,
    both_signs: bool = False,
) -> dict[str, Any]:
    """Batch-17 isosurface options for render_*; {} without a volume (defaults unchanged)."""
    if not volume:
        return {}
    return {
        "volume": volume,
        "isovalue": isovalue,
        "iso_color": color,
        "iso_color_negative": color_negative,
        "iso_opacity": opacity,
        "iso_both_signs": bool(both_signs),
    }


def _ellipsoid_kwargs(style: str | None, adp: Any = None, probability: float | None = None) -> dict[str, Any]:
    """Batch-16 ellipsoid options (adp = CIF path, probability %); {} unless style is ellipsoid."""
    from shido.renderer import normalize_style

    if normalize_style(style) != "ellipsoid":
        return {}
    return {"adp": adp, "probability": probability}


def _molecule_render_kwargs(cutoff_planes: Any = None, whole_molecules: bool = False, hbonds: Any = None) -> dict[str, Any]:
    """Batch-23 drawing options for the renderer; {} when none is given (default look)."""
    out: dict[str, Any] = {}
    if cutoff_planes:
        out["cutoff_planes"] = [cutoff_planes] if isinstance(cutoff_planes, str) else list(cutoff_planes)
    if whole_molecules:
        out["whole_molecules"] = True
    if hbonds:
        out["hbonds"] = [hbonds] if isinstance(hbonds, str) else list(hbonds)
    return out


def _overlay_specs(
    paths: Any = None,
    shifts: Any = None,
    rotates: Any = None,
    shift_unit: str | None = None,
    opacity: float | None = None,
    color: str | None = None,
    about: str | None = None,
) -> list[dict[str, Any]] | None:
    """Batch-21 overlay specs (JSON-friendly; renderer loads the files). The i-th shift /
    rotate goes with the i-th path (missing ones = none); None when no path is given."""
    if not paths:
        return None
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    shifts = list(shifts or [])
    rotates = [rotates] if isinstance(rotates, str) else list(rotates or [])
    if len(shifts) > len(paths) or len(rotates) > len(paths):
        raise ValueError("more --overlay-shift / --overlay-rotate than --overlay files.")
    specs: list[dict[str, Any]] = []
    for i, p in enumerate(paths):
        if not Path(p).is_file():
            raise FileNotFoundError(f"overlay file not found: {p}")
        spec: dict[str, Any] = {"path": str(p)}
        if i < len(shifts) and shifts[i] is not None:
            spec["shift"] = [float(v) for v in (shifts[i].replace(",", " ").split()
                                                if isinstance(shifts[i], str) else shifts[i])]
        if i < len(rotates) and rotates[i]:
            spec["rotate"] = rotates[i]
        for key, val in (("shift_unit", shift_unit), ("opacity", opacity), ("color", color), ("about", about)):
            if val is not None:
                spec[key] = val
        specs.append(spec)
    return specs


def _write_molecule_outputs(structure: Any, output_dir: str | Path, molecules: bool, hbonds: Any) -> dict[str, Any]:
    """数値/molecules.csv (molecules) and 数値/hbonds.csv (hbonds); counts for the result."""
    from collections import Counter

    from shido.core.molecules import find_hbonds, find_molecules, write_hbonds_csv, write_molecules_csv

    data_dir = Path(output_dir) / DATA_DIR
    out: dict[str, Any] = {}
    if molecules:
        mols = find_molecules(structure)
        out["molecules_csv"] = str(write_molecules_csv(structure, mols, data_dir / "molecules.csv").resolve())
        out["num_molecules"] = len(mols)
        out["molecule_formulas"] = dict(Counter(m["formula"] for m in mols))
        out["num_extended_networks"] = sum(1 for m in mols if m["extended"])
    if hbonds:
        hb = find_hbonds(structure, [hbonds] if isinstance(hbonds, str) else list(hbonds))
        out["hbonds_csv"] = str(write_hbonds_csv(structure, hb, data_dir / "hbonds.csv").resolve())
        out["num_hbonds"] = len(hb)
    return out


def _load_structure_or_volume(path: str | Path) -> tuple[Any, list[str], str | None]:
    """(structure, warnings, volume path or None): a CHGCAR / LOCPOT / ELFCAR / Cube /
    XSF grid file gives the structure stored in it (and itself as the volume)."""
    from shido.core.volumetric import is_volume_file, read_volume

    p = Path(path)
    if is_volume_file(p):
        grid = read_volume(p)
        if grid.structure is None:
            raise ValueError(f"{p.name} has no atoms; give a CIF and --volume {p.name}.")
        return grid.structure, [f"Structure read from the volumetric file {p.name}."], str(p)
    struct, warns = _unpack_robust_load(load_structure_robust(p))
    return struct, warns, None


def _polyhedra_setting(show_polyhedra: bool | None) -> bool | str:
    """show_polyhedra as written in はじめに読む.html (None = auto)."""
    return POLYHEDRA_AUTO_LABEL if show_polyhedra is None else bool(show_polyhedra)


def run_bulk_analysis(
    structure: Any,
    output_dir: str | Path,
    show_polyhedra: bool | None = None,
    parse_warnings: list[str] | None = None,
    transparent: bool | None = None,
    poscar_cell: str = "input",
    xrd: bool = False,
    wavelength: str | float | None = None,
    stl: bool = False,
    stl_split: bool = False,
    stl_max_mm: float | None = None,
    html: bool = False,
    style: str | None = None,
    view_dir: list[float] | None = None,
    view_normal: list[float] | None = None,
    perspective: bool = False,
    labels: str | None = None,
    bonds: list[str] | str | None = None,
    polyhedra_center: list[str] | str | None = None,
    bond_color: str | None = None,
    frac_range: list[float] | None = None,
    image_formats: str | list[str] | None = None,
    show_planes: list[Any] | str | None = None,
    xrd_options: dict[str, Any] | None = None,
    vectors: list[Any] | str | None = None,
    vector_basis: str | None = None,
    vector_scale: float | None = None,
    vector_color: str | None = None,
    vector_radius: float | None = None,
    volume: str | Path | None = None,
    isovalue: Any = None,
    iso_color: Any = None,
    iso_color_negative: Any = None,
    iso_opacity: float | None = None,
    iso_both_signs: bool = False,
    adp: Any = None,
    probability: float | None = None,
    animate: str | None = None,
    animate_frames: int | None = None,
    animate_axis: str | None = None,
    cutoff_planes: list[str] | str | None = None,
    molecules: bool = False,
    whole_molecules: bool = False,
    hbonds: list[str] | str | None = None,
    overlays: list[Any] | None = None,
    supercell: Any = None,
    export_formats: str | list[str] | None = None,
) -> dict[str, Any]:
    """Run quantitative bulk analysis and multi-angle PyVista rendering.

    ``overlays`` (see renderer.normalize_overlays, e.g. [{"path": "molecule.cif",
    "shift": [0, 0, 5], "rotate": "0 0 1 45"}]) draws those structures on every view.

    ``cutoff_planes`` (e.g. ["1 1 1@0.5"]) draws only atoms with h*x+k*y+l*z <= D;
    ``molecules`` writes 数値/molecules.csv (covalent connected components, composition,
    centre of mass); ``whole_molecules`` also draws every molecule whole; ``hbonds``
    (e.g. ["O-H...O:2.5:120"]) writes 数値/hbonds.csv and draws them dashed.

    ``animate`` 'gif' / 'mp4' also writes ``図/rotation.gif`` (.mp4): ``animate_frames``
    (72) views turning once around ``animate_axis`` ('up' = screen vertical, a/b/c, x/y/z),
    seen from the isometric view (or view_dir / view_normal), same style / range / labels.

    ``style='ellipsoid'`` draws displacement ellipsoids of ``probability`` % (default 50)
    from the ADPs of the CIF ``adp`` (path); without ``adp`` atoms are small spheres.

    ``volume`` (CHGCAR / LOCPOT / ELFCAR / *.cube / *.xsf / *.grd) overlays its
    isosurfaces at ``isovalue`` (e.g. "0.05,0.1"; CHGCAR in e/Å^3) on every view;
    ``iso_color`` / ``iso_color_negative`` / ``iso_opacity`` / ``iso_both_signs`` (±level).

    ``vectors`` (e.g. ["Mn:0 0 4"] or ["magmom"] for mCIF moments, see core.vectors)
    draws an arrow through those atoms on every view and in the HTML viewer;
    ``vector_basis`` 'crystal' (default, along a/|a| b/|b| c/|c| like mCIF) / 'cartesian',
    ``vector_scale`` Å per unit (1.0), ``vector_color``, ``vector_radius`` Å (0.08).

    ``xrd_options`` (e.g. {"profile": "pseudo_voigt", "fwhm": 0.1, "radiation": "neutron",
    "caglioti": (U, V, W), "observed": "meas.xy"}) is passed on to ``xrd.compute_xrd``.

    ``show_planes`` (e.g. ["1 1 1", "1 1 0@0.5:#ff0000:0.4"]) draws those (hkl)
    planes on every view ('h k l[@d][:color][:opacity]', see renderer.normalize_planes).

    ``frac_range`` [x0, x1, y0, y1, z0, z1] draws all atoms in that fractional box
    (e.g. -0.5..1.5); ``image_formats`` "tiff,jpeg,pdf,eps,svg" also writes every
    figure in those formats (PDF / EPS / SVG embed the bitmap, they are not vector).

    ``show_polyhedra``: None (default) = auto, polyhedra only for structures with an
    anion-forming element (or with ``polyhedra_center``); True / False forces it.

    ``bonds`` (e.g. ["Mn-Ga:0:3.0"]) replaces the automatic bond search (CSV, angles,
    images, STL all follow it); ``polyhedra_center`` (e.g. "Mn") picks the polyhedron
    centres; ``bond_color`` "split" colours ball-stick bonds by element.

    ``style``: 'ball-stick' (default, None), 'space-filling', 'stick' or 'wireframe'.
    ``view_dir`` [u v w] / ``view_normal`` (h k l) add ``図/view_custom.png`` seen
    from that direction; ``perspective`` and ``labels`` ('site' / 'element') apply to all views.

    v3 options (all off by default, so the default outputs are unchanged):
    ``xrd`` (powder XRD CSV + figure, ``wavelength`` = 'CuKa' / 'MoKa' / Å),
    ``stl`` (3D-print model ``3D/structure.stl``; ``stl_split`` adds one STL per
    element, ``stl_max_mm`` shrinks it to fit), ``html`` (rotatable offline
    ``3D/structure.html``), ``poscar_cell`` ('input' / 'conventional' / 'primitive').

    Returns a JSON-serializable dictionary excluding raw numpy arrays.
    """
    transparent = bool(_or_setting(transparent, "render.transparent_background"))
    supercell_rep: dict[str, Any] | None = None
    if supercell is not None:
        structure, supercell_rep = _transform_cell(structure, supercell=supercell)
        if supercell_rep.get("warnings"):
            parse_warnings = [*list(parse_warnings or []), *supercell_rep["warnings"]]
    show_polyhedra = resolve_show_polyhedra(
        show_polyhedra, structure, normalize_polyhedra_center(polyhedra_center)
    )
    data_res = analyze_structure_data(
        structure,
        output_dir,
        parse_warnings=parse_warnings,
        poscar_cell=poscar_cell,
        bond_specs=bonds,
    )
    vec_kw = _vector_kwargs(vectors, vector_basis, vector_scale, vector_color, vector_radius)
    vol_kw = _volume_kwargs(volume, isovalue, iso_color, iso_color_negative, iso_opacity, iso_both_signs)
    if vol_kw:
        from shido.core.volumetric import read_volume

        vol_grid = read_volume(volume)
        vol_kw["volume"] = vol_grid
    img_map = render_bulk_views(
        structure,
        output_dir,
        render_bonds=data_res["render_bonds"],
        polyhedra_envs=data_res["polyhedra_envs"],
        show_polyhedra=show_polyhedra,
        summary=data_res["summary"],
        transparent_background=transparent,
        style=style,
        view_dir=view_dir,
        view_normal=view_normal,
        perspective=perspective,
        labels=labels,
        polyhedra_center=polyhedra_center,
        bond_color=bond_color,
        frac_range=frac_range,
        image_formats=image_formats,
        planes=show_planes,
        **vec_kw,
        **vol_kw,
        **_ellipsoid_kwargs(style, adp, probability),
        **_molecule_render_kwargs(cutoff_planes, whole_molecules, hbonds),
        **({"overlays": overlays} if overlays else {}),
    )
    figure_overview =img_map.get("figure_overview") or img_map.get("overview")
    result: dict[str, Any] = {
        "output_dir": data_res["output_dir"],
        "lattice_summary_csv": data_res["lattice_summary_csv"],
        "bond_lengths_csv": data_res["bond_lengths_csv"],
        "bond_angles_csv": data_res.get("bond_angles_csv"),
        "atomic_positions_csv": data_res.get("atomic_positions_csv"),
        "cif_file": data_res.get("cif_file"),
        "poscar_file": data_res.get("poscar_file"),
        "poscar_cell": data_res.get("poscar_cell", "input"),
        "poscar_num_sites": data_res.get("poscar_num_sites"),
        "figure_overview": figure_overview,
        "figure_overview_svg": img_map.get("figure_overview_svg"),
        "view_custom": img_map.get("view_custom"),
        "images": list(img_map.values()),
        "summary": data_res["summary"],
        "num_bonds": data_res["num_bonds"],
        "xrd_peaks_csv": None,
        "xrd_figure_png": None,
        "xrd_figure_svg": None,
        "xrd_n_peaks": None,
        "xrd_warnings": [],
        "stl_file": None,
        "stl_split_files": [],
        "html_file": None,
    }
    if supercell_rep is not None:
        result["supercell"] = supercell_rep["supercell"]
    if vol_kw:
        from shido.core.volumetric import normalize_isovalues

        result["volume"] = {
            **vol_grid.summary(),
            "isovalues": normalize_isovalues(isovalue, vol_grid),
            "both_signs": bool(iso_both_signs),
        }
    if molecules or whole_molecules or hbonds:
        result.update(_write_molecule_outputs(structure, output_dir, bool(molecules or whole_molecules), hbonds))

    if xrd:
        xrd_res = compute_xrd(structure, output_dir, wavelength=wavelength, **(xrd_options or {}))
        result.update(
            xrd_peaks_csv=xrd_res.get("xrd_peaks_csv"),
            xrd_figure_png=xrd_res.get("figure_png"),
            xrd_figure_svg=xrd_res.get("figure_svg"),
            xrd_n_peaks=xrd_res.get("n_peaks"),
            xrd_warnings=list(xrd_res.get("warnings") or []),
        )
        if xrd_res.get("profile_csv"):
            result["xrd_profile_csv"] = xrd_res["profile_csv"]
        if xrd_res.get("observed"):
            result["xrd_observed"] = xrd_res["observed"]

    models_dir = Path(output_dir) / MODEL_DIR
    if stl or stl_split:
        models_dir.mkdir(parents=True, exist_ok=True)
        stl_path = export_stl(
            structure,
            models_dir / "structure.stl",
            render_bonds=data_res["render_bonds"],
            split_by_element=bool(stl_split),
            max_size_mm=stl_max_mm,
        )
        result["stl_file"] = str(Path(stl_path).resolve())
        if stl_split:
            result["stl_split_files"] = [
                str(p.resolve()) for p in split_stl_paths(stl_path, structure) if p.is_file()
            ]
    if html:
        models_dir.mkdir(parents=True, exist_ok=True)
        html_path = export_html(
            structure,
            models_dir / "structure.html",
            render_bonds=data_res["render_bonds"],
            **vec_kw,
        )
        result["html_file"] = str(Path(html_path).resolve())
    if export_formats:
        from shido.core.exporters import export_structures

        exp_res = export_structures([("structure", structure)], export_formats, Path(output_dir) / STRUCT_DIR)
        exp_files = dict(exp_res["structures"][0]["files"])
        wrl = exp_files.get("wrl")
        if wrl:
            models_dir.mkdir(parents=True, exist_ok=True)
            dest = models_dir / Path(wrl).name
            Path(wrl).replace(dest)
            exp_files["wrl"] = str(dest.resolve())
        if exp_res.get("multicif"):
            exp_files["multicif"] = exp_res["multicif"]
        result["exported_files"] = exp_files
        if exp_res.get("warnings"):
            result["export_warnings"] = list(exp_res["warnings"])
    if animate:
        from shido.core.animation import render_rotation_animation
        from shido.renderer import _format_bulk_caption, resolve_view_direction

        if view_dir is not None or view_normal is not None:
            cam_dir, up_dir, _ = resolve_view_direction(structure, view_dir=view_dir, view_normal=view_normal)
        else:  # same oblique direction as view_isometric.png
            a_vec, b_vec, c_vec = structure.lattice.matrix
            cam_dir, up_dir = a_vec + 0.36 * b_vec + 0.24 * c_vec, c_vec
        result["animation"] = render_rotation_animation(
            structure, Path(output_dir) / FIG_DIR / "rotation.gif", cam_dir, up_dir,
            fmt=animate, frames=animate_frames, axis=animate_axis,
            render_bonds=data_res["render_bonds"], polyhedra_envs=data_res["polyhedra_envs"],
            show_polyhedra=show_polyhedra, polyhedra_center=polyhedra_center, bond_color=bond_color,
            caption_text=_format_bulk_caption(structure, summary=data_res["summary"]),
            transparent_background=transparent, style=style, perspective=perspective,
            labels=labels, frac_range=frac_range, planes=show_planes,
            **vec_kw, **vol_kw, **_ellipsoid_kwargs(style, adp, probability),
        )
    return result


def run_demo(
    miller_index: tuple[int, int, int] = (1, 1, 1),
    base_dir: str | Path | None = None,
    show_polyhedra: bool | None = None,
    filter_polar: bool | None = None,
    deduplicate: bool | None = None,
    transparent: bool | None = None,
    **bulk_options: Any,
) -> dict[str, Any]:
    """Generate the Ni2MnGa L21 Heusler demo CIF and run bulk + slab analysis.

    ``bulk_options`` (xrd, wavelength, stl, stl_split, stl_max_mm, html, poscar_cell)
    are passed to `run_bulk_analysis`.

    Without ``base_dir`` everything goes to a new folder ``出力/<日時>_demo_Ni2MnGa/``
    (slabs in its ``slab_<hkl>/``); with it, ``<base_dir>/output_Ni2MnGa`` and
    ``<base_dir>/output_slab_<hkl>`` as before.
    """
    transparent = bool(_or_setting(transparent, "render.transparent_background"))
    h, k, l = miller_index
    if base_dir:
        root = Path(base_dir)
        bulk_dir = root / "output_Ni2MnGa"
        slab_dir = root / f"output_slab_{h}{k}{l}"
    else:
        bulk_dir = new_run_dir("demo", "Ni2MnGa")
        slab_dir = bulk_dir / f"slab_{h}{k}{l}"

    struct, cif_path = generate_demo_cif(bulk_dir / STRUCT_DIR / "Ni2MnGa_L21.cif")
    bulk_res = run_bulk_analysis(
        struct,
        bulk_dir,
        show_polyhedra=show_polyhedra,
        transparent=transparent,
        **bulk_options,
    )
    slab_res = cut_slabs(
        struct,
        miller_index,
        slab_dir,
        filter_polar=filter_polar,
        deduplicate=deduplicate,
        transparent_background=transparent,
    )

    settings = {"input": "Ni2MnGa L21（デモ用に自動で作った構造）", "miller_index": [h, k, l],
                "show_polyhedra": _polyhedra_setting(show_polyhedra), "transparent": transparent,
                **_shown_options(bulk_options)}
    return _finish_run(
        {"demo_cif": str(cif_path), "bulk_analysis": bulk_res, "slab_generation": slab_res},
        bulk_dir, "デモ：Ni2MnGa の解析とスラブ切り出し", source="Ni2MnGa_L21.cif（デモ）",
        settings=settings,
    )


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    miller_index: tuple[int, int, int] | None = None,
    show_polyhedra: bool | None = None,
    filter_polar: bool | None = None,
    deduplicate: bool | None = None,
    transparent: bool | None = None,
    all_blocks: bool = False,
    **bulk_options: Any,
) -> dict[str, Any]:
    """Batch-analyze all CIF files under input_dir with fault tolerance.

    Generates per-structure outputs, a consolidated ``batch_summary.csv``, and
    a multi-panel ``batch_gallery.png`` overview. ``all_blocks=True`` processes every
    data block of a multi-block CIF into its own folder (``<NNN>_<stem>_<block>``).
    ``bulk_options`` (xrd, wavelength, stl, stl_split, stl_max_mm, html, poscar_cell)
    are passed to `run_bulk_analysis`.
    """
    transparent = bool(_or_setting(transparent, "render.transparent_background"))
    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Batch input directory not found: {in_dir}")

    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else new_run_dir("batch", sanitize_name(in_dir.name))
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cif_files = _discover_cifs(in_dir, exclude_dir=out_dir)

    rows: list[dict[str, Any]] = []
    gallery_entries: list[tuple[str, str]] = []
    items: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0

    def _record_error(idx: int, cif_file: Path, struct_out: Path, exc: Exception, block: str) -> None:
        err_msg = f"{type(exc).__name__}: {exc}"
        row = {col: None for col in BATCH_SUMMARY_COLUMNS}
        row.update(
            index=idx,
            cif_file=str(cif_file.resolve()),
            status="error",
            formula="",
            reduced_formula="",
            spacegroup_symbol="",
            parse_warnings="",
            error=err_msg,
            data_block=block,
        )
        rows.append(row)
        items.append(
            {
                "index": idx,
                "cif_file": str(cif_file.resolve()),
                "data_block": block or None,
                "status": "error",
                "output_dir": str(struct_out.resolve()),
                "parse_warnings": [],
                "error": err_msg,
            }
        )

    for idx, cif_file in enumerate(cif_files, start=1):
        base_name = f"{idx:03d}_{sanitize_name(structure_stem(cif_file))}"
        try:
            if all_blocks:
                entries = load_structures_all(cif_file)
            else:
                one, one_warns = _unpack_robust_load(load_structure_robust(cif_file))
                entries = [(one, one_warns, "")]
        except Exception as exc:
            _record_error(idx, cif_file, out_dir / base_name, exc, "")
            failed += 1
            continue

        multi = len(entries) > 1
        for struct, parse_warnings, block in entries:
            block = block if multi else ""
            struct_out = out_dir / (f"{base_name}_{sanitize_name(block)}" if multi else base_name)
            try:
                per_bulk_options = dict(bulk_options)
                if "adp" in per_bulk_options and per_bulk_options["adp"] is None:
                    per_bulk_options["adp"] = str(cif_file)
                bulk_res = run_bulk_analysis(
                    struct,
                    struct_out,
                    show_polyhedra=show_polyhedra,
                    parse_warnings=parse_warnings,
                    transparent=transparent,
                    **per_bulk_options,
                )

                slab_res: dict[str, Any] | None = None
                if miller_index is not None:
                    h, k, l = miller_index
                    slab_out = struct_out / f"slab_{h}{k}{l}"
                    slab_res = cut_slabs(
                        struct,
                        (h, k, l),
                        slab_out,
                        filter_polar=filter_polar,
                        deduplicate=deduplicate,
                        transparent_background=transparent,
                    )

                summary = bulk_res.get("summary", {})
                reduced_formula = str(
                    summary.get(
                        "reduced_formula",
                        getattr(struct.composition, "reduced_formula", ""),
                    )
                )
                is_ordered = summary.get("is_ordered", bool(getattr(struct, "is_ordered", True)))
                disordered_sites = summary.get(
                    "disordered_sites",
                    sum(1 for site in struct if not getattr(site, "is_ordered", True)),
                )
                warnings_str = (
                    summary.get("parse_warnings")
                    if isinstance(summary.get("parse_warnings"), str)
                    else "; ".join(str(w) for w in parse_warnings)
                )

                rows.append(
                    {
                        "index": idx,
                        "cif_file": str(cif_file.resolve()),
                        "status": "ok",
                        "formula": summary.get("formula", ""),
                        "reduced_formula": reduced_formula,
                        "spacegroup_symbol": summary.get("spacegroup_symbol", ""),
                        "spacegroup_number": summary.get("spacegroup_number", None),
                        "a_angstrom": summary.get("a_angstrom", None),
                        "b_angstrom": summary.get("b_angstrom", None),
                        "c_angstrom": summary.get("c_angstrom", None),
                        "volume_angstrom3": summary.get("volume_angstrom3", None),
                        "density_g_cm3": summary.get("density_g_cm3", None),
                        "is_ordered": is_ordered,
                        "disordered_sites": disordered_sites,
                        "parse_warnings": warnings_str,
                        "error": "",
                        "data_block": block,
                    }
                )

                preview_png = _select_gallery_image(bulk_res)
                if preview_png:
                    label = f"{structure_stem(cif_file)}:{block}" if multi else structure_stem(cif_file)
                    gallery_entries.append((f"{label} ({reduced_formula})", preview_png))

                item_entry: dict[str, Any] = {
                    "index": idx,
                    "cif_file": str(cif_file.resolve()),
                    "data_block": block or None,
                    "status": "ok",
                    "output_dir": str(struct_out.resolve()),
                    "parse_warnings": parse_warnings,
                    "bulk_analysis": bulk_res,
                    "error": None,
                }
                if slab_res is not None:
                    item_entry["slab_generation"] = slab_res
                items.append(item_entry)
                succeeded += 1
            except Exception as exc:
                _record_error(idx, cif_file, struct_out, exc, block)
                failed += 1

    batch_summary_csv = out_dir / DATA_DIR / "batch_summary.csv"
    batch_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=BATCH_SUMMARY_COLUMNS).to_csv(
        batch_summary_csv, index=False
    )

    batch_gallery_target = out_dir / FIG_DIR / "batch_gallery.png"
    batch_gallery_target.parent.mkdir(parents=True, exist_ok=True)
    batch_gallery_png: str | None = None
    if gallery_entries:
        try:
            rendered = render_batch_gallery(gallery_entries, batch_gallery_target)
            if rendered:
                batch_gallery_png = str(rendered)
            elif batch_gallery_target.is_file():
                batch_gallery_png = str(batch_gallery_target.resolve())
        except Exception:
            batch_gallery_png = (
                str(batch_gallery_target.resolve())
                if batch_gallery_target.is_file()
                else None
            )

    settings = {"input": str(in_dir.resolve()), "miller_index": list(miller_index) if miller_index else None,
                "show_polyhedra": _polyhedra_setting(show_polyhedra), "transparent": transparent, "all_blocks": all_blocks or None,
                **_shown_options(bulk_options)}
    return _finish_run(
        {
            "output_dir": str(out_dir),
            "total_cifs": len(cif_files),
            "succeeded": succeeded,
            "failed": failed,
            "batch_summary_csv": str(batch_summary_csv.resolve()),
            "batch_gallery_png": batch_gallery_png,
            "items": items,
        },
        out_dir,
        f"まとめて解析：{in_dir.name}（{len(cif_files)} 個の CIF、成功 {succeeded}・失敗 {failed}）",
        source=in_dir.resolve(),
        settings=settings,
    )


def create_mcp_server():
    """Create and configure the MCP server (compatible with mcp 2.x MCPServer and 1.x FastMCP)."""
    try:
        from mcp.server.mcpserver import MCPServer as _MCPServer
    except ModuleNotFoundError:
        from mcp.server.fastmcp import FastMCP as _MCPServer

    mcp = _MCPServer("CrystalAnalyzerMCP")
    # Every @mcp.tool() below records its name and arguments, so the run folder gets recipe.json.
    _plain_tool = mcp.tool
    mcp.tool = _recipe.recording_tool_decorator(_plain_tool)

    @_plain_tool()  # not recorded itself: the recipe is the call it repeats
    def rerun_recipe(recipe_path: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make the same result again from a recipe (a run folder or its recipe.json) into a NEW folder.

        overrides change only what you name, e.g. {"style": "space-filling"}
        (a tool argument / command option), {"render.dpi": 600} (a dotted name = assumed parameter).
        Everything else uses the recorded values and settings. A changed input file or a different
        library version gives warnings (in result["rerun"]["warnings"]), not an error.
        """
        return _recipe.rerun(recipe_path, overrides)

    @mcp.tool()
    def analyze_structure(
        cif_path: str,
        output_dir: str | None = None,
        show_polyhedra: bool | None = None,
        transparent: bool | None = None,
        poscar_cell: str = "input",
        style: str = "ball-stick",
        view_dir: list[float] | None = None,
        view_normal: list[float] | None = None,
        perspective: bool = False,
        labels: str | None = None,
        bonds: list[str] | None = None,
        polyhedra_center: str | None = None,
        bond_color: str | None = None,
        frac_range: list[float] | None = None,
        image_formats: str | None = None,
        show_planes: list[str] | None = None,
        vectors: list[str] | None = None,
        vector_basis: str | None = None,
        vector_scale: float | None = None,
        vector_color: str | None = None,
        vector_radius: float | None = None,
        volume_path: str | None = None,
        isovalue: list[float] | str | float | None = None,
        iso_color: str | None = None,
        iso_color_negative: str | None = None,
        iso_opacity: float | None = None,
        iso_both_signs: bool = False,
        probability: float | None = None,
        cutoff_planes: list[str] | None = None,
        molecules: bool = False,
        whole_molecules: bool = False,
        hbonds: list[str] | None = None,
    ) -> dict[str, Any]:
        """Analyze a crystal structure CIF and render multi-angle publication views.

        show_polyhedra: None (default, auto) draws coordination polyhedra only when the
        structure has an anion-forming element (O, N, S, halogens, P, As, Se, Te, C, B, H)
        or polyhedra_center is given, so metals / intermetallics get none; True / False forces it.

        cutoff_planes ["h k l@D", ...] (e.g. ["1 1 1@1.5"]) draws only atoms with
        h*x+k*y+l*z <= D. molecules=True writes 数値/molecules.csv (atoms joined by covalent
        bonds across cell edges: composition, centre of mass, extended network yes/no);
        whole_molecules=True also draws each molecule whole. hbonds ["D-H...A:dmax:angle"]
        (e.g. ["O-H...O:2.5:120"]) writes 数値/hbonds.csv and draws them as dashed lines.

        style "ellipsoid" draws displacement (thermal) ellipsoids from the CIF's U/B/beta
        ADPs at probability % (default 50) with the 3 principal ellipses; non-positive-
        definite ADPs become spheres (warning), atoms without ADPs small spheres.

        volume_path (VASP CHGCAR / LOCPOT / ELFCAR, Gaussian *.cube, XSF DATAGRID_3D, VESTA
        *.grd) overlays isosurfaces at isovalue (e.g. [0.05, 0.1] or "0.05"; CHGCAR in
        e/Å^3 = file value / V_cell, Cube as stored; default 10 % of max |value|) on every
        view. iso_color (default #F2C200; "c1,c2" one per level), iso_color_negative
        (#2F9BD6), iso_opacity (0.55), iso_both_signs also draws -isovalue. cif_path may
        itself be a CHGCAR / Cube / XSF: its structure is drawn with its isosurfaces.

        vectors ["TARGET:X Y Z", ...] (TARGET = element "Mn", site label "Mn1" or index "#3";
        e.g. ["Mn:0 0 4"]) or ["magmom"] (moments of an mCIF / magmom site property) draws an
        arrow through those atoms (magnetic moments) on every view and in the HTML viewer.
        vector_basis "crystal" (default: components along a/|a|, b/|b|, c/|c| in mu_B, as
        mCIF crystalaxis_x/y/z) or "cartesian"; vector_scale Å per unit (default 1.0),
        vector_color (default #E4572E), vector_radius shaft radius Å (default 0.08).

        show_planes ["h k l[@d][:color][:opacity]", ...] (e.g. ["1 1 1", "1 1 0@0.5:#ff0000:0.4"])
        draws translucent (hkl) planes in the unit cell on every view; d is the plane
        h*x+k*y+l*z = d (fractional, i.e. d * d_hkl from the origin; omitted = default).

        frac_range [x0, x1, y0, y1, z0, z1] (e.g. [-0.5, 1.5, -0.5, 1.5, -0.5, 1.5]) draws
        every atom in that fractional range; image_formats "tiff,jpeg,pdf,eps,svg" also
        writes each figure in those formats (PDF / EPS / SVG embed the bitmap, not vector).

        bonds ["A-B:dmin:dmax", ...] (e.g. ["Mn-Ga:0:3.0"]) draws/counts only those bonds
        instead of the automatic search; polyhedra_center "Mn" (or "Mn,Ga") picks the
        polyhedron centres; bond_color "split" colours bonds half-and-half by element.

        style: "ball-stick" (default), "space-filling", "stick", "wireframe" or "ellipsoid".
        view_dir [u, v, w] or view_normal [h, k, l] adds 図/view_custom.png from that
        direction; perspective=True uses a perspective camera; labels "site" / "element"
        writes atom labels on every view.

        poscar_cell selects the cell written to POSCAR: "input" (as given, default),
        "conventional" or "primitive". For partially occupied (disordered) structures
        "input" is recommended, because a standard cell can change the composition.
        """
        transparent = bool(_or_setting(transparent, "render.transparent_background"))
        cif_file = Path(cif_path)
        struct, parse_warnings, own_volume = _load_structure_or_volume(cif_file)
        out_dir = Path(output_dir) if output_dir else new_run_dir("", structure_stem(cif_file))
        vol_kw = _volume_kwargs(
            volume_path or own_volume, isovalue, iso_color, iso_color_negative, iso_opacity,
            iso_both_signs,
        )
        opts: dict[str, Any] = dict(
            poscar_cell=poscar_cell,
            style=style,
            view_dir=view_dir,
            view_normal=view_normal,
            perspective=perspective,
            labels=labels,
            bonds=bonds,
            polyhedra_center=polyhedra_center,
            bond_color=bond_color,
            frac_range=frac_range,
            image_formats=image_formats,
            show_planes=show_planes,
            **_vector_kwargs(vectors, vector_basis, vector_scale, vector_color, vector_radius),
            **vol_kw,
            **_ellipsoid_kwargs(style, str(cif_file), probability),
            **{k: v for k, v in dict(cutoff_planes=cutoff_planes, molecules=molecules,
                                     whole_molecules=whole_molecules, hbonds=hbonds).items() if v},
        )
        res = run_bulk_analysis(
            struct,
            out_dir,
            show_polyhedra=show_polyhedra,
            parse_warnings=parse_warnings,
            transparent=transparent,
            **opts,
        )
        return _finish_run(
            res, out_dir, f"CIF の解析：{cif_file.name}", source=cif_file.resolve(),
            settings={"show_polyhedra": _polyhedra_setting(show_polyhedra), "transparent": transparent, **_shown_options(opts)},
        )

    @mcp.tool()
    def render_view(
        cif_path: str,
        output_path: str | None = None,
        view_dir: list[float] | None = None,
        view_normal: list[float] | None = None,
        perspective: bool = False,
        labels: str | None = None,
        style: str = "ball-stick",
        transparent: bool | None = None,
        bonds: list[str] | None = None,
        polyhedra_center: str | None = None,
        bond_color: str | None = None,
        frac_range: list[float] | None = None,
        image_formats: str | None = None,
        show_planes: list[str] | None = None,
        vectors: list[str] | None = None,
        vector_basis: str | None = None,
        vector_scale: float | None = None,
        vector_color: str | None = None,
        vector_radius: float | None = None,
        volume_path: str | None = None,
        isovalue: list[float] | str | float | None = None,
        iso_color: str | None = None,
        iso_color_negative: str | None = None,
        iso_opacity: float | None = None,
        iso_both_signs: bool = False,
        probability: float | None = None,
        animate: str | None = None,
        animate_frames: int = 72,
        animate_axis: str = "up",
        cutoff_planes: list[str] | None = None,
        whole_molecules: bool = False,
        hbonds: list[str] | None = None,
    ) -> dict[str, Any]:
        """cutoff_planes / whole_molecules / hbonds: as in analyze_structure (drawing only).

        animate "gif" / "mp4" also writes <image stem>_rotation.gif (.mp4; GIF if no
        ffmpeg) next to the image: animate_frames views turning once around animate_axis
        ("up" = screen vertical, "a"/"b"/"c", "x"/"y"/"z"), with every option below.

        Render one view of a CIF looking along [u, v, w] (view_dir) or the (h k l) plane
        normal (view_normal); exactly one of them. perspective=True uses a perspective
        camera (no scale bar); labels "site" (e.g. Ni1) / "element" labels the atoms.
        bonds / bond_color: as in analyze_structure; polyhedra_center "Mn" also turns
        the polyhedra on (off by default in this tool). frac_range / image_formats /
        show_planes: as in analyze_structure (extra files are listed under "images").
        vectors / vector_basis / vector_scale / vector_color / vector_radius: arrows
        (magnetic moments), as in analyze_structure. volume_path / isovalue / iso_*:
        isosurfaces of a CHGCAR / Cube / XSF / grd, as in analyze_structure.
        style "ellipsoid" + probability (%, default 50): displacement ellipsoids from the CIF ADPs.
        Default output: a new folder 出力/<date>_view_<cif stem>/図/view_custom.png.
        """
        from shido.analyzer import compute_bonds
        from shido.renderer import _format_bulk_caption, render_single_view, resolve_view_direction

        transparent = bool(_or_setting(transparent, "render.transparent_background"))
        cif_file = Path(cif_path)
        struct, _, own_volume = _load_structure_or_volume(cif_file)
        cam_dir, up_dir, view_label = resolve_view_direction(
            struct, view_dir=view_dir, view_normal=view_normal
        )
        run_dir = None if output_path else new_run_dir("view", structure_stem(cif_file))
        out_path = Path(output_path) if output_path else run_dir / FIG_DIR / "view_custom.png"
        _, render_bonds, polyhedra_envs = compute_bonds(struct, bonds)
        from shido.core.animation import RotationHook, normalize_animate

        anim_fmt = normalize_animate(animate)
        hook = RotationHook(frames=animate_frames, axis=animate_axis) if anim_fmt else None
        image = render_single_view(
            struct,
            out_path,
            frame_hook=hook,
            camera_dir=cam_dir,
            view_up=up_dir,
            render_bonds=render_bonds,
            polyhedra_envs=polyhedra_envs,
            show_polyhedra=polyhedra_center is not None,
            polyhedra_center=polyhedra_center,
            bond_color=bond_color,
            orientation_label=view_label,
            caption_text=_format_bulk_caption(struct),
            transparent_background=transparent,
            style=style,
            perspective=perspective,
            labels=labels,
            frac_range=frac_range,
            image_formats=image_formats,
            planes=show_planes,
            **_vector_kwargs(vectors, vector_basis, vector_scale, vector_color, vector_radius),
            **_volume_kwargs(
                volume_path or own_volume, isovalue, iso_color, iso_color_negative, iso_opacity,
                iso_both_signs,
            ),
            **_ellipsoid_kwargs(style, str(cif_file), probability),
            **_molecule_render_kwargs(cutoff_planes, whole_molecules, hbonds),
        )
        from shido.renderer import _IMAGE_EXT, normalize_image_formats

        extra = [
            str(Path(image).with_suffix(_IMAGE_EXT[f]))
            for f in normalize_image_formats(image_formats)
        ]
        anim = (
            hook.write(Path(image).with_name(Path(image).stem + "_rotation.gif"), anim_fmt)
            if hook
            else None
        )
        return _finish_run(
            {
                "image": str(image),
                "images": [str(image), *extra],
                "view_label": view_label,
                "camera_dir": [float(v) for v in cam_dir],
                **({"animation": anim} if anim else {}),
            },
            run_dir or Path(image).parent,
            f"1 方向から見た図：{cif_file.name}",
            source=cif_file.resolve(),
            settings={"view_dir": view_dir, "view_normal": view_normal, "style": style},
            write_index=run_dir is not None,
        )

    @mcp.tool()
    def render_overlay(
        cif_path: str,
        overlay_paths: list[str],
        shifts: list[list[float]] | None = None,
        rotates: list[str] | None = None,
        shift_unit: str = "angstrom",
        about: str = "origin",
        opacity: float | None = None,
        color: str | None = None,
        output_path: str | None = None,
        view_dir: list[float] | None = None,
        view_normal: list[float] | None = None,
        style: str = "ball-stick",
        transparent: bool | None = None,
        frac_range: list[float] | None = None,
        labels: str | None = None,
        image_formats: str | None = None,
    ) -> dict[str, Any]:
        """Draw more structures over a base structure in one picture (molecule on a
        surface, grain boundary). overlay_paths: CIF / POSCAR / XSF / .vesta files;
        shifts[i] [x, y, z] (Å, or fractions of the base cell with shift_unit "frac") and
        rotates[i] "ax ay az angle_deg" (Cartesian axis; or 9 numbers = matrix) go with
        overlay_paths[i]; the rotation is about the origin (about "center": the overlay's
        cell centre) and comes before the shift. opacity 0-1 (e.g. 0.5) / color (one
        colour for the overlays). View as in render_view (default isometric).
        Default output: 出力/<date>_overlay_<cif stem>/図/overlay.png.
        """
        from shido.analyzer import compute_bonds
        from shido.renderer import _format_bulk_caption, render_single_view, resolve_view_direction

        transparent = bool(_or_setting(transparent, "render.transparent_background"))
        cif_file = Path(cif_path)
        struct, _, _ = _load_structure_or_volume(cif_file)
        specs = _overlay_specs(overlay_paths, shifts, rotates, shift_unit, opacity, color, about)
        if not specs:
            raise ValueError("render_overlay needs at least one overlay path.")
        if view_dir is not None or view_normal is not None:
            cam_dir, up_dir, view_label = resolve_view_direction(
                struct, view_dir=view_dir, view_normal=view_normal
            )
        else:
            a_vec, b_vec, c_vec = struct.lattice.matrix
            cam_dir, up_dir, view_label = a_vec + 0.36 * b_vec + 0.24 * c_vec, c_vec, "Isometric"
        run_dir = None if output_path else new_run_dir("overlay", structure_stem(cif_file))
        out_path = Path(output_path) if output_path else run_dir / FIG_DIR / "overlay.png"
        _, render_bonds, polyhedra_envs = compute_bonds(struct)
        image = render_single_view(
            struct, out_path, camera_dir=cam_dir, view_up=up_dir,
            render_bonds=render_bonds, polyhedra_envs=polyhedra_envs, show_polyhedra=False,
            orientation_label=view_label, caption_text=_format_bulk_caption(struct),
            transparent_background=transparent, style=style, labels=labels,
            frac_range=frac_range, image_formats=image_formats, overlays=specs,
        )
        return _finish_run(
            {"image": str(image), "view_label": view_label,
             "overlays": [Path(s["path"]).name for s in specs]},
            run_dir or Path(image).parent,
            f"構造を重ねた図：{cif_file.name}",
            source=cif_file.resolve(),
            settings={"overlays": specs, "style": style},
            write_index=run_dir is not None,
        )

    @mcp.tool()
    def compute_xrd_pattern(
        cif_path: str,
        output_dir: str | None = None,
        wavelength: str | None = None,
        two_theta_min: float | None = None,
        two_theta_max: float | None = None,
        label_top_n: int | None = None,
        radiation: str = "xray",
        profile: str | None = None,
        fwhm: float | None = None,
        caglioti_u: float | None = None,
        caglioti_v: float | None = None,
        caglioti_w: float | None = None,
        eta: float | None = None,
        observed_path: str | None = None,
    ) -> dict[str, Any]:
        """Compute the powder XRD pattern of a CIF and write xrd_peaks.csv and a figure (PNG/SVG).

        wavelength is a source key ('CuKa', 'MoKa', 'CuKa1', ...) or a number in Angstrom
        given as text (e.g. '1.5406'). Omitted values use the assumed parameters (xrd.*).
        radiation 'neutron' uses nuclear scattering lengths (neutron_peaks.csv, figure_neutron.png).
        profile 'gaussian' / 'lorentzian' / 'pseudo_voigt' draws a curve (xrd_profile.csv) with a
        constant fwhm (deg 2θ, default 0.1) or Caglioti caglioti_u/v/w (deg^2); eta = Lorentzian
        fraction. observed_path (.xy / .csv, 2θ and intensity) overlays a measured pattern with the
        scaled calculation and the difference (figure_xrd_observed.png). Omitted -> sticks as before.
        """
        cif_file = Path(cif_path)
        struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif_file))
        out_dir = Path(output_dir) if output_dir else new_run_dir("xrd", structure_stem(cif_file))
        wl: str | float | None = wavelength
        if isinstance(wavelength, str):
            try:
                wl = float(wavelength)
            except ValueError:
                wl = wavelength
        two_theta_range = None
        if two_theta_min is not None or two_theta_max is not None:
            two_theta_range = (
                float(_or_setting(two_theta_min, "xrd.two_theta_min")),
                float(_or_setting(two_theta_max, "xrd.two_theta_max")),
            )
        res = compute_xrd(
            struct,
            out_dir,
            wavelength=wl,
            two_theta_range=two_theta_range,
            label_top_n=label_top_n,
            **_xrd_extension_options(
                radiation, profile, fwhm, (caglioti_u, caglioti_v, caglioti_w), eta, observed_path
            ),
        )
        res["parse_warnings"] = parse_warnings
        return _finish_run(
            res, out_dir, f"粉末回折パターン：{cif_file.name}", source=cif_file.resolve(),
            settings={"wavelength": wavelength, "xrd_options": {"radiation": radiation, "profile": profile}},
        )

    @mcp.tool()
    def export_3d_model(
        cif_path: str,
        format: str = "stl",
        output_path: str | None = None,
        split_by_element: bool = False,
        max_size_mm: float | None = None,
        scale_mm_per_angstrom: float = 10.0,
        include_unit_cell: bool = True,
    ) -> dict[str, Any]:
        """Export a 3D model of a CIF: format="stl" (3D-print mesh in mm) or "html" (offline rotatable viewer).

        For STL, split_by_element also writes one STL per element plus one for bonds and
        the unit cell (for multi-colour printing); max_size_mm shrinks the model to fit.
        """
        fmt = str(format).strip().lower()
        if fmt not in ("stl", "html"):
            raise ValueError(f"format must be 'stl' or 'html', got {format!r}.")
        cif_file = Path(cif_path)
        struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif_file))
        run_dir = None if output_path else new_run_dir("3d", structure_stem(cif_file))
        out_path = Path(output_path) if output_path else run_dir / MODEL_DIR / f"structure.{fmt}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _done(res: dict[str, Any]) -> dict[str, Any]:
            return _finish_run(
                res, run_dir or out_path.parent, f"3D モデル（{fmt.upper()}）：{cif_file.name}",
                source=cif_file.resolve(), write_index=run_dir is not None,
            )

        if fmt == "html":
            written = export_html(struct, out_path, include_unit_cell=include_unit_cell)
            return _done({
                "format": "html",
                "html_file": str(Path(written).resolve()),
                "parse_warnings": parse_warnings,
            })
        written = export_stl(
            struct,
            out_path,
            include_unit_cell=include_unit_cell,
            scale=scale_mm_per_angstrom,
            split_by_element=split_by_element,
            max_size_mm=max_size_mm,
        )
        split_files = (
            [str(p.resolve()) for p in split_stl_paths(written, struct) if p.is_file()]
            if split_by_element
            else []
        )
        return _done({
            "format": "stl",
            "stl_file": str(Path(written).resolve()),
            "stl_split_files": split_files,
            "parse_warnings": parse_warnings,
        })

    @mcp.tool()
    def generate_slabs(
        cif_path: str,
        h: int,
        k: int,
        l: int,
        output_dir: str | None = None,
        min_slab_size: float | None = None,
        min_vacuum_size: float | None = None,
        filter_polar: bool | None = None,
        deduplicate: bool | None = None,
    ) -> dict[str, Any]:
        """Generate surface slabs for a given Miller index (h, k, l) from a CIF file. Omitted values use the assumed parameters (get_assumed_parameters)."""
        cif_file = Path(cif_path)
        struct, _ = _unpack_robust_load(load_structure_robust(cif_file))
        out_dir = Path(output_dir) if output_dir else new_run_dir(f"slab_{h}{k}{l}", structure_stem(cif_file))
        res = cut_slabs(
            struct,
            (h, k, l),
            out_dir,
            min_slab_size=_or_setting(min_slab_size, "slab.min_slab_size"),
            min_vacuum_size=_or_setting(min_vacuum_size, "slab.min_vacuum_size"),
            filter_polar=_or_setting(filter_polar, "slab.filter_polar"),
            deduplicate=_or_setting(deduplicate, "slab.deduplicate"),
        )
        return _finish_run(
            res, out_dir, f"スラブ（表面）の切り出し ({h} {k} {l})：{cif_file.name}",
            source=cif_file.resolve(), settings={"miller_index": [h, k, l]},
        )

    @mcp.tool()
    def run_ni2mnga_demo(
        h: int = 1,
        k: int = 1,
        l: int = 1,
        base_dir: str | None = None,
        filter_polar: bool | None = None,
        deduplicate: bool | None = None,
        transparent: bool | None = None,
    ) -> dict[str, Any]:
        """Run the end-to-end Ni2MnGa L21 Heusler bulk and slab analysis demo."""
        transparent = bool(_or_setting(transparent, "render.transparent_background"))
        return run_demo(
            miller_index=(h, k, l),
            base_dir=base_dir,
            filter_polar=filter_polar,
            deduplicate=deduplicate,
            transparent=transparent,
        )

    @mcp.tool()
    def batch_analyze_cifs(
        input_dir: str,
        output_dir: str | None = None,
        h: int | None = None,
        k: int | None = None,
        l: int | None = None,
        filter_polar: bool | None = None,
        transparent: bool | None = None,
        style: str = "ball-stick",
        view_dir: list[float] | None = None,
        view_normal: list[float] | None = None,
        perspective: bool = False,
        labels: str | None = None,
        bonds: list[str] | None = None,
        polyhedra_center: str | None = None,
        bond_color: str | None = None,
        frac_range: list[float] | None = None,
        image_formats: str | None = None,
        show_planes: list[str] | None = None,
        vectors: list[str] | None = None,
        vector_basis: str | None = None,
        vector_scale: float | None = None,
        vector_color: str | None = None,
        vector_radius: float | None = None,
    ) -> dict[str, Any]:
        """Batch-analyze all CIF files in a directory and generate summary CSV and gallery PNG.

        style: "ball-stick" (default), "space-filling", "stick" or "wireframe".
        view_dir / view_normal / perspective / labels / bonds / polyhedra_center /
        bond_color / frac_range / image_formats / show_planes / vectors / vector_basis /
        vector_scale / vector_color / vector_radius: as in analyze_structure.
        """
        transparent = bool(_or_setting(transparent, "render.transparent_background"))
        hkl_vals = (h, k, l)
        if all(v is not None for v in hkl_vals):
            miller_index: tuple[int, int, int] | None = (int(h), int(k), int(l))  # type: ignore[arg-type]
        elif all(v is None for v in hkl_vals):
            miller_index = None
        else:
            raise ValueError("Provide all three of h, k, l or leave all three as None.")

        return run_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            miller_index=miller_index,
            filter_polar=filter_polar,
            transparent=transparent,
            style=style,
            view_dir=view_dir,
            view_normal=view_normal,
            perspective=perspective,
            labels=labels,
            bonds=bonds,
            polyhedra_center=polyhedra_center,
            bond_color=bond_color,
            frac_range=frac_range,
            image_formats=image_formats,
            show_planes=show_planes,
            **_vector_kwargs(vectors, vector_basis, vector_scale, vector_color, vector_radius),
        )

    @mcp.tool()
    def compare_transformation_phases(
        parent_cif: str,
        martensite_cif: str,
        output_dir: str | None = None,
        transparent: bool | None = None,
    ) -> dict[str, Any]:
        """Compare parent (austenite) and martensite CIFs side-by-side at identical scale with Bain subcell overlay and strain CSV."""
        transparent = bool(_or_setting(transparent, "render.transparent_background"))
        parent_struct, _ = _unpack_robust_load(load_structure_robust(parent_cif))
        mart_struct, _ = _unpack_robust_load(load_structure_robust(martensite_cif))
        out_dir = Path(output_dir) if output_dir else new_run_dir("phase_compare", Path(parent_cif).stem)
        res = compare_phases(
            parent_struct,
            mart_struct,
            out_dir,
            transparent_background=transparent,
        )
        return _finish_run(
            res, out_dir, "変態前後の 2 相の比較", source=f"{parent_cif} / {martensite_cif}",
            settings={"transparent": transparent},
        )

    @mcp.tool()
    def generate_heusler_sublattice_models(
        formula: str | None = None,
        ordering: str | None = None,
        num_patterns: int | None = None,
        a: float | None = None,
        output_dir: str | None = None,
        highlight_elem1: str | None = None,
        highlight_elem2: str | None = None,
        transparent: bool | None = None,
    ) -> dict[str, Any]:
        """Generate Heusler supercell CIF/POSCAR patterns, 4a/4b/8c sublattice schematic, and nearest-neighbor pair highlight figure. ordering: "L21" or "B2" (Fm-3m, X->8c, Y->4a, Z->4b), or "XA" (ordered inverse Heusler, F-43m, X->4a+4d, Y->4c, Z->4b; ordered only, no XA disorder). Omitted values use the assumed parameters."""
        formula = _or_setting(formula, "heusler.formula")
        ordering = _or_setting(ordering, "heusler.ordering")
        pair = list(cfg.get("heusler.highlight_pair"))
        highlight_elem1 = highlight_elem1 or pair[0]
        highlight_elem2 = highlight_elem2 or pair[1]
        out_dir = (
            Path(output_dir) if output_dir
            else new_run_dir("heusler", f"{sanitize_name(formula)}_{ordering}")
        )
        res = generate_heusler_models(
            formula=formula,
            ordering=ordering,
            num_patterns=_or_setting(num_patterns, "heusler.num_patterns"),
            a=_or_setting(a, "lattice.heusler_a"),
            supercell=tuple(cfg.get("heusler.supercell")),
            seed=int(cfg.get("heusler.seed")),
            output_dir=out_dir,
            highlight_pair=(highlight_elem1, highlight_elem2),
            transparent_background=transparent,
        )
        return _finish_run(
            res, out_dir, f"ホイスラー合金の副格子モデル：{formula}",
            settings={"formula": formula, "ordering": ordering},
        )

    @mcp.tool()
    def generate_dft_ca_sweep(
        cif_path: str,
        output_dir: str | None = None,
        ca_min: float | None = None,
        ca_max: float | None = None,
        ca_step: float | None = None,
    ) -> dict[str, Any]:
        """Generate constant-volume Bain c/a sweep POSCAR/CIF files and sweep table CSV for DFT calculations."""
        struct, _ = _unpack_robust_load(load_structure_robust(cif_path))
        out_dir = Path(output_dir) if output_dir else new_run_dir("ca_sweep", Path(cif_path).stem)
        res = generate_ca_sweep(
            struct,
            out_dir,
            ca_min=_or_setting(ca_min, "ca_sweep.ca_min"),
            ca_max=_or_setting(ca_max, "ca_sweep.ca_max"),
            ca_step=_or_setting(ca_step, "ca_sweep.ca_step"),
        )
        return _finish_run(
            res, out_dir, f"c/a スイープ（DFT 用の構造）：{Path(cif_path).name}",
            source=Path(cif_path).resolve(),
            settings={"c/a の範囲": [_or_setting(ca_min, "ca_sweep.ca_min"), _or_setting(ca_max, "ca_sweep.ca_max")],
                      "c/a の刻み": _or_setting(ca_step, "ca_sweep.ca_step")},
        )

    @mcp.tool()
    def get_assumed_parameters() -> dict[str, Any]:
        """Return the assumed parameters (lattice constants, c/a range, slab sizes, display atomic radii, render settings, material presets) currently in effect."""
        return cfg.load_settings()

    @mcp.tool()
    def set_assumed_parameters(values: dict[str, Any]) -> dict[str, Any]:
        """Change assumed parameters with dotted keys, e.g. {"lattice.a_austenite": 5.80, "render.dpi": 600, "atomic_radii.Ni": 1.25}. Saved to settings_user.yaml (same as the GUI tab)."""
        return cfg.update(values)  # 1項目ずつ確認し、おかしければ ValueError（何も保存しない）

    @mcp.tool()
    def reset_assumed_parameters() -> dict[str, Any]:
        """Discard user changes and return to the default assumed parameters."""
        cfg.reset_user_settings()
        return cfg.load_settings()

    @mcp.tool()
    def list_color_themes() -> dict[str, Any]:
        """List available color themes and the one currently used for figures."""
        return {"themes": list_themes(), "active": cfg.get("render.theme")}

    @mcp.tool()
    def set_color_theme(theme: str) -> dict[str, Any]:
        """Set the color theme used for all following figures (saved as render.theme)."""
        if theme not in list_themes():
            raise ValueError(f"Unknown theme {theme!r}. Choose from {list_themes()}")
        set_active_theme(theme)
        cfg.update({"render.theme": theme})
        return {"active": theme}

    @mcp.tool()
    def list_major_planes() -> list[dict[str, Any]]:
        """List major crystallographic plane presets (Miller indices and aliases)."""
        return [
            {"key": p["key"], "hkl": list(p["hkl"]), "label": p["label"], "aliases": p["aliases"]}
            for p in MAJOR_PLANES
        ]

    @mcp.tool()
    def plot_dft_ca_energy_curve(
        energy_csv: str,
        cif_path: str | None = None,
        output_dir: str | None = None,
        n_atoms: int | None = None,
    ) -> dict[str, Any]:
        """Plot publication-grade delta-E vs c/a curve with automatic crystal structure thumbnails at energy minima (PNG + SVG). Give n_atoms (atoms per cell) or cif_path when the CSV has only total energy_eV; otherwise the axis is meV/cell."""
        base_struct = (
            _unpack_robust_load(load_structure_robust(cif_path))[0]
            if cif_path
            else None
        )
        if output_dir:
            return plot_ca_energy_curve(energy_csv, base_struct, Path(output_dir), n_atoms=n_atoms)
        out_dir = new_run_dir("ca_energy", Path(energy_csv).stem)
        res = plot_ca_energy_curve(energy_csv, base_struct, out_dir, n_atoms=n_atoms)
        return _finish_run(res, out_dir, "c/a とエネルギーの曲線", source=Path(energy_csv).resolve())

    @mcp.tool()
    def measure_geometry(
        cif_path: str,
        atoms: list[str] | None = None,
        planes: list[str] | None = None,
        best_plane_atoms: list[str] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Measure geometry in a CIF (periodic images handled). The result is also saved as
        measure.json (output_dir, default a new folder 出力/<date>_measure_<cif>/数値/).

        atoms: 2 atoms -> distance (A), 3 -> angle at the middle atom, 4 -> dihedral (deg).
        Each atom is "<site>[@<image>]": a site index ("0"), a CIF label ("Na1") or
        element+ordinal ("Si2"); "@1,0,0" picks the image in the cell shifted by
        (1,0,0). Without "@", the image nearest to the previous atom is used.
        planes: two Miller indices (e.g. ["1 0 0", "1 1 0"]) -> angle between normals.
        best_plane_atoms: >= 3 atoms -> least-squares plane, deviations and approx (hkl).
        """
        struct, parse_warnings = _unpack_robust_load(load_structure_robust(Path(cif_path)))
        res = _measure_geometry(
            struct, atoms=atoms, planes=planes, best_plane_atoms=best_plane_atoms
        )
        res["parse_warnings"] = parse_warnings
        return _save_numeric_run(
            "measure", cif_path, output_dir, res, "距離・角度の測定",
            lambda d: {"json": str(_write_json(res, d / "measure.json"))},
        )

    @mcp.tool()
    def get_symmetry_info(
        cif_path: str,
        symprec: float = 0.01,
        output_dir: str | None = None,
        include_operations: bool = True,
    ) -> dict[str, Any]:
        """Symmetry table of a CIF (spglib): space group number/symbol, symmetry
        operations as "x,y,z" strings (in the input cell, incl. centring), and per site /
        per orbit the Wyckoff letter, multiplicity, site symmetry and equivalent positions.
        output_dir: also write symmetry_info.json, symmetry_sites.csv, symmetry_operations.csv.
        include_operations=False drops the (possibly 192-long) operation list from the reply.
        """
        struct, parse_warnings = _unpack_robust_load(load_structure_robust(Path(cif_path)))
        info = compute_symmetry_info(struct, symprec=symprec)
        _save_numeric_run(
            "symmetry", cif_path, output_dir, info, "対称性（空間群・Wyckoff）",
            lambda d: write_symmetry_report(info, d),
        )
        if not include_operations:
            info.pop("operations", None)
        info["parse_warnings"] = parse_warnings
        return info

    @mcp.tool()
    def transform_cell(
        cif_path: str,
        cell: str = "input",
        transform: str | list[float] | None = None,
        origin_shift: str | list[float] | None = None,
        supercell: str | list[int] | None = None,
        symprec: float = 0.01,
        output_dir: str | None = None,
        cif_symprec: float | None = None,
    ) -> dict[str, Any]:
        """Change the cell (VESTA-like) and optionally write the new CIF + POSCAR.

        Order: cell -> transform/origin_shift -> supercell.
        cell: "input" | "niggli" (Niggli cell of the primitive lattice) |
        "conventional"/"standard" (spglib standard) | "primitive".
        transform: matrix P, 9 numbers row-major "P11 P12 P13 P21 ... P33" (fractions OK,
        e.g. F->P "0 1/2 1/2 1/2 0 1/2 1/2 1/2 0") or 3 numbers (diagonal);
        (a',b',c') = (a,b,c)P, x' = P^-1(x - p). origin_shift: p in old fractional coords.
        supercell: integer "2 2 1" or 9 numbers. Checks: composition, site count = n*|det|;
        det < 0 gives a handedness warning. output_dir writes <name>_transformed.cif/.vasp
        (CIF in P1 unless cif_symprec is given).
        """
        struct, parse_warnings = _unpack_robust_load(load_structure_robust(Path(cif_path)))
        new, report = _transform_cell(
            struct, cell=cell, transform=transform, origin_shift=origin_shift,
            supercell=supercell, symprec=symprec,
        )
        report["parse_warnings"] = parse_warnings
        return _save_numeric_run(
            "transform", cif_path, output_dir, report, "格子の変換",
            lambda d: write_transformed(new, d, stem=f"{Path(cif_path).stem}_transformed",
                                        cif_symprec=cif_symprec),
            sub=STRUCT_DIR, settings={"cell": cell, "transform": transform, "origin_shift": origin_shift,
                                      "supercell": supercell},
        )

    @mcp.tool()
    def analyze_polyhedra(
        cif_path: str,
        bond_specs: str | list[str] | None = None,
        madelung: bool = False,
        oxidation_states: dict[str, float] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Coordination-polyhedron numbers per site (VESTA-like), with the same bonds as
        bond_lengths.csv: volume, mean bond length, Baur distortion index, quadratic
        elongation and bond-angle variance (n = 4, 6, 8), ECoN, bond valence sum (BVS).
        bond_specs: e.g. ["Na-Cl:0:3.0"] (same as the bond search). madelung=True adds
        the Ewald (Madelung) energy, site potentials and, for AB-type two-element
        compounds, the Madelung constant (NaCl 1.7476). oxidation_states: e.g.
        {"Na": 1, "Cl": -1} (default: guessed). output_dir writes polyhedra.csv/.json.
        """
        from shido.core.polyhedra_metrics import analyze_polyhedra as _analyze_polyhedra, write_polyhedra_report

        struct, parse_warnings = _unpack_robust_load(load_structure_robust(Path(cif_path)))
        df, report = _analyze_polyhedra(
            struct, bond_specs=bond_specs, madelung=madelung, oxidation_states=oxidation_states
        )
        report["parse_warnings"] = parse_warnings
        return _save_numeric_run(
            "polyhedra", cif_path, output_dir, report, "配位多面体の解析",
            lambda d: write_polyhedra_report(df, report, d), settings={"madelung": madelung},
        )

    @mcp.tool()
    def compute_structure_factors(
        cif_path: str,
        dmin: float = 0.8,
        radiation: str = "xray",
        wavelength: str | None = None,
        debye_waller_factors: dict[str, float] | None = None,
        include_extinct: bool = True,
        output_dir: str | None = None,
        max_rows: int = 200,
    ) -> dict[str, Any]:
        """Structure factor table F(hkl) (VESTA-like) for every hkl family with d >= dmin.

        Columns: h k l, d, 2θ, multiplicity, F_real/F_imag, |F|, phase (deg), F², Lp,
        intensity = multiplicity·|F|²·Lp (relative values match compute_xrd_pattern),
        extinct ('systematic' = space-group absence, 'structure' = zero from the atom
        positions, e.g. Si (2 2 2)), and f_<El> per element. radiation 'xray' (pymatgen
        form factors, F in electrons) or 'neutron' (scattering lengths, F in fm).
        wavelength: 'CuKa', 'MoKa', ... or Å as text (only 2θ/Lp depend on it).
        debye_waller_factors: {element: B [Å^2]}. output_dir writes structure_factors.csv/.json
        (neutron: structure_factors_neutron.*). Only the first max_rows rows are returned.
        """
        from shido.core.structure_factors import (
            compute_structure_factors as _compute_sf,
            write_structure_factor_report,
        )

        struct, parse_warnings = _unpack_robust_load(load_structure_robust(Path(cif_path)))
        df, report = _compute_sf(
            struct, dmin=dmin, radiation=radiation, wavelength=_parse_wavelength(wavelength),
            debye_waller_factors=debye_waller_factors, include_extinct=include_extinct,
        )
        _save_numeric_run(
            "structure_factors", cif_path, output_dir, report, "構造因子 F(hkl)",
            lambda d: write_structure_factor_report(df, report, d),
            settings={"dmin": dmin, "radiation": radiation, "wavelength": wavelength},
        )
        if max_rows is not None and len(report["reflections"]) > int(max_rows):
            report["reflections"] = report["reflections"][: max(0, int(max_rows))]
            report["reflections_truncated"] = True
        report["parse_warnings"] = parse_warnings
        return report

    @mcp.tool()
    def analyze_displacement(
        cif_path: str,
        probability: float = 50.0,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Thermal displacement parameters (ADP, ORTEP-like ellipsoids) read from the CIF:
        _atom_site_aniso_U_ij / B_ij / beta_ij and U_iso / B_iso per atom label.
        Per atom: U_cif (U11..U23, A^2), U_eq = trace(U_cart)/3, principal mean-square
        displacements U1<=U2<=U3, rms, ellipsoid semi-axes at `probability` percent
        (radius factor sqrt(chi2_3^-1(p)), 50% -> 1.5382), Cartesian axis directions,
        non-positive-definite flag. output_dir writes displacement.csv/.json.
        """
        from shido.core.adp import analyze_displacement as _analyze_adp, write_displacement_report

        struct, parse_warnings = _unpack_robust_load(load_structure_robust(Path(cif_path)))
        df, report = _analyze_adp(cif_path, probability=probability, structure=struct)
        report["parse_warnings"] = parse_warnings
        return _save_numeric_run(
            "displacement", cif_path, output_dir, report, "熱振動（原子変位パラメータ）",
            lambda d: write_displacement_report(df, report, d), settings={"probability": probability},
        )

    @mcp.tool()
    def slice_volume(
        volume_path: str,
        slice: str | list[str] | None = "0 0 1@0.5",
        contours: float | str | list[float] | None = None,
        line_profile: str | list[str] | None = None,
        resolution: float | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """2D sections and line profiles of volumetric data (VESTA "2D data display").

        volume_path: CHGCAR / LOCPOT / ELFCAR / *.cube / XSF DATAGRID_3D / *.grd.
        slice: "h k l@pos" (plane h x + k y + l z = pos, pos in d_hkl), one or a list;
        PNG colour map with contour lines (axes in Å, dashed = unit-cell cut) + CSV.
        contours: levels (e.g. 0.05 or "0.05,0.1"); None = 8 automatic levels, "none" = no lines.
        line_profile: "x1 y1 z1 x2 y2 z2" (fractional), one or a list -> PNG + CSV.
        Periodic trilinear interpolation, oblique cells handled. resolution: pixel in Å.
        """
        return _slice_volume_impl(volume_path, slice, contours, line_profile, resolution, output_dir)

    @mcp.tool()
    def volume_model(
        input_path: str,
        model: str = "xray",
        grid: int | str | list[int] | None = None,
        b_iso: float | dict[str, float] | None = None,
        patterson: bool = False,
        diff_path: str | None = None,
        export_formats: str | list[str] = "cube",
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Make volumetric data (VESTA model ED / Patterson / difference) and write it as
        Gaussian Cube and/or VESTA *.grd, ready for slice_volume and volume_path.

        input_path: a CIF (or any structure file) -> model density by Fourier synthesis of
        the calculated F(hkl): model "xray" = electron density (e/Å^3, integral = number
        of electrons) or "neutron" = nuclear scattering density (fm/Å^3); or a volumetric
        file (CHGCAR / *.cube / *.xsf / *.grd), used as it is.
        grid: points per axis (64 or "48,48,64"); None = about 0.1 Å (even, 16..128).
        b_iso: isotropic B in Å^2 (one value or {element: B}); default 0.5.
        patterson: also write the Patterson map P(u) = ∫ρ(r)ρ(r+u)dr.
        diff_path: another volume file (or a CIF -> its model on the same grid); writes
        input − diff after checking grid size, lattice and origin.
        export_formats: "cube", "grd", "cube,grd" or "all".
        """
        return _volume_model_impl(input_path, model, grid, b_iso, patterson, diff_path, export_formats, output_dir)

    @mcp.tool()
    def render_morphology(
        cif_path: str,
        faces: str | list[Any],
        view_dir: list[float] | None = None,
        export_formats: str | list[str] = "stl,html",
        size_mm: float = 30.0,
        expand: bool = True,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Crystal shape (VESTA "morphology"): convex polyhedron from {hkl} and their
        central distances, with a PNG (faces coloured per {hkl}), the area table
        (morphology_faces.csv per (hkl), morphology_families.csv per {hkl}) and STL / HTML.

        faces: "1 0 0:1.0,1 1 1:0.9" (h k l : distance; only the ratios matter) or a list
        like [{"hkl": [1,0,0], "distance": 1.0}]. Each {hkl} is expanded to its
        symmetry-equivalent planes (expand=False: only the given plane).
        view_dir: Cartesian direction towards the camera. export_formats: "stl", "html",
        "stl,html" or "none". size_mm: longest side of the STL.
        """
        return _morphology_impl(cif_path, faces, view_dir, export_formats, size_mm, expand, output_dir)

    @mcp.tool()
    def export_structure(
        cif_path: str | list[str],
        formats: str | list[str] = "xyz,p1,xtl,pdb",
        output_dir: str | None = None,
        multi_name: str | None = None,
    ) -> dict[str, Any]:
        """Export structure data (VESTA "Export Data") for one or more CIFs.

        formats: any of "xyz" (Cartesian Å), "p1" (<name>_P1.cif, all atoms),
        "xtl" (fractional), "pdb" (CRYST1 + ATOM with occupancy), "wrl" (colour VRML,
        mm, same model as STL), "multicif" (every structure / data block in one
        <multi_name>_all.cif), "vesta" (<name>.vesta: cell, space group, sites,
        occupancies, bonds, colours; opens in VESTA), or "all". cif_path may be a list; every data block of a
        multi-block CIF is exported. output_dir defaults to a new folder 出力/<date>_export_<cif>/
        (構造/; VRML in 3D/).
        """
        return _export_structure_impl(cif_path, formats, output_dir, multi_name)

    @mcp.tool()
    def edit_structure(
        cif_path: str,
        operations: list[dict[str, Any]] | str,
        merge_tol: float | None = None,
        poscar: bool = False,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Edit atoms / sites (VESTA "Edit Data") and save <name>_edited.cif (構造/).

        operations (applied in order; site indices are 0-based, counted at that step):
        {"op":"add","element":"Li","coords":[x,y,z]} (+ "occupancy", "species":{"Mn":0.5,"Ga":0.5}, "cartesian"),
        {"op":"remove","site":3} / {"op":"remove","element":"Cl"},
        {"op":"replace","from":"Na","to":"K"} (or "site":[0,1], "to":{"K":0.5,"Na":0.5}),
        {"op":"move","site":0,"coords":[x,y,z]} or "shift":[dx,dy,dz],
        {"op":"occupancy","element":"Cl","value":0.5} or "species":{...},
        {"op":"merge","tol":0.01} (duplicates closer than tol A become one site).
        merge_tol: also merge duplicates after the last operation. The result is checked
        (overlap < 0.6 A, empty cell, occupancy) before anything is written.
        """
        return _edit_structure_impl(cif_path, operations, merge_tol, poscar, output_dir)

    @mcp.tool()
    def create_structure(
        spacegroup: str | int,
        lattice: list[float] | dict[str, float] | str,
        sites: list[Any] | str,
        name: str | None = None,
        merge_tol: float | None = 0.01,
        poscar: bool = False,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Build a structure from a space group + lattice + asymmetric unit (from_spacegroup).

        lattice: [a,b,c,alpha,beta,gamma], {"a":5.64} (missing values follow the crystal
        system, e.g. cubic needs only a), or a 3x3 matrix.
        sites: [{"element":"Na","coords":[0,0,0]}, ...] (+ "occupancy"/"species"), or
        [["Na",0,0,0],["Cl",0.5,0.5,0.5]]. Writes <name>_created.cif in 構造/ after the check.
        """
        return _create_structure_impl(spacegroup, lattice, sites, name, merge_tol, poscar, output_dir)

    @mcp.tool()
    def make_journal_figure(
        journal: str,
        panels: str | list[str],
        width: str = "single",
        cif_path: str | None = None,
        format: str | None = None,
        dpi: float | None = None,
        columns: int | None = None,
        labels: bool = True,
        output_dir: str | None = None,
        view_labels: bool = False,
        caption_bar: bool = False,
        legend: bool = True,
    ) -> dict[str, Any]:
        """Finish a multi-panel figure to a journal's rules and check it (journals/*.yaml).

        journal: id or abbreviation (see list_journals), e.g. "acta_materialia", "prb", "nature".
        panels: "bulk:iso,xrd" — bulk:iso / bulk:a / bulk:c (structure views of cif_path) and
        xrd or xrd:MoKa are drawn at the finished size in the journal's fonts (scale bar,
        a/b/c arrows, hkl in exact pt; one shared element legend under the panels unless
        legend=False); bulk:overview or image file paths (slices, graphs) are scaled.
        view_labels / caption_bar also write the view name / formula line in the
        structure panels (off by default: they belong in the caption). width: single / one_half / double or mm.
        Lays the panels out at the column width (mm x dpi pixels), adds panel labels in the
        journal's style, writes the journal's format (TIFF/PDF/EPS/PNG) to 図/, and returns
        "check" (width, height, dpi, format, smallest text and line: ok/fail/warn/unknown).
        Unverified journal rules use safe defaults, listed in "defaults_used"/"warnings".
        """
        from shido.journal_figure import make_journal_figure_impl
        return make_journal_figure_impl(panels, journal, width, cif_path, format, dpi, columns, labels, output_dir,
                                        view_labels, caption_bar, legend)

    @mcp.tool()
    def list_journals() -> list[dict[str, Any]]:
        """Journals with figure rules (journals/*.yaml): widths mm, max height, min text pt,
        min line pt, dpi, formats, panel label style and the rules not yet verified ("未確認")."""
        from shido.core.journal_spec import list_journals as _list_journals
        return _list_journals()

    return mcp


def _write_json(data: Any, path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path.resolve()


def _slice_volume_impl(
    volume_path: str | Path,
    slices: Any,
    contours: Any,
    line_profiles: Any,
    resolution: float | None,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    """Shared by the MCP tool `slice_volume` and `server.py slice` (batch 18).

    output_dir given: PNG and CSV straight into it. Otherwise a new run folder with the
    maps in 図/ and the CSV in 数値/ plus はじめに読む.html."""
    from shido.core.volumetric import read_volume, write_slice_outputs

    vp = Path(volume_path)
    grid = read_volume(vp)
    if not slices and not line_profiles:
        raise ValueError("Give at least one slice ('h k l@pos') or line profile.")
    if output_dir:
        run, fig_dir, data_dir = Path(output_dir), Path(output_dir), Path(output_dir)
    else:
        run = new_run_dir("slice", vp.stem or vp.name)
        fig_dir, data_dir = run / FIG_DIR, run / DATA_DIR
    report = write_slice_outputs(
        grid, slices=slices, line_profiles=line_profiles, contours=contours,
        fig_dir=fig_dir, data_dir=data_dir, resolution=resolution,
    )
    settings = {"volume": str(vp), "slice": slices, "contours": contours, "line_profile": line_profiles}
    return _finish_run(
        report, run, f"体積データの断面・ラインプロファイル：{vp.name}", source=vp.resolve(),
        settings=settings, write_index=not output_dir,
    )


def _load_volume_source(path: str | Path) -> Any:
    """A volumetric file -> VolumetricGrid; anything else -> Structure (robust loader)."""
    from shido.core.volumetric import is_volume_file, read_volume

    p = Path(path)
    if not p.is_file():
        raise ValueError(f"file not found: {p}")
    if is_volume_file(p):
        return read_volume(p)
    struct, _ = _unpack_robust_load(load_structure_robust(p))
    return struct


def _volume_model_impl(
    input_path: str | Path,
    model: str | None,
    grid: Any,
    b_iso: Any,
    patterson: bool,
    diff_path: str | Path | None,
    export_formats: Any,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    """Shared by the MCP tool `volume_model` and `server.py volume-model` (batch 19).

    output_dir given: files straight into it. Otherwise a new run folder with the
    volumes in 数値/ plus はじめに読む.html."""
    from shido.core.volumetric import build_volumes

    ip = Path(input_path)
    source = _load_volume_source(ip)
    diff = _load_volume_source(diff_path) if diff_path else None
    if output_dir:
        run, data_dir = Path(output_dir), Path(output_dir)
    else:
        run = new_run_dir("volume", ip.stem or ip.name)
        data_dir = run / DATA_DIR
    report = build_volumes(
        source, model=model, grid=grid, b_iso=b_iso, patterson_map=patterson, diff=diff,
        export_formats=export_formats, out_dir=data_dir,
    )
    settings = {"input": str(ip), "model": model if report["source"] != "input" else None,
                "grid": report["grid"], "b_iso": b_iso, "patterson": patterson,
                "diff": str(diff_path) if diff_path else None, "export": report["export_formats"]}
    settings = {k: v for k, v in settings.items() if v not in (None, False)}
    return _finish_run(
        report, run, f"体積データの作成（モデル密度・パターソン・差）：{ip.name}", source=ip.resolve(),
        settings=settings, write_index=not output_dir,
    )


def _morphology_impl(
    cif_path: str | Path,
    faces: Any,
    view_dir: Any = None,
    export_formats: Any = "stl,html",
    size_mm: float = 30.0,
    expand: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Shared by the MCP tool `render_morphology` and `server.py morphology` (batch 20).

    output_dir given: files straight into it. Otherwise a new run folder: PNG in 図/,
    CSV in 数値/, STL / HTML in 3D/, plus はじめに読む.html."""
    from shido.core.morphology import build_morphology, write_morphology_csv
    from shido.export3d import export_morphology_html, export_morphology_stl
    from shido.renderer import render_morphology as _render_morphology

    cp = Path(cif_path)
    if not cp.is_file():
        raise ValueError(f"file not found: {cp}")
    struct, _ = _unpack_robust_load(load_structure_robust(cp))
    morph = build_morphology(struct, faces, expand=expand)
    fmts = export_formats if isinstance(export_formats, (list, tuple)) else str(export_formats or "").split(",")
    fmts = {f.strip().lower() for f in fmts if f and f.strip()}
    if "all" in fmts:
        fmts = {"stl", "html"}
    fmts.discard("none")
    unknown = fmts - {"stl", "html"}
    if unknown:
        raise ValueError(f"unknown export format(s): {sorted(unknown)} (use stl, html, none)")
    if output_dir:
        run = fig_dir = data_dir = model_dir = Path(output_dir)
    else:
        run = new_run_dir("morphology", cp.stem)
        fig_dir, data_dir, model_dir = run / FIG_DIR, run / DATA_DIR, run / MODEL_DIR
    report = morph.report()
    vd = None if view_dir is None else [float(v) for v in (view_dir.split() if isinstance(view_dir, str) else view_dir)]
    files = [str(_render_morphology(morph, fig_dir / "morphology.png", view_dir=vd))]
    files += [str(p) for p in write_morphology_csv(morph, data_dir)]
    if "stl" in fmts:
        files.append(str(export_morphology_stl(morph, model_dir / "morphology.stl", size_mm=size_mm).resolve()))
    if "html" in fmts:
        title = f"{cp.stem} crystal shape"
        files.append(str(export_morphology_html(morph, model_dir / "morphology.html", title=title).resolve()))
    report["files"] = files
    settings = {"faces": faces, "expand": expand, "view_dir": vd, "export": sorted(fmts) or None}
    settings = {k: v for k, v in settings.items() if v is not None}
    return _finish_run(
        report, run, f"結晶外形（モルフォロジー）：{cp.name}", source=cp.resolve(),
        settings=settings, write_index=not output_dir,
    )


def _save_numeric_run(
    kind: str,
    cif_path: str | Path,
    output_dir: str | Path | None,
    result: dict[str, Any],
    title: str,
    writer: Any,
    sub: str = DATA_DIR,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save the files of a measure / symmetry / ... run and fill result["files"].

    ``output_dir`` given: files go straight into it as before (no はじめに読む.html).
    Otherwise: a new folder ``出力/<日時>_<kind>_<cif>/`` with the files in ``sub``.
    """
    cif = Path(cif_path)
    if output_dir:
        result["files"] = writer(Path(output_dir))
        return _finish_run(result, output_dir, title, write_index=False)
    run = new_run_dir(kind, cif.stem)
    result["files"] = writer(run / sub)
    return _finish_run(result, run, f"{title}：{cif.name}", source=cif.resolve(), settings=settings)


def _export_structure_impl(
    cif_paths: str | list[str],
    formats: str | list[str] | None,
    output_dir: str | Path | None,
    multi_name: str | None = None,
) -> dict[str, Any]:
    """Shared by the MCP tool `export_structure` and `server.py export`."""
    from shido.core.exporters import export_structures

    paths = [Path(cif_paths)] if isinstance(cif_paths, (str, Path)) else [Path(p) for p in cif_paths]
    if not paths:
        raise ValueError("cif_path: give at least one file.")
    entries = []
    parse_warnings: list[str] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"CIF file not found: {p}")
        blocks = load_structures_all(p)
        for struct, warns, block in blocks:
            name = p.stem if len(blocks) == 1 else f"{p.stem}_{block}"
            entries.append((name, struct))
            parse_warnings.extend(f"{name}: {w}" for w in warns)
    if multi_name is None:
        multi_name = paths[0].stem if len(paths) == 1 else "structures"
    if output_dir:
        res = export_structures(entries, formats, Path(output_dir), multi_name=multi_name)
        res["parse_warnings"] = parse_warnings
        return _finish_run(res, output_dir, "構造ファイルの書き出し", write_index=False)
    run = new_run_dir("export", paths[0].stem if len(paths) == 1 else f"{paths[0].stem}_ほか")
    res = export_structures(entries, formats, run / STRUCT_DIR, multi_name=multi_name)
    # VRML is a 3D model: move it next to the other 3D files
    for item in res["structures"]:
        wrl = item["files"].get("wrl")
        if wrl:
            dest = run / MODEL_DIR / Path(wrl).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            Path(wrl).replace(dest)
            item["files"]["wrl"] = str(dest.resolve())
    res["parse_warnings"] = parse_warnings
    return _finish_run(
        res, run, "構造ファイルの書き出し", source=", ".join(str(p.resolve()) for p in paths),
        settings={"formats": res.get("formats")},
    )


def _save_structure_run(
    kind: str, name: str, struct: Any, stem: str, rep: dict[str, Any], title: str,
    poscar: bool, output_dir: str | Path | None, source: Any = None, settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Batch 22: validate + write <stem>.cif (.vasp) into output_dir or a new run's 構造/."""
    from shido.core.editor import write_edited

    if output_dir:
        rep["files"] = write_edited(struct, Path(output_dir), stem=stem, poscar=poscar)
        return _finish_run(rep, output_dir, title, write_index=False)
    run = new_run_dir(kind, name)  # rep came from edit_report, which already validated
    rep["files"] = write_edited(struct, run / STRUCT_DIR, stem=stem, poscar=poscar)
    return _finish_run(rep, run, title, source=source, settings=settings)


def _edit_structure_impl(
    cif_path: str | Path,
    operations: Any,
    merge_tol: float | None = None,
    poscar: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Shared by the MCP tool `edit_structure` and `server.py edit` (batch 22)."""
    from shido.core.editor import apply_operations, edit_report

    cif = Path(cif_path)
    if not cif.is_file():
        raise FileNotFoundError(f"Structure file not found: {cif}")
    struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif))
    new, log = apply_operations(struct, operations, merge_tol=merge_tol)
    rep = edit_report(struct, new, log)
    rep["parse_warnings"] = parse_warnings
    return _save_structure_run(
        "edit", cif.stem, new, f"{cif.stem}_edited", rep, f"原子の編集：{cif.name}", poscar, output_dir,
        source=cif.resolve(), settings={"operations": len(log), "formula": rep["output"]["formula"]},
    )


def _create_structure_impl(
    spacegroup: str | int,
    lattice: Any,
    sites: Any,
    name: str | None = None,
    merge_tol: float | None = 0.01,
    poscar: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Shared by the MCP tool `create_structure` and `server.py create` (batch 22)."""
    from shido.core.editor import create_structure as _create, edit_report

    struct = _create(spacegroup, lattice, sites, merge_tol=merge_tol)
    rep = edit_report(None, struct, [f"from_spacegroup {spacegroup}: {len(struct)} sites"])
    stem = name or struct.composition.reduced_formula
    return _save_structure_run(
        "create", stem, struct, f"{stem}_created", rep, f"構造の新規作成：{stem}", poscar, output_dir,
        settings={"formula": rep["output"]["formula"]},
    )


def run_demo_v2(
    base_dir: str | Path | None = None,
    transparent: bool | None = None,
) -> dict[str, Any]:
    """Run all 3 v2.md killer features end-to-end (Design 1, 2, and 3)."""
    transparent = bool(_or_setting(transparent, "render.transparent_background"))
    is_new = not base_dir
    root = Path(base_dir) if base_dir else new_run_dir("demo_v2")
    pre = "" if is_new else "output_"

    # Design 1: Austenite (L2_1) vs Martensite (L1_0, c/a=1.25)
    a0 = float(cfg.get("lattice.a_austenite"))
    parent = build_ni2mnga_structure(a0)
    mart = build_ni2mnga_martensite_l10(a0, c_over_a=float(cfg.get("lattice.martensite_c_over_a")))
    d1_res = compare_phases(
        parent,
        mart,
        root / f"{pre}phase_compare",
        transparent_background=transparent,
    )

    # Design 2: Heusler off-stoichiometric Ni50Mn35In15 L2_1 sublattice & Mn-Mn pair highlight
    h = cfg.get("heusler")
    d2_res = generate_heusler_models(
        formula=h["formula"],
        ordering=h["ordering"],
        supercell=tuple(h["supercell"]),
        num_patterns=int(h["num_patterns"]),
        a=float(cfg.get("lattice.heusler_a")),
        output_dir=root / f"{pre}heusler_{sanitize_name(h['formula'])}_{h['ordering']}",
        highlight_pair=tuple(h["highlight_pair"]),
        seed=int(h["seed"]),
        transparent_background=transparent,
    )

    # Design 3: DFT c/a sweep (0.85..1.35) + synthetic Bain double-well energy curve + thumbnails
    ca_dir = root / f"{pre}ca_sweep_Ni2MnGa"
    d3_pre = generate_ca_sweep(
        parent,
        ca_dir,
        ca_min=float(cfg.get("ca_sweep.ca_min")),
        ca_max=float(cfg.get("ca_sweep.ca_max")),
        ca_step=float(cfg.get("ca_sweep.ca_step")),
    )
    demo_e_csv = generate_demo_ca_energy_csv(
        d3_pre["sweep_table_csv"],
        ca_dir / DATA_DIR / "demo_ca_energy.csv",
    )
    d3_post = plot_ca_energy_curve(demo_e_csv, parent, ca_dir, points_label="Model energies (demo, not DFT)")

    return _finish_run(
        {
            "design1_phase_compare": d1_res,
            "design2_heusler_models": d2_res,
            "design3_ca_sweep": d3_pre,
            "design3_ca_plot": d3_post,
        },
        root,
        "デモ v2：2 相比較・ホイスラー合金モデル・c/a スイープ",
        settings={"transparent": transparent},
        write_index=is_new,
    )


def run_mcp_server() -> None:
    """Start the MCP stdio server exposing crystal analysis tools."""
    print(f"[CrystalAnalyzerMCP v{__version__}] MCP server starting (stdio)...", file=sys.stderr)
    mcp = create_mcp_server()
    mcp.run()


def _print_bulk_result(bulk_res: dict[str, Any]) -> None:
    """Print a clean CLI summary of bulk analysis outputs."""
    print(f"Bulk output dir     : {bulk_res['output_dir']}")
    print(f"Lattice summary CSV : {bulk_res['lattice_summary_csv']}")
    print(f"Bond lengths CSV    : {bulk_res['bond_lengths_csv']}")
    print(f"Detected bonds      : {bulk_res['num_bonds']}")
    if bulk_res.get("cif_file"):
        print(f"Exported CIF        : {bulk_res['cif_file']}")
    if bulk_res.get("poscar_file"):
        print(f"Exported POSCAR     : {bulk_res['poscar_file']}")
    if bulk_res.get("figure_overview"):
        print(f"Overview figure PNG : {bulk_res['figure_overview']}")
    if bulk_res.get("figure_overview_svg"):
        print(f"Overview figure SVG : {bulk_res['figure_overview_svg']}")
    if bulk_res.get("poscar_cell") and bulk_res.get("poscar_cell") != "input":
        print(f"POSCAR cell         : {bulk_res['poscar_cell']} ({bulk_res.get('poscar_num_sites')} sites)")
    if bulk_res.get("xrd_peaks_csv"):
        print(f"XRD peaks CSV       : {bulk_res['xrd_peaks_csv']} ({bulk_res.get('xrd_n_peaks')} peaks)")
    if bulk_res.get("xrd_figure_png"):
        print(f"XRD figure PNG      : {bulk_res['xrd_figure_png']}")
    if bulk_res.get("xrd_profile_csv"):
        print(f"XRD profile CSV     : {bulk_res['xrd_profile_csv']}")
    if bulk_res.get("xrd_observed"):
        obs = bulk_res["xrd_observed"]
        print(f"XRD vs observed PNG : {obs.get('figure_png')} (R = {obs.get('r_factor_percent'):.2f} %)")
    for w in bulk_res.get("xrd_warnings") or []:
        print(f"XRD warning         : {w}")
    if bulk_res.get("stl_file"):
        print(f"STL model           : {bulk_res['stl_file']}")
    for f in bulk_res.get("stl_split_files") or []:
        print(f"STL (split)         : {f}")
    if bulk_res.get("html_file"):
        print(f"3D viewer HTML      : {bulk_res['html_file']}")
    for fmt, path in (bulk_res.get("exported_files") or {}).items():
        print(f"Exported {fmt.upper():<10} : {path}")
    if bulk_res.get("animation"):
        print(f"Rotation animation  : {bulk_res['animation'].get('path')}")
    summary = bulk_res.get("summary", {})
    if summary.get("parse_warnings"):
        print(f"Parse warnings      : {summary['parse_warnings']}")
    for img_path in bulk_res.get("images", []):
        print(f"Rendered image      : {img_path}")


def _print_slab_result(slab_res: dict[str, Any]) -> None:
    """Print a clean CLI summary of slab generation outputs."""
    if "output_dir" in slab_res:
        print(f"Slab output dir     : {slab_res['output_dir']}")
    if "d_hkl_angstrom" in slab_res:
        print(f"d_hkl (Angstrom)    : {slab_res['d_hkl_angstrom']}")
    if "raw_terminations" in slab_res:
        print(f"Raw terminations    : {slab_res['raw_terminations']}")
    if "after_dedup" in slab_res:
        print(f"After dedup         : {slab_res['after_dedup']}")
    if "polar_filtered_count" in slab_res:
        print(f"Polar filtered      : {slab_res['polar_filtered_count']}")
    if "polar_fallback" in slab_res:
        print(f"Polar fallback used : {slab_res['polar_fallback']}")
    if "num_terminations" in slab_res:
        print(f"Terminations        : {slab_res['num_terminations']}")
    if "surface_summary_csv" in slab_res:
        print(f"Surface summary CSV : {slab_res['surface_summary_csv']}")
    for cif in slab_res.get("cif_files", []):
        print(f"Slab CIF            : {cif}")
    for poscar in slab_res.get("poscar_files", []):
        print(f"Slab POSCAR         : {poscar}")
    for img in slab_res.get("images", []):
        print(f"Slab image          : {img}")


def _print_batch_result(batch_res: dict[str, Any]) -> None:
    """Print a clean CLI summary of batch analysis outputs."""
    print(f"Batch output dir    : {batch_res['output_dir']}")
    print(f"Total CIFs found    : {batch_res['total_cifs']}")
    print(f"Succeeded           : {batch_res['succeeded']}")
    print(f"Failed              : {batch_res['failed']}")
    print(f"Batch summary CSV   : {batch_res['batch_summary_csv']}")
    if batch_res.get("batch_gallery_png"):
        print(f"Batch gallery PNG   : {batch_res['batch_gallery_png']}")


def _parse_wavelength(value: str | None) -> str | float | None:
    """'CuKa' stays a key; '1.5406' becomes a float in Angstrom."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _xrd_extension_options(
    radiation: str | None = None,
    profile: str | None = None,
    fwhm: float | None = None,
    caglioti: Any = None,
    eta: float | None = None,
    observed: str | None = None,
) -> dict[str, Any]:
    """Only the batch-12 XRD options that were given (so defaults stay byte-identical)."""
    opts: dict[str, Any] = {}
    if radiation and str(radiation).strip().lower() not in ("xray", "x-ray"):
        opts["radiation"] = radiation
    if profile:
        opts["profile"] = profile
    if fwhm is not None:
        opts["fwhm"] = float(fwhm)
    if isinstance(caglioti, str):
        caglioti = [p for p in caglioti.replace(",", " ").split() if p]
    if caglioti is not None and any(v is not None for v in caglioti):
        if len(caglioti) != 3 or any(v is None for v in caglioti):
            raise ValueError(f"Caglioti needs all three of U, V, W, got {caglioti!r}.")
        opts["caglioti"] = tuple(float(v) for v in caglioti)
    if eta is not None:
        opts["eta"] = float(eta)
    if observed:
        opts["observed"] = observed
    return opts


def _bulk_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """v3 output options for `run_bulk_analysis` (all off unless the flag is given)."""
    xrd_options = _xrd_extension_options(
        args.radiation, args.xrd_profile, args.fwhm, args.caglioti, args.eta, args.xrd_observed
    )
    return {
        "poscar_cell": args.poscar_cell,
        "xrd": bool(args.xrd or args.wavelength is not None or xrd_options),
        "xrd_options": xrd_options,
        "wavelength": _parse_wavelength(args.wavelength),
        "stl": bool(args.stl or args.stl_split or args.stl_max_mm is not None),
        "stl_split": bool(args.stl_split),
        "stl_max_mm": args.stl_max_mm,
        "html": bool(args.html),
        "style": args.style,
        "view_dir": args.view_dir,
        "view_normal": args.view_normal,
        "perspective": bool(args.perspective),
        "labels": None if args.labels == "none" else args.labels,
        "bonds": args.bond,
        "polyhedra_center": args.polyhedra_center,
        "bond_color": args.bond_color,
        "frac_range": args.frac_range,
        "image_formats": args.image_format,
        "show_planes": args.plane_show,
        "cutoff_planes": args.cutoff_plane,
        "molecules": bool(args.molecules),
        "whole_molecules": bool(args.whole_molecules),
        "hbonds": args.hbond,
        **({"overlays": ov} if (ov := _overlay_specs(
            args.overlay, args.overlay_shift, args.overlay_rotate,
            "frac" if args.overlay_frac else None, args.overlay_opacity, args.overlay_color,
            args.overlay_about)) else {}),
        **_vector_kwargs(
            args.vector, args.vector_basis, args.vector_scale, args.vector_color, args.vector_radius
        ),
        **_volume_kwargs(
            args.volume, args.isovalue, args.iso_color, args.iso_color_negative, args.iso_opacity,
            args.iso_both_signs,
        ),
        **_ellipsoid_kwargs(args.style, None, args.probability),  # adp = the input CIF, set in main
        **({"animate": args.animate, "animate_frames": args.frames, "animate_axis": args.animate_axis}
           if args.animate else {}),
        **({"supercell": list(args.supercell)} if args.supercell and list(args.supercell) != [1, 1, 1] else {}),
        **({"export_formats": args.export_formats} if args.export_formats else {}),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP",
        description=(
            f"CrystalAnalyzerMCP v{__version__}: crystal structure analyzer, renderer, "
            "slab cutter, and batch processor. "
            "Run with no arguments to start the FastMCP server."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"CrystalAnalyzerMCP v{__version__}",
    )
    parser.add_argument(
        "target",
        help=(
            "'demo' to run the Ni2MnGa L21 demo, 'batch' followed by a CIF directory, "
            "a directory path (auto-batch mode), or path to a single .cif file."
        ),
    )
    parser.add_argument(
        "batch_dir",
        nargs="?",
        default=None,
        help="Input directory of .cif files when using 'batch <cif_dir>'.",
    )
    parser.add_argument(
        "--hkl",
        nargs=3,
        type=int,
        metavar=("H", "K", "L"),
        default=None,
        help="Miller index (h k l) for slab cutting (default: 1 1 1 in demo mode).",
    )
    parser.add_argument(
        "--plane",
        type=str,
        default=None,
        help=(
            "Major plane preset or shorthand (e.g. '111', '(110)', 'slip-bcc', 'habit', "
            "or 'list' to display presets)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base output directory (default: a new folder 出力/<date>_<name>/ in crystal-mcp).",
    )
    parser.add_argument(
        "--no-polyhedra",
        action="store_true",
        help="Disable coordination polyhedra rendering.",
    )
    parser.add_argument(
        "--polyhedra",
        action="store_true",
        help="Always draw coordination polyhedra (default: only when the structure has an "
             "anion-forming element such as O, N, S or a halogen; metals get none).",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Render transparent-background PNGs for publication slides/overlays.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Color theme name from themes.yaml (e.g., 'acta_modern', 'okabe_ito_safe').",
    )
    parser.add_argument(
        "--keep-polar",
        action="store_true",
        help="Keep polar (asymmetric/dipolar) slab terminations instead of filtering them.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable symmetry/structural deduplication of slab terminations.",
    )
    parser.add_argument(
        "--min-slab-size",
        type=float,
        default=None,
        help="Minimum slab thickness in Angstroms (default: slab.min_slab_size in settings).",
    )
    parser.add_argument(
        "--min-vacuum-size",
        type=float,
        default=None,
        help="Minimum vacuum thickness in Angstroms (default: slab.min_vacuum_size in settings).",
    )
    v3 = parser.add_argument_group("v3 outputs (off unless given)")
    v3.add_argument("--xrd", action="store_true",
                    help="Also compute the powder XRD pattern (XRD/xrd_peaks.csv, XRD/figure_xrd.png/svg).")
    v3.add_argument("--wavelength", type=str, default=None,
                    help="XRD source key (CuKa, MoKa, CuKa1, ...) or wavelength in Angstrom; implies --xrd.")
    v3.add_argument("--radiation", choices=["xray", "neutron"], default=None,
                    help="Powder diffraction radiation; 'neutron' writes neutron_peaks.csv / figure_neutron.png. Implies --xrd.")
    v3.add_argument("--xrd-profile", nargs="?", const="pseudo_voigt", default=None,
                    choices=["gaussian", "lorentzian", "pseudo_voigt"],
                    help="Draw the pattern as a curve with this peak shape (default pseudo_voigt; XRD/xrd_profile.csv). Implies --xrd.")
    v3.add_argument("--fwhm", type=float, default=None,
                    help="Constant peak FWHM in deg 2theta for --xrd-profile (default 0.1). Implies --xrd-profile.")
    v3.add_argument("--caglioti", type=str, default=None, metavar="U,V,W",
                    help="Caglioti widths FWHM^2 = U tan^2(theta) + V tan(theta) + W (deg^2), e.g. '0.02,-0.01,0.01'. Implies --xrd-profile.")
    v3.add_argument("--eta", type=float, default=None,
                    help="Lorentzian fraction of the pseudo-Voigt profile (0..1, default 0.5).")
    v3.add_argument("--xrd-observed", type=str, default=None, metavar="FILE",
                    help="Measured pattern (.xy / .csv: 2theta, intensity) to overlay with the scaled calculation and difference. Implies --xrd.")
    v3.add_argument("--stl", action="store_true",
                    help="Also export a 3D-print model (3D/structure.stl, mm).")
    v3.add_argument("--stl-split", action="store_true",
                    help="Also write one STL per element plus bonds/cell (multi-colour printing); implies --stl.")
    v3.add_argument("--stl-max-mm", type=float, default=None,
                    help="Shrink the STL model to fit this size in mm on every axis; implies --stl.")
    v3.add_argument("--style", choices=list(RENDER_STYLES), default="ball-stick",
                    help="Display style of the bulk views: ball-stick (default), space-filling, stick, wireframe "
                         "or ellipsoid (displacement ellipsoids from the CIF's ADPs).")
    v3.add_argument("--probability", type=float, default=None, metavar="P",
                    help="Probability of the --style ellipsoid ellipsoids in percent (default 50).")
    view_grp = v3.add_mutually_exclusive_group()
    view_grp.add_argument("--view-dir", nargs=3, type=float, metavar=("U", "V", "W"), default=None,
                          help="Also render 図/view_custom.png looking along [u v w] (e.g. 1 1 0).")
    view_grp.add_argument("--view-normal", nargs=3, type=float, metavar=("H", "K", "L"), default=None,
                          help="Also render 図/view_custom.png looking along the (h k l) plane normal.")
    v3.add_argument("--perspective", action="store_true",
                    help="Perspective camera instead of parallel projection (no scale bar).")
    v3.add_argument("--labels", choices=["none", "site", "element"], default="none",
                    help="Atom labels: site (CIF site label, e.g. Ni1), element, or none (default).")
    v3.add_argument("--bond", action="append", default=None, metavar="A-B:DMIN:DMAX",
                    help="Only these bonds, e.g. Mn-Ga:0:3.0 (repeatable; A-B:DMAX means DMIN=0). "
                         "Replaces the automatic search in the images and the bond CSVs.")
    v3.add_argument("--polyhedra-center", type=str, default=None, metavar="EL[,EL]",
                    help="Centre element(s) of the coordination polyhedra, e.g. Mn (default: least frequent element).")
    v3.add_argument("--bond-color", choices=["single", "split"], default=None,
                    help="Ball-stick bond colour: single theme colour (default) or split half-and-half by element.")
    v3.add_argument("--range", dest="frac_range", nargs=6, type=float, default=None,
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="Draw every atom with fractional coords in this box, e.g. -0.5 1.5 -0.5 1.5 -0.5 1.5.")
    v3.add_argument("--supercell", nargs="+", type=int, default=None, metavar="N",
                    help="Repeat the cell before analysis/rendering: 3 integers (e.g. 2 2 1) or 9 (3x3 matrix).")
    v3.add_argument("--export", dest="export_formats", type=str, default=None, metavar="FMT[,FMT]",
                    help="Also export structure files in 構造/ (and 3D/ for wrl): xyz, p1, xtl, pdb, wrl, vesta.")
    v3.add_argument("--animate", choices=["gif", "mp4"], default=None,
                    help="Also write 図/rotation.gif (or .mp4; needs ffmpeg / imageio-ffmpeg, else GIF): "
                         "the isometric (or --view-dir) view turning once, with the same style/range/labels.")
    v3.add_argument("--frames", type=int, default=72, help="Frames per turn for --animate (default 72).")
    v3.add_argument("--animate-axis", choices=["up", "a", "b", "c", "x", "y", "z"], default="up",
                    help="Rotation axis for --animate: up (screen vertical, default), a/b/c or x/y/z.")
    v3.add_argument("--image-format", type=str, default=None, metavar="FMT[,FMT]",
                    help="Also write the figures as tiff, jpeg, pdf, eps and/or svg (comma list; PNG is always "
                         "written). PDF/EPS/SVG embed the bitmap, they are not vector drawings.")
    v3.add_argument("--plane-show", action="append", default=None, metavar="'H K L[@D][:COLOR][:OPACITY]'",
                    help="Draw this (hkl) lattice plane in the unit cell on the bulk views (repeatable), "
                         "e.g. '1 1 1' or '1 1 0@0.5:#ff0000:0.4'. D: plane h*x+k*y+l*z=D (default: D=1 "
                         "for 111); COLOR: #rrggbb or a name; OPACITY 0-1.")
    v3.add_argument("--vector", action="append", default=None, metavar="'TARGET:X Y Z' | magmom",
                    help="Draw an arrow (magnetic moment) through these atoms on the bulk views and the HTML "
                         "viewer (repeatable). TARGET: element (Mn), site label (Mn1) or index (#3); X Y Z "
                         "along a/|a|, b/|b|, c/|c| in mu_B like mCIF (see --vector-basis), e.g. 'Mn:0 0 4'. "
                         "'magmom' takes the moments of an mCIF (_atom_site_moment.*).")
    v3.add_argument("--vector-basis", choices=["crystal", "cartesian"], default=None,
                    help="Basis of the --vector components: crystal (default) or cartesian x y z (a along x).")
    v3.add_argument("--vector-scale", type=float, default=None, metavar="A_PER_UNIT",
                    help="Arrow length in Angstrom per unit (mu_B) of the vector (default 1.0).")
    v3.add_argument("--vector-color", type=str, default=None, help="Arrow colour (#rrggbb or a name; default #E4572E).")
    v3.add_argument("--vector-radius", type=float, default=None, metavar="ANGSTROM",
                    help="Arrow shaft radius in Angstrom (default 0.08; the head is 2.2x).")
    v3.add_argument("--volume", type=str, default=None, metavar="FILE",
                    help="Overlay isosurfaces of this volumetric file on the bulk views: VASP CHGCAR / LOCPOT / "
                         "ELFCAR, Gaussian *.cube, XSF (DATAGRID_3D) or VESTA *.grd. A CHGCAR / Cube / XSF "
                         "given as the target is drawn with its own structure and isosurfaces.")
    v3.add_argument("--isovalue", type=str, default=None, metavar="V[,V]",
                    help="Isosurface level(s), e.g. 0.05 or 0.05,0.1 (CHGCAR in e/A^3 = file value / V_cell; "
                         "Cube as stored). Default: 10%% of the max |value|.")
    v3.add_argument("--iso-color", type=str, default=None, metavar="COLOR[,COLOR]",
                    help="Isosurface colour(s), one per level (cycled); default #F2C200.")
    v3.add_argument("--iso-color-negative", type=str, default=None,
                    help="Colour of negative levels (default #2F9BD6).")
    v3.add_argument("--iso-opacity", type=float, default=None, help="Isosurface opacity 0-1 (default 0.55).")
    v3.add_argument("--iso-both-signs", action="store_true",
                    help="Also draw -V for each level (two colours; e.g. charge differences, LOCPOT).")
    v3.add_argument("--cutoff-plane", action="append", default=None, metavar="'H K L@D'",
                    help="Draw only the atoms with h*x+k*y+l*z <= D (fractional; repeatable, all must hold), "
                         "e.g. '1 1 1@1.5'. Bonds and polyhedra follow. '-1 -1 -1@-1.5' keeps the other side.")
    v3.add_argument("--molecules", action="store_true",
                    help="Find molecules (atoms joined by covalent bonds, across cell edges) and write "
                         "数値/molecules.csv (composition, centre of mass).")
    v3.add_argument("--whole-molecules", action="store_true",
                    help="Draw every molecule whole (not cut at the cell edge; centre of mass in the cell or "
                         "--range). Implies --molecules.")
    v3.add_argument("--hbond", action="append", default=None, metavar="'D-H...A:DMAX:ANGLE'",
                    help="Find hydrogen bonds D-H...A with H...A <= DMAX A and angle D-H...A >= ANGLE deg "
                         "(repeatable), e.g. 'O-H...O:2.5:120'. Writes 数値/hbonds.csv and draws them dashed.")
    v3.add_argument("--overlay", action="append", default=None, metavar="FILE",
                    help="Draw this structure (CIF / POSCAR / XSF / .vesta) in the same bulk views (repeatable; "
                         "e.g. a molecule on a surface, the other grain of a boundary).")
    v3.add_argument("--overlay-shift", action="append", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Shift of the matching --overlay (n-th shift = n-th overlay) in Angstrom "
                         "(or fractions of the main cell with --overlay-frac).")
    v3.add_argument("--overlay-rotate", action="append", default=None, metavar="'AX AY AZ DEG'",
                    help="Rotation of the matching --overlay: Cartesian axis and angle in degrees, e.g. "
                         "'0 0 1 90' (or 9 numbers: a rotation matrix). Applied before the shift.")
    v3.add_argument("--overlay-frac", action="store_true",
                    help="--overlay-shift is in fractions of the main cell instead of Angstrom.")
    v3.add_argument("--overlay-about", choices=["origin", "center"], default=None,
                    help="Rotate the overlays about the origin (default) or the centre of their own cell.")
    v3.add_argument("--overlay-opacity", type=float, default=None,
                    help="Opacity 0-1 of the overlaid structures (default 1; e.g. 0.5).")
    v3.add_argument("--overlay-color", type=str, default=None,
                    help="Paint the overlaid structures in one colour (#rrggbb or a name; default element colours).")
    v3.add_argument("--html", action="store_true",
                    help="Also export an offline mouse-rotatable 3D viewer (3D/structure.html).")
    v3.add_argument("--poscar-cell", choices=list(POSCAR_CELL_CHOICES), default="input",
                    help="Cell written to POSCAR: input (as given, default), conventional or primitive.")
    v3.add_argument("--all-blocks", action="store_true",
                    help="Process every data block of a multi-block CIF into its own output folder.")
    return parser


def _run_interactive_cli(base_dir: Path | None) -> int:
    """Terminal interactive wizard for selecting plane presets, elements, and theme."""
    print("=== CrystalAnalyzerMCP 対話メニュー ===")
    print("実行モードを選んでください:")
    print("  1. 基本解析 + スラブ切り出し (Ni2MnGa デモ or CIF指定)")
    print("  2. 変態前後 2相比較 (Ni2MnGa L21 vs L10)")
    print("  3. ホイスラー副格子・置換モデル (組成・元素ペア指定)")
    print("  4. DFT c/a スイープ + エネルギー曲線")
    mode = input("番号 [1-4, Enter=1]: ").strip() or "1"

    themes = list_themes()
    print("\nカラーテーマ:")
    for i, t in enumerate(themes, start=1):
        print(f"  {i}. {t}")
    t_in = input(f"番号 [1-{len(themes)}, Enter=1]: ").strip()
    if t_in.isdigit() and 1 <= int(t_in) <= len(themes):
        set_active_theme(themes[int(t_in) - 1])

    if mode == "1":
        cif_in = input("\nCIFファイルパス (空欄で Ni2MnGa デモ): ").strip()
        print("\n切り出し面 (ミラー指数):")
        for i, p in enumerate(MAJOR_PLANES, start=1):
            print(f"  {i}. {p['label']}")
        p_in = input(f"番号 [1-{len(MAJOR_PLANES)}] または 'h k l' 直接入力 (Enter=1): ").strip() or "1"
        if p_in.isdigit() and 1 <= int(p_in) <= len(MAJOR_PLANES):
            hkl = tuple(MAJOR_PLANES[int(p_in) - 1]["hkl"])
        else:
            hkl = resolve_miller_index(p_in)
        if not cif_in:
            res = run_demo(miller_index=hkl, base_dir=base_dir)
            _print_bulk_result(res["bulk_analysis"])
            _print_slab_result(res["slab_generation"])
            _print_saved(res)
        else:
            cif_p = Path(cif_in)
            struct, warns = _unpack_robust_load(load_structure_robust(cif_p))
            stem = sanitize_name(cif_p.stem)
            hkl_tag = f"{hkl[0]}{hkl[1]}{hkl[2]}"
            if base_dir is None:
                bulk_dir = new_run_dir("", cif_p.stem)
                slab_dir = bulk_dir / f"slab_{hkl_tag}"
            else:
                bulk_dir = base_dir / f"output_{stem}"
                slab_dir = base_dir / f"output_slab_{stem}_{hkl_tag}"
            bulk_res = run_bulk_analysis(struct, bulk_dir, parse_warnings=warns)
            slab_res = cut_slabs(struct, hkl, slab_dir)
            _print_bulk_result(bulk_res)
            _print_slab_result(slab_res)
            _print_saved(_finish_run({}, bulk_dir, f"CIF の解析：{cif_p.name}", source=cif_p.resolve(),
                                     settings={"miller_index": list(hkl)}))
        return 0

    if mode == "2":
        out = base_dir / "output_phase_compare" if base_dir else new_run_dir("phase_compare", "Ni2MnGa")
        compare_phases(build_ni2mnga_structure(), build_ni2mnga_martensite_l10(), out)
        _print_saved(_finish_run({}, out, "変態前後の 2 相の比較（Ni2MnGa L21 と L10）"))
        return 0

    if mode == "3":
        hd = cfg.get("heusler")
        formula = input(f"\n合金組成 (Enter={hd['formula']}): ").strip() or hd["formula"]
        ordering = (input(f"規則度 L21 / B2 / XA（逆ホイスラー・規則のみ） (Enter={hd['ordering']}): ").strip() or hd["ordering"]).upper()
        el1 = input(f"強調する元素ペア1 (Enter={hd['highlight_pair'][0]}): ").strip() or hd["highlight_pair"][0]
        el2 = input(f"強調する元素ペア2 (Enter={hd['highlight_pair'][1]}): ").strip() or hd["highlight_pair"][1]
        out = (
            base_dir / f"output_heusler_{sanitize_name(formula)}_{ordering}" if base_dir
            else new_run_dir("heusler", f"{sanitize_name(formula)}_{ordering}")
        )
        generate_heusler_models(formula=formula, ordering=ordering, output_dir=out, highlight_pair=(el1, el2))
        _print_saved(_finish_run({}, out, f"ホイスラー合金の副格子モデル：{formula}",
                                 settings={"formula": formula, "ordering": ordering}))
        return 0

    if mode == "4":
        parent = build_ni2mnga_structure(float(cfg.get("lattice.a_austenite")))
        ca_dir = base_dir / "output_ca_sweep_Ni2MnGa" if base_dir else new_run_dir("ca_sweep", "Ni2MnGa")
        sw = generate_ca_sweep(
            parent,
            ca_dir,
            ca_min=float(cfg.get("ca_sweep.ca_min")),
            ca_max=float(cfg.get("ca_sweep.ca_max")),
            ca_step=float(cfg.get("ca_sweep.ca_step")),
        )
        e_csv = generate_demo_ca_energy_csv(sw["sweep_table_csv"], ca_dir / DATA_DIR / "demo_ca_energy.csv")
        plot_ca_energy_curve(e_csv, parent, ca_dir, points_label="Model energies (demo, not DFT)")
        _print_saved(_finish_run({}, ca_dir, "c/a スイープとエネルギー曲線（Ni2MnGa）"))
        return 0

    print("無効な番号です。", file=sys.stderr)
    return 1


def _run_measure_cli(argv: list[str]) -> int:
    """`server.py measure <cif> --atoms ... --planes ... --best-plane ...`"""
    import json

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP measure",
        description=(
            "Measure distance (2 atoms), angle (3), dihedral (4), the angle between two "
            "(hkl) planes, or a best plane. Atom = site index, CIF label or element+ordinal, "
            "optionally with an image: 'Na1@1,0,0'."
        ),
    )
    p.add_argument("cif", help="Path to a .cif file.")
    p.add_argument("--atoms", nargs="+", default=None, metavar="ATOM",
                   help="2-4 atoms, e.g. Si1 Si2 Si5@-1,-1,0")
    p.add_argument("--planes", nargs=2, default=None, metavar="HKL",
                   help="Two Miller indices, e.g. \"1 0 0\" \"1 1 0\"")
    p.add_argument("--best-plane", nargs="+", default=None, metavar="ATOM",
                   help=">= 3 atoms for a least-squares plane.")
    p.add_argument("--out-dir", default=None,
                   help="Write measure.json here (default: a new folder 出力/<date>_measure_<cif>/数値/).")
    args = p.parse_args(argv)
    cif = Path(args.cif)
    if not cif.is_file():
        print(f"Error: CIF file not found: {cif}", file=sys.stderr)
        return 1
    struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif))
    try:
        res = _measure_geometry(
            struct, atoms=args.atoms, planes=args.planes, best_plane_atoms=args.best_plane
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for w in parse_warnings:
        print(f"[warning] {w}")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    saved = _save_numeric_run(
        "measure", cif, args.out_dir, {}, "距離・角度の測定",
        lambda d: {"json": str(_write_json(res, d / "measure.json"))},
    )
    _print_saved(saved)
    return 0


def _run_symmetry_cli(argv: list[str]) -> int:
    """`server.py symmetry <cif> [--symprec 0.01] [--out-dir DIR]`"""
    import json

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP symmetry",
        description=(
            "Symmetry table: space group, symmetry operations (x,y,z), Wyckoff letter, "
            "multiplicity and site symmetry per site. Writes JSON + CSV with --out-dir."
        ),
    )
    p.add_argument("cif", help="Path to a .cif file.")
    p.add_argument("--symprec", type=float, default=0.01, help="spglib tolerance (A).")
    p.add_argument("--out-dir", default=None,
                   help="Write symmetry_info.json / symmetry_sites.csv / symmetry_operations.csv here.")
    args = p.parse_args(argv)
    cif = Path(args.cif)
    if not cif.is_file():
        print(f"Error: CIF file not found: {cif}", file=sys.stderr)
        return 1
    struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif))
    try:
        info = compute_symmetry_info(struct, symprec=args.symprec)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for w in parse_warnings:
        print(f"[warning] {w}")
    print(f"{info['spacegroup_symbol']} (No. {info['spacegroup_number']}), "
          f"{info['n_operations']} operations in this cell")
    for o in info["orbits"]:
        print(f"  {o['label'] or o['element']}: {o['wyckoff']}  site symmetry {o['site_symmetry']}"
              f"  ({o['n_in_cell']} in cell)")
    saved = _save_numeric_run("symmetry", cif, args.out_dir, {}, "対称性（空間群・Wyckoff）",
                              lambda d: write_symmetry_report(info, d))
    if args.out_dir:
        print(json.dumps(saved["files"], ensure_ascii=False, indent=2))
    _print_saved(saved)
    return 0


def _run_transform_cli(argv: list[str]) -> int:
    """`server.py transform <cif> [--cell niggli] [--transform P..] [--origin-shift p] [--supercell 2 2 1]`"""
    import json

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP transform",
        description=(
            "Cell transformation: Niggli / standard / primitive cell, matrix P + origin shift "
            "((a',b',c') = (a,b,c)P, x' = P^-1(x-p)), supercell. Writes CIF + POSCAR with --out-dir."
        ),
    )
    p.add_argument("cif", help="Path to a .cif file.")
    p.add_argument("--cell", default="input",
                   choices=["input", "niggli", "conventional", "standard", "primitive"])
    p.add_argument("--transform", default=None, metavar="P",
                   help='9 numbers row-major (fractions OK) or 3 (diagonal), e.g. "0 1/2 1/2 1/2 0 1/2 1/2 1/2 0"')
    p.add_argument("--origin-shift", default=None, metavar="p", help='e.g. "1/2 1/2 1/2"')
    p.add_argument("--supercell", nargs="+", default=None, metavar="N", help="e.g. 2 2 1")
    p.add_argument("--symprec", type=float, default=0.01, help="spglib tolerance (A).")
    p.add_argument("--cif-symprec", type=float, default=None,
                   help="Write the CIF with symmetry found at this tolerance (default: P1).")
    p.add_argument("--out-dir", default=None, help="Write <name>_transformed.cif / .vasp here.")
    args = p.parse_args(argv)
    cif = Path(args.cif)
    if not cif.is_file():
        print(f"Error: CIF file not found: {cif}", file=sys.stderr)
        return 1
    struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif))
    try:
        new, rep = _transform_cell(
            struct, cell=args.cell, transform=args.transform, origin_shift=args.origin_shift,
            supercell=" ".join(args.supercell) if args.supercell else None, symprec=args.symprec,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for w in parse_warnings + rep["warnings"]:
        print(f"[warning] {w}")
    a, b = rep["input"], rep["output"]
    print(f"{a['formula']}: {a['n_sites']} -> {b['n_sites']} sites, det(P) = {rep['det_P_total']}")
    print(f"  a b c = {b['a']} {b['b']} {b['c']} A, alpha beta gamma = "
          f"{b['alpha']} {b['beta']} {b['gamma']} deg, V = {b['volume']} A^3, "
          f"{b['spacegroup_symbol']} (No. {b['spacegroup_number']})")
    saved = _save_numeric_run(
        "transform", cif, args.out_dir, {}, "格子の変換",
        lambda d: write_transformed(new, d, stem=f"{cif.stem}_transformed", cif_symprec=args.cif_symprec),
        sub=STRUCT_DIR,
        settings={"cell": args.cell, "transform": args.transform, "origin_shift": args.origin_shift,
                  "supercell": args.supercell},
    )
    if args.out_dir:
        print(json.dumps(saved["files"], ensure_ascii=False, indent=2))
    _print_saved(saved)
    return 0


def _run_export_cli(argv: list[str]) -> int:
    """`server.py export <cif> [<cif> ...] [--formats xyz,p1,xtl,pdb,wrl,multicif,vesta] [--out-dir D]`"""
    import json

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP export",
        description=(
            "Export structure data: XYZ, P1 CIF, XTL, PDB, VRML (*.wrl), and all structures "
            "in one CIF (multicif)."
        ),
    )
    p.add_argument("cif", nargs="+", help="One or more .cif files (all data blocks are used).")
    p.add_argument("--formats", "--export", dest="formats", default="xyz,p1,xtl,pdb",
                   help='Comma list of xyz,p1,xtl,pdb,wrl,multicif,vesta or "all" (default: xyz,p1,xtl,pdb).')
    p.add_argument("--out-dir", default=None,
                   help="Output folder (default: a new folder 出力/<date>_export_<cif>/構造/).")
    p.add_argument("--multi-name", default=None, help="Stem of <name>_all.cif for multicif.")
    args = p.parse_args(argv)
    try:
        res = _export_structure_impl(args.cif, args.formats, args.out_dir, args.multi_name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for w in res["parse_warnings"] + res["warnings"]:
        print(f"[warning] {w}")
    for s in res["structures"]:
        print(f"{s['name']}: {s['formula']}, {s['n_sites']} sites")
    print(json.dumps({"structures": {s["name"]: s["files"] for s in res["structures"]},
                      "multicif": res["multicif"]}, ensure_ascii=False, indent=2))
    _print_saved(res)
    return 0


def _print_edit_result(res: dict[str, Any]) -> None:
    import json

    for w in res.get("parse_warnings", []):
        print(f"[warning] {w}")
    for line in res["operations"]:
        print(f"  {line}")
    o = res["output"]
    print(f"{o['formula']}: {o['n_sites']} sites, a b c = {o['a']} {o['b']} {o['c']} A, "
          f"V = {o['volume']} A^3, {o['spacegroup_symbol']} (No. {o['spacegroup_number']}), "
          f"d_min = {res['min_distance']} A")
    print(json.dumps(res["files"], ensure_ascii=False, indent=2))
    _print_saved(res)


def _run_edit_cli(argv: list[str]) -> int:
    """`server.py edit <cif> [--ops ops.json | --add "Li 0 0 0.5" --remove 3 --replace Na:K ...]`"""
    from shido.core.validator import ScientificValidationError

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP edit",
        description="Edit atoms / sites (add, remove, replace, move, occupancy, merge) and save a checked CIF.",
    )
    p.add_argument("cif", help="Structure file (.cif, POSCAR, ...).")
    p.add_argument("--ops", default=None,
                   help='JSON file or JSON text: [{"op":"replace","from":"Na","to":"K"}, ...] (applied first).')
    p.add_argument("--add", action="append", default=[], metavar='"El x y z [occ]"',
                   help="Add an atom at fractional x y z (repeatable).")
    p.add_argument("--remove", action="append", default=[], metavar="SITE|El",
                   help="Remove a site (0-based index) or every site of an element (repeatable).")
    p.add_argument("--replace", action="append", default=[], metavar="FROM:TO",
                   help="Replace an element (Na:K) or a site (3:K) (repeatable).")
    p.add_argument("--move", action="append", default=[], metavar='"SITE x y z"',
                   help="Move a site to fractional x y z (repeatable).")
    p.add_argument("--occupancy", action="append", default=[], metavar="SITE|El:VALUE",
                   help="Set the occupancy of a site or an element (Cl:0.5) (repeatable).")
    p.add_argument("--merge-tol", type=float, default=None,
                   help="Merge duplicate atoms closer than this (A) at the end.")
    p.add_argument("--poscar", action="store_true", help="Also write <name>_edited.vasp.")
    p.add_argument("--out-dir", default=None,
                   help="Output folder (default: a new folder 出力/<date>_edit_<cif>/構造/).")
    args = p.parse_args(argv)

    def sel(text: str) -> dict[str, Any]:
        t = text.strip()
        return {"site": int(t)} if t.lstrip("-").isdigit() else {"element": t}

    try:
        from shido.core.editor import normalize_operations

        ops: list[dict[str, Any]] = normalize_operations(args.ops) if args.ops else []
        for t in args.add:
            parts = t.split()
            op = {"op": "add", "element": parts[0], "coords": parts[1:4]}
            if len(parts) > 4:
                op["occupancy"] = float(parts[4])
            ops.append(op)
        for t in args.replace:
            a, b = t.split(":", 1)
            ops.append({"op": "replace", **({"site": int(a)} if a.strip().isdigit() else {"from": a.strip()}), "to": b.strip()})
        for t in args.move:
            parts = t.split()
            ops.append({"op": "move", "site": int(parts[0]), "coords": parts[1:4]})
        for t in args.occupancy:
            a, b = t.rsplit(":", 1)
            ops.append({"op": "occupancy", **sel(a), "value": float(b)})
        for t in args.remove:
            ops.append({"op": "remove", **sel(t)})
        if not ops and args.merge_tol is None:
            print("Error: give --ops or at least one of --add/--remove/--replace/--move/--occupancy/--merge-tol.",
                  file=sys.stderr)
            return 1
        res = _edit_structure_impl(args.cif, ops, args.merge_tol, args.poscar, args.out_dir)
    except (FileNotFoundError, ValueError, ScientificValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _print_edit_result(res)
    return 0


def _run_create_cli(argv: list[str]) -> int:
    """`server.py create --spacegroup Fm-3m --lattice 5.64 --site "Na 0 0 0" --site "Cl 0.5 0.5 0.5"`"""
    from shido.core.validator import ScientificValidationError

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP create",
        description="Build a structure from a space group, lattice and asymmetric unit, and save a checked CIF.",
    )
    p.add_argument("--spacegroup", "--sg", required=True, help="Symbol (Fm-3m) or number (225).")
    p.add_argument("--lattice", nargs="+", required=True, metavar="V",
                   help="a [b c alpha beta gamma]; missing values follow the crystal system (cubic: a only).")
    p.add_argument("--site", action="append", default=[], metavar='"El x y z [occ]"',
                   help="One atom of the asymmetric unit (repeatable).")
    p.add_argument("--sites", default=None, help='JSON file or text: [["Na",0,0,0],["Cl",0.5,0.5,0.5]].')
    p.add_argument("--name", default=None, help="File stem (default: the formula).")
    p.add_argument("--merge-tol", type=float, default=0.01, help="Merge duplicates closer than this (A).")
    p.add_argument("--poscar", action="store_true", help="Also write <name>_created.vasp.")
    p.add_argument("--out-dir", default=None,
                   help="Output folder (default: a new folder 出力/<date>_create_<name>/構造/).")
    args = p.parse_args(argv)
    try:
        import json

        sites: list[Any] = []
        if args.sites:
            text = Path(args.sites).read_text(encoding="utf-8-sig") if Path(args.sites).is_file() else args.sites
            sites += json.loads(text)
        for t in args.site:
            parts = t.split()
            sites.append([parts[0], *parts[1:]])
        names = ("a", "b", "c", "alpha", "beta", "gamma")
        lat = [float(v) for v in args.lattice]
        lattice = dict(zip(names, lat)) if len(lat) < 6 else lat
        if len(lat) == 2:  # a c (tetragonal / hexagonal)
            lattice = {"a": lat[0], "c": lat[1]}
        res = _create_structure_impl(args.spacegroup, lattice, sites, args.name, args.merge_tol,
                                     args.poscar, args.out_dir)
    except (FileNotFoundError, ValueError, ScientificValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _print_edit_result(res)
    return 0


def _run_polyhedra_cli(argv: list[str]) -> int:
    """`server.py polyhedra <cif> [--bond Na-Cl:0:3.0] [--madelung] [--oxi Na=1,Cl=-1] [--out-dir D]`"""
    from shido.core.polyhedra_metrics import analyze_polyhedra as _analyze_polyhedra, write_polyhedra_report

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP polyhedra",
        description=(
            "Polyhedron numbers per site (volume, distortion index, quadratic elongation, "
            "bond-angle variance, ECoN), bond valence sum, and with --madelung the Ewald "
            "energy / Madelung constant. Writes polyhedra.csv / .json with --out-dir."
        ),
    )
    p.add_argument("cif", help="Path to a .cif file.")
    p.add_argument("--bond", "--bond-spec", dest="bond_spec", action="append", default=None,
                   metavar="A-B:dmin:dmax", help="Bond search range (repeatable), same as the main CLI --bond.")
    p.add_argument("--madelung", action="store_true", help="Also compute the Madelung (Ewald) energy.")
    p.add_argument("--oxi", default=None, metavar="El=q,...", help='Oxidation states, e.g. "Na=1,Cl=-1".')
    p.add_argument("--out-dir", default=None, help="Write polyhedra.csv / polyhedra.json here.")
    args = p.parse_args(argv)
    cif = Path(args.cif)
    if not cif.is_file():
        print(f"Error: CIF file not found: {cif}", file=sys.stderr)
        return 1
    oxi = None
    try:
        if args.oxi:
            oxi = {k.strip(): float(v) for k, v in (x.split("=") for x in args.oxi.split(",") if x.strip())}
        struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif))
        df, rep = _analyze_polyhedra(struct, bond_specs=args.bond_spec, madelung=args.madelung,
                                     oxidation_states=oxi)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for w in parse_warnings:
        print(f"[warning] {w}")
    cols = ["site", "element", "coordination", "volume_A3", "mean_bond_length_A", "distortion_index",
            "quadratic_elongation", "bond_angle_variance_deg2", "ECoN", "bvs"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    mad = rep.get("madelung")
    if mad:
        if "error" in mad:
            print(f"[warning] Madelung: {mad['error']}")
        else:
            print(f"Madelung: E = {mad['total_energy_eV']:.4f} eV/cell, "
                  f"{mad['energy_per_formula_unit_eV']:.4f} eV/f.u. (oxidation {mad['oxidation_states']})")
            if mad.get("madelung_constant") is not None:
                print(f"  Madelung constant = {mad['madelung_constant']:.4f} (r0 = {mad['r0_A']:.4f} A)")
            if mad.get("warning"):
                print(f"[warning] {mad['warning']}")
    saved = _save_numeric_run("polyhedra", cif, args.out_dir, {}, "配位多面体の解析", lambda d: write_polyhedra_report(df, rep, d))
    files = saved["files"]
    print(f"Wrote {files['csv']} and {files['json']}")
    _print_saved(saved)
    return 0


def _run_structure_factors_cli(argv: list[str]) -> int:
    """`server.py structure-factors <cif> [--dmin 0.8] [--radiation neutron] [--b Na=0.5,Cl=0.6] [--out-dir D]`"""
    from shido.core.structure_factors import compute_structure_factors as _compute_sf, write_structure_factor_report

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP structure-factors",
        description=(
            "Structure factor table F(hkl): |F|, phase, F^2, d, 2theta, multiplicity, extinct "
            "(systematic / structure), X-ray or neutron. Writes structure_factors.csv / .json with --out-dir."
        ),
    )
    p.add_argument("cif", help="Path to a .cif file.")
    p.add_argument("--dmin", type=float, default=0.8, help="Smallest d-spacing in A (default 0.8).")
    p.add_argument("--radiation", default="xray", choices=["xray", "neutron"])
    p.add_argument("--wavelength", default=None, help="'CuKa', 'MoKa', ... or A (default: settings xrd.wavelength).")
    p.add_argument("--b", "--debye-waller", dest="b_iso", default=None, metavar="El=B,...",
                   help='Isotropic B in A^2 per element, e.g. "Na=0.5,Cl=0.6".')
    p.add_argument("--no-extinct", action="store_true", help="Leave out the zero (extinct) reflections.")
    p.add_argument("--out-dir", default=None, help="Write structure_factors.csv / .json here.")
    args = p.parse_args(argv)
    cif = Path(args.cif)
    if not cif.is_file():
        print(f"Error: CIF file not found: {cif}", file=sys.stderr)
        return 1
    try:
        dw = None
        if args.b_iso:
            dw = {k.strip(): float(v) for k, v in (x.split("=") for x in args.b_iso.split(",") if x.strip())}
        struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif))
        df, rep = _compute_sf(struct, dmin=args.dmin, radiation=args.radiation,
                              wavelength=_parse_wavelength(args.wavelength), debye_waller_factors=dw,
                              include_extinct=not args.no_extinct)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for w in parse_warnings + rep["warnings"]:
        print(f"[warning] {w}")
    print(f"{rep['formula']} ({rep['spacegroup_symbol']}), {rep['radiation']}, lambda = "
          f"{rep['wavelength_angstrom']:.5f} A, d >= {rep['dmin_angstrom']} A, F in {rep['F_unit']}: "
          f"{rep['n_families']} families ({rep['n_extinct_systematic']} systematic, "
          f"{rep['n_extinct_structure']} structure zeros)")
    cols = ["hkl", "d_angstrom", "two_theta_deg", "multiplicity", "F_abs", "phase_deg", "F_squared",
            "intensity_rel", "extinct"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    saved = _save_numeric_run("structure_factors", cif, args.out_dir, {}, "構造因子 F(hkl)", lambda d: write_structure_factor_report(df, rep, d))
    files = saved["files"]
    print(f"Wrote {files['csv']} and {files['json']}")
    _print_saved(saved)
    return 0


def _run_displacement_cli(argv: list[str]) -> int:
    """`server.py displacement <cif> [--probability 50] [--out-dir D]`"""
    from shido.core.adp import analyze_displacement as _analyze_adp, write_displacement_report

    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP displacement",
        description=(
            "Thermal displacement parameters from the CIF (aniso U/B/beta, U_iso/B_iso): U_eq, "
            "principal axes and ellipsoid semi-axes. Writes displacement.csv / .json with --out-dir."
        ),
    )
    p.add_argument("cif", help="Path to a .cif file.")
    p.add_argument("--probability", type=float, default=50.0, help="Ellipsoid probability in percent (default 50).")
    p.add_argument("--out-dir", default=None, help="Write displacement.csv / displacement.json here.")
    args = p.parse_args(argv)
    cif = Path(args.cif)
    if not cif.is_file():
        print(f"Error: CIF file not found: {cif}", file=sys.stderr)
        return 1
    try:
        struct, parse_warnings = _unpack_robust_load(load_structure_robust(cif))
        df, rep = _analyze_adp(cif, probability=args.probability, structure=struct)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for w in parse_warnings + rep["warnings"]:
        print(f"[warning] {w}")
    print(f"{rep['n_atoms']} atoms ({rep['n_anisotropic']} aniso, {rep['n_isotropic']} iso, "
          f"{rep['n_without_adp']} without ADP), probability {rep['probability_percent']:g}% "
          f"-> radius factor {rep['radius_scale']:.4f}")
    cols = ["label", "element", "adp_type", "U_eq", "rms1", "rms2", "rms3",
            "semi_axis1", "semi_axis2", "semi_axis3", "positive_definite", "n_sites_in_cell"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    saved = _save_numeric_run("displacement", cif, args.out_dir, {}, "熱振動（原子変位パラメータ）", lambda d: write_displacement_report(df, rep, d))
    files = saved["files"]
    print(f"Wrote {files['csv']} and {files['json']}")
    _print_saved(saved)
    return 0


def _run_slice_cli(argv: list[str]) -> int:
    """`server.py slice <volume> --slice "0 0 1@0.5" [--contours 0.05 ...] [--line-profile "x1 y1 z1 x2 y2 z2"]`"""
    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP slice",
        description=(
            "2D sections (colour map + contour lines, axes in A) and line profiles of volumetric "
            "data (CHGCAR / LOCPOT / ELFCAR / *.cube / *.xsf / *.grd)."
        ),
    )
    p.add_argument("volume", help="Volumetric data file.")
    p.add_argument("--slice", action="append", default=None, metavar='"H K L@POS"',
                   help="Plane h x + k y + l z = POS (POS in d_hkl). Repeat for several.")
    p.add_argument("--contours", action="append", default=None, metavar="LEVEL",
                   help="Contour level(s), e.g. 0.05 or 0.05,0.1 (repeatable). Default 8 automatic; 'none' = no lines.")
    p.add_argument("--line-profile", action="append", default=None, metavar='"X1 Y1 Z1 X2 Y2 Z2"',
                   help="Line in fractional coordinates (repeatable) -> PNG + CSV.")
    p.add_argument("--resolution", type=float, default=None, help="Pixel size of the map in A (default: half the grid spacing).")
    p.add_argument("--out-dir", default=None, help="Write the files here instead of a new 出力/ folder.")
    args = p.parse_args(argv)
    vol = Path(args.volume)
    if not vol.is_file():
        print(f"Error: volume file not found: {vol}", file=sys.stderr)
        return 1
    slices = args.slice
    if not slices and not args.line_profile:
        slices = ["0 0 1@0.5"]
    try:
        rep = _slice_volume_impl(vol, slices, args.contours, args.line_profile, args.resolution, args.out_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    v = rep["volume"]
    print(f"{vol.name}: {v['format']} grid {v['grid']}, {v['min']:.4g} .. {v['max']:.4g} ({v['unit']})")
    for s in rep["slices"]:
        print(f"  slice ({' '.join(str(h) for h in s['hkl'])})@{s['position']:g}: "
              f"{s['min']:.4g} .. {s['max']:.4g} -> {s['png']}")
    for lp in rep["line_profiles"]:
        print(f"  line {lp['start_frac']} -> {lp['end_frac']} ({lp['length_angstrom']:.3f} A): "
              f"{lp['min']:.4g} .. {lp['max']:.4g} -> {lp['png']}")
    _print_saved(rep)
    return 0


def _run_volume_model_cli(argv: list[str]) -> int:
    """`server.py volume-model <cif|volume> [--model-density xray|neutron] [--grid 64] [--patterson] [--volume-diff B] [--export-volume cube,grd]`"""
    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP volume-model",
        description=(
            "Make volumetric data: model electron density (X-ray) or nuclear density (neutron) "
            "from a structure, Patterson map, difference of two grids; written as Cube / VESTA grd "
            "(usable with 'slice' and --volume)."
        ),
    )
    p.add_argument("input", help="CIF / structure file (-> model density) or a volumetric file (used as it is).")
    p.add_argument("--model-density", "--model", dest="model", default="xray", choices=["xray", "neutron"],
                   help="Model for a structure input: xray = electrons/A^3, neutron = fm/A^3 (default xray).")
    p.add_argument("--grid", default=None, metavar="N | NX,NY,NZ",
                   help="Grid points per axis (default: about 0.1 A, even, 16..128).")
    p.add_argument("--b-iso", type=float, default=None, help="Isotropic B in A^2 for every atom (default 0.5).")
    p.add_argument("--patterson", action="store_true", help="Also write the Patterson map.")
    p.add_argument("--volume-diff", "--diff", dest="diff", default=None, metavar="OTHER",
                   help="Write input - OTHER (volume file, or a CIF -> its model on the same grid).")
    p.add_argument("--export-volume", "--export", dest="export", default="cube", metavar="FMT",
                   help="cube, grd, cube,grd or all (default cube).")
    p.add_argument("--out-dir", default=None, help="Write the files here instead of a new 出力/ folder.")
    args = p.parse_args(argv)
    if not Path(args.input).is_file():
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        return 1
    try:
        rep = _volume_model_impl(args.input, args.model, args.grid, args.b_iso, args.patterson,
                                 args.diff, args.export, args.out_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{Path(args.input).name}: grid {rep['grid']}")
    for name, v in rep["volumes"].items():
        unit = str(v["unit"]).replace("Å", "A")
        print(f"  {name}: {v['min']:.4g} .. {v['max']:.4g}, integral {v['integral']:.6g} ({unit})")
        for f in v["files"]:
            print(f"    -> {f}")
    _print_saved(rep)
    return 0


def _run_morphology_cli(argv: list[str]) -> int:
    """`server.py morphology <cif> --faces "1 0 0:1.0,1 1 1:0.9" [--view x y z] [--export stl,html|none]`"""
    p = argparse.ArgumentParser(
        prog="CrystalAnalyzerMCP morphology",
        description=(
            "Crystal shape (morphology): convex polyhedron from {hkl} and central distances "
            "(symmetry-equivalent planes added), PNG, area table (CSV), STL and HTML."
        ),
    )
    p.add_argument("cif", help="CIF / structure file (lattice and point group).")
    p.add_argument("--faces", action="append", required=True, metavar='"H K L:D,..."',
                   help='Faces and central distances, e.g. "1 0 0:1.0,1 1 1:0.9" (repeatable; only the ratios matter).')
    p.add_argument("--no-expand", action="store_true", help="Use only the given (hkl), not the symmetry-equivalent planes.")
    p.add_argument("--view", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                   help="Cartesian viewing direction (towards the camera).")
    p.add_argument("--export", default="stl,html", metavar="FMT", help="stl, html, stl,html or none (default stl,html).")
    p.add_argument("--size-mm", type=float, default=30.0, help="Longest side of the STL in mm (default 30).")
    p.add_argument("--out-dir", default=None, help="Write the files here instead of a new 出力/ folder.")
    args = p.parse_args(argv)
    if not Path(args.cif).is_file():
        print(f"Error: file not found: {args.cif}", file=sys.stderr)
        return 1
    try:
        rep = _morphology_impl(args.cif, args.faces, args.view, args.export, args.size_mm,
                               not args.no_expand, args.out_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{Path(args.cif).name}: {rep['n_faces']} faces, {rep['n_vertices']} vertices, "
          f"volume {rep['volume']:.4g}, area {rep['total_area']:.4g}")
    for fam in rep["families"]:
        print(f"  {fam['family']} d={fam['distance']:g}: {fam['n_on_crystal']}/{fam['n_faces']} faces, "
              f"area {fam['area']:.4g} ({100 * fam['area_fraction']:.1f}%)")
    for f in rep["files"]:
        print(f"  -> {f}")
    _print_saved(rep)
    return 0


def _run_rerun_cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        prog="server.py rerun",
        description="レシピ（出力フォルダの recipe.json）から同じ結果をもう一度、新しいフォルダに作る。",
    )
    p.add_argument("recipe", help="出力フォルダ、またはその中の recipe.json")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="一部だけ変える。style=space-filling（コマンドの設定）/ render.dpi=600・render.theme=okabe_ito_safe（点の入った名前は前提数値）。何回でも")
    args = p.parse_args(argv)
    res = _recipe.rerun(args.recipe, _recipe.parse_overrides(args.overrides))
    info = res["rerun"]
    for w in info["warnings"]:
        print(f"警告: {w}")
    if info.get("new_recipe"):
        print(f"新しいレシピ: {info['new_recipe']}")
    return int(res.get("exit_code", 0) or 0)


def main(argv: list[str] | None = None) -> int:
    """Entry point for both FastMCP server mode (no args) and CLI mode."""
    args_list = sys.argv[1:] if argv is None else list(argv)
    if len(args_list) == 0:
        run_mcp_server()
        return 0
    if args_list[0].lower() == "rerun":
        return _run_rerun_cli(args_list[1:])
    if args_list[0].lower() in {"interactive", "menu"}:  # answers typed in: nothing to replay
        return _main_cli(args_list)
    with _recipe.recording_cli(args_list):  # every run folder of this command gets recipe.json
        return _main_cli(args_list)


def _main_cli(args_list: list[str]) -> int:
    """CLI commands (``main`` records them for recipe.json)."""
    if args_list[0].lower() == "measure":
        return _run_measure_cli(args_list[1:])
    if args_list[0].lower() == "symmetry":
        return _run_symmetry_cli(args_list[1:])
    if args_list[0].lower() == "transform":
        return _run_transform_cli(args_list[1:])
    if args_list[0].lower() == "export":
        return _run_export_cli(args_list[1:])
    if args_list[0].lower() == "polyhedra":
        return _run_polyhedra_cli(args_list[1:])
    if args_list[0].lower() in {"structure-factors", "sf"}:
        return _run_structure_factors_cli(args_list[1:])
    if args_list[0].lower() in {"displacement", "adp"}:
        return _run_displacement_cli(args_list[1:])
    if args_list[0].lower() in {"slice", "volume-slice"}:
        return _run_slice_cli(args_list[1:])
    if args_list[0].lower() == "edit":
        return _run_edit_cli(args_list[1:])
    if args_list[0].lower() == "create":
        return _run_create_cli(args_list[1:])
    if args_list[0].lower() in {"volume-model", "density"}:
        return _run_volume_model_cli(args_list[1:])
    if args_list[0].lower() in {"morphology", "crystal-shape"}:
        return _run_morphology_cli(args_list[1:])
    if args_list[0].lower() in {"figure", "journal-figure"}:
        from shido.journal_figure import run_figure_cli
        return run_figure_cli(args_list[1:])

    parser = _build_arg_parser()
    args = parser.parse_args(args_list)
    print(f"[CrystalAnalyzerMCP v{__version__}]")
    bulk_options = _bulk_options_from_args(args)
    # 既定（どちらも付けない）は自動：陰イオンになる元素がある構造だけ多面体を描く
    show_polyhedra: bool | None = False if args.no_polyhedra else (True if args.polyhedra else None)
    # フラグを付けなかったものは None にして、前提数値（settings）の値を使う
    filter_polar = False if args.keep_polar else None
    deduplicate = False if args.no_dedup else None
    transparent = True if args.transparent else None
    if args.theme:
        active_theme = set_active_theme(args.theme)
        print(f"[CrystalAnalyzerMCP] Active color theme: {active_theme}")
    base_dir = Path(args.out_dir) if args.out_dir else None

    if args.plane and args.plane.strip().lower() == "list":
        print("主要な結晶面プリセット (--plane に指定可能):")
        for p in MAJOR_PLANES:
            print(f"  {p['key']:<4}  {p['label']}  (別名: {', '.join(p['aliases'])})")
        return 0

    if args.plane and args.hkl is None:
        args.hkl = list(resolve_miller_index(args.plane))

    if args.target.lower() in {"interactive", "menu"}:
        return _run_interactive_cli(base_dir)

    if args.target.lower() == "planes":
        print("主要な結晶面プリセット (--plane に指定可能):")
        for p in MAJOR_PLANES:
            print(f"  {p['key']:<4}  {p['label']}  (別名: {', '.join(p['aliases'])})")
        return 0

    if args.target.lower() == "demo-v2":
        print("[CrystalAnalyzerMCP] Running v2.md all-in-one showcase (Design 1, 2, 3)...")
        v2_res = run_demo_v2(base_dir=base_dir, transparent=transparent)
        print(f"Design 1 (Phase Compare) PNG : {v2_res['design1_phase_compare'].get('figure_phase_compare_png')}")
        print(f"Design 1 (Phase Compare) SVG : {v2_res['design1_phase_compare'].get('figure_phase_compare_svg')}")
        print(f"Design 2 (Heusler Model) PNG : {v2_res['design2_heusler_models']['image_files'].get('figure_png')}")
        print(f"Design 2 (Heusler Model) SVG : {v2_res['design2_heusler_models']['image_files'].get('figure_svg')}")
        print(f"Design 3 (c/a Sweep) steps   : {v2_res['design3_ca_sweep'].get('num_steps')}")
        print(f"Design 3 (c/a Curve) PNG     : {v2_res['design3_ca_plot'].get('figure_ca_energy_png')}")
        print(f"Design 3 (c/a Curve) SVG     : {v2_res['design3_ca_plot'].get('figure_ca_energy_svg')}")
        print("[CrystalAnalyzerMCP] v2.md showcase completed successfully.")
        _print_saved(v2_res)
        return 0

    if args.target.lower() == "demo":
        miller_index = tuple(args.hkl) if args.hkl is not None else (1, 1, 1)
        print(
            f"[CrystalAnalyzerMCP] Running Ni2MnGa L21 demo "
            f"(Miller index {miller_index})..."
        )
        res = run_demo(
            miller_index=miller_index,
            base_dir=base_dir,
            show_polyhedra=show_polyhedra,
            filter_polar=filter_polar,
            deduplicate=deduplicate,
            transparent=transparent,
            **bulk_options,
        )
        print(f"Demo CIF            : {res['demo_cif']}")
        _print_bulk_result(res["bulk_analysis"])
        _print_slab_result(res["slab_generation"])
        print("[CrystalAnalyzerMCP] Demo completed successfully.")
        _print_saved(res)
        return 0

    if args.target.lower() == "batch":
        if not args.batch_dir:
            print(
                "Error: 'batch' mode requires a directory path: python server.py batch <cif_dir>",
                file=sys.stderr,
            )
            return 1
        batch_input = Path(args.batch_dir)
        if not batch_input.is_dir():
            print(f"Error: Batch directory not found: {batch_input}", file=sys.stderr)
            return 1
        miller_index = tuple(args.hkl) if args.hkl is not None else None
        print(f"[CrystalAnalyzerMCP] Running batch analysis on {batch_input}...")
        batch_res = run_batch(
            input_dir=batch_input,
            output_dir=args.out_dir,
            miller_index=miller_index,
            show_polyhedra=show_polyhedra,
            filter_polar=filter_polar,
            deduplicate=deduplicate,
            transparent=transparent,
            all_blocks=args.all_blocks,
            **bulk_options,
        )
        _print_batch_result(batch_res)
        print("[CrystalAnalyzerMCP] Batch analysis completed.")
        _print_saved(batch_res)
        return 0

    target_path = Path(args.target)
    if target_path.is_dir():
        miller_index = tuple(args.hkl) if args.hkl is not None else None
        print(
            f"[CrystalAnalyzerMCP] Directory detected; running batch analysis on {target_path}..."
        )
        batch_res = run_batch(
            input_dir=target_path,
            output_dir=args.out_dir,
            miller_index=miller_index,
            show_polyhedra=show_polyhedra,
            filter_polar=filter_polar,
            deduplicate=deduplicate,
            transparent=transparent,
            all_blocks=args.all_blocks,
            **bulk_options,
        )
        _print_batch_result(batch_res)
        print("[CrystalAnalyzerMCP] Batch analysis completed.")
        _print_saved(batch_res)
        return 0

    if not target_path.is_file():
        print(f"Error: CIF file or directory not found: {target_path}", file=sys.stderr)
        return 1

    print(f"[CrystalAnalyzerMCP] Loading structure from {target_path}...")
    if args.all_blocks:
        entries = load_structures_all(target_path)
    else:
        one, one_warns, own_volume = _load_structure_or_volume(target_path)
        entries = [(one, one_warns, "")]
        if own_volume and not bulk_options.get("volume"):  # a CHGCAR / Cube alone
            bulk_options.update(_volume_kwargs(
                own_volume, args.isovalue, args.iso_color, args.iso_color_negative,
                args.iso_opacity, args.iso_both_signs,
            ))
    if "adp" in bulk_options and bulk_options["adp"] is None:  # --style ellipsoid: ADPs of this CIF
        bulk_options["adp"] = str(target_path)
    multi = len(entries) > 1
    if multi:
        print(f"[CrystalAnalyzerMCP] {len(entries)} data blocks found; each goes to its own folder.")
    stem = sanitize_name(structure_stem(target_path))
    # 既定は 出力/<日時>_<CIF名>/（データブロックが複数なら、その中に 1 つずつ）
    run_dir = None if base_dir is not None else new_run_dir("", structure_stem(target_path))
    settings = {"show_polyhedra": _polyhedra_setting(show_polyhedra), "transparent": bool(transparent),
                "miller_index": args.hkl, "all_blocks": args.all_blocks or None,
                **_shown_options(bulk_options)}
    saved: dict[str, Any] = {}

    for struct, parse_warnings, block in entries:
        out_stem = f"{stem}_{sanitize_name(block)}" if multi else stem
        if run_dir is not None:
            bulk_dir = run_dir / sanitize_name(block) if multi else run_dir
        else:
            bulk_dir = base_dir / f"output_{out_stem}"

        print(f"[CrystalAnalyzerMCP] Running bulk analysis into {bulk_dir}...")
        bulk_res = run_bulk_analysis(
            struct,
            bulk_dir,
            show_polyhedra=show_polyhedra,
            parse_warnings=parse_warnings,
            transparent=transparent,
            **bulk_options,
        )
        _print_bulk_result(bulk_res)
        _recipe.note_result(bulk_res)  # its warnings go into recipe.json

        if args.hkl is not None:
            h, k, l = args.hkl
            slab_dir = (
                bulk_dir / f"slab_{h}{k}{l}" if run_dir is not None
                else base_dir / f"output_slab_{out_stem}_{h}{k}{l}"
            )
            print(
                f"[CrystalAnalyzerMCP] Cutting ({h}{k}{l}) slabs into {slab_dir}..."
            )
            slab_res = cut_slabs(
                struct,
                (h, k, l),
                slab_dir,
                min_slab_size=args.min_slab_size,
                min_vacuum_size=args.min_vacuum_size,
                filter_polar=filter_polar,
                deduplicate=deduplicate,
                transparent_background=transparent,
            )
            _print_slab_result(slab_res)
        if run_dir is None:  # --out-dir: はじめに読む.html は output_<名前>/ に置く
            saved = _finish_run({}, bulk_dir, f"CIF の解析：{target_path.name}",
                                source=target_path.resolve(), settings=settings)

    if run_dir is not None:
        saved = _finish_run({}, run_dir, f"CIF の解析：{target_path.name}",
                            source=target_path.resolve(), settings=settings)
    print("[CrystalAnalyzerMCP] Analysis completed successfully.")
    _print_saved(saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# %%
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/avega17/PLAN6068_PV_Project.git"
REPO_BRANCH = "nblink-publish"
COLAB_REPO_DIR = Path("/content/PLAN6068_PV_Project")
COLAB_REQUIRED_PACKAGES = {
    "folium": "folium",
    "geopandas": "geopandas",
    "ipywidgets": "ipywidgets",
    "lonboard": "lonboard",
    "plotly": "plotly",
    "pyarrow": "pyarrow",
}


def in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def resolve_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    markers = ("project_rules.md", ".git")
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in markers):
            return candidate
    return current


def ensure_colab_repo(repo_dir: Path = COLAB_REPO_DIR) -> Path:
    if repo_dir.exists() and not (repo_dir / "project_rules.md").exists():
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                REPO_BRANCH,
                REPO_URL,
                str(repo_dir),
            ],
            check=True,
        )
    return repo_dir


def ensure_colab_packages() -> list[str]:
    missing = [package for module_name, package in COLAB_REQUIRED_PACKAGES.items() if importlib.util.find_spec(module_name) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing))], check=True)
    return missing


def prepare_runtime_workspace() -> Path:
    project_root = resolve_project_root()
    if any((project_root / marker).exists() for marker in ("project_rules.md", ".git")):
        os.chdir(project_root)
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        return project_root

    if not in_colab():
        return project_root

    repo_dir = ensure_colab_repo()
    ensure_colab_packages()
    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    print(f"Google Colab workspace ready at {repo_dir}")
    return repo_dir


PROJECT_ROOT = prepare_runtime_workspace()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MAPS_DIR = OUTPUTS_DIR / "maps"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PRESENTATION_DIR = OUTPUTS_DIR / "presentation"

import folium
import geopandas as gpd
import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from folium.plugins import MarkerCluster
from IPython.display import HTML, Image, display
from lonboard import Map, PolygonLayer
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from plotly.subplots import make_subplots


def artifact(relative_path: str) -> Path:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing presentation artifact: {relative_path}")
    return path


def show_png(relative_path: str, width: int | None = None) -> None:
    display(Image(filename=str(artifact(relative_path)), width=width))


def read_manifest() -> dict[str, object]:
    path = PRESENTATION_DIR / "presentation_data_manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


presentation_manifest = read_manifest()
selected_building_export = presentation_manifest.get("selected_building_export", {})

display(
    HTML(
        """
        <style>
        :root {
            --pv-ink: #12211f;
            --pv-muted: #52635f;
            --pv-green: #0f766e;
            --pv-gold: #d97706;
            --pv-sky: #2563eb;
            --pv-paper: #f7f3e8;
            --pv-line: #c9d8d2;
        }
        .jp-RenderedHTMLCommon h1,
        .jp-RenderedHTMLCommon h2,
        .jp-RenderedHTMLCommon h3 {
            color: var(--pv-ink);
            letter-spacing: 0;
        }
        .jp-RenderedHTMLCommon h1 { font-size: 2.05rem; }
        .jp-RenderedHTMLCommon h2 { font-size: 1.45rem; border-bottom: 2px solid var(--pv-line); padding-bottom: .25rem; }
        .presentation-note { color: var(--pv-muted); font-size: 0.95rem; }
        .tight-list li { margin-bottom: 0.32rem; }
        .asset-frame { max-width: 100%; border: 1px solid var(--pv-line); border-radius: 6px; }
        .figure-caption {
            color: var(--pv-muted);
            font-size: 0.92rem;
            line-height: 1.35;
            margin: .35rem 0 1rem;
        }
        .analysis-callout {
            border-left: 5px solid var(--pv-gold);
            background: #fff8e6;
            padding: .85rem 1rem;
            border-radius: 6px;
            color: var(--pv-ink);
            margin: 1rem 0;
        }
        .quad-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .6rem;
            margin: .9rem 0;
        }
        .quad-grid div {
            background: #f7faf8;
            border: 1px solid var(--pv-line);
            border-radius: 6px;
            padding: .7rem;
        }
        .cover-slide {
            min-height: 640px;
            display: grid;
            align-content: center;
            gap: 1.2rem;
            padding: 3rem 4rem;
            border-radius: 8px;
            background:
                linear-gradient(115deg, rgba(10, 74, 68, 0.92), rgba(21, 111, 96, 0.78)),
                radial-gradient(circle at 15% 15%, rgba(245, 158, 11, 0.45), transparent 36%),
                linear-gradient(45deg, #10231f, #f7f3e8);
            color: #fffaf0;
            box-shadow: 0 18px 48px rgba(18, 33, 31, 0.18);
        }
        .cover-slide h1 {
            max-width: 1040px;
            margin: 0;
            color: #fffaf0;
            font-size: clamp(2.4rem, 5vw, 4.5rem);
            line-height: 0.98;
        }
        .cover-kicker {
            color: #fde68a;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .cover-subtitle {
            max-width: 860px;
            color: #e7f4ef;
            font-size: 1.35rem;
            line-height: 1.35;
        }
        .cover-meta {
            color: #dbece7;
            display: grid;
            gap: .25rem;
            font-size: 1rem;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
            margin: 1rem 0;
        }
        .deck-stat {
            border-left: 5px solid var(--pv-green);
            background: #f7faf8;
            padding: .8rem .95rem;
            border-radius: 6px;
        }
        .deck-stat strong { display: block; color: var(--pv-ink); font-size: 1.35rem; }
        .deck-stat span { color: var(--pv-muted); }
        .two-col {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            align-items: start;
        }
        @media (max-width: 800px) {
            .cover-slide { padding: 2rem; min-height: 560px; }
            .stat-grid, .two-col, .quad-grid { grid-template-columns: 1fr; }
        }
        </style>
        """
    )
)

# %%
case_boundaries = gpd.read_file(PRESENTATION_DIR / "case_study_municipalities.geojson").to_crs("EPSG:4326")
case_pv = gpd.read_file(PRESENTATION_DIR / "case_study_osm_pv_polygons.geojson").to_crs("EPSG:4326")
case_barrios = gpd.read_file(PRESENTATION_DIR / "case_study_high_coverage_barrios.geojson").to_crs("EPSG:4326")

HEIGHT_FROM_FLOOR_METERS = 3.2
MIN_PREVIEW_ELEVATION_METERS = 1.0
MAX_PREVIEW_ELEVATION_METERS = 75.0


def caption(text: str) -> None:
    display(HTML(f'<p class="figure-caption">{text}</p>'))


def build_case_study_micro_map(marker_limit: int = 1200) -> folium.Map:
    bounds = case_boundaries.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    micro_map = folium.Map(location=center, zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Esri WorldImagery",
        overlay=False,
        control=True,
        max_zoom=20,
    ).add_to(micro_map)

    folium.GeoJson(
        case_boundaries.__geo_interface__,
        name="Case-study municipality boundaries",
        style_function=lambda _: {"fillColor": "#00000000", "color": "#f8fafc", "weight": 3},
        tooltip=folium.GeoJsonTooltip(fields=["municipality_name"], aliases=["Municipality"]),
    ).add_to(micro_map)

    if not case_barrios.empty:
        folium.GeoJson(
            case_barrios.__geo_interface__,
            name="High-coverage barrio context",
            style_function=lambda _: {"fillColor": "#f59e0b", "color": "#78350f", "weight": 2, "fillOpacity": 0.35},
            tooltip=folium.GeoJsonTooltip(fields=["study_area_label"], aliases=["Area"]),
        ).add_to(micro_map)

    folium.GeoJson(
        case_pv.__geo_interface__,
        name=f"All OSM rooftop PV polygons ({len(case_pv):,})",
        style_function=lambda _: {"fillColor": "#34d399", "color": "#064e3b", "weight": 1, "fillOpacity": 0.62},
        smooth_factor=1,
    ).add_to(micro_map)

    high_coverage_pv = gpd.sjoin(
        case_pv,
        case_barrios[["study_area_label", "geometry"]],
        predicate="intersects",
        how="inner",
    )
    high_coverage_ids = set(high_coverage_pv["feature_id"].tolist())
    high_coverage_cluster = MarkerCluster(
        name=f"All PV centroids in high-coverage barrios ({len(high_coverage_pv):,})",
        overlay=True,
        control=True,
    ).add_to(micro_map)
    for row in high_coverage_pv.itertuples(index=False):
        point = row.geometry.centroid
        area_m2 = getattr(row, "pv_area_m2", np.nan)
        popup_text = (
            f"feature_id: {row.feature_id}<br>"
            f"municipality: {row.municipality_name}<br>"
            f"area_m2: {float(area_m2):,.1f}"
        )
        folium.Marker(
            location=[point.y, point.x],
            popup=folium.Popup(popup_text, max_width=280),
            icon=folium.Icon(color="green", icon="bolt", prefix="fa"),
        ).add_to(high_coverage_cluster)

    remainder = case_pv[~case_pv["feature_id"].isin(high_coverage_ids)]
    marker_sample = remainder.sample(n=min(marker_limit, len(remainder)), random_state=6068) if len(remainder) else remainder
    sample_cluster = MarkerCluster(
        name=f"Sampled PV centroids outside high-coverage barrios ({len(marker_sample):,} of {len(remainder):,})",
        overlay=True,
        control=True,
    ).add_to(micro_map)
    for row in marker_sample.itertuples(index=False):
        point = row.geometry.centroid
        area_m2 = getattr(row, "pv_area_m2", np.nan)
        popup_text = (
            f"feature_id: {row.feature_id}<br>"
            f"municipality: {row.municipality_name}<br>"
            f"area_m2: {float(area_m2):,.1f}"
        )
        folium.Marker(
            location=[point.y, point.x],
            popup=folium.Popup(popup_text, max_width=280),
            icon=folium.Icon(color="blue", icon="bolt", prefix="fa"),
        ).add_to(sample_cluster)

    micro_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    folium.LayerControl(collapsed=False).add_to(micro_map)
    return micro_map


def display_case_study_micro_map() -> None:
    display(build_case_study_micro_map())
    caption(
        "Interactive case-study map. All 11,412 OSM rooftop PV polygons are included; "
        "the centroid marker layer includes every PV label inside Puerto Nuevo and Mora, plus a sample outside those high-coverage barrios."
    )


def display_capacity_chart() -> None:
    capacity = pd.read_csv(FIGURES_DIR / "pr_pv_capacity_quarterly_2017_2025.csv")
    capacity["quarter_end"] = pd.to_datetime(capacity["quarter_end"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=capacity["quarter_end"],
            y=capacity["capacity_mw"],
            mode="lines+markers",
            name="Installed MW",
            line={"color": "#0f766e", "width": 3},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=capacity["quarter_end"],
            y=capacity["client_count"],
            name="NM clients",
            marker_color="#d97706",
            opacity=0.45,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Quarterly distributed PV capacity and net-metered clients",
        template="plotly_white",
        height=560,
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 60, "r": 60, "t": 90, "b": 40},
    )
    fig.update_yaxes(title_text="Installed capacity (MW)", secondary_y=False)
    fig.update_yaxes(title_text="Registered clients", secondary_y=True)
    fig.show()
    caption("Distributed PV capacity and net-metered customer counts show the exponential post-Maria growth of residential Photovoltaic energy infrastructure.")


def display_building_export_candidates() -> None:
    candidate_df = pd.DataFrame(presentation_manifest.get("building_export_candidates", []))
    if candidate_df.empty:
        display(candidate_df)
        return
    selected_scope = selected_building_export.get("scope")
    candidate_df = candidate_df.assign(
        buildings=lambda frame: frame["rows"].map(lambda value: f"{int(value):,}"),
        role=lambda frame: np.where(frame["scope"].eq(selected_scope), "Selected interactive preview", "Alternative scope reviewed"),
    )[["label", "buildings", "role"]]
    display(candidate_df)
    caption("Building-footprint scopes reviewed for the interactive preview. The selected San Juan/Puerto Nuevo scope preserves the dense urban case area while still allowing responsive browser exploration.")


def load_building_sources() -> dict[str, gpd.GeoDataFrame]:
    sources: dict[str, gpd.GeoDataFrame] = {}
    full_buildings_path = PRESENTATION_DIR / "case_study_overture_buildings.parquet"
    sample_buildings_path = PRESENTATION_DIR / "case_study_overture_buildings_preview.parquet"

    if full_buildings_path.exists():
        label = selected_building_export.get("label", "selected compressed export")
        sources[f"Full selected export: {label}"] = gpd.read_parquet(full_buildings_path).to_crs("EPSG:4326")
    if sample_buildings_path.exists():
        sources["Two-municipality preview sample"] = gpd.read_parquet(sample_buildings_path).to_crs("EPSG:4326")
    if not sources:
        raise FileNotFoundError("No presentation building-footprint parquet extract found.")
    return sources


building_sources = load_building_sources()
available_municipalities = sorted(
    {
        municipality
        for gdf in building_sources.values()
        for municipality in gdf["municipality_name"].dropna().unique().tolist()
    }
)


def prepare_preview_heights(preview_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    prepared = preview_gdf.copy()
    prepared["height"] = pd.to_numeric(prepared.get("height"), errors="coerce")
    prepared["num_floors"] = pd.to_numeric(prepared.get("num_floors"), errors="coerce")
    inferred_height = prepared["num_floors"] * HEIGHT_FROM_FLOOR_METERS
    prepared["preview_height_m"] = prepared["height"].where(prepared["height"].notna(), inferred_height)
    prepared["preview_height_m"] = prepared["preview_height_m"].fillna(MIN_PREVIEW_ELEVATION_METERS)
    prepared["preview_height_m"] = prepared["preview_height_m"].clip(
        lower=MIN_PREVIEW_ELEVATION_METERS,
        upper=MAX_PREVIEW_ELEVATION_METERS,
    )
    stats = {
        "rows_total": int(len(prepared)),
        "rows_with_explicit_height": int(prepared["height"].notna().sum()),
        "rows_inferred_from_num_floors": int(prepared["height"].isna().mul(prepared["num_floors"].notna()).sum()),
        "rows_min_fallback": int(prepared["preview_height_m"].eq(MIN_PREVIEW_ELEVATION_METERS).sum()),
    }
    return prepared, stats


def height_colors(heights_meters: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(heights_meters.astype(np.float64, copy=False), nan=1.0)
    safe = np.where(safe > 0, safe, 1.0)
    max_height = float(np.nanmax(safe)) if safe.size else 1.0
    norm = LogNorm(vmin=1.0, vmax=max(max_height, 1.01), clip=True)(safe)
    return np.clip(np.round(colormaps["YlOrRd"](norm) * 255), 0, 255).astype(np.uint8)


def render_building_preview(source: str, municipality: str = "All", max_features: int = 6000) -> None:
    subset = building_sources[source]
    if municipality != "All":
        subset = subset[subset["municipality_name"].eq(municipality)]
    if subset.empty:
        print(f"No buildings available for {municipality} in {source}.")
        return

    render_subset = subset
    if len(render_subset) > max_features:
        render_subset = render_subset.sample(n=max_features, random_state=6068)
    render_subset, stats = prepare_preview_heights(render_subset)
    heights_m = render_subset["preview_height_m"].to_numpy(dtype=np.float32, na_value=MIN_PREVIEW_ELEVATION_METERS)
    layer = PolygonLayer.from_geopandas(
        render_subset,
        get_fill_color=height_colors(heights_m),
        get_line_color=[88, 28, 8, 220],
        get_elevation=heights_m,
        extruded=True,
        line_width_min_pixels=1,
        pickable=True,
    )
    display(Map(layers=[layer]))
    print(
        f"{source} | {municipality}: rendered {len(render_subset):,} of {len(subset):,} buildings; "
        f"explicit heights={stats['rows_with_explicit_height']:,}, "
        f"inferred heights={stats['rows_inferred_from_num_floors']:,}, "
        f"fallback minimum={stats['rows_min_fallback']:,}."
    )


def display_building_preview_widget() -> None:
    widgets.interact(
        render_building_preview,
        source=widgets.Dropdown(options=list(building_sources.keys()), description="Dataset"),
        municipality=widgets.Dropdown(options=["All", *available_municipalities], description="Municipio"),
        max_features=widgets.IntSlider(value=6000, min=1000, max=150000, step=1000, description="Render cap"),
    )
    caption("Lonboard preview of the selected building-footprint scope. Heights are explicit where available, otherwise inferred from floors or assigned a minimal preview elevation.")


def display_database_summary_table() -> None:
    db_summary = pd.read_csv(PRESENTATION_DIR / "research_database_summary.csv")
    db_summary = db_summary.assign(
        records=lambda frame: frame["record_count"].map(lambda value: f"{int(value):,}")
    )[["summary_group", "component", "source_table", "records", "planning_use"]]
    display(db_summary)
    caption("Full research database summary. These tables are the integrated data backbone behind the maps, model outputs, and spatial statistics shown in the deck.")


def display_model_summary_table() -> None:
    model_summary = pd.read_csv(REPORTS_DIR / "pv_building_detections_by_model_arch_summary.csv")
    display(model_summary.pivot(index="model_arch_display", columns="municipio", values="detected_buildings").fillna(0).astype(int))
    caption("Detected building counts by model architecture and municipality, interpreted as building-level evidence rather than pixel-perfect panel segmentation.")


def display_global_morans_table() -> None:
    morans = pd.read_csv(REPORTS_DIR / "case_study_esda_global_morans_by_variable.csv")
    display(
        morans[["municipio", "metric_label", "n_tracts", "morans_I", "p_sim"]]
        .sort_values(["municipio", "metric_label"])
        .round({"morans_I": 3, "p_sim": 3})
    )
    caption("Global Moran's I is the screening step: positive significant values indicate clustering, negative values suggest dispersion, and non-significant results should not be interpreted as a spatial pattern.")


def display_bivariate_morans_table() -> None:
    bivariate_morans = pd.read_csv(REPORTS_DIR / "case_study_esda_bivariate_morans.csv")
    display(
        bivariate_morans[["municipio", "relationship", "n_tracts", "moran_bv_I", "p_sim"]]
        .sort_values(["municipio", "relationship"])
        .round({"moran_bv_I": 3, "p_sim": 3})
    )
    caption("Bivariate Moran summaries compare tract PV density with neighboring tract context variables; they are screening results, not causal estimates.")

# %%
presentation_manifest = read_manifest()
selected_building_export = presentation_manifest.get("selected_building_export", {})
building_sources = load_building_sources()
available_municipalities = sorted(
    {
        municipality
        for gdf in building_sources.values()
        for municipality in gdf["municipality_name"].dropna().unique().tolist()
    }
)
MAX_BUILDING_RENDER_CAP = max(1000, ((max(len(gdf) for gdf in building_sources.values()) + 999) // 1000) * 1000)


def display_building_export_candidates() -> None:
    candidate_df = pd.DataFrame(presentation_manifest.get("building_export_candidates", []))
    if candidate_df.empty:
        display(candidate_df)
        return
    selected_scope = selected_building_export.get("scope")
    candidate_df = candidate_df.assign(
        buildings=lambda frame: frame["rows"].map(lambda value: f"{int(value):,}"),
        role=lambda frame: np.where(frame["scope"].eq(selected_scope), "Selected interactive preview", "Alternative scope reviewed"),
    )[["label", "buildings", "role"]]
    display(candidate_df)
    caption("Building-footprint scopes reviewed for the interactive preview. The selected full San Juan plus Isabela scope now retains both municipalities while the render cap keeps browser exploration manageable.")


def display_building_preview_widget() -> None:
    widgets.interact(
        render_building_preview,
        source=widgets.Dropdown(options=list(building_sources.keys()), description="Dataset"),
        municipality=widgets.Dropdown(options=["All", *available_municipalities], description="Municipio"),
        max_features=widgets.IntSlider(value=min(6000, MAX_BUILDING_RENDER_CAP), min=1000, max=MAX_BUILDING_RENDER_CAP, step=1000, description="Render cap"),
    )
    caption("Lonboard preview of the full selected building-footprint scope. Heights are explicit where available, otherwise inferred from floors or assigned a minimal preview elevation.")

# %% [markdown]
# ## Google Colab Setup Preface
# 
# This presentation is now intended to run from the [nblink-publish branch](https://github.com/avega17/PLAN6068_PV_Project/tree/nblink-publish) in Google Colab.
# 
# 1. Run the first code cell. In Colab it clones the repo into `/content/PLAN6068_PV_Project`, installs any missing presentation packages, switches the working directory to the repo root, and makes the helper modules plus presentation artifacts available.
# 2. Run the second code cell to load the compact presentation artifacts and helper functions used by the slides.
# 3. Focus the cover slide and start slideshow mode from that cell with `View > Start slideshow`, `Ctrl` + `Shift` + `P` then `Start notebook slideshow`, or `Alt` + `V`.
# 4. To restart from the first cell, use `View > Start slideshow from beginning` or `Alt` + `Shift` + `V`.
# 5. Appending `#slideshowMode=true` to a shared Colab URL opens directly in slideshow mode.
# 
# If slideshow mode closes, press `Escape` or the `x` button, focus the desired cell again, and restart the slideshow. For stability across presentations, Colab also lets you pin a specific runtime version from `Runtime > Change runtime type`.

# %% [markdown]
# <div class="cover-slide">
#   <div class="cover-kicker" align="right">PLAN 6068 Final Project</div>
#   <h1 align="center">Geospatial Data Pipelines for Rooftop Solar Detection in Puerto Rico</h1>
#   <div class="cover-subtitle" align="center"> <h4> A Python notebook slide deck on crowdsourced PV labels, building footprints, computer vision experiments, and exploratory spatial analysis for municipal energy planning.</h4></div>
#   <div class="cover-meta">
#     <div align="center"><strong>Alejandro S. Vega-Nogales</strong> (801-13-7956) | M.S. Candidate, Computer Science, University of Puerto Rico - Río Piedras</div>
#     <div align="center">PLAN 6068: AI Applications in Planning | Prof. J. Ayala Hernandez</div>
#   </div>
# </div>

# %% [markdown]
# # 1. Introduction

# %% [markdown]
# ## 1.1 Problem Statement
# 
# Puerto Rico's distributed solar transition is moving faster than the public planning data infrastructure built to understand it. After Hurricane Maria in 2017 and Hurricane Fiona in 2022, recurring outages and voltage instability turned rooftop solar from a niche sustainability choice into a household energy resilience strategy.
# 
# The planning problem is now spatial: agencies, utilities, and municipalities need to know where PV systems are likely installed, which neighborhoods are missing from the transition, and how adoption intersects with vulnerability, energy equity, and solar resource availability.

# %% [markdown]
# ## 1.1.1 Local Context in Puerto Rico
# 
# <ul class="tight-list">
# <li><strong>Centralized grid fragility:</strong> major hurricane impacts exposed long-standing weaknesses in transmission, distribution, and emergency restoration capacity.</li>
# <li><strong>Grassroots solar adoption:</strong> private rooftop PV and batteries expanded as a bottom-up response to unreliable service, outpacing many centralized planning workflows.</li>
# <li><strong>Unequal resilience:</strong> the households most able to finance PV may not be the households facing the highest outage and heat exposure risks.</li>
# <li><strong>Data gap:</strong> public PV inventories are partial, delayed, or aggregated; planning requires a finer spatial signal.</li>
# </ul>

# %%
display_capacity_chart()

# %% [markdown]
# ## 1.1.2: Relevance to Urban Planning and Grid Operations
# 
# Rooftop PV detection is not just an image-classification problem. It is an urban planning and grid operations problem with several linked uses:
# 
# <ul class="tight-list">
# <li><strong>Hosting capacity:</strong> local feeders need better estimates of distributed generation density before interconnection constraints become visible through failures.</li>
# <li><strong>Resilience planning:</strong> emergency management can prioritize critical facilities and vulnerable neighborhoods when likely PV/battery availability is mapped.</li>
# <li><strong>Equity analysis:</strong> PV evidence can be compared with income, education, language, vulnerability, and urban/rural indicators.</li>
# <li><strong>Municipal implementation:</strong> planners can connect rooftop opportunity, adoption signals, and local permitting workflows at the block-group or barrio scale.</li>
# </ul>

# %% [markdown]
# ## 1.2: Updated Research Questions
# 
# The project was rescoped after the first island-wide segmentation plan ran into two practical barriers: uneven public imagery quality and the time cost of building reliable training data across all municipalities. The final design narrows to San Juan and Isabela, where contrasting urban morphologies and strong OSM label coverage make a higher-fidelity workflow possible.
# 
# 1. **GeoAI detection:** Can pre-trained computer-vision models be adapted to identify building-level rooftop PV evidence in high-digitization barrios?
# 2. **Exploratory spatial data analysis:** Does PV evidence exhibit spatial autocorrelation, and does that structure align with socioeconomic or vulnerability indicators?
# 3. **Geophysical context:** Do irradiance and temperature summaries help explain where PV evidence appears, or are social and built-environment variables more informative at this scale?
# 4. **AI-assisted Urban Planning:** Can a reproducible geospatial pipeline developed with Agentic AI turn heterogeneous open data into presentation-ready planning insights without requiring a full research or development team?

# %% [markdown]
# # 2. Datasets

# %% [markdown]
# ## 2.1: Geographic Scope and Case Study Selection
# 
# The final case-study frame keeps two municipalities because they expose different planning and modeling challenges while still satisfying the scope for our project.
# 
# <div class="stat-grid">
# <div class="deck-stat"><strong>San Juan</strong><span>: dense metropolitan morphology; largest OSM PV label concentration</span></div>
# <div class="deck-stat"><strong>Isabela</strong><span>: rural/coastal morphology; lower density but strong local test areas</span></div>
# <div class="deck-stat"><strong>11,412</strong><span>: all case-study OSM PV polygons retained in the lightweight map extract</span></div>
# </div>
# 
# The initial SNIC/superpixel island-wide segmentation methodology from [8] was set aside because the engineering burden was not the segmentation algorithm alone. The harder problem was consistent imagery, clean geometry alignment, availability of 4-band imagery, and enough trustworthy labels to evaluate building-level usefulness.

# %%
# needs to be re-run due to platform size constraints
display_case_study_micro_map()

# %% [markdown]
# ## 2.2: US Census Bureau Data
# 
# Census geographies provide the common spatial frame for planning interpretation. The project uses counties (municipios), tracts, and block groups, then joins socioeconomic and vulnerability attributes at the most reliable shared geography.
# 
# <ul class="tight-list">
# <li><strong>ACS 2020 5-year estimates:</strong> income, education, language, household, and population indicators.</li>
# <li><strong>CDC Social Vulnerability Index:</strong> percentile-based vulnerability indicators derived from ACS variables.</li>
# <li><strong>Urban/rural classification:</strong> context for interpreting roof opportunity, density, and infrastructure exposure.</li>
# <li><strong>Aggregation target:</strong> block groups and tracts used for exploratory spatial statistics.</li>
# </ul>
# 
# <img src="outputs/figures/pr_case_study_census_hierarchy.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Census geography hierarchy for the case-study municipalities. Block groups and tracts provide the tractable spatial units for joining ACS, SVI, PV evidence, and modeled detections.</p>

# %% [markdown]
# ## 2.3: OpenStreetMap Overpass API Vectors
# 
# OSM PV polygons are the project's most important weak-supervision source. They are not a complete adoption inventory; they are a spatially explicit label layer created by crowdsourced digitization with uneven coverage and quality.
# 
# <ul class="tight-list">
# <li>Island-wide OSM PV labels outside San Juan and Isabela support training-data sampling.</li>
# <li>Case-study labels support validation and exploratory comparison.</li>
# <li>Puerto Nuevo in San Juan and Mora in Isabela serve as high-digitization test contexts.</li>
# <li>The label-bias problem is explicit: mapped PV density partly reflects mapper effort.</li>
# </ul>
# 
# <img src="outputs/maps/pr_macro_pv_map.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Island-wide OSM PV label distribution. The map is useful as weak supervision context, but the density of labels also reflects uneven crowdsourced mapping effort.</p>

# %% [markdown]
# ## 2.4: Overture Maps Building Footprints
# 
# Overture building footprints help define the land area search space and give the detection workflow a standard scale and unit: 
# PV installation evidence per building, per H3 cell, and per Census geography.
# 
# <ul class="tight-list">
# <li>Raw footprints were ingested locally with DuckDB Spatial and Overture tooling.</li>
# <li>The interactive preview prioritizes urban density: all Puerto Nuevo buildings plus a reduced deterministic San Juan sample.</li>
# <li>Footprints provide the denominator for comparing PV evidence across neighborhoods instead of relying on raw detection counts.</li>
# </ul>

# %%
display_building_export_candidates()

# %% [markdown]
# The interactive 3D preview below uses the San Juan building-footprint subset. The render cap keeps interaction responsive while still letting the viewer zoom into dense Puerto Nuevo building vectors.

# %%
display_building_preview_widget()

# %% [markdown]
# ## 2.5: Open Access Catalogs of High-Resolution Aerial and Satellite Imagery
# 
# Open imagery was essential for feasibility testing, but it was not reliable enough to be treated as a single homogeneous modeling source.
# 
# <ul class="tight-list">
# <li><strong>USDA NAIP:</strong> useful coverage where available, but inconsistent timing and partial Puerto Rico coverage.</li>
# <li><strong>Maxar/Vantor Open Data:</strong> valuable disaster-period imagery, but limited spatial and temporal coverage.</li>
# <li><strong>Esri World Imagery:</strong> visually useful for inspection and tile-based sampling, but not a reproducible bulk public download source.</li>
# <li><strong>Google Solar API:</strong> high-resolution and consistent for sampled properties, but constrained by API budgeting and operational complexity.</li>
# </ul>
# 
# <img src="outputs/maps/PR_STAC_raster_footprints.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Open STAC raster footprints over Puerto Rico. Coverage, cloud contamination, and sensor inconsistency explain why public imagery was not enough for a reliable island-wide segmentation workflow.</p>
# 
# <p class="figure-caption">Raster footprint and tile-planning figures. These show why open STAC imagery was useful for scoping coverage and providing inference imagery for our case study but is not comprehensive enough to serve as the single imagery source; model training ultimately used clipped high-resolution basemap tiles prepared through Contextily-style tile workflows and aligned to H3/building sampling units.</p>
# 
# <img src="outputs/maps/dataset_split_plan_overview.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Dataset split plan showing how training, validation, and test areas are separated to reduce <a href="https://en.wikipedia.org/wiki/Data_leakage_(machine_learning)">data leakage</a> between weak labels and evaluation areas.</p>

# %% [markdown]
# ## 2.6: National Solar Radiation Database (NSRDB)
# 
# NREL's NSRDB provides the geophysical context for PV opportunity: global horizontal irradiance, temperature, and related meteorological variables at moderate spatial resolution (2km) and sub-hourly temporal resolution.
# 
# In this presentation the NSRDB branch is used as contextual evidence rather than a causal model. Its main role is to support future bivariate spatial autocorrelation analysis against PV evidence, vulnerability, and urban morphology.
# 
# <img src="outputs/maps/nsrdb_case_study_multi-year_means.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Multi-year NSRDB irradiance and meteorological summaries for San Juan and Isabela. These variables give geophysical context for later bivariate spatial analysis.</p>

# %% [markdown]
# ## 2.7: AI-Assisted Dataset Collection and Management
# 
# AI assistance was most useful before formal analysis: building ingestion scripts, validating joins against the actual database, managing asynchronous API clients, and converting source-specific files into shared analysis tables, maps, and notebook cells.
# 
# <ul class="tight-list">
# <li><strong>GitHub Copilot agent mode:</strong> accelerated boilerplate for Census, OSM, NSRDB, raster catalog, and export scripts. <br> <em>Note this tool is <a href="https://www.digitalocean.com/resources/articles/github-copilot-vs-microsoft-copilot#github-copilot-vs-microsoft-365-copilot">substantially different than the Microsoft 365 Copilot products</a> and offers <a href="https://docs.github.com/en/copilot/reference/ai-models/supported-models">models from multiple providers</a>.</em></li>
# <li><strong>Documentation-grounded prompting:</strong> prompts regularly included direct links to API references, method docs, tutorials, and similar code patterns so the agent could adapt current library syntax rather than guessing from "memory".</li>
# <li><strong>DuckDB/MCP validation:</strong> database-aware <a href="https://motherduck.com/blog/faster-data-pipelines-with-mcp-duckdb-ai/#mcp-closing-the-feedback-loop">MCP tool</a> let the coding assistant inspect schemas, run SQL, and show query results before code changes were trusted, closing the feedback loop between model's code and real data.</li>
# <li><strong>Human verification:</strong> row counts, map previews, sampled records, and visual checks were used to decide whether generated outputs was accurate and reliable.</li>
# </ul>

# %% [markdown]
# # 3. Methodology

# %% [markdown]
# ## 3.1: Software Stack and Agentic AI Development
# 
# The project was built as a sequence of reproducible steps ([notebooks 01 - 15](https://github.com/avega17/PLAN6068_PV_Project/tree/nblink-publish/notebooks)) backed by shared Python utilities for code reuse and to avoid notebook bloat.
# 
# <ul class="tight-list">
# <li><strong>Python geospatial stack:</strong> GeoPandas, Shapely, DuckDB + Spatial extension (PostGIS-like), PySAL, Rasterio, Contextily, Folium, Plotly, lonboard, and the STAC (SpatioTemporal Asset Catalog) specification.</li>
# <li><strong>Modeling stack:</strong> segmentation and feature-extraction experiments around Meta's SAM3 and DINOv3 models, and proven semantic segmentation architectures.</li>
# <li><strong>Agentic workflow:</strong> Github Copilot <a href="https://github.blog/ai-and-ml/github-copilot/agent-mode-101-all-about-github-copilots-powerful-mode/">Agent mode</a> was used for implementation drafts, refactors, and validation scaffolding; every high-risk step was checked against real local data outputs using our agent's access to our database.</li>
# <li><strong>Documentation context:</strong> library docs and external examples were treated as part of the prompt, especially for geospatial packages whose APIs change quickly.</li>
# <li><strong>Google Colab sharing:</strong> the deck can clone the <code>nblink-publish</code> branch into <code>/content</code>, load compact presentation artifacts, and use slideshow mode to present from the cover slide or any focused cell.</li>
# </ul>

# %% [markdown]
# ## 3.2: Geospatial Data Pipelines: Ingestion
# 
# The ingestion pipeline follows the notebook order: Census and OSM vectors first, Overture buildings second, tabular socioeconomic layers next, then raster and NSRDB context. [DuckDB Spatial](https://motherduck.com/blog/geospatial-for-beginner-duckdb-spatial-motherduck/) serves as the local integration point because it can hold large geometry tables while still supporting high-performance SQL-based operations and easy validation.
# 
# <!-- side by side map figures -->
# <div class="two-col">
#     <img src="outputs/maps/cabo_rojo_raster_census_samples.png" class="asset-frame" style="width: 48%; height:600px;" align="left">
#     <img src="outputs/maps/cabo_rojo_rasters_h3_pv.png" class="asset-frame" style="width: 49%; height: 600px;" align="right">
# </div>
# <p class="figure-caption" align="bottom">
# Pipeline diagnostic examples from the raster/H3 workflow for a randomly sampled municipality. They show how large imagery sources are clipped into analysis tiles and aligned with Census geometry, buildings, and PV labels.
# </p>

# %% [markdown]
# ## 3.3 Preprocessing: Standardization and Aggregation of Datasets
# 
# 
# 
# Preprocessing focused on making heterogeneous layers comparable:
# 
# <ul class="tight-list">
# <li>standardize CRS and validate geometries before overlays;</li>
# <li>clip and aggregate OSM PV labels to municipality, tract, block group, and H3 units;</li>
# <li>derive building denominators from Overture footprints;</li>
# <li>prepare model train/validation/test splits that separate case-study evaluation areas from broader training labels.</li>
# </ul>

# %% [markdown]
# ## 3.4 AI Model Training and Evaluation
# 
# <img src="outputs/figures/pr_pv_three_panel_example.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Ideal evaluation target: a consistent pixel-level signal that can be aggregated to buildings. In practice, this project treats model output conservatively as building-level rooftop PV evidence.</p>
# <img src="outputs/presentation/model_examples/dinov3_test_8a4cee7823a7fff_review.png" class="asset-frame" style="width: 100%;">
# <img src="outputs/presentation/model_examples/smp_pan_efficientnet_b6_test_8a4cee791b2ffff_review.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Representative model review panels. The examples emphasize why patchy pixel masks can still be informative when converted into building-level evidence, but also why the results should not be presented as final solar-panel delineations.</p>
# 
# 
# The model-evaluation frame is deliberately conservative. Pixel-level performance can overstate planning value when the dominant class is background and when the practical question is whether a building has useful rooftop PV evidence.
# 
# <ul class="tight-list">
# <li>OSM PV polygons support weak supervision and sampling.</li>
# <li>High-digitization barrios create a more reliable test dataset.</li>
# <li>Model outputs are interpreted at building-level instead of pixel-level.</li>
# <li>Future validation should compare detections against CRIM cadastres, interconnection data from <em> <a href="https://energia.pr.gov/en/dockets/?docket=nepr-mi-2019-0016"> Negociado de Energía de Puerto Rico</a></em> and US <a href="https://data.openei.org/submissions/5749">DoE's PR100 study</a>, or collaboration with <a href="http://cohemisferico.uprm.edu/pr100/">relevant stakeholders and organizations</a>.</li>
# </ul>

# %% [markdown]
# ## 3.5 Exploratory Spatial Data Analysis
# 
# ESDA converts model and label outputs into planning questions about clustering, inequality, and context. The unit of analysis here is the Census tract, run separately for San Juan and Isabela. Our spatial weights use <a href="https://geographicdata.science/book/notebooks/04_spatial_weights.html?highlight=queen#contiguity-weights">row-standardized Queen contiguity</a>, so tracts are neighbors when they share a direct edge or a vertex in common. The spatial lag summarizes neighboring values:
# 
# $$W y_i = \sum_j w_{ij} y_j$$
# 
# Global Moran's $I$ asks whether a variable is spatially clustered overall. <a href="https://geographicdata.science/book/notebooks/07_local_autocorrelation.html?highlight=spatial+lag#motivating-local-spatial-autocorrelation">Local Indicators of Spatial Association (LISA)</a> then classify each tract by comparing its own value with the average value (<a href="https://geographicdata.science/book/notebooks/12_feature_engineering.html?highlight=spatial+lag#what-is-spatial-feature-engineering">spatial lag</a>) of its neighbors.
# 
# <div class="quad-grid">
# <div><strong>High-High (HH)</strong><br>A high-value tract surrounded by high-value neighbors. Interpreted as a local hot spot.</div>
# <div><strong>Low-Low (LL)</strong><br>A low-value tract surrounded by low-value neighbors. Interpreted as a local cold spot.</div>
# <div><strong>High-Low (HL)</strong><br>A high-value tract surrounded by low-value neighbors. Interpreted as a high spatial outlier.</div>
# <div><strong>Low-High (LH)</strong><br>A low-value tract surrounded by high-value neighbors. Interpreted as a low spatial outlier.</div>
# </div>
# 
# These quadrants are the main preliminary spatial result because they move beyond a single island-wide correlation and identify where planning conditions are locally clustered or locally statistically significant. Bivariate Local Moran extends the same idea by comparing PV density in each tract with the spatial lag of a neighboring contextual variable, such as median income, SVI, English-language proficiency, or solar irradiance.

# %% [markdown]
# # 4. Results

# %% [markdown]
# ## 4.1 Consolidated Geospatial Database for Puerto Rico
# 
# The main result is an integrated DuckDB Spatial research database that connects raw geospatial sources, model outputs, and planning-scale aggregates. The summary below shows the analytical backbone behind the maps and tables in this presentation.

# %%
display_database_summary_table()

# %% [markdown]
# ## 4.2 Choropleth Maps
# 
# <div class="two-col">
# <div><img src="outputs/maps/pv_any_evidence_per_1000_buildings_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# <div><img src="outputs/maps/pv_model_detected_per_1000_buildings_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# </div>
# <div><img src="outputs/maps/pv_detection_rate_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# <p class="figure-caption">PV evidence choropleths. These maps normalize OSM labels and model detections by the number of building units so dense urban areas are not interpreted only through raw counts.</p>
# 
# These maps are best read as exploratory indicators, not final adoption estimates. They reveal where OSM labels and model detections are spatially concentrated after normalizing by number of building units per Census unit.

# %% [markdown]
# <div class="two-col">
# <div><img src="outputs/maps/acs_income_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# <div><img src="outputs/maps/acs_education_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# <div><img src="outputs/maps/acs_spanish_english_not_at_all_18_64_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# <div><img src="outputs/maps/acs_spanish_english_well_18_64_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# </div>
# <p class="figure-caption">ACS socioeconomic context variables: income, education, and two English-language proficiency indicators. The language indicators are included because prior Lab 3 work with the CDC SVI highlighted language-access variables as important correlates for socioeconomic vulnerability and service access.</p>
# 
# <div><img src="outputs/maps/cdc_svi_2020_overall_percentile_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# <p class="figure-caption">Census-derived SVI context for interpreting PV evidence. Together, ACS and SVI variables form the main socioeconomic comparison layer for the spatial analysis.</p>
# 
# <div><img src="outputs/maps/nsrdb_ghi_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# <div><img src="outputs/maps/urban_land_share_choropleth.png" class="asset-frame" style="width: 100%;"></div>
# <p class="figure-caption">Geophysical and built-environment context choropleths. These variables are used in bivariate spatial autocorrelation analysis to see if they help explain PV evidence patterns, but they are not expected to be the causal or even dominant factors at this scale.</p>

# %% [markdown]
# ## 4.3 Model Performance
# 
# <img src="outputs/maps/pv_building_detections_by_model_arch.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Building-level detections summarized by model architecture. These counts are preliminary and depend on the unvalidated review threshold used to translate masks or similarity signals into building evidence.</p>

# %%
display_model_summary_table()

# %% [markdown]
# ## 4.4 Preliminary Spatial Analysis: LISA and Bivariate Moran
# 
# <img src="outputs/maps/case_study_esda_lisa_variable_grid.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Local Moran/LISA variable grid. HH and LL cells mark local clusters; HL and LH cells mark spatial outliers. Non-significant tracts are not interpreted as local clusters under the current permutation test threshold.</p>
# 
# <div class="analysis-callout"><strong>How to read the result:</strong> an HH PV tract is a tract with high PV evidence surrounded by tracts that also have high PV evidence; an LL PV tract is part of a low-evidence cold spot. HL and LH are especially useful for municipal targeting, outreach prioritization, feeder-level review, and manual label audits because they flag places whose local condition differs from the surrounding neighborhood context.</div>
# 
# These results are **exploratory, tract-scale, and non-causal**. They should be treated as a screening layer for follow-up investigation and as a preliminary stepping stone in my intended thesis, not as any conclusive proof that specific socioeconomic or geophysical variables drive PV adoption.

# %%
display_global_morans_table()

# %% [markdown]
# <img src="outputs/maps/case_study_esda_bivariate_moran_pv_covariates.png" class="asset-frame" style="width: 100%;">
# <p class="figure-caption">Bivariate Local Moran maps comparing tract PV evidence with neighboring tract covariates. The quadrant labels have the same HH/LL/HL/LH logic, but the neighboring value is a contextual variable rather than PV itself.</p>
# 
# Bivariate Moran compares PV density in each tract with the spatial lag of a contextual variable in neighboring tracts. For example, an HH bivariate tract means high PV density in the focal tract near high neighboring values of the selected covariate. It is useful for screening neighborhood alignment, but it is not the same as same-tract correlation.

# %%
display_bivariate_morans_table()

# %% [markdown]
# # 5. Conclusion and Discussion

# %% [markdown]
# ## 5.1 Key Findings and Data Artifacts
# 
# <ul class="tight-list">
# <li>The ongoing project now has a reproducible local geospatial database connecting OSM PV labels, Overture buildings, Census/SVI variables, raster catalogs, and NSRDB context.</li>
# <li>Google Colab makes the work shareable as a browser-based project workspace: reviewers can clone the branch into <code>/content</code>, inspect repository files, and present or rerun the interactive slides from one notebook.</li>
# <li>San Juan and Isabela provide a reasonable contrastive case-study proof-of-concept for my intended thesis project.</li>
# <li>OSM labels are valuable but must be treated as biased weak supervision that enables model training and evaluation, but with inconsistent coverage and label definitions rather than ground truth adoption counts.</li>
# </ul>

# %% [markdown]
# ## 5.2 Data and Analysis Limitations
# 
# <ul class="tight-list">
# <li><strong>Imagery:</strong> open, cloud-free, high-resolution imagery over Puerto Rico remains uneven and hard to automate at scale.</li>
# <li><strong>Labels:</strong> crowdsourced PV polygons encode mapper attention and areas-of-interest, not a direct signal of solar adoption.</li>
# <li><strong>Models:</strong> current detections are limited to building-level analysis which might suffice for exploratory purposes, but falls far short of any capacity estimation or generation potential assessment.</li>
# <li><strong>Integration:</strong> CRS, geometry validity, temporal mismatch, and resolution mismatch are core methodological risks and not fully controlled at this stage.</li>
# <li><strong>Reproducibility:</strong> full-database runs still require the project pipeline and source data; the shared browser workspace demonstrates the workflow and selected interactive outputs for review.</li>
# </ul>

# %% [markdown]
# ## 5.3 Future Work (Thesis)
# 
# <ul class="tight-list">
# <li>Significantly improve model performance against the high-digitization barrios and document label completeness assumptions or gaps.</li>
# <li>Compare model detections against installed capacity from interconnection data, or manually audited regional samples.</li>
# <li>Refine spatial analysis between PV evidence and socioeconomic + geophysical variables to go beyond exploratory insights.</li>
# <li>Extend building-level aggregation to individual "Panel Row"[8] features, adapt existing methodologies for capacity estimation via high-temporal-resolution irradiance data.</li>
# <li>Develop the PV-S3 <a href="https://motherduck.com/learn/what-is-a-data-lakehouse/#what-is-a-data-lakehouse">data lakehouse</a> and public web notebook interface: a local-first, cloud-scalable catalog of imagery, PV labels, building context, and model outputs that can publish queryable maps and analysis-ready PV inventories.</li>
# </ul>

# %% [markdown]
# ## 5.4 General AI Limitations & Particular Reflections on Agentic Coding
# 
# AI agents made the project faster, but **only when paired with strict validation habits**.
# 
# <ul class="tight-list">
# <li><strong>API hallucination risk:</strong> niche geospatial libraries change quickly, and agents often mix old and new API methods and conventions. Providing the agent with specific library documentation that matches installed versions is crucial.</li>
# <li><strong>Silent data risks:</strong> a script can run successfully while producing wrong joins, invalid geometries, or biased samples that go unnoticed without direct inspection of our dataset.</li>
# <li><strong>Model-tier sensitivity:</strong> smaller (more cost-effective) models struggled with long geospatial state and multi-file reasoning.</li>
# <li><strong>Cost pressure:</strong> the <a href="https://www.linkedin.com/posts/eat-sleep-cloud-repeat_github-just-changed-the-rules-for-copilot-share-7454668305744883712-zeN1/">industry-wide shift</a> to token-metered agentic coding makes clean repositories, small artifacts, and executable checks crucial for cost effective usage of AI development tools.</li>
# <li><strong>Best pattern:</strong> let AI generate and refactor code, but require row counts, map previews, file manifests, and focused smoke tests run directly on the dataset before trusting model outputs or descriptions of work performed.</li>
# </ul>

# %% [markdown]
# # 6. References
# 
# [1] M. Alipour, H. Salim, R. A. Stewart, and O. Sahin, "Predictors, taxonomy of predictors, and correlations of predictors with the decision behaviour of residential solar photovoltaics adoption: A review," Renewable and Sustainable Energy Reviews, vol. 123, p. 109749, 2020. doi: https://doi.org/10.1016/j.rser.2020.109749  
# [2] M. Baggu and R. Burton, "Puerto Rico Grid Resilience and Transitions to 100% Renewable Energy Study (PR100): Final Report," National Renewable Energy Laboratory (NREL), Golden, CO, Tech. Rep. NREL/TP-6A20-88384, Mar. 2024. [Online]. Available: https://www.nrel.gov/docs/fy24osti/88384.pdf  
# [3] W. Hu, K. Bradbury, J. M. Malof, B. Li, B. Huang, A. Streltsov, K. S. Fujita, and B. Hoen, "What you get is not always what you see—pitfalls in solar array assessment using overhead imagery," Applied Energy, vol. 327, p. 120143, 2022. doi: https://doi.org/10.1016/j.apenergy.2022.120143  
# [4] S. Y. Kim, K. Ganesan, C. Soderman, and R. O'Rourke, "Spatial distribution of solar PV deployment: An application of the region-based convolutional neural network," EPJ Data Science, vol. 12, no. 1, p. 25, 2023. doi: https://doi.org/10.1140/epjds/s13688-023-00399-1  
# [5] C. A. Peña-Becerra, W. A. Pacheco-Cano, D. F. Aragones-Vargas, A. Irizarry-Rivera, and M. Castro-Sitiriche, "Barrio-Level Assessment of Solar Rooftop Energy and Initial Insights into Energy Inequalities in Puerto Rico," Solar, vol. 5, p. 28, 2025. doi: https://doi.org/10.3390/solar5020028  
# [6] C. Robinson et al., "Global Renewables Watch: A Temporal Dataset of Solar and Wind Energy Derived from Satellite Imagery," arXiv preprint arXiv:2503.14860, 2025. doi: https://doi.org/10.48550/ARXIV.2503.14860
# [7] T. Sanzillo and C. Kunkel, "Solar at a Crossroads in Puerto Rico: Oversight Board, Power Plant Operator Threaten Renewable Energy Transformation," Institute for Energy Economics and Financial Analysis (IEEFA), Tech. Rep., June 2024. https://ieefa.org/resources/solar-crossroads-puerto-rico  
# [8] J. T. Stid, A. D. Kendall, A. Anctil, J. Rapp, J. C. Bingaman, and D. W. Hyndman, "A harmonized dataset of ground-mounted solar energy in the US with enhanced metadata," Scientific Data, vol. 12, no. 1, p. 1586, 2025. doi: https://doi.org/10.1038/s41597-025-05862-4  
# [9] Y. Trémenbert, G. Kasmi, L. Dubus, Y.-M. Saint-Drenan, and P. Blanc, "PyPVRoof: a Python package for extracting the characteristics of rooftop PV installations using remote sensing data," arXiv preprint arXiv:2309.07143, 2023. doi: https://doi.org/10.48550/ARXIV.2309.07143  
# [10] J. Yu, "DeepSolar: A Machine Learning Framework to Efficiently Construct a Solar Deployment Database in the United States," Joule, vol. 2, no. 12, pp. 2605–2617, 2018. doi: https://doi.org/10.1016/j.joule.2018.11.021  
# [11] Project repository: https://github.com/avega17/PLAN6068_PV_Project  
# [12] Poster presentation on initial data ingest work presented in April 2026: https://drive.google.com/file/d/119Q8_dt981HzuIwGAf882aEVYjrnkVvt/view?usp=sharing



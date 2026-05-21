# *PLAN 6068 AI Research Proposal — Revised Draft: Detecting Rooftop PV in Puerto Rico from Google Solar API Imagery and Fine-Tuned GeoAI Detectors*

##### **Alejandro S. Vega Nogales; Prof. J. Ayala Hernández**

> **Revision note.** This document replaces the original `ASVN_AI_Planning_Proposal_Final_Draft.md` following a mid-project pivot (April 2026). The original file is preserved unchanged for reference. The pivot (a) drops the panel-row superpixel segmentation research question, (b) narrows the geographic scope from island-wide to the two municipalities of **San Juan** and **Isabela**, (c) swaps the multi-source high-resolution imagery catalog for a **Google Solar API Data Layers**-only pipeline with a hard **1,999 Data-Layers-call budget cap**, and (d) introduces a **fine-tuned GeoAI rooftop PV detector** seeded by OSM labels as the core detection step. RQ2 and RQ3 are rescoped to these two municipalities and aggregated at the Census Block Group level.

### **1\. Research Problem**

#### **Local Context**

Following the catastrophic electrical grid failures experienced during Hurricanes Maria and Fiona, Puerto Rico has undergone a rapid and largely uncoordinated proliferation of decentralized rooftop photovoltaic (PV) solar installations. Driven by a need for resilience against an unreliable, centralized grid, the number of grid-connected PV systems grew nearly tenfold between 2017 and 2024 (NEPR, 2024). By June 2025 roughly 1.2 GW across 117,000+ net-metered residential and commercial installations accounted for >10% of the island's electricity consumption, up from 2–3% pre-Maria (IEEFA, 2024, 2025). This private-citizen-led transition unfolds against stagnant utility-scale progress (PR100 2023); although Puerto Rico's Renewable Portfolio Standard mandates 40% renewable generation by 2025 and 100% by 2050, every interim milestone to date has been missed (IEEFA, 2024).

#### **Problem Statement**

The decentralized nature of Puerto Rico's rooftop-PV boom creates a spatial-information gap: neither grid operators nor urban planners maintain a validated, publicly accessible geospatial inventory of residential PV. Grid operators lack the array-level granularity required for short-term site-level forecasting and local-load balancing; planners cannot quantify or target energy-justice gaps, even though high up-front costs and roof ownership asymmetries concentrate PV ownership among higher-income households (IEEFA, 2024; Peña-Becerra et al., 2025).

Published computer-vision work on PV detection (Kruitwagen et al., 2021; Robinson et al., 2025) is trained on large-array and utility-scale contexts and faces significant distribution shift when applied to small residential installations in dense tropical urban environments. As Hu et al. (2024) show, such studies commonly suffer from **four pitfalls: distribution shift, poor test-data quality, inappropriate spatial aggregation, and proprietary data**. This project addresses all four directly by (a) **fine-tuning** a pretrained detector on Puerto-Rico-specific chips, (b) using **OSM-seeded, human-review-gated** training labels, (c) aggregating at the **Census Block Group** level, and (d) relying on the **publicly priced Google Solar API** so every experiment is reproducible at a declared cost.

#### **Research Questions (Revised)**

1. **PV Detection.** Can a rooftop-PV instance-segmentation model fine-tuned from `geoai.SolarPanelDetector` (pretrained on 0.6 m NAIP imagery of Davis, CA) — using OSM-seeded labels over Google-Solar-API RGB imagery at 0.1 m (HIGH quality, San Juan metro) and 0.25 m (MEDIUM/BASE quality, Isabela) — reliably identify residential rooftop PV arrays across two heterogeneous imagery-quality regimes? How does detection performance stratify across HIGH vs. MEDIUM/BASE tiles?
2. **Equity EDA.** Is there a statistically significant spatial correlation at the Census Block Group level between detected PV presence (or PV-per-housing-unit) and ACS 2022 5-yr sociodemographic indicators (median household income, education attainment, tenure, race) for the two municipalities of San Juan and Isabela?
3. **Solar-Potential EDA.** To what extent does Google Solar API's **annual-flux** layer (replacing the previously scoped NSRDB + land-cover covariates) correlate with observed PV presence, relative to the socioeconomic signal from RQ2?

The original RQ1 — replicating the Stid et al. (2025) panel-row superpixel methodology — has been deprecated. The superpixel experiment produced weak boundaries on the types of small residential arrays that dominate in PR and is not competitive with supervised instance segmentation for this task. It is removed from scope and archived in `notebooks/rasters/legacy/`.

### **2\. Data to be Used**

* **Building Footprints.** [Overture Maps buildings](https://docs.overturemaps.org/guides/buildings/) (2026-03-18 release) clipped to San Juan and Isabela, ingested into a DuckDB spatial table (`pr_overture_buildings`) with Hilbert ordering and R-Tree indexing. Municipio attribution is derived from the US Census 2022 county-equivalent TIGER/Line boundaries.
* **Census Geometries.** US Census 2022 TIGER/Line **municipalities**, **tracts**, and **block groups** (state FIPS `72`), ingested via `censusdis` into `pr_municipalities`, `pr_census_tracts`, `pr_block_groups`.
* **Rooftop PV Seed Labels.** [OpenStreetMap rooftop-PV polygons](https://docs.overturemaps.org/) extracted via `osmnx` Overpass queries (`generator:source=solar`) and persisted as `pr_osm_rooftop_pv_polygons`. Used both to **seed** training chips and as a **validation** cohort.
* **High-Resolution Imagery & Solar Potential.** The primary imagery stack is now [Google Solar API — Data Layers](https://developers.google.com/maps/documentation/solar/data-layers), endpoint `solar.googleapis.com/v1:getDataLayers`, view `IMAGERY_AND_ANNUAL_FLUX_LAYERS` (RGB + building mask + DSM + annual flux). Requests are issued at ≤175 m radius per tile; a fraction are HIGH (0.1 m, concentrated in San Juan metro), the remainder are MEDIUM/BASE (0.25 m, including Isabela). This ingest path is already implemented with tile manifests, imagery-quality probes, a cost ledger, and local GeoTIFF caching under `data/rasters/solar/`. An auxiliary local STAC raster branch (NAIP, Maxar Open Data, and other available sources) is also in place for municipality-level coverage QA, visual cross-checks, and fallback inspection, but it is not the core covariate source for the final project.
* **Sociodemographic Data.** ACS **2020-2024 5-yr** estimates at BG level, fetched via `censusdis`, pending investigation on whether the 2020 vs 2024 census geography geometries have changed, otherwise we will have ACS pinned to 2020 so the data are co-temporal with the 2020 decennial geography and the 2020 Urban Areas product. Variables: `B19013_001E` (median household income), `B25001_001E` / `B25003` (housing units & tenure), **`B03002`** (Hispanic-or-Latino origin by race, the canonical input for the Census 2020 Diversity Index), `B15003` (educational attainment). A **Diversity Index** is derived per BG following Census 2020 methodology, $DI = 1 - \sum_i p_i^2$, over the seven mutually-exclusive non-Hispanic race groups plus Hispanic/Latino of any race.
* **Geophysical Variables:** Solar irradiance and ground temperature from the [NREL’s National Solar Radiation Database (NSRDB)](https://www.nlr.gov/hpc/nsrdb-dataset), and (tentatively) Land Cover data [from Overture Maps](https://docs.overturemaps.org/blog/2024/05/16/land-cover/) derived from ESA’s 10m WorldCover rasters. The NSRDB branch has progressed to a first-draft GOES Full Disc ingest planner using documented `wkt` geometry requests, `site-count` preflight checks, request-weight-aware batching, and a parquet request ledger; however, the bulk Puerto Rico pull and normalization workflow are not complete yet, so NSRDB remains exploratory rather than a guaranteed final-project input.  
* **Urban vs. Rural Classification.** [Census 2020 Urban Areas — Blocks file](https://www2.census.gov/geo/docs/reference/ua/2020_UA_BLOCKS.txt) filtered to state FIPS `72`, aggregated to BG level (`pr_bg_urban_flags`): a BG is classified as **rural** if it contains no 2020 urban blocks. The 2020 release is the latest available (per [Census geo-areas guidance](https://www.census.gov/programs-surveys/geography/guidance/geo-areas/urban-rural.html)) and is the reason ACS vintage is pinned to 2020.

### **3\. Methodology and AI Tool(s)**

The pipeline is implemented as Jupytext `.py` notebooks (paired with `.ipynb`) in five phases. All ingest steps use DuckDB + the spatial extension, and all tiles/polygons flow through the centralized `data/PR_PV_plan_data.duckdb` database.

#### **3.1 Pipeline Overview**

| Phase | Notebook(s) | Output table / artifact |
| :---- | :---- | :---- |
| 1 — Core vectors | `vectors/01_census_geometries_ingest.py`, `vectors/02_osm_pv_ingestion_and_viz.py`, `vectors/03_overture_buildings_ingest.py`, `vectors/06_h3_tile_manifest.py` | Census municipio / tract / BG geometries, `pr_osm_rooftop_pv_polygons`, `pr_overture_buildings`, `pr_solar_tile_manifest` |
| 2 — Raster ingest and QA | `rasters/05_solar_quality_probe.py`, `rasters/08_pr_stac_municipality_fetch.py`, `rasters/09_google_solar_api_ingest.py`, `rasters/10_solar_raster_catalog.py` | Google Solar GeoTIFFs under `data/rasters/solar/`, local STAC cache under `data/rasters/stac/local/`, Solar API ledger, raster coverage manifests, `pr_solar_catalog_items.parquet` |
| 3 — GeoAI training and inference | `rasters/11_geoai_training_data.py`, `rasters/12_geoai_solar_finetune.py`, `rasters/13_geoai_solar_inference.py` | Training chips, Mask R-CNN checkpoint, preview artifacts, `pr_solar_pv_detections` |
| 4 — Tabular enrichment and exploratory analysis | `tabular/04_acs_5year_ingest.py`, `tabular/05_urban_blocks_2020_ingest.py`, `tabular/13_pv_building_join.py`, `tabular/14_pv_bg_aggregation.py` | ACS covariates, `pr_bg_urban_flags`, `pr_buildings_with_pv`, `pr_pv_bg_aggregates`, choropleths, Moran's I |
| 5 — Auxiliary NSRDB and narrative | `tabular/06_nsrdb_goes_full_disc_ingest.py` (draft), `00_e2e_master_narrative.py` | NSRDB request plan / ledger artifacts, end-to-end story, figures |

#### **3.2 Tile Manifest and Quality Probe**

Each Census Block Group intersecting San Juan or Isabela is covered by a snapped UTM-19N grid of ≤175 m-radius tiles spaced ~247 m (=175 · √2) so that 175 m disks exactly tile the plane. Tiles are assigned `priority_score ∈ {1, 2, 3}`: **3** for tiles overlapping the two seed neighborhoods (Puerto Nuevo, Barrio Mora), **2** for tiles whose BG contains ≥1 OSM PV polygon, **1** for coverage-sweep tiles. A pre-spend probe calls `buildingInsights:findClosest` (free tier, 10K/month) to tag each BG's imagery quality, collapsing tiles by best available quality per BG; 404 tiles are marked `no_coverage` and dropped.

#### **3.3 Rooftop PV Detection**

`geoai.SolarPanelDetector` (Davis, CA NAIP 0.6 m baseline) is warm-started and fine-tuned via `geoai.train_MaskRCNN_model(num_classes=2, num_epochs=30, batch_size=4, learning_rate=1e-4, val_split=0.2)` on 512×512 chips exported from priority-3 tiles with OSM polygons as seed masks (`geoai.export_geotiff_tiles_batch(buffer_radius=0.5, class_value_field="class", skip_empty_tiles=True)`). Inference runs `geoai.object_detection(window_size=400, overlap=100, confidence_threshold=0.4)` across every Solar RGB tile, followed by `geoai.orthogonalize(epsilon=0.2)` → `geoai.add_geometric_properties` → a geometry filter (`3 ≤ area_m² ≤ 500` and `elongation < 10`) → persist to `pr_solar_pv_detections`. Detection metrics are reported stratified by imagery quality (HIGH vs. MEDIUM/BASE). Operationally, the success criterion is not perfect polygon fidelity but a detector that is reliable enough to support downstream binary building assignment: a building is treated as `has_PV` when one or more accepted detections intersect its footprint.

#### **3.4 Building- and BG-Level Aggregation**

Each Overture building is joined to (a) OSM PV polygons (`has_pv_osm`), (b) model detections, where `has_pv_detected` is a binary indicator set when any accepted detection intersects the building footprint and `pv_detected_area` stores the intersected detected area, and (c) annual-flux zonal statistics (mean / p10 / p90 / pixel-count) from the Solar API `annualFlux` GeoTIFFs, and persisted to `pr_buildings_with_pv`. Aggregation to Block Group uses spatial containment of building centroids, yielding `pr_pv_bg_aggregates` with detection rate per building, OSM-labeled count, pixel-weighted BG flux mean, the 2020 urban/rural flag, and ACS 2016-2020 5-yr covariates including the derived Diversity Index. Choropleth maps (matplotlib + `mapclassify` quantile binning) are produced for each variable across the two project municipalities — detection rate, annual flux, median household income, Bachelor's-or-higher share, Diversity Index, and urban/rural classification. Spatial autocorrelation of `detected_pv_rate` is measured per-municipality via Moran's I (`libpysal.weights.Queen` + `esda.Moran`, 999 permutations).

#### **3.5 Python Libraries and Geospatial Stack**

| Category | Libraries |
| :---- | :---- |
| Geospatial | GeoPandas, Shapely, DuckDB (spatial), `censusdis`, `osmnx` |
| Raster | Rasterio, rioxarray, `rasterstats` |
| CV & GeoAI | `geoai-py` (Mask R-CNN Solar Panel Detector, DINOv3 similarity validation) |
| APIs | `google-maps-solar` (plus direct REST via `requests` when the SDK omits API-key propagation) |
| Spatial Stats | `libpysal`, `esda`, `mapclassify` |
| Viz | Matplotlib, Seaborn, `contextily`, `lonboard`, Folium |

#### **3.6 AI Developer-Assistant Workflow**

Implementation leverages the [GitHub Copilot Coding Agent](https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-coding-agent) inside VS Code, combined with Claude Opus for deep coding tasks. The assistant accelerates three kinds of work: (i) REST-fallback wrappers for the Solar SDK, (ii) DuckDB SQL over the spatial extension, and (iii) boilerplate for visualization and data-class definitions. Planning and change-capture are surfaced as Markdown documents under `outputs/reports/`.

#### **3.7 Current Implementation Progress and Risk (May 2026)**

The core ingest stack is materially further along than the original proposal text implied. The vector foundation for San Juan and Isabela is in place, the Google Solar tile manifest and Data Layers ingest are operational, and the raster branch has been hardened for reruns with cache-aware manifests, local GeoTIFF persistence, and coverage QA. In parallel, an auxiliary STAC workflow now clips one local raster per unique asset, exports square tiles for occupied H3 cells, and visualizes municipality-level land-area and building-coverage footprints so raster coverage gaps can be diagnosed before training or inference.

The NSRDB / NREL branch has also moved from concept to a first draft: the project now has a documented request planner for GOES Full Disc that uses polygon `wkt`, `site-count` preflight, request-weight-aware batching, and a parquet ledger. That is real implementation progress, but it is still not a completed production ingest. For the final project, Google Solar `annualFlux` remains the primary solar-potential variable, while NSRDB should be treated as an exploratory extension that is included only if the pull, normalization, and aggregation steps are finished in time.

The most critical blocker is no longer basic data acquisition; it is inference reliability. The practical requirement for the project is to classify every building as `has_PV` versus `no_PV` by counting any building with one or more accepted detections. If the fine-tuned detector is unstable across HIGH versus MEDIUM/BASE imagery, or if OSM-seeded labels introduce too much noise, then every downstream building-level and BG-level quantity inherits that uncertainty. The near-term priority is therefore a reviewed, thresholded inference workflow that is reliable enough for binary building assignment, even if polygon boundaries remain imperfect.

Because of that risk profile, the concluding quantitative work will likely be intentionally limited. Rather than promising a broad modeling agenda, the project will soon formalize a smaller set of exploratory spatial and spatial-correlation analyses centered on descriptive BG-level mapping, building-level `has_PV` aggregation, and a narrow set of municipal-scale correlation or Moran-style diagnostics that can be defended given the final detection quality.

### **4\. Expected Results**

Given the ~2-month course scope, the deliverables are a working prototype plus evidence of feasibility:

* **End-to-End Pipeline.** A reproducible, budget-bounded ingest → detection → aggregation pipeline over San Juan and Isabela, re-runnable against any pair of PR municipalities via a single constants block.
* **Fine-Tuned PV Detector.** A Mask R-CNN checkpoint and inference workflow evaluated against held-out OSM labels and reviewed overlays, with IoU / precision / recall reported **separately for HIGH (0.1 m) and MEDIUM/BASE (0.25 m) tiles** to quantify the resolution effect, and with the practical benchmark being whether the model is reliable enough to support a binary building-level `has_PV` versus `no_PV` assignment.
* **BG-Level Equity & Flux Maps.** A deliberately limited exploratory set of choropleth figures and spatial-correlation diagnostics for the two municipalities covering detected-PV rate, annual-flux mean, median household income, and share of adults with a Bachelor's degree or higher, plus Moran's I or closely related diagnostics for spatial autocorrelation of detection rate. Published to `outputs/maps/` and `outputs/figures/`.
* **Cost Transparency.** A published ledger of every Solar API call with imagery quality, date, and estimated cost in USD.

### **References**

1. J. T. Stid et al., *"A harmonized dataset of ground-mounted solar energy in the US with enhanced metadata,"* Nature Scientific Data, vol. 12, no. 1, p. 1586, Sep. 2025, doi: [10.1038/s41597-025-05862-4](https://doi.org/10.1038/s41597-025-05862-4).
2. W. Hu et al., *"What you get is not always what you see — pitfalls in solar array assessment using overhead imagery,"* Applied Energy, vol. 327, p. 120143, Dec. 2022, doi: [10.1016/j.apenergy.2022.120143](https://doi.org/10.1016/j.apenergy.2022.120143).
3. IEEFA, *"Solar at a crossroads in Puerto Rico."* Available: [ieefa.org/resources/solar-crossroads-puerto-rico](https://ieefa.org/resources/solar-crossroads-puerto-rico)
4. IEEFA, *"Rooftop Solar in Puerto Rico Reaches 10%, Grid Reliability Continues to Wane."* Available: [ieefa.org/resources/rooftop-solar-puerto-rico-reaches-10-grid-reliability-continues-wane](https://ieefa.org/resources/rooftop-solar-puerto-rico-reaches-10-grid-reliability-continues-wane)
5. C. A. Peña-Becerra et al., *"Barrio-Level Assessment of Solar Rooftop Energy and Initial Insights into Energy Inequalities in Puerto Rico,"* Solar, vol. 5, no. 2, p. 28, Jun. 2025, doi: [10.3390/solar5020028](https://doi.org/10.3390/solar5020028).
6. L. Kruitwagen et al., *"A global inventory of photovoltaic solar energy generating units,"* Nature, vol. 598, no. 7882, pp. 604–610, Oct. 2021, doi: [10.1038/s41586-021-03957-7](https://doi.org/10.1038/s41586-021-03957-7).
7. C. Robinson et al., *"Global Renewables Watch: A Temporal Dataset of Solar and Wind Energy Derived from Satellite Imagery,"* 2025, arXiv. doi: [10.48550/ARXIV.2503.14860](https://doi.org/10.48550/arXiv.2503.14860).
8. J. Yu et al., *"DeepSolar: A Machine Learning Framework to Efficiently Construct a Solar Deployment Database in the United States,"* Joule, vol. 2, no. 12, pp. 2605–2617, Dec. 2018, doi: [10.1016/j.joule.2018.11.021](https://doi.org/10.1016/j.joule.2018.11.021).
9. M. Baggu et al., *"Puerto Rico Grid Resilience and Transitions to 100% Renewable Energy Study (PR100): Final Report,"* NREL/TP-6A20-88384, Mar. 2024, doi: [10.2172/2335361](https://doi.org/10.2172/2335361).
10. Google, *"Solar API — Data Layers,"* Google Maps Platform documentation. Available: [developers.google.com/maps/documentation/solar/data-layers](https://developers.google.com/maps/documentation/solar/data-layers)
11. Q. Wu, *"GeoAI: Geospatial Artificial Intelligence,"* opengeoai.org. Available: [opengeoai.org](https://opengeoai.org/)
12. A. Lamstein, *"censusdis — Python functions for multi-year ACS data."* Available: [arilamstein.com/blog/2025/01/29/new-python-functions-for-working-with-multi-year-acs-data/](https://arilamstein.com/blog/2025/01/29/new-python-functions-for-working-with-multi-year-acs-data/)

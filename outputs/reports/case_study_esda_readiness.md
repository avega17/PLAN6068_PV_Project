# Case-Study ESDA Readiness

## Core Decision

Freeze the core analysis unit at the census block-group aggregate for San Juan and Isabela. The current repo is ready on geometry, ACS fallback data, and urban context, but it is not yet ready on the materialized building join and BG aggregate tables that the core ESDA workflow depends on.

Current gate:
- Rebuild `pr_buildings_with_pv` from the building-level join notebook.
- Rebuild `pr_pv_bg_aggregates` after that join exists.
- Only then run the core ESDA maps, correlations, and Moran diagnostics.

## Dataset Status

| dataset | status | evidence | notes |
| --- | --- | --- | --- |
| Case-study municipality polygons | ready | pr_census_counties: Isabela: 1, San Juan: 1 | Target municipality boundaries are the geometry anchor for the BG core. |
| Census block-group geometries | ready | pr_census_block_groups: Isabela: 29, San Juan: 362 | This is the frozen core analysis unit for ESDA and correlations. |
| ACS 2024 5-year block-group attributes | ready | table=False (pr_acs_2024_block_groups), artifact=True (data/tabular/acs_2024_5yr_block_groups.parquet) | Ready via local parquet fallback even though the DuckDB table is not currently materialized. |
| Urban context flags | ready | pr_bg_urban_flags: 2,513 rows | Block-group urban context is materialized and joinable now. |
| Overture buildings | ready | pr_overture_buildings: Isabela: 38,969, San Juan: 126,088 | Building footprints are present and scoped to the case-study municipalities. |
| OSM rooftop PV polygons | ready | pr_osm_rooftop_pv_polygons: Isabela: 1,474, San Juan: 9,938 | Usable as the conservative fallback PV signal once joined back to buildings and BGs. |
| GeoAI PV detections | partial | pr_solar_pv_detections: total rows=1, by target=n/a | Detections exist but are too sparse for the core narrative; retain as sensitivity only. |
| Building-level PV + annual-flux join | partial | pr_buildings_with_pv: materialized=True | This is the immediate gate. The join table is missing, so BG PV and flux outcomes are not audit-ready yet. |
| Block-group aggregate surface | partial | pr_pv_bg_aggregates: materialized=True | Core ESDA should remain BG-based, but the aggregate must be rebuilt before maps, correlations, or Moran diagnostics can run. |
| NSRDB multiyear add-on | partial | normalized parquet files: nsrdb_site_6393384_2018_30min.parquet, nsrdb_site_6393384_2019_30min.parquet, nsrdb_site_6393384_2020_30min.parquet, nsrdb_site_6393384_2021_30min.parquet, nsrdb_site_6393384_2022_30min.parquet | Keep NSRDB outside the core ESDA gate until the BG join path is restored; use it later as a comparison branch. |

Ready now: Case-study municipality polygons, Census block-group geometries, ACS 2024 5-year block-group attributes, Urban context flags, Overture buildings, OSM rooftop PV polygons.
Partial / blocked surfaces: GeoAI PV detections, Building-level PV + annual-flux join, Block-group aggregate surface, NSRDB multiyear add-on.
Exploratory add-ons: none.

## Variable Audit

| variable | status | recommended_role | notes |
| --- | --- | --- | --- |
| detected_pv_count | partial | Sensitivity only | Blocked by missing building join and materially weakened by sparse detections. |
| detected_pv_rate | partial | Sensitivity only | Same dependency as detected_pv_count; do not use as the lead outcome until the join is rebuilt and zero-inflation is checked. |
| osm_pv_count / overlap_count fallback | partial | Preferred conservative PV outcome | OSM polygons are present now, but the building and BG joins still need to be materialized. |
| annual_flux_mean_kwh_per_kw_yr | partial | Core covariate once join is rebuilt | Blocked on the missing building-level join; do not claim BG-level flux readiness yet. |
| flux_pixel_count | partial | Coverage QC metric | Needed to assess flux coverage defensibility after the join is rebuilt. |
| median_household_income_usd | ready | Core contextual covariate | Available via the ACS 2024 parquet fallback and ready to join into the BG aggregate. |
| pct_bachelor_plus | ready | Core contextual covariate | Derived directly from the ACS 2024 block-group slice. |
| pct_owner_occupied | ready | Core contextual covariate | Derived directly from the ACS 2024 block-group slice. |
| diversity_index | ready | Core contextual covariate | Derivable now from ACS B03002 once the BG aggregate is rebuilt. |
| is_urban / pct_urban_population | ready | Core contextual covariate | Materialized in pr_bg_urban_flags and ready for the BG join. |

Primary interpretation for the 3-day core:
- Keep the BG aggregate as the analysis unit.
- Prefer a conservative OSM or overlap-style PV outcome if the rebuilt detection outputs remain sparse or zero-inflated.
- Treat detections as sensitivity only unless the rebuilt join shows materially better coverage than the current 58-row detection table suggests.

## Method Matrix

| method_family | method | status | implementation_notes | suggested_modules |
| --- | --- | --- | --- | --- |
| Core | Fixed-classification municipality choropleths | recommended | Use shared breaks across San Juan and Isabela; keep municipality small multiples instead of pooling the disconnected geography. | geopandas.plot, mapclassify |
| Core | Pearson / Spearman matrix and targeted scatterplots | recommended | Use BG-level outcomes plus ACS and urban covariates once the aggregate exists. | pandas, scipy.stats, seaborn |
| Core | Queen weights, Moran scatter, global Moran's I | recommended | Run primarily per municipality; pooled results are descriptive only. | libpysal.weights.Queen, esda.moran.Moran |
| Core | Local Moran / LISA | recommended | Produce one main PV-outcome LISA map and one optional contextual map once the BG outcome is defensible. | esda.moran.Moran_Local |
| Sensitivity | Queen-versus-Rook weights check | recommended | Retain only if the main spatial interpretation changes materially. | libpysal.weights.Queen, libpysal.weights.Rook |
| Appendix | Correlogram or one focused bivariate/partial Moran question | optional | Only after the core ESDA tables and maps land cleanly. | libpysal, esda |
| Appendix | DBSCAN on NSRDB or other point surfaces | optional | Keep off the BG core. If used, treat it as a separate point-based appendix rather than a BG clustering result. | sklearn.cluster.DBSCAN |
| Excluded from 3-day core | Point-pattern methods or DBSCAN on BG centroids | exclude | Not aligned with the frozen BG analysis unit for the core deliverable. | n/a |
| Excluded from 3-day core | Full spatial econometrics, GWR, predictive ML | exclude | Too far beyond the current data readiness and timeline. | n/a |

## Core ESDA Execution Order

1. Materialize `pr_buildings_with_pv` and then `pr_pv_bg_aggregates`.
2. Export one audited BG table for San Juan and Isabela with PV, flux, ACS, and urban fields.
3. Build municipality small-multiple choropleths with fixed classification across both municipalities.
4. Compute Pearson and Spearman correlations plus targeted scatterplots for the chosen PV outcome against flux, income, education, tenure, diversity, and urban context.
5. Build Queen weights per municipality, inspect neighbor counts and isolates, then run global Moran's I primarily per municipality.
6. Produce one main LISA map for the primary PV outcome and one optional contextual LISA map if the first result is stable.
7. Run a Queen-versus-Rook sensitivity check only if the interpretation changes materially.

## NSRDB Add-On Path

Treat NSRDB as non-blocking for the core BG sprint. Once normalized site or summary parquet files exist, aggregate them to BG or municipio and compare whether they materially change the story relative to the Google-flux path. Until then, keep NSRDB in the report as a future or appendix branch rather than a core finding.

## Report Structure

1. Project status by dataset.
2. Analysis-unit choice and caveats.
3. Core ESDA workflow.
4. PySAL implementation notes.
5. Optional clustering appendix.
6. Contingent NSRDB add-on path.

## 3-Day Schedule

1. Day 1: rebuild the building join and BG aggregate; audit the candidate PV outcome and flux coverage fields.
2. Day 2: produce choropleths, correlation tables, weights diagnostics, and Moran results per municipality.
3. Day 3: finish LISA outputs, write the guide, and add only the smallest optional appendix that survives the readiness checks.

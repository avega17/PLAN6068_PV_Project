# Tract Indicator Guide

This note explains the tract maps and Moran diagnostics exported by `14_pv_bg_aggregation.py`.
Public-facing PV indicators are scaled per 1,000 buildings so readers do not have to interpret very small decimals.

## Buildings with any PV evidence per 1,000 buildings

- Output file: `outputs/maps/pv_any_evidence_per_1000_buildings_choropleth.png`
- What it measures: Counts buildings tagged either by OSM rooftop PV labels or by the GeoAI detection workflow, standardized by 1,000 buildings in each tract.
- How to read it: This is the broadest rooftop-PV prevalence indicator in the project and is the best first map for non-technical readers.
- Observed range across case-study tracts: San Juan: 0.00 to 209.09; Isabela: 21.97 to 109.91

## Model-detected PV buildings per 1,000 buildings

- Output file: `outputs/maps/pv_model_detected_per_1000_buildings_choropleth.png`
- What it measures: Counts only buildings flagged by the inference model, standardized by 1,000 buildings in each tract.
- How to read it: Compare this map against the broader any-evidence surface to see where model detections diverge from label-assisted evidence.
- Observed range across case-study tracts: San Juan: 0.00 to 209.09; Isabela: 21.97 to 52.38

## NSRDB multiyear GHI mean (W/m2)

- Output file: `outputs/maps/nsrdb_ghi_choropleth.png`
- What it measures: A tract-average rooftop irradiance context layer derived from the nearest NSRDB 30-minute site summaries attached to buildings.
- How to read it: Higher values indicate stronger long-run solar resource potential, but not necessarily more rooftop PV adoption.
- Observed range across case-study tracts: San Juan: 218.22 to 240.81; Isabela: 230.60 to 243.57

## Distinct occupied H3 cells

- Output file: `outputs/maps/occupied_h3_cell_count_choropleth.png`
- What it measures: Counts distinct H3 cells containing buildings within each tract.
- How to read it: This is a simple spatial-coverage diagnostic: larger counts often indicate tracts with more distributed building footprints and inference opportunities.
- Observed range across case-study tracts: San Juan: 11.00 to 620.00; Isabela: 117.00 to 957.00

## ACS median household income (USD)

- Output file: `outputs/maps/acs_income_choropleth.png`
- What it measures: Median household income from the locally available ACS tract slice used in the project pipeline.
- How to read it: Income remains a core planning covariate for testing whether rooftop-PV evidence clusters with higher-earning neighborhoods.
- Observed range across case-study tracts: San Juan: 2499.00 to 85633.00; Isabela: 17273.00 to 29157.00

## Share age 25+ with Bachelor's or higher

- Output file: `outputs/maps/acs_education_choropleth.png`
- What it measures: Share of adults age 25+ with a bachelor's degree or higher.
- How to read it: Educational attainment is still a useful socioeconomic context layer for rooftop-PV adoption analysis.
- Observed range across case-study tracts: San Juan: 0.01 to 0.84; Isabela: 0.13 to 0.33

## Share of Spanish-speaking adults 18-64 who speak English well

- Output file: `outputs/maps/acs_spanish_english_well_18_64_choropleth.png`
- What it measures: Among Spanish-speaking adults age 18-64, the share reporting that they speak English well.
- How to read it: This is the positive bilingual-capacity measure requested for the planning narrative because it can proxy access to higher-paying jobs and service networks.
- Observed range across case-study tracts: San Juan: 0.01 to 0.56; Isabela: 0.14 to 0.42

## Share of Spanish-speaking adults 18-64 who speak no English

- Output file: `outputs/maps/acs_spanish_english_not_at_all_18_64_choropleth.png`
- What it measures: Among Spanish-speaking adults age 18-64, the share reporting that they do not speak English at all.
- How to read it: This is the opposite bilingual-capacity extreme and helps interpret where language access barriers may align with lower rooftop-PV uptake.
- Observed range across case-study tracts: San Juan: 0.00 to 0.67; Isabela: 0.15 to 0.46

## CDC SVI 2020 overall percentile (0-100)

- Output file: `outputs/maps/cdc_svi_2020_overall_percentile_choropleth.png`
- What it measures: The CDC/ATSDR Social Vulnerability Index overall percentile for 2020 Puerto Rico tracts, scaled from 0 to 100.
- How to read it: This replaces the earlier diversity-index map with an established public-health vulnerability measure that is much more interpretable in the Puerto Rico planning context.
- Observed range across case-study tracts: San Juan: 0.33 to 99.67; Isabela: 22.45 to 83.95

## Urban land-area share

- Output file: `outputs/maps/urban_land_share_choropleth.png`
- What it measures: Share of tract land area classified as urban by the Census 2020 urban-area overlay workflow.
- How to read it: This is more informative than a binary urban/rural map because all target tracts are urban to some extent; the gradient still reveals how built-up each tract is.
- Observed range across case-study tracts: San Juan: 0.89 to 1.00; Isabela: 0.15 to 1.00

## Moran's I diagnostics

The Moran's I figure summarizes whether neighboring tracts have similar PV prevalence values.
Positive Moran's I means similar values cluster together in space; values near zero suggest little spatial structure; negative values indicate local contrast.
The `p` labels on each bar are permutation-test probabilities, so smaller values indicate stronger evidence that the spatial pattern is not random.

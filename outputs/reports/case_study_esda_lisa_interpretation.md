# Case-Study ESDA Interpretation Notes

## LISA Cluster Labels

- High-High (HH): a high-value tract surrounded by high-value neighbors; interpreted as a hot spot.
- Low-Low (LL): a low-value tract surrounded by low-value neighbors; interpreted as a cold spot.
- High-Low (HL): a high-value tract surrounded by low-value neighbors; interpreted as a high spatial outlier.
- Low-High (LH): a low-value tract surrounded by high-value neighbors; interpreted as a low spatial outlier.
- Not significant: the Local Moran pseudo p-value is not below the configured threshold.

Main threshold: p < 0.05; permutations: 999; weights: row-standardized Queen contiguity, run separately by municipality.

## Slide Variables

- `any_pv_evidence_buildings_per_1000`: Buildings with any PV evidence per 1,000 buildings. Broadest rooftop-PV prevalence signal, combining OSM labels and model detections per 1,000 buildings.
- `median_household_income_usd`: ACS median household income (USD). Socioeconomic purchasing-power context from the tract-level ACS slice.
- `pct_bachelor_plus`: Share age 25+ with Bachelor's or higher. Educational-attainment context for rooftop-PV adoption patterns.
- `pct_spanish_english_well_18_64`: Share of Spanish-speaking adults 18-64 who speak English well. Bilingual-capacity proxy for access to labor markets and service networks.
- `cdc_svi_2020_overall_percentile`: CDC SVI 2020 overall percentile rank (0-100). CDC/ATSDR overall vulnerability percentile, rescaled to 0-100.
- `nsrdb_multiyear_ghi_mean`: NSRDB multiyear GHI mean (W/m2). Nearest-site NSRDB long-run global horizontal irradiance context attached through buildings.

## Bivariate Moran Caution

Bivariate Moran compares one variable in each tract with the spatial lag of another variable in neighboring tracts. It is not the same as same-tract Pearson or Spearman correlation, and it should be interpreted as exploratory neighborhood alignment rather than causality.

Local bivariate backup maps: PV density vs neighboring Median income, Education, English proficiency, No English, Social vulnerability, Irradiance.

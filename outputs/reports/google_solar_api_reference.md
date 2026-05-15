# Google Solar API — Reference Summary

Compiled from the official Google Maps Platform Solar API docs, the Python SDK
reference, and the Leafmap convenience helpers. Scope: only the Data Layers +
Building Insights endpoints; Aggregate Insights is out of scope for this
project.

## 1. Endpoints and quotas (Essentials tier)

| Endpoint | Purpose | Free / month | Paid pricing (≤100K/mo) |
|---|---|---|---|
| `buildingInsights:findClosest` | Closest building + solar potential JSON for a lat/lon | 10,000 | $5 / 1K |
| `dataLayers:get` | Multi-layer GeoTIFF package (RGB, DSM, mask, annual/monthly flux, hourly shade) | 1,000 | $75 / 1K |

Both endpoints return `404 NOT_FOUND` where Google has no coverage. The
interactive coverage map is only an approximation — always treat 404s as real.

Budget note for this project: ~2,000 Data Layers calls (≈ $50 of credits + a
user-covered delta at $75/1K). Building Insights is effectively free at our
scale and is used as a cheap coverage + quality probe.

## 2. `buildingInsights:findClosest`

GET request with `location.latitude`, `location.longitude`, optional
`requiredQuality` (`HIGH|MEDIUM|BASE`). Response keys we care about:

- `name` — closest building ID (can be ignored).
- `center.latitude`, `center.longitude` — snapped centroid.
- `imageryQuality` — `HIGH | MEDIUM | BASE`.
- `imageryDate`, `imageryProcessedDate`.
- `solarPotential.maxArrayPanelsCount`, `.maxArrayAreaMeters2`,
  `.maxSunshineHoursPerYear`, `.carbonOffsetFactorKgPerMwh`.
- `solarPotential.roofSegmentStats[]` — per-roof-plane azimuth, pitch, usable
  area, annual sunshine — useful for per-building flux priors.

We use this purely to tag each BG with `imageryQuality` before spending Data
Layers quota. The JSON is small (<20 KB typical) so caching is trivial.

## 3. `dataLayers:get`

GET request with:

- `location.latitude`, `location.longitude`
- `radiusMeters` — integer.
- `view` — `DataLayerView` enum (see §4).
- `requiredQuality` — `HIGH | MEDIUM | BASE`.
- `pixelSizeMeters` — one of `0.1`, `0.25`, `0.5`, `1.0`.
- `experiments=EXPANDED_COVERAGE` — required to receive BASE-tier responses.

Response is a JSON envelope listing short-lived GeoTIFF URLs. URLs expire after
~1 hour but downloaded GeoTIFFs may be cached locally up to 30 days per ToS.

### 3.1 Radius rules

The single hard constraint per the official docs and verified empirically
against the live API:

- `radiusMeters ≤ 100`: always valid (any supported pixel size).
- `radiusMeters > 100`: **must satisfy `radiusMeters ≤ pixelSizeMeters * 1000`**
  (e.g. up to 250 m at 0.25 m/pix, 500 m at 0.5 m/pix, 1,000 m at 1.0 m/pix).
- `radiusMeters > 175` additionally forbids the `IMAGERY_AND_ALL_FLUX_LAYERS`
  and `FULL_LAYERS` views (monthly flux and hourly shade).

**Known gotcha:** `radiusMeters=175` + `pixelSizeMeters=0.1` (HIGH native)
**fails every call** with HTTP 400 `INVALID_ARGUMENT` because
`0.1 × 1000 = 100 < 175`. `utils/solar_api.py::enforce_radius_pixel_constraint`
auto-bumps `pixelSizeMeters` to the smallest supported value satisfying the
constraint (0.25 m in this case) rather than silently failing.

Our tiles use **radius=100 m** (HIGH, 0.1 m native) or **radius=175 m** with
**pixel=0.25 m** (MEDIUM/BASE, or HIGH with auto-adjust) to maximize coverage
per request while keeping RGB, building mask and annual flux.

### 3.2 Pixel size per layer (native resolution)

| Layer | Native pixel size | Format |
|---|---|---|
| RGB (aerial) | 0.1 m | 3-band uint8 GeoTIFF |
| Building mask | 0.1 m | 1-bit mask |
| DSM (surface model) | 0.1 m | float32, nodata = -9999 |
| Annual flux (kWh/kW/yr) | 0.1 m | float32 |
| Monthly flux | 0.5 m | 12-band float32 |
| Hourly shade | 1.0 m | 24 GeoTIFFs × 366 bands (bit-packed) |

Server resamples to `pixelSizeMeters`; asking for 0.1 m on a MEDIUM area still
returns 0.1 m pixels but visually at 0.25 m quality — pay attention to
`imageryQuality` in the response, not the pixel grid.

## 4. `DataLayerView` enum

| Value | Bundles |
|---|---|
| `DSM_LAYER` | DSM only |
| `IMAGERY_LAYERS` | RGB + mask + DSM |
| `IMAGERY_AND_ANNUAL_FLUX_LAYERS` | RGB + mask + DSM + annual flux | **← our default** |
| `IMAGERY_AND_ALL_FLUX_LAYERS` | + monthly flux (forces radius ≤ 100 m or ≤ pixel×1000) |
| `FULL_LAYERS` | + hourly shade |

## 5. Imagery quality tiers

| Quality | Source | Effective GSD | PR coverage |
|---|---|---|---|
| `HIGH` | low-altitude aerial, 0.1 m/px | 0.1 m | San Juan metro + small pockets |
| `MEDIUM` | high-altitude aerial, 0.25 m/px | 0.25 m | large parts of PR including some coastal municipalities |
| `BASE` (experimental) | satellite, 0.25 m/px | 0.25 m | most remaining covered areas |

BASE requires `experiments=EXPANDED_COVERAGE`. In PR, Isabela and many rural
municipalities are expected to return MEDIUM or BASE. The pipeline requests
`requiredQuality = <expected_quality_from_probe>` and sets `pixelSizeMeters`
accordingly (0.1 m for HIGH, 0.25 m for MEDIUM/BASE).

## 6. Python SDK usage (`google-maps-solar`)

```python
from google.maps import solar_v1
from google.maps.solar_v1 import DataLayerView, ImageryQuality

client = solar_v1.SolarClient()  # uses ADC or API key via ClientOptions

# Probe
insights = client.find_closest_building_insights(request={
    "location": {"latitude": lat, "longitude": lon},
    "required_quality": ImageryQuality.HIGH,
})

# Data Layers
layers = client.get_data_layers(request={
    "location": {"latitude": lat, "longitude": lon},
    "radius_meters": 175,
    "view": DataLayerView.IMAGERY_AND_ANNUAL_FLUX_LAYERS,
    "required_quality": ImageryQuality.HIGH,
    "pixel_size_meters": 0.1,
    "exact_quality_required": False,
})

# layers.rgb_url, layers.mask_url, layers.dsm_url, layers.annual_flux_url, ...
geotiff = client.get_geo_tiff(request={"id": layers.rgb_url.split("id=")[-1]})
with open("rgb.tif", "wb") as fh:
    fh.write(geotiff.content)
```

Auth: either Application Default Credentials (`gcloud auth
application-default login` + `set-quota-project`), or API key passed via
`ClientOptions(api_key=...)`. This project uses the API key in `.env`
(`SOLAR_API_KEY`) for Data Layers and ADC for billing attribution.

An async client exists at `google.maps.solar_v1.SolarAsyncClient` with the
same method names.

## 7. REST fallback

All endpoints accept a plain `key=<SOLAR_API_KEY>` query parameter:

- `GET https://solar.googleapis.com/v1/buildingInsights:findClosest?location.latitude=...&location.longitude=...&requiredQuality=HIGH&key=...`
- `GET https://solar.googleapis.com/v1/dataLayers:get?location.latitude=...&location.longitude=...&radiusMeters=175&view=IMAGERY_AND_ANNUAL_FLUX_LAYERS&requiredQuality=HIGH&pixelSizeMeters=0.1&key=...`
- `GET <rgbUrl>&key=...` to download the GeoTIFF.

The REST fallback is useful in environments where the gRPC SDK refuses to
attach the API key (as of the `google-maps-solar` 0.x series,
`client_options=ClientOptions(api_key=...)` only applies to unary HTTP+REST
transport).

## 8. Leafmap convenience helper

```python
import leafmap
leafmap.get_solar_data(
    lat, lon,
    radius=175,
    view="IMAGERY_AND_ANNUAL_FLUX_LAYERS",
    quality="HIGH",
    pixel_size=0.1,
    out_dir="data/rasters/solar/prototype",
    api_key=os.environ["SOLAR_API_KEY"],
)
```

Leafmap handles the URL dance and file naming. Useful for prototyping a single
tile; for the 2K-tile pipeline we use our own rate-limited fetcher in
`utils/solar_api.py` to get deterministic paths + a ledger.

## 9. Caching + retention

- GeoTIFF URLs are valid for ~1 hour — always download immediately.
- Downloaded assets may be cached up to 30 days per the Solar API terms.
- We cache at two levels: (a) the small JSON response from `dataLayers:get`
  under `data/rasters/solar/cache/` keyed by
  `(round(lon,6), round(lat,6), radius, view, pixel_size, quality)`; (b) the
  downloaded GeoTIFFs at their canonical project paths
  `data/rasters/solar/{municipio}/{bg_geoid}/{tile_id}_{layer}_{quality}.tif`
  with `.json` sidecars carrying the response metadata.

## 10. Error taxonomy

| Status | Meaning | Action |
|---|---|---|
| `404 NOT_FOUND` | No coverage at this lat/lon | Drop tile, record in `no_coverage` ledger column |
| `400 INVALID_ARGUMENT` | Radius too large for view/pixel combo | Log + halve radius or switch view |
| `403 PERMISSION_DENIED` | API key missing, invalid, or quota disabled | Fail loudly |
| `429 RESOURCE_EXHAUSTED` | Rate-limit hit | Exponential backoff, cap retries at 4 |
| `500 / 503` | Transient | Retry up to 3 times with jitter |

## 11. Key references

- <https://developers.google.com/maps/documentation/solar/overview>
- <https://developers.google.com/maps/documentation/solar/data-layers>
- <https://developers.google.com/maps/documentation/solar/concepts>
- <https://developers.google.com/maps/documentation/solar/reference/rest>
- <https://developers.google.com/maps/documentation/solar/expanded-coverage>
- Python SDK: `pip install google-maps-solar` — types under
  `google.maps.solar_v1.types`.

# 36 Waugh Street, Griffith NSW 2680 — site information

Desktop site lookup for a planned house build. Data fetched 12 October 2026 directly from NSW
government ArcGIS REST services (raw responses in `data/`). Coordinates are GDA2020 / WGS84;
areas and lengths are computed in GDA2020 MGA Zone 55 (EPSG:7855).

![Aerial site map](site_aerial.png)

Imagery and cadastre © State of New South Wales (Spatial Services), licensed CC BY 4.0.

## Title and lot

| Item | Value | Source |
|---|---|---|
| Lot / plan | **Lot 624 DP821594** (`lotidstring` 624//DP821594) | NSW_Cadastre/MapServer/9 (Lot) |
| Address point | 146.077937, −34.264453 (GURAS address 36 WAUGH STREET GRIFFITH, single match) | NSW_Geocoded_Addressing_Theme/MapServer/1 |
| Lot area | **≈1,093 m²** (1,092.8 m² planar MGA55; 1,093.4 m² ellipsoidal) | Computed from the cadastre polygon |
| Perimeter | ≈137.7 m | Computed |
| Boundaries | NW (rear) 36.9 m · NE side 45.0 m · SW side 32.9 m · street frontage a curve of ≈25.5 m | Computed |

The cadastre's own `planlotarea` field is empty and its `shape_Area` (1,604.8) is in Web Mercator
units, which inflate area at this latitude, so neither should be quoted. The computed area comes from the
digital cadastre, which is not survey-accurate; the registered plan DP821594 gives the legal
dimensions and area.

## Planning (Griffith LEP 2014 and SEPPs)

| Control | Result |
|---|---|
| Zoning | **R1 General Residential** across the whole lot (Griffith LEP 2014, Amendment No 7). The rear (NW) boundary adjoins **C2 Environmental Conservation** land (0 m² overlap with the lot). |
| Minimum lot size | **700 m²** (map class 700–749, symbol Q, cl. 4.1) |
| Height of buildings / FSR | No LEP map control applies |
| Heritage | None mapped |
| **Bushfire prone land** | **Yes: Vegetation Buffer** (data updated 30 May 2023). It covers ≈546 m², about half the lot, on the rear/uphill side. Expect a BAL assessment under Planning for Bush Fire Protection. |
| Flood planning | Not mapped |
| Landslide risk, acid sulfate soils, salinity, riparian, biodiversity, wetlands, scenic protection | Not mapped |
| Land reservation acquisition | None |
| DCP | An LGA-based Development Control Plan applies (Griffith) |

Sources: `mapprod3.environment.nsw.gov.au` `Planning/EPI_Primary_Planning_Layers`,
`Planning/Hazard`, `Planning/Protection`, `ePlanning/Planning_Portal_Hazard`,
`ePlanning/Planning_Portal_Local_Provisions` and `ePlanning/Planning_Portal_Development_Control`.
These are NSW Department of Planning services; the Planning Portal host
`mapservices.planningportal.nsw.gov.au` did not respond from this environment (HTTP 502), so it was **not**
used. **Unverified:** results have not been cross-checked against a Planning Portal property report or a
Section 10.7 planning certificate from Griffith City Council. Get the certificate before relying on them.

## Levels (heights)

| Item | Value |
|---|---|
| Ground levels at corners | ≈160.3 (W) · ≈158.8 (N) · ≈154.5 (E, street) · ≈156.1 (S, street) m AHD |
| Range over lot | ≈154.5 to ≈160.2 m AHD, so ≈5.7 m of fall |
| Average slope | ≈14% (≈8°), falling to the east-south-east (toward the street) |
| Wider setting | Contours within 200 m run from 144 to 194 m AHD. The land rises steeply behind the lot into the C2 bushland. |

Levels are sampled from the NSW 5 m DEM (`public/NSW_5M_Elevation/ImageServer`, tile
Griffith-DEM-AHD_55_5m, derived from stereo imagery), resampled to 1 m. The 2 m contours are from
`NSW_Elevation_and_Depth_Theme/MapServer/2`. Both are **indicative only**: the 5 m DEM smooths the
ground and its vertical accuracy is metre-scale. No finer LiDAR product was found through these
services. Get a detail and level survey by a registered surveyor before design.

## Imagery

The aerial photo is the NSW_Imagery tile cache at zoom 20, ≈0.12 m/pixel. Per `NSW_Imagery_Dates` it is
the Griffith ADS40 town capture, 10 cm resolution, dated **17 February 2014**. The lot appeared
vacant then; current conditions may differ.

## Reproducing

- `data/` holds the raw GeoJSON/JSON responses, the DEM export (`dem5m_mga.tif`, `dem_crop.npy`) and
  the imagery tiles.
- `scripts/render_site_map.py` redraws `site_aerial.png` from `data/` (requires `pyproj`, `shapely`,
  `numpy`, `Pillow`).

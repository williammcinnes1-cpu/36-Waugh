# 10 Mancini Drive, Lake Wyangan NSW 2680

Site lookup from NSW public spatial services (fetched with curl/Python), for siting a pool shed.
Data © State of New South Wales (Spatial Services), CC BY 4.0. GDA2020 / MGA zone 55, metres, m AHD.

| Item | Value | Source |
|---|---|---|
| Address point | 146.0376377, −34.2279034 (GURAS "10 MANCINI DRIVE LAKE WYANGAN", single match) | NSW_Geocoded_Addressing_Theme/MapServer/1 |
| Lot / plan | **Lot 5 DP285693** | SIX Maps NSW_Cadastre/MapServer/9 |
| Lot area | ≈8,029 m² (0.80 ha), computed in MGA55 (the layer's `shape_Area` of 11,781 is in web-mercator units) | derived |
| Boundary | north 89.4 m, west 92.6 m (28.2 + 64.4), south 84.5 m, east ≈93.5 m along Mancini Drive | derived |
| Zoning | R5 Large Lot Residential, Griffith LEP 2014 | mapprod3 EPI_Primary_Planning_Layers/2 |
| Minimum lot size | two overlapping features returned: "Area C" (CA) and 5 ha (AA) — check the LEP map | EPI_Primary_Planning_Layers/4 |
| Heritage, FSR, height of building | none mapped | EPI_Primary_Planning_Layers/0, 1, 5 |
| Bushfire prone, flood, landslide | none mapped on the lot | ePlanning/Planning_Portal_Hazard/229, 230, 232 |
| Ground levels | 116.3–117.1 m AHD; nearly flat, ≈0.4 % fall towards about 259° (west-south-west) | NSW 5 m DEM |
| Aerial | NSW Imagery, Griffith ADS40 block, Feb 2014, 10 cm | NSW_Imagery, NSW_Imagery_Dates |

No official contours exist for the lot in `NSW_Elevation_and_Depth_Theme/MapServer/2` (empty result), so the
0.25 m and 1 m contours on the plan are generated from the 5 m DEM. The DEM comes from stereo imagery and is
indicative only (vertical accuracy around a metre); get a detail and level survey before design.

## House and pool from House.IGS

`House.IGS` (SolidWorks, mm) contains the lot surface, the house (to a 4.6 m ridge) and a T-shaped pool (≈1.9 m deep).
It was meshed with gmsh, split by height, and the house footprint was fitted to the roof in the 2014 aerial
(rotation −17.0°, 91 % overlap). The model's own lot outline, placed with the same transform, overlaps the cadastre
by 90 %. The placed house and pool are in `data/placed_model.geojson`; the fitted transform is `data/model_to_mga.json`.
The pool is not in the 2014 imagery (built later or proposed).

## Files
- `site_plan.png`: aerial, boundary with lengths, contours, spot heights, house and pool.
- `scripts/fetch.py`: tiles, DEM, contours, imagery date. `scripts/site_plan.py`: the plan. `scripts/common.py`: helpers.

## 3D viewer
`viewer/index.html` (with `viewer/site.json` and `viewer/aerial.jpg`, built by `scripts/build_viewer.py` from the gmsh
mesh of House.IGS): terrain draped with the aerial, lot boundary (red), a line 10 m inside the boundary (blue),
0.25 m contours, and the house and pool. Published privately at https://claude.ai/artifact/WRS6y62avyobSXoHSsqBj3.

## Pool as built
The IGES pool position did not match the pool as built. The as-built pool and paved surround were traced from a recent
Google Maps screenshot supplied by the client, registered to the NSW aerial with four tie points (house east end
corners, cross-wing corner, driveway crossover; residuals 0.7–2.1 m): water ≈12.0 × 4.6 m, surround ≈15.2 × 10.0 m
(`data/pool_asbuilt.json`, `data/google_to_mga.json`). Positional accuracy is about ±2 m.

## Known issue: house roof
House.IGS has walls, eaves and a flat ceiling but no roof surfaces, so the viewer shows the house without its hipped
roof. House.SLDPRT (SolidWorks native) cannot be read without SolidWorks; a STEP export (.step/.stp) is needed.

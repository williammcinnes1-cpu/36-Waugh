# Design brief: 36 Waugh Street, Griffith NSW (Lot 624 DP821594)

This is the shared brief for every house concept. Each designer gets a separate STYLE section in their prompt. Read this whole file first.

## Client must-haves
1. **Pool and big glass face north.** Put a private courtyard in the rear (uphill) part of the lot. Living rooms open onto it through large black-framed glazing.
2. **Drive-in garage at street level under the house.**
   - Double garage, about 6.4 × 7.5 m, floor at about 155.0 m AHD.
   - The driveway comes straight off the street at the low north-east end (u ≈ 16–23).
   - There is no existing crossover; a new one will be built.
3. **Privacy to the street** (a client named Jodee asked for this): ground-floor street windows stay small or moderate.
4. **Upper floor takes in the view.** The second floor gets generous windows to the street (SSE) and to the east, e.g. a Juliet or small balcony.
   - East in the street frame is the direction (u, n) ≈ (+0.89, −0.45).
   - Upper windows facing the NE neighbour's yard need privacy treatment (high sills, fixed louvres or obscure glass) while keeping the long view.
5. **Use the whole block, out to the lot boundary on every side**, including both rear corners.
   - Use terraces with retaining walls where the slope needs them. No leftover unlevelled ground.
   - Nothing may extend outside the lot.

## Griffith Residential DCP rules (all must pass; assert them in your site script)
- **Front setback (4.7.1a):** dwelling walls, including verandah or porch posts, at least 6.0 m from the street boundary. Measure to the ARC, not the chord.
- **Garage door setback (4.7.1c):** at least 5.5 m from the street boundary.
- **Articulation zone (4.7.2):** a porch or entry element may sit up to 1.5 m forward of the 6 m line, i.e. down to 4.5 m.
  - Total area in that zone: no more than 25% of the zone, about 8 m².
  - Never a garage, carport or main living floor area in the zone.
- **Side setbacks (4.7.3):**
  - Single storey: 0.9 m, with eaves at least 0.45 m.
  - Upper storey: 1.5 m, with eaves at least 0.9 m.
  - Measure perpendicular to the slanted boundary lines.
- **Rear setback (4.7.4):** ground floor 2 m, upper storey 3 m, measured to the diagonal rear boundary.
- **Height (4.5a):** every roof point at most 9.0 m above NATURAL ground directly below it.
  - Natural ground is data/3d/dem_240m.tif. See scripts/dcp_check.py and its `natural(u, n)`.
  - Natural ground is lowest at the front NE corner (about 155.4–155.7), so tall parts go to the middle or rear.
  - Aim for at least 0.2 m of margin, because the 5 m DEM is approximate.
- **Façade (4.4b, 4.4c):** street-facing walls no longer than 5 m without a physical change (a step of 0.3 m or more, a bay, or a material change). Side walls visible from the street no longer than 10 m without a change.
- **Streetscape (4.3):**
  - The street façade needs at least two of: porch or entry feature, window awnings, recessed or projecting elements, open verandah, bay windows.
  - It also needs ONE major window to a habitable room facing the street. Screen it with a verandah or planting for privacy.
  - A garage wider than 50% of the front elevation must be recessed behind the building line.

## Geometry (street frame, metres, heights in m AHD)
- **Axes:** u runs along the frontage chord from the SW end (u=0) to the NE end (u=22.9). n runs into the lot.
- **Street boundary:** an arc from (0,0) to (22.9,0), bulging to n≈0.87. It is a circle with centre (11.45, −74.10) and radius 75.0.
- **Side boundaries:** SW (0,0)→(−5.1,32.5); NE (22.9,0)→(29.8,44.5).
- **Rear boundary:** (−5.1,32.5)→(29.8,44.5).
- **Orientation:** the street side faces about 153° (SSE), so +n points about 333° (NNW).
- **Natural ground:** about 154.5–156.1 at the street, 158.8 at the NE rear corner and 160.3 at the SW rear corner. The rear half is bushfire-prone.
- **Levels:** floor FFL 157.6, courtyard 157.45, garage floor 155.0. An upper garden terrace about 158.6–159.2 works well at the rear.
- **Front:** a low street wall plus a planted bank handles the roughly 2 m rise.

## Site context, from the client's photos in site_photos/ (read NOTES.md and look at the JPGs)
- **The view:** wide and open to the SE and E over Griffith and flat irrigation plains.
- **NE neighbour:** single-storey white render with a red terracotta hip roof, lower than our lot. Shared fence is BEIGE Colorbond, about 1.8 m.
- **SW neighbour:** SAGE-GREEN Colorbond fence, about 1.8 m, with a big red bottlebrush behind it.
- **Model these:** both fences, and simple massing of the two neighbour houses. Use the cadastre lots in data/neighbour_lots.geojson: 625 is NE, 623 is SW. The 2014 aerial shows their footprints. These make the street views read as a real street.

## Repo and tools
- **Copy and adapt; don't edit the shared scripts.**
  - Site fit: scripts/farmhouse_site.py. It has DESIGN, pad_polygon, designed_ground, lawn_triangles, cut_fill_volumes, and the ground/finish/plan commands.
  - Scene: scripts/farmhouse_cycles.py. It has wall/window/gable/roof helpers, plinths, lanterns, fences, garden, export_glb, export_mesh.
  - Reference implementations: scripts/designs/*_site.py and *_cycles.py from earlier designers. Black barn, wraparound and sage_stone are good.
- **Shared, read-only:** scripts/cycles_render.py, house_massing.py, build_3d_render.py, dcp_check.py, shadow_diagrams.py, and the photo viewer files.
- **Workflow:**
  1. Run `python3 scripts/designs/<id>_site.py ground`.
  2. Run `python3 scripts/build_3d_render.py`.
  3. Run your Blender script.
- **Blender:** bpy 4.2 in /tmp/claude-0/-home-user-36-Waugh/5d310e2a-c88e-5d61-a0d7-d8065c69872b/scratchpad/bvenv/bin/python.
  - Prefix every run with `nice -n 10`, because other designers share the 4 cores.
  - Budget: one test at --scale 25 --samples 12, then finals at --scale 50 --samples 32. About 6 renders maximum.
- **Do not commit** data/3d/house_massing.json, data/3d/render_scene.json, site_render.html or site_3d.html. Revert them before committing, to avoid merge conflicts.

## Quality bar (a reviewer will score the street render out of 10; you need at least 6)
- **Street view:** a real-estate-style photo from across the road at eye level, about 1.6–2.5 m. The house fills about 55–70% of the frame width, with road and kerb in the foreground and sky above.
- **Lighting:** golden hour, warm sun from the west, warm interior glow behind the glass. Clear blue-to-warm sky, not grey.
- **Must not happen:**
  - shaded walls rendering nearly black;
  - floating or clipping geometry;
  - bare golden pasture texture on the lot itself (lawn, paving, garden beds and mulch should cover it);
  - roofs running through each other.
- **Courtyard view:** shows the pool, the north glazing and the rear terraces. An eye-level or modest-height view is preferred over a high aerial.
- **Self-check:** before finishing, look at your renders with Read and fix the worst three problems.

## Deliverables (inside your worktree)
1. scripts/designs/<id>_site.py and scripts/designs/<id>_cycles.py.
2. designs/<id>_street.png, designs/<id>_courtyard.png and designs/<id>_plan.png.
3. data/3d/designs/<id>.txt: the GLB from `--export-glb` (it removes interiors), base64-encoded, under 14 MB. Delete the .glb.
4. data/3d/designs/<id>.json:
   ```
   {"id","name","file":"designs/<id>.txt","note":<one sentence>,"facts":[
     ["Lot area","≈1,093 m²"],["Zoning","R1 General Residential"],
     ["House + garage footprint","<n> m² (<pct>)"],["Floor level","157.6 m AHD"],
     ["Earthworks depth","cut ≤x m · fill ≤y m"],
     ["Cut / fill volume","≈C / F m³"],["Net soil","≈N m³ to remove|to bring in"],["Pool excavation","≈P m³ extra"],
     ["Bushfire prone","rear half (vegetation buffer)"]]}
   ```
5. data/3d/designs/<id>_mesh.json, from the --export-mesh equivalent (house triangles, u/n/z), so shadow diagrams can be made.
6. Commit everything in the worktree. Do not push.

## Final reply (under 250 words)
- Worktree path and branch.
- File paths.
- Footprint, volumes and every ridge RL with its height above natural ground.
- Garage and closest-dwelling setbacks, and the minimum side and rear setbacks.
- Which rooms get the street and east views.
- Known flaws.

## QA lessons from the first round (apply these)
- **Sky:** copy `golden_sky()` from scripts/designs/sage_stone_cycles.py (about line 1092) and use a 10° sun elevation. The flat grey skies scored badly.
- **No bare paddock:** don't let any render show the lot floating in bare brown pasture.
  - Model the two neighbour houses: 625 NE is single-storey white render with a terracotta hip roof; 623 SW has a tile roof.
  - Add trees and hedges along the boundaries.
  - Keep courtyard cameras close: an eye at about 6 m height inside the lot, looking at the pool and glazing with a 24 mm lens.
- **Lawn and water levels:** lawn triangles must not cover the pool. Cut lawn around the pool and paved areas, or keep the water above the lawn level.
- **No flat beige ground** in the street foreground. Use lawn, gravel or mulch beds with planting.
- **Final renders:** use about 96 samples. 32 left the trees and stone blotchy.
- **Roofs:** charcoal about #3a3d40, roughness about 0.45, not pure black. Exposure about 1.0–1.05.
- **Garage in frame:** keep the garage in the street view; aim the camera so the NE end isn't cut off.

## Roof rule (the client flagged this; it must never happen)
- **One clean roof per part of the house.** Never stack two roof planes over the same area, and never let a lower roof slide under or through another roof.
  - Example of the fault: the old farmhouse had a lean-to roof and a verandah roof sitting one above the other at the front, showing as two parallel roof lines.
- **Where roofs meet,** use a proper valley or a junction against a wall with flashing. Don't let them intersect at random.
- **Check before finishing:** look at the aerial view and at a close-up from a front corner. Verify every roof edge is a single, deliberate line, with no doubled fascias, overlapping eaves or roofs poking through walls.

## Arch rule (client feedback: "these arches are just too much")
- **At most one or two arched openings on the whole house,** for example the front door or one feature window.
- **Everything else rectangular,** in black steel-framed glazing.
- **Never** several arched gables or rows of arched windows. This applies even to the Mediterranean and French styles.

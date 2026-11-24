# osm_alsancak_v0: PLACEHOLDER SUMO network fixture

> **Placeholder.** This is **not** the project's official digital twin. The real Alsancak–Konak network is owned by Öykü Doğa Dabak in [`simulation/`](../../../simulation/). This fixture exists only so the `ai/` prototype and tests can run until that network is available. It is offered to Öykü as a possible starting point.

**Map data © OpenStreetMap contributors, available under the Open Database License (ODbL 1.0).** <https://www.openstreetmap.org/copyright>. This network is a derivative database of OpenStreetMap data.

## Contents

| File | Description |
|---|---|
| `osm.net.xml.gz` | SUMO network. The osmWebWizard export from 2026-10-21 was later edited manually in **netedit** (saved 2026-10-21 15:38); this is a gzip of the edited version. |
| `osm.passenger.trips.xml` | 2305 random passenger trips over 3600 s (`randomTrips.py --seed 42`) |
| `osm.sumocfg` | SUMO config from the wizard (writes `tripinfos.xml`, `stats.xml`, `edgeData.xml`, all gitignored) |
| `osm.netccfg` | netconvert configuration used by the wizard |
| `osm.ogcfg`, `osm.view.xml`, `output.add.xml` | wizard / GUI / edgeData helper files |
| `build.bat` | Despite the extension, a **bash** script that regenerates the trips with `randomTrips.py` |
| `run.bat` | `sumo-gui -c osm.sumocfg` |

Run it with: `cd ai/sim_fixtures/osm_alsancak_v0 && sumo -c osm.sumocfg` (or `sumo-gui`).

## Known limitations

- The bounding box (38.393–38.455 N, 27.067–27.174 E) is **larger than the pilot area** (38.4150–38.4410 N, 27.1250–27.1550 E).
- Only three signalised intersections are configured for the prototype: Lozan, Montrö and 9 Eylül Meydanı (used as Basmane, **ASSUMPTION**). See [`ai/prototype/configs/intersections_config.json`](../../prototype/configs/intersections_config.json).
- The netedit changes were not recorded. **ASSUMPTION:** they adjusted traffic-light programs (Lozan is `delay_based`). The diff against the wizard output is about 1900 lines.
- Demand is random, not calibrated to IZUM counts.

## How to regenerate

The raw OSM extract is **not committed**. Keep it locally at `data/raw/osm/osm_bbox.osm.xml.gz` (gitignored).

1. Get the OSM extract. Either run SUMO's wizard (`python "$SUMO_HOME/tools/osmWebWizard.py"`) and select the area, or download the bounding box from the OpenStreetMap API / an Overpass mirror into `data/raw/osm/osm_bbox.osm.xml.gz`.
2. Build the network. Copy `osm.netccfg` next to the extract, then run `netconvert -c osm.netccfg`, which reads `osm_bbox.osm.xml.gz` and writes `osm.net.xml.gz`.
3. Repeat the manual netedit edits. They are not scripted, so this step is **not reproducible yet**.
4. Regenerate the trips: `bash build.bat` (uses `--seed 42`).

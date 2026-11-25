# toy_corridor: PLACEHOLDER 6-junction arterial

> **Placeholder.** This is a synthetic network for developing and testing `ai/`. It is **not** the Alsancak–Konak digital twin (owned by Öykü Doğa Dabak in `simulation/`). Results on it are relative improvements on a placeholder fixture only.

| Property | Value |
|---|---|
| Layout | Straight east-west arterial W → J1 … J6 → E, with a north and a south side street at each junction |
| Length | 500 m junction spacing + 250 m end stubs = **3.0 km** W→E (the emergency-KPI corridor length) |
| Lanes | Arterial 2 per direction; side streets 1 per direction; 50 km/h |
| Signals | J1…J6, netconvert static default programs: 2 green phases (arterial, side), permissive left turns. Only used as the set of allowed green states; timing comes from the phase machine |
| Demand | Generated per seed by `corridor_ai.sim.demand` (`toy_od`, Poisson): 500 veh/h through traffic per arterial direction, 15 veh/h from each arterial end to each side exit, 40 veh/h from each side entry to each arterial end and straight across (scale 1.0, **ASSUMPTION**) |
| Agent mapping | `alsancak_station`=J1, `lozan`=J2, `montro`=J3, `basmane`=J4, `cankaya`=J5, `konak_pier`=J6 (**ASSUMPTION**, corridor order only) |

Files: `toy.nod.xml` and `toy.edg.xml` (plain inputs) and `toy.net.xml.gz` (netconvert output). Regenerate them with:

```bash
python ai/scripts/make_toy_corridor.py
```

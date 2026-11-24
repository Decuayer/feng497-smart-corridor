# data/: data sources

> **No raw data is committed.** `data/raw/` and `data/processed/` are gitignored except for `.gitkeep`. Raw files may be committed only after Demir confirms the licence allows redistribution (**open item**). Each member downloads the files locally as described below.

## Layout

```
data/
├── raw/          # original downloads, unchanged (gitignored)
│   └── osm/      # OpenStreetMap extract for the SUMO fixture (gitignored)
└── processed/    # cleaned / derived data (gitignored)
```

Recommended local file names are given below. Code should read paths from configuration, not hard-code them.

## Sources

All municipal datasets come from **İzmir Metropolitan Municipality (İzmir Büyükşehir Belediyesi) open data**, published on the municipality's open-data portal. **ASSUMPTION:** the portal is <https://acikveri.bizizmir.com>. Verify each dataset's page and licence terms before use.

### 1. IZUM main-artery vehicle counts

- **Local file:** `data/raw/izbb-izum-anaarter-arac-sayim-guncel.xlsx`
- **Columns:** `TARIH, ANA_ARTER_ADI, ISTIKAMET, BASLAMA, BITIS, ARAC_SAYISI, EK_SERIT, ACIKLAMA`
- **Coverage:** 2020-08 to 2026-10. Arteries: Anadolu Cd., Cumhuriyet Blv., Yeşildere Cd. Mostly peak-hour windows of 15–60 min.
- **Used by:** O/D calibration (`simulation/`), daily/seasonal profile checks for forecasting (`ai/`), analysis (`analysis/`).

### 2. Segment speed / duration data

- **Local file:** `data/raw/izbb-hiz-sure-verileri.xlsx`
- **Columns:** `TARIH, SAAT, BASLAMA, BITIS, HIZ, SURE`
- **Coverage:** snapshots at 08:15, 09:15 and 19:05 for selected segments (e.g. Marina Kavşağı ↔ Konak Pier).
- **Used by:** travel-time calibration and KPI sanity checks.

### 3. ESHOT bus lines

- **Local file:** `data/raw/eshot-otobus-hatlari.csv` (`;`-separated, UTF-8)
- **Columns:** `HAT_NO, HAT_ADI, GUZERGAH_ACIKLAMA, ACIKLAMA, HAT_BASLANGIC, HAT_BITIS`
- **Used by:** bus demand in scenarios (`simulation/`).

### 4. ESHOT bus schedules

- **Local file:** `data/raw/eshot-otobus-hareketsaatleri.csv` (`;`-separated, about 5.5 MB)
- **Columns:** `HAT_NO, TARIFE_ID, GIDIS_SAATI, DONUS_SAATI, SIRA, GIDIS_ENGELLI_DESTEGI, DONUS_ENGELLI_DESTEGI, BISIKLETLI_GIDIS, BISIKLETLI_DONUS, GIDIS_ELEKTRIKLI_OTOBUS, DONUS_ELEKTRIKLI_OTOBUS`
- **Used by:** bus departures in scenarios (`simulation/`).

### 5. OpenStreetMap extract

- **Local file:** `data/raw/osm/osm_bbox.osm.xml.gz`
- **Licence:** © OpenStreetMap contributors, ODbL 1.0.
- **How to obtain:** see [`ai/sim_fixtures/osm_alsancak_v0/README.md`](../ai/sim_fixtures/osm_alsancak_v0/README.md#how-to-regenerate), or SUMO's `osmWebWizard.py`.

### 6. IZUM camera streams

Live streams are used only by the edge prototype and configured through `IZUM_STREAM_URL` in `.env`. Check IZUM's terms of use. Recorded video or images must not be committed.

## Known limitations

- The IZUM counts are **coarse**: peak-hour windows only, 3 arteries, and they do not cover all six pilot intersections. Short-term (15-minute) forecasting will therefore be validated mainly on **SUMO detector output** (E1/E2 aggregated per 1–5 min). IZUM data will be used for calibration and daily/seasonal profile checks.
- The speed/duration data has only three daily snapshots.

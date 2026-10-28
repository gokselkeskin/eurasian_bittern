# Eurasian Bittern – GPS-GSM track analysis

Analysis code for the spring migration of a Eurasian Bittern (*Botaurus stellaris*) tagged at the Aras River Ornithological Station, Türkiye, in 2022.

## Run

```
pip install -r requirements.txt
python bittern_216839_analysis.py --input data/216839_20221011_174700.csv
```

Options: `--outdir` (default `outputs`), `--no-show` (save figures without opening them).

Country borders (Natural Earth 1:10 m) and the Figure 2 basemap (Esri World Imagery) are downloaded on first run.

## Outputs

| File | Content |
|---|---|
| 01_fix_statistics.csv | logged records, failed fixes, coverage of the hourly schedule |
| 02_interval_vs_battery.csv | fix interval length vs battery charge (Table S4) |
| 03_data_gaps.csv | intervals > 24.5 h (Table S3) |
| 04_daily_summary.csv | observed daily path and day classification |
| 05_phases.csv | stopovers, halts and movements |
| 06_table1.csv | Table 1 |
| 07_flights.csv | flight windows, speeds, night fraction, sunset/sunrise (UTC) |
| 08/09_sensitivity_*.csv | threshold sensitivity (Tables S1, S2) |
| 10_steps_labelled.csv | all fixes with step metrics and labels |
| 11_manuscript_numbers.csv | every number reported in the manuscript |
| Figure2–4.png | figures (300 dpi) |

## Definitions

- Failed fixes: latitude = longitude = 0.
- Data gap: interval > 24.5 h.
- Movement: step speed > 3 km/h (intervals ≤ 6.5 h); intervals of 6.5–24.5 h count as movement only on migratory days.
- Migratory day: observed daily path (UTC) > 30 km.
- Movement events < 30 km are local movements within stopovers.
- Stopover: stationary ≥ 24 h; halt: < 24 h.
- Night: solar elevation < −0.833° (Michalsky 1988).

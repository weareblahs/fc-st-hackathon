# EDA and Feature Engineering Plan

This plan outlines how we will engineer the target feature "route efficiency" from lat/lon, speed, and direction, then conduct a thorough EDA on three CSVs individually and on a merged dataset. We will infer column types (numeric, categorical, datetime, geospatial), generate appropriate charts automatically, compute correlations, and map relevant features against our target. We’ll favor lightweight, reproducible methods (seaborn/matplotlib for bulk charts, Plotly/Folium for maps), optimize performance via downsampling and dtype tuning, and export key visuals. Assumptions about schemas and keys are noted, with validation steps up front.

---

## Decisions (from your selections)

- Analyze both per-file and merged datasets (1/c)
- Merge strategy: rows ↔ trips on trip_id, then join combined_data on shared keys (trip_id and timestamp if available) (2/a)
- Route efficiency: instantaneous forward-progress projection (3/a)
- Heading/direction is degrees [0, 360) (5/a)
- Speed is km/h (convert to m/s) (6/a)
- Timestamps exist and orderable within trips (7/a)
- Coordinates are lat/lon in WGS84 (8/a)
- Column typing via auto inference with simple patterns and overrides (9/a)
- Plotting hybrid: seaborn/matplotlib for bulk charts; Plotly for key maps (to enable PDF export) with Folium as optional fallback (10/c, 11/a)
- Target vs features: descriptive EDA (scatter/box + Pearson/Spearman/Kruskal) (12/a)
- Performance: downsample for plotting, full data for stats, cache Parquet with optimized dtypes (13/b)
- Outputs: export key plots to `eda_outputs/` as PNG/HTML (14/b)
- Exclusions: skip ID-like and high-cardinality categoricals from bar plots; show top-20 categories (15/a)

---

## Data and Merge Strategy

Assumptions to validate:

- Primary keys: `trip_id` exists in both `safetruck_data_iter1_rows.csv` (row-level telemetry) and `safetruck_data_iter1_trips.csv` (trip-level aggregates/metadata).
- `combined_data.csv` contains fields that overlap with the above or additional enriched signals; ideally shares `trip_id` and possibly `timestamp` for row alignment.
- Time column exists (e.g., `timestamp`) and is sortable within each `trip_id`.

Merge plan:

1. Load all three CSVs with memory-conscious dtypes (categorical for IDs, float32 for continuous where safe).
2. Standardize column names (lowercase, underscores) and parse datetime fields.
3. Join rows ↔ trips on `trip_id` (left join from rows to attach trip-level metadata).
4. Join with `combined_data.csv`:
   - If `combined_data.csv` is row-granular with `trip_id`+`timestamp`, join on both.
   - If it's trip-level, join on `trip_id` only.
   - If it already represents the merged canonical dataset, still perform validation and use it as the merged basis.

Outputs:

- Keep three frames: `df_rows`, `df_trips`, `df_combined`.
- Create `df_merged` for target and correlation analysis.

Validation checks:

- Key coverage (percentage of rows matched), duplicate keys, row count changes.
- Sanity checks on coordinates (lat ∈ [-90,90], lon ∈ [-180,180]) and speeds.

---

## Feature Engineering: Route Efficiency (Target)

Goal: quantify forward progress efficiency using only lat/lon, speed, and direction.

Definitions:

- Convert speed from km/h to m/s: `speed_mps = speed_kmh / 3.6`.
- Compute per-trip sequential bearing `bearing_track(t)` from point t-1 → t using lat/lon (Haversine + initial bearing formula).
- Use given direction column (degrees [0,360)) as `heading_sensor(t)`.
- Angular difference (wrap-aware): `delta = smallest_angle(|bearing_track - heading_sensor|)` mapped to [0, π].
- Instantaneous forward progress: `fp(t) = max(0, speed_mps(t) * cos(delta(t)))`.
- Stationary threshold: if `speed_mps < 0.5` then `fp = 0`.
- GPS jump/outlier guard: compute implied speed from Haversine between consecutive points; if > 60 m/s (~216 km/h), mark as unreliable and set `bearing_track` to NaN for that step.

Normalization:

- Compute a global scaling factor as the 95th percentile of `speed_mps` across `df_merged` after basic cleaning: `scale = P95(speed_mps)`.
- Normalized route efficiency: `route_eff = clip(fp / scale, 0, 1)`.

Edge cases:

- First point per trip (no previous position): `bearing_track = NaN` → `route_eff = NaN` (or forward-fill after some steps if needed).
- Missing lat/lon/speed/direction: `route_eff = NaN`.
- Stopped or idling (low speed): `route_eff = 0`.

Deliverables:

- Add `route_eff` to `df_rows` (and thus to `df_merged`).
- Summary stats and distribution plot of `route_eff`.

---

## Column Typing Strategy

Automatic inference rules with overrides capability:

- Datetime: columns parsable as datetime (e.g., `timestamp`) → `datetime`.
- Geospatial: columns named like `lat`, `latitude`, `lon`, `longitude` → `geospatial`.
- Numeric: float/int columns excluding ID-like detected by high uniqueness ratio (e.g., > 0.9 unique) or name patterns (`id`, `uuid`, `vin`).
- Categorical: object/category dtype or numeric with low cardinality (e.g., unique <= 50) that aren’t ID-like.
- High-cardinality categorical threshold: > 100 uniques or uniqueness ratio > 0.5 → skip full bar plot; plot top-20 only.

We will allow a manual overrides list if any column is misclassified after the first run.

---

## EDA Workflow (Per-file and Merged)

We will repeat univariate EDA for each of: `df_rows`, `df_trips`, `df_combined`, then perform correlation/target mapping on `df_merged`.

Univariate EDA (per dataset):

- Numeric columns: histogram + KDE; summary stats (count, mean, std, min, max, percentiles); outlier flags via IQR.
- Categorical columns: bar chart of top-20 categories; frequency table; category coverage.
- Datetime: distribution by hour-of-day/day-of-week; optional time series sampling.
- Geospatial: basic map (sampled points) to validate GPS quality and spatial coverage.

Target mapping (on `df_merged`):

- Numeric vs `route_eff`: scatter with reg/LOWESS trendline; Pearson and Spearman correlation coefficients annotated.
- Categorical vs `route_eff`: box/violin plots; Kruskal–Wallis p-values (non-parametric) annotated; group means table.
- Datetime vs `route_eff`: time-of-day/day-of-week averages; line/bar plots.

Correlation analysis (numeric):

- Pearson and Spearman matrices on numeric features including `route_eff` (mask upper triangle for readability).
- Heatmap with value annotations (optional) and clustering disabled by default for interpretability.

Exclusions:

- Skip ID-like columns from correlations and univariate plots.
- Treat `lat/lon` primarily as geospatial (excluded from correlation heatmap by default); show their distributions and maps instead.

---

## Performance and Data Handling

- Read CSVs with dtype hints (categorical for IDs, nullable integer types where suitable, float32 for continuous variables).
- Downsample for plotting (e.g., sample up to 100k rows per dataset) while using the full dataset for summary stats and correlations.
- Cache cleaned and typed data to Parquet (`data/cache/rows.parquet`, `trips.parquet`, `combined.parquet`, `merged.parquet`) for faster re-runs.
- Guardrails: chunked reading optional if needed; avoid OOM by incremental aggregation for correlations if datasets are very large.

---

## Outputs and Artifacts

- Display all plots in the notebook.
- Export key plots to `eda_outputs/` as PDF with naming convention:
  - `univariate/{dataset}_{column}.pdf`
  - `maps/{dataset}_map.pdf` (via Plotly + kaleido), with HTML fallback if PDF is not feasible
  - `correlations/{dataset}_corr_heatmap.pdf`
  - `target/{feature}_vs_route_eff.pdf`
- Save a compact summary table (CSV) of correlations and test statistics to `eda_outputs/summaries/`.

Notes:

- Matplotlib/Seaborn support native PDF export via `savefig`.
- Plotly requires `kaleido` installed to export to PDF (`fig.write_image(..., format='pdf')`). If `kaleido` isn’t available, we’ll temporarily save maps as HTML and optionally convert later.

---

## Notebook Structure (planned cells)

1. Markdown: Title and overview
2. Code: Imports, plotting style, paths, constants (e.g., sample sizes, thresholds)
3. Code: Load CSVs with dtype hints; preview shapes, dtypes, head
4. Code: Cleaning (renames, datetime parsing, basic sanity checks)
5. Code: Merge frames to create `df_merged` (join strategy per above; validation checks)
6. Code: Route efficiency feature engineering (functions + application), add `route_eff`
7. Code: Column typing helper and overrides
8. Code: Univariate EDA generator (numeric/categorical/datetime/geo) for each dataset; export selected plots
9. Code: Correlation matrices (Pearson & Spearman) on `df_merged` numeric features; heatmaps
10. Code: Target vs feature plots (numeric scatter + trend, categorical box/violin + Kruskal); stats tables
11. Code: Geospatial maps (sampled) via Folium or Plotly Express
12. Markdown: Key findings and next steps

---

## Implementation Details

Bearing & distance:

- Use Haversine for distance and the initial bearing formula.
- Bearing wrap: normalize to [0, 360); compute minimal angular difference.

Stat tests:

- Numeric vs target: report Pearson and Spearman r with p-values.
- Categorical vs target: Kruskal–Wallis across groups with p-values.

Maps:

- Prefer Plotly Express `scatter_geo` (token-free) to enable direct PDF export via kaleido; supports sampled points or simple trajectories.
- Folium remains an optional fallback (HTML export) if PDF export isn’t available or necessary for specific maps.

Logging and resilience:

- Try/except around chart generation per column to avoid single-column failures halting the run.
- Skip plotting columns with all-NaN or single-unique values.

---

## Open Items (to verify when implementing)

- Confirm exact column names for: `trip_id`, `timestamp`, `lat`/`lon`, `speed` (km/h), and `direction` (degrees).
- Sampling rate per trip (helps in interpreting efficiency distributions).
- Whether `combined_data.csv` is row- or trip-level.

---

## Acceptance Criteria

- Route efficiency computed and added to `df_merged` with normalization and guards.
- Clear, reproducible EDA covering per-file univariate analysis and merged correlation/target mapping.
- Charts auto-generated by column type with sensible exclusions; key plots exported to `eda_outputs/`.
- Correlation heatmaps (Pearson & Spearman) across numeric features including `route_eff`.
- Notebook sections structured as listed and ready for execution.

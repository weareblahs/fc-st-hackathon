We will build a robust, idempotent feature-engineering pipeline over `data/Safetruck Hackathon Data/combined_data.csv` to compute trip-level and per-truck metrics: Trip_Distance_km, Trip_Duration_min, Avg_Speed, Idle_Time, Moving_Time, Idle_Percentage, submetrics (Distance/Time/Idle efficiencies and Signal Reliability), and the “Age of Truck (months)” using daily odometer deltas. The logic will prioritize odometer-based distance, segment trips by Engine ON→OFF, operate in UTC, cap long sampling gaps to avoid inflated idle time, and finally persist outputs as CSVs under `data/Safetruck Hackathon Data/` with both trip-level and slim per-row tables. Existing or similar columns will be detected and skipped to keep the pipeline idempotent.

## Decisions locked (from your selections)

- Output format: CSV in `data/Safetruck Hackathon Data/` (1/a)
- Schema mapping: you’ll provide explicit column name mapping (2/a)
- Timezone: UTC (3/a)
- Trip definition: Engine ON → Engine OFF (4/a)
- Idle threshold: 1 km/h (5/b)
- Gap handling: cap any single Δt to 5 minutes (6/a)
- Distance method: Odometer-only (7/a)
- Time efficiency baseline speed: 80 km/h (8/b)
- Fuel efficiency: skip for now (9/d)
- Refuel handling: detect via fuel increase threshold; exclude those intervals if later added (10/a)
- Signal Reliability: non-null lat/long counts as “Located” (11/a)
- Age-of-Truck window: last 60 days for monthly S estimate (12/b)
- Multiple vehicles: yes, compute per-vehicle (13/a)
- Outputs: separate trip-level table + slim per-row file (14/b)
- Implementation location: `fte.ipynb` (15/a)
- Scale: ~1–10M rows; moderate sampling (16/b)

Additional clarifications just confirmed:

- Output filenames (2/a):
  - Trip-level: `data/Safetruck Hackathon Data/safetruck_data_iter1_trips.csv`
  - Row-level: `data/Safetruck Hackathon Data/safetruck_data_iter1_rows.csv`
- Engine status source (3/b): column name is `Engine Status` with string values `OFF` / `ON` mapped to boolean `engine_on`.

## Data contract and normalization

To make the logic unambiguous, we’ll normalize key inputs using your mapping (2/a):

- timestamp: UTC datetime (parsed from your timestamp column)
- vehicle_id: truck identifier
- lat, lon: float degrees
- speed_kmh: numeric speed in km/h
- engine_on: boolean derived from the `Engine Status` column (3/b): map `ON` → True, `OFF` → False (case-insensitive)
- odometer_km: numeric, expected monotonically non-decreasing (per vehicle)
- fuel_level: optional; skipped for now (9/d)
- gps_valid: optional; not needed since we’ll use non-null lat/lon (11/a)

Assumptions if needed:

- Engine status values will be mapped to boolean (ON=1/OFF=0 or ON/OFF strings).
- Speed is in km/h; if in m/s, convert.
- Odometer is in km; if in meters or miles, convert before use.

## Trip segmentation (Engine ON → OFF)

1. Sort by (vehicle_id, timestamp) and drop exact duplicates per vehicle.
2. Map engine flag to boolean engine_on.
3. Identify transitions: a trip starts when engine_on toggles OFF→ON, and ends when ON→OFF.
   - If data starts with engine_on==ON, begin a trip at the first ON row.
   - If a trip never sees OFF (open-ended), close it at the last available row.
4. Assign a sequential trip_id per vehicle.

## Time deltas, capping, and state windows

For each vehicle, compute per-row Δt_sec to the next row. Cap each Δt to 300 seconds (5 minutes) to avoid inflated durations from sparse logging or connectivity gaps. Within each trip:

- idle_flag = engine_on AND speed_kmh ≤ 1.0
- moving_flag = engine_on AND speed_kmh > 1.0
- Idle_Time_min = sum(Δt_sec where idle_flag) / 60
- Moving_Time_min = sum(Δt_sec where moving_flag) / 60

## Distance, duration, and basic KPIs

- Trip_Distance_km (odometer-only): odometer_km(last) − odometer_km(first)
  - Guardrails: if negative or if a large negative jump is detected (reset), set to NaN for the trip.
  - If odometer missing for a trip, set Trip_Distance_km to NaN (we are using odometer-only per 7/a).
- Trip_Duration_min: (timestamp_last − timestamp_first) in minutes
- Avg_Speed: Trip_Distance_km / (Trip_Duration_min / 60.0) with safe handling for zero duration
- Idle_Percentage: clamp(Idle_Time_min / Trip_Duration_min, 0, 1)

## Submetrics

- Distance Efficiency: Optimal / Actual, where Optimal is the great-circle (haversine) distance between (lat_start, lon_start) and (lat_end, lon_end). Actual is Trip_Distance_km.
- Time Efficiency: Expected / Actual time, where Expected = Optimal / 80 km/h, Actual = Trip_Duration_min / 60 (hours). Clamp to a sensible range to avoid extreme ratios.
- Idle Efficiency: 1 − (Idle_Time_min / Trip_Duration_min)
- Signal Reliability: Located / Total within the trip, where Located = rows with non-null lat & lon.
- Fuel Efficiency: skipped for now (9/d). If later enabled, we’ll detect refuels (10/a) and exclude those intervals.

## Target: Age of Truck (months)

For each vehicle:

1. For each calendar day in the last 60 days with data, take the day’s first and last odometer readings (closest to 00:00 and 23:59 or earliest/latest available that day).
2. Compute daily distance S_d = odo_last − odo_first; ignore negative/anomalous days.
3. Let S = sum of valid S_d over the 60-day window.
4. Let O_final = most recent odometer value available for the vehicle.
5. Age_months = floor(O_final / S) if S > 0 else NaN (equivalent to repeated subtraction of S).
   - Notes: This treats S as an approximate monthly distance proxy from the last 60 days. If desired, we can scale S to a 30-day normalized month before division.

## Idempotency: "ignore if exists or too similar"

Before creating any output column, check for an existing column with a near-identical name or semantic (e.g., case-insensitive name match or known aliases). If found, skip creating the new one and proceed to the next feature. Log/print a short note indicating the skip for traceability.

## Outputs and file layout (CSV)

We’ll produce two CSV outputs under `data/Safetruck Hackathon Data/` (per 2/a):

1. Trip-level table (final name): `safetruck_data_iter1_trips.csv`

   - Keys: vehicle_id, trip_id
   - Columns: Trip_Distance_km, Trip_Duration_min, Avg_Speed, Idle_Time_min, Moving_Time_min, Idle_Percentage,
     Distance_Efficiency, Time_Efficiency, Idle_Efficiency, Signal_Reliability, timestamps (start/end), start/end lat/lon, and any QA flags.

2. Slim per-row table (final name): `safetruck_data_iter1_rows.csv`
   - Keys: vehicle_id, timestamp, trip_id
   - Columns: projected minimal set to support recompute/QA (e.g., speed_kmh, engine_on, lat, lon), plus optional derived per-row states. Heavy intermediate columns (like Δt_sec) are not persisted.

## QA and validation

- Distributions: histograms/quantiles for Trip_Distance_km, Avg_Speed, Idle_Percentage
- Outliers: top/bottom 1% trips by Avg_Speed and Idle_Percentage; trips with NaN distances
- Consistency checks: proportion of trips with (Idle_Time_min + Moving_Time_min) close to Trip_Duration_min
- Spot-checks: compare odometer-based distance to haversine for a sample; verify ON→OFF segmentation visually for a few trips

## Performance and scalability

- Chunked reading if CSV > 5–10M rows; or convert to Parquet for faster iterations
- Group-by operations per vehicle; prefer vectorized pandas/numpy
- Avoid per-row Python loops; use cythonized haversine or numpy-based implementation for Optimal distance

## Implementation notes (in `fte.ipynb`)

- Organize reusable helpers: parsing, segmentation, Δt computation, odometer guards, haversine, submetrics, age-of-truck
- Add a small validation cell with summary prints/plots
- Ensure all functions are deterministic and idempotent (skip if column exists)

## Next actions

1. Receive exact column name mapping for: timestamp, vehicle_id, lat, lon, speed_kmh, odometer_km, (optional) fuel_level. Note: engine_on will be derived from `Engine Status` with `OFF`/`ON`.
2. Implement the pipeline in `fte.ipynb` following the above steps.
3. Generate and QA the two CSV outputs in `data/Safetruck Hackathon Data/`.

Pending from 1/a: Please provide the exact column names for the above fields (or paste the CSV header row) so we can lock the mapping without ambiguity.

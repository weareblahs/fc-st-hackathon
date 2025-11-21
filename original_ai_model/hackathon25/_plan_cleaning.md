Safetruck Malaysia — Data Cleaning and Feature Engineering Plan

1. Canonical schema and assumptions
   Canonical columns
   timestamp_utc: tz-aware UTC datetime
   vehicle_id: normalized from plate/asset
   engine_on: boolean
   speed_kmh: numeric km/h
   odometer_km: numeric km
   lat, lon: float degrees
   gps_ok: boolean (from GPSlocated)
   Units: speed km/h, odometer km
   Daily aggregation timezone: MYT (UTC+8) [confirm in Q3]
   Critical fields per row: timestamp_utc, vehicle_id, engine_on, speed_kmh, and at least one of [odometer_km, (lat & lon)]
2. Ingestion and normalization
   Load data/Safetruck Hackathon Data/combined_data.csv with dtype hints and low_memory=False.
   Rename/mapping to the canonical column names listed above.
   Parse timestamps to pandas tz-aware UTC.
   Sort by [vehicle_id, timestamp_utc]; drop exact duplicates by [vehicle_id, timestamp_utc] (keep first).
   Coerce numeric columns; invalid parses → NaN.
3. GPS smoothing and step filtering
   Rolling median or mean smoothing (window=3, center=True) on speed_kmh; optionally on lat/lon if needed.
   Compute gps_step_km using haversine between consecutive points per vehicle.
   Teleport/spike filter: if implied speed from gps_step_km over the time delta > 140 km/h, set gps_step_km = NaN (do not use for distance).
   If gps_ok is False, optionally reduce trust: either set gps_step_km = NaN or use a higher threshold (recommend NaN).
4. Odometer deltas and resets
   Compute odo_step_km = diff(odometer_km) per vehicle.
   Mark odometer resets or anomalies:
   Negative deltas or unrealistically large deltas (e.g., > 50 km in < 60s) → invalid_odo_step.
   Use only non-negative, non-anomalous odo_step_km for calibration.
5. Fused distance (GPS-corrected odometer)
   Per vehicle, compute a robust scale factor s = median(gps_step_km / odo_step_km) over valid segments where:
   gps_ok == True, gps_step_km is finite, 0.01 ≤ odo_step_km ≤ 5.0, and implied speed ≤ 120 km/h.
   Clip s to a reasonable band to prevent overcorrection (e.g., 0.8 ≤ s ≤ 1.2).
   Fused per-interval distance:
   If valid odometer: fused_step_km = odo_step_km \* s.
   Else if valid GPS: fused_step_km = gps_step_km.
   Else: fused_step_km = 0 and mark interval as distance_missing.
   Sum fused_step_km for trip distances.
6. Outlier labelling and exclusion policy
   Labels:
   flagged_over120 = (gps_ok == False) and (speed_kmh > 120)
   speeding_over120 = (gps_ok == True) and (speed_kmh > 120)
   Exclusion for metrics [choose policy in Q2]:
   Recommended: exclude flagged_over120 from all metrics; include or exclude speeding_over120 per chosen policy (b recommended).
   Always keep labels for KPI counts.
7. Trip segmentation (hybrid)
   Primary: ignition-based—trip starts when engine_on changes False→True; ends when True→False.
   Fallback segmentation:
   New trip if time gap between consecutive points ≥ 15 minutes.
   New trip if continuous engine_on == True but no movement (speed_kmh < 1 km/h and fused_step_km ≈ 0) for ≥ 15 minutes.
   Assign monotonically increasing trip_id per vehicle.
8. Feature calculations (per trip)
   Trip_Start_UTC, Trip_End_UTC: first and last timestamps in trip
   Trip_Duration_min: (Trip_End_UTC - Trip_Start_UTC) in minutes
   Trip_Distance_km: sum fused_step_km within trip (respect exclusion policy)
   Idle_Time_min: sum of time deltas where engine_on == True and speed_kmh < 1 (respect exclusion policy)
   Moving_Time_min: sum of time deltas where engine_on == True and speed_kmh ≥ 1 (respect exclusion policy)
   Avg_Speed_kmh: Trip_Distance_km / (Trip_Duration_min / 60)
   Idle_Percentage: Idle_Time_min / Trip_Duration_min
   Validity filter: keep trips with Trip_Duration_min ≥ 5 and Trip_Distance_km ≥ 1
9. Daily aggregates (vehicle-day in MYT)
   Convert timestamps to MYT (UTC+8) and derive local date.
   Aggregate per vehicle_id + date_myt:
   Total_Distance_km, Total_Duration_min, Total_Idle_min, Total_Moving_min
   Daily_Avg_Speed_kmh = Total_Distance_km / (Total_Duration_min / 60)
   Daily_Idle_Percentage = Total_Idle_min / Total_Duration_min
   Trip_Count
   Speeding_Count = count of speeding_over120
   Flagged_Count = count of flagged_over120
10. Known quirks handling
    Odometer resets: drop or repair negative odo_step_km; do not use for fused distance.
    GPS teleports: drop gps_step_km segments with implied speed > 140 km/h or huge jumps.
    Data gaps: segments with dt > 15 min start a new trip; do not interpolate for distance.
11. Outputs (dataframes)
    df_trips: one row per trip with features above, plus vehicle_id
    df_daily: one row per vehicle-day (MYT) with KPIs above
    Optional: df_events with speeding/flagged events for dashboard overlays
12. Validation checks
    Consistency: sum(df_trips.Trip_Distance_km) ≈ sum valid fused_step_km by vehicle-day
    Sanity bands:
    Avg_Speed_kmh typically 20–90 km/h; investigate tails
    Idle_Percentage 0–60% typical; investigate > 80%
    Spot-check 3–5 trips against raw odometer start/end where available
13. Modeling target prep (Idle_Percentage)
    Label: next-trip Idle_Percentage for each (vehicle_id, trip_id)
    Candidate features:
    Recent history: last N trips idle%, moving%, distance, duration
    Time features: hour-of-day, day-of-week (MYT)
    Route proxy: median speed, stop density (idle events per hour)
    Vehicle profile: per-vehicle idle baseline (rolling mean)

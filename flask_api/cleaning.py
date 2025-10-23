#!/usr/bin/env python
# coding: utf-8

# # Safetruck Malaysia — Data Cleaning and Feature Engineering
# 
# This notebook implements the complete data cleaning pipeline according to the plan, including:
# 1. Data ingestion and normalization
# 2. GPS smoothing and filtering
# 3. Odometer processing and fusion
# 4. Trip segmentation
# 5. Feature calculation
# 6. Daily aggregation

# ## 1. Import Libraries and Setup

# In[17]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# For distance calculations
from math import radians, cos, sin, asin, sqrt

def cleaning(uuid):
    print("Libraries imported successfully")

    # ## 2. Load and Normalize Data

    # In[18]:


    # Load the combined data
    data_path = os.getenv('COMBINED_DATA_PATH', f'combined_data.csv')
    df = pd.read_csv(data_path, low_memory=False)

    print(f"Loaded {len(df):,} rows")
    print(f"\nOriginal columns: {list(df.columns)}")
    df.head()


    # In[19]:


    # Normalize column names to canonical schema
    # We need to inspect the actual columns first - let's create a flexible mapping

    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower or 'date' in col_lower:
            column_mapping[col] = 'timestamp_utc'
        elif 'plate' in col_lower or 'vehicle' in col_lower or 'asset' in col_lower:
            column_mapping[col] = 'vehicle_id'
        elif 'engine' in col_lower or 'ignition' in col_lower:
            column_mapping[col] = 'engine_on'
        elif 'speed' in col_lower:
            column_mapping[col] = 'speed_kmh'
        elif 'odometer' in col_lower or 'odo' in col_lower:
            column_mapping[col] = 'odometer_km'
        elif 'lat' in col_lower and 'lon' not in col_lower:
            column_mapping[col] = 'lat'
        elif 'lon' in col_lower or 'lng' in col_lower:
            column_mapping[col] = 'lon'
        elif 'gps' in col_lower and 'locat' in col_lower:
            column_mapping[col] = 'gps_ok'

    print("Column mapping:")
    for old, new in column_mapping.items():
        print(f"  {old} -> {new}")

    df.rename(columns=column_mapping, inplace=True)
    print(f"\nNew columns: {list(df.columns)}")


    # In[20]:


    # Parse timestamps to UTC
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True, errors='coerce')

    # Sort by vehicle and timestamp
    df.sort_values(['vehicle_id', 'timestamp_utc'], inplace=True)

    # Drop exact duplicates
    initial_count = len(df)
    df.drop_duplicates(subset=['vehicle_id', 'timestamp_utc'], keep='first', inplace=True)
    print(f"Dropped {initial_count - len(df):,} duplicate rows")

    # Coerce numeric columns
    numeric_cols = ['speed_kmh', 'odometer_km', 'lat', 'lon']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert engine_on to boolean
    if 'engine_on' in df.columns:
        df['engine_on'] = df['engine_on'].astype(bool)

    # Convert gps_ok to boolean
    if 'gps_ok' in df.columns:
        df['gps_ok'] = df['gps_ok'].astype(bool)

    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    # Convert vehicle_id to categorical for faster groupby operations
    df['vehicle_id'] = df['vehicle_id'].astype('category')

    print(f"\nFinal shape: {df.shape}")
    print(f"Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    print(f"Number of vehicles: {df['vehicle_id'].nunique()}")
    df.info()


    # ## 3. GPS Smoothing and Step Filtering

    # In[21]:


    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        Returns distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # Radius of earth in kilometers
        r = 6371

        return c * r

    print("Haversine function defined")


    # In[22]:


    # Apply rolling median smoothing to speed per vehicle
    df['speed_smoothed'] = df.groupby('vehicle_id')['speed_kmh'].transform(
        lambda x: x.rolling(window=3, center=True, min_periods=1).median()
    )

    print("Speed smoothing applied")
    print(f"Original speed range: {df['speed_kmh'].min():.2f} - {df['speed_kmh'].max():.2f} km/h")
    print(f"Smoothed speed range: {df['speed_smoothed'].min():.2f} - {df['speed_smoothed'].max():.2f} km/h")


    # In[23]:


    # Compute GPS step distance and time delta per vehicle
    # Optimize: group once and reuse
    g = df.groupby('vehicle_id', sort=False)
    df['lat_prev'] = g['lat'].shift(1)
    df['lon_prev'] = g['lon'].shift(1)
    df['time_prev'] = g['timestamp_utc'].shift(1)

    # Calculate GPS step distance using haversine - VECTORIZED
    mask = df['lat'].notna() & df['lon'].notna() & df['lat_prev'].notna() & df['lon_prev'].notna()
    
    # Vectorized haversine calculation
    lat1 = np.radians(df.loc[mask, 'lat_prev'].to_numpy())
    lon1 = np.radians(df.loc[mask, 'lon_prev'].to_numpy())
    lat2 = np.radians(df.loc[mask, 'lat'].to_numpy())
    lon2 = np.radians(df.loc[mask, 'lon'].to_numpy())
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    df.loc[mask, 'gps_step_km'] = 2 * 6371.0 * np.arcsin(np.sqrt(a))

    # Calculate time delta in hours
    df['time_delta_hours'] = (df['timestamp_utc'] - df['time_prev']).dt.total_seconds() / 3600

    print(f"GPS steps calculated: {df['gps_step_km'].notna().sum():,} valid steps")
    print(f"Mean GPS step: {df['gps_step_km'].mean():.4f} km")
    print(f"Median GPS step: {df['gps_step_km'].median():.4f} km")


    # In[24]:


    # Filter teleports and spikes
    # If implied speed > 140 km/h, mark as invalid - OPTIMIZED
    valid_dt = df['time_delta_hours'] > 0
    df['implied_speed_kmh'] = np.nan
    df.loc[valid_dt, 'implied_speed_kmh'] = df.loc[valid_dt, 'gps_step_km'] / df.loc[valid_dt, 'time_delta_hours']

    # Mark teleports
    teleport_mask = df['implied_speed_kmh'] > 140
    teleport_count = teleport_mask.sum()

    # Set gps_step_km to NaN for teleports
    df.loc[teleport_mask, 'gps_step_km'] = np.nan

    # If gps_ok is False, also set gps_step_km to NaN
    if 'gps_ok' in df.columns:
        bad_gps_mask = df['gps_ok'] == False
        df.loc[bad_gps_mask, 'gps_step_km'] = np.nan
        print(f"Invalidated {bad_gps_mask.sum():,} GPS steps where gps_ok=False")

    print(f"Filtered {teleport_count:,} teleport/spike events (>140 km/h)")
    print(f"Valid GPS steps remaining: {df['gps_step_km'].notna().sum():,}")


    # ## 4. Odometer Deltas and Fusion

    # In[25]:


    # Compute odometer step per vehicle
    df['odo_step_km'] = df.groupby('vehicle_id')['odometer_km'].diff()

    # Mark invalid odometer steps
    # Negative deltas or unrealistically large deltas (> 50 km in < 60s)
    df['invalid_odo_step'] = False

    # Negative deltas
    negative_mask = df['odo_step_km'] < 0
    df.loc[negative_mask, 'invalid_odo_step'] = True

    # Large deltas in short time
    large_delta_mask = (df['odo_step_km'] > 50) & (df['time_delta_hours'] < (60/3600))
    df.loc[large_delta_mask, 'invalid_odo_step'] = True

    print(f"Odometer steps calculated: {df['odo_step_km'].notna().sum():,}")
    print(f"Invalid odometer steps: {df['invalid_odo_step'].sum():,}")
    print(f"  - Negative: {negative_mask.sum():,}")
    print(f"  - Large delta: {large_delta_mask.sum():,}")


    # In[26]:


    # Calculate GPS-corrected odometer scale factor per vehicle
    # Use valid segments where:
    # - gps_ok == True
    # - gps_step_km is finite
    # - 0.01 <= odo_step_km <= 5.0
    # - implied_speed <= 120 km/h

    calibration_mask = (
        (df['gps_ok'] == True) &
        (df['gps_step_km'].notna()) &
        (df['odo_step_km'] >= 0.01) &
        (df['odo_step_km'] <= 5.0) &
        (df['invalid_odo_step'] == False) &
        (df['implied_speed_kmh'] <= 120)
    )

    df['odo_gps_ratio'] = np.where(
        calibration_mask & (df['odo_step_km'] > 0),
        df['gps_step_km'] / df['odo_step_km'],
        np.nan
    )

    # Calculate median scale factor per vehicle
    scale_factors = df[df['odo_gps_ratio'].notna()].groupby('vehicle_id')['odo_gps_ratio'].median()

    # Clip to reasonable range [0.8, 1.2]
    scale_factors = scale_factors.clip(0.8, 1.2)

    print(f"Calibration segments used: {calibration_mask.sum():,}")
    print(f"Vehicles with scale factors: {len(scale_factors)}")
    print(f"\nScale factor statistics:")
    print(scale_factors.describe())

    # Map scale factors back to dataframe
    # Convert vehicle_id to string temporarily to avoid categorical issues
    df['scale_factor'] = df['vehicle_id'].astype(str).map(scale_factors.to_dict()).fillna(1.0)


    # In[27]:


    # Create fused distance
    df['fused_step_km'] = 0.0
    df['distance_missing'] = False

    # Priority 1: Valid odometer (scaled)
    valid_odo_mask = (df['odo_step_km'].notna()) & (df['invalid_odo_step'] == False) & (df['odo_step_km'] >= 0)
    df.loc[valid_odo_mask, 'fused_step_km'] = df.loc[valid_odo_mask, 'odo_step_km'] * df.loc[valid_odo_mask, 'scale_factor']

    # Priority 2: Valid GPS (when odometer not available)
    valid_gps_mask = (~valid_odo_mask) & (df['gps_step_km'].notna())
    df.loc[valid_gps_mask, 'fused_step_km'] = df.loc[valid_gps_mask, 'gps_step_km']

    # Mark missing distance
    df['distance_missing'] = (df['fused_step_km'] == 0) & (df['odo_step_km'].isna() | (df['invalid_odo_step'] == True)) & (df['gps_step_km'].isna())

    print(f"Fused distance calculation:")
    print(f"  - From odometer (scaled): {valid_odo_mask.sum():,}")
    print(f"  - From GPS: {valid_gps_mask.sum():,}")
    print(f"  - Missing: {df['distance_missing'].sum():,}")
    print(f"\nTotal fused distance: {df['fused_step_km'].sum():,.2f} km")
    print(f"Mean step: {df['fused_step_km'].mean():.4f} km")
    print(f"Median step: {df['fused_step_km'].median():.4f} km")


    # ## 5. Outlier Labeling

    # In[28]:


    # Create outlier flags
    df['flagged_over120'] = (df['gps_ok'] == False) & (df['speed_smoothed'] > 120)
    df['speeding_over120'] = (df['gps_ok'] == True) & (df['speed_smoothed'] > 120)

    # Exclusion policy: exclude flagged_over120 from metrics
    # For speeding_over120, we'll keep them but track separately (policy b)
    df['exclude_from_metrics'] = df['flagged_over120']

    print("Outlier labeling completed:")
    print(f"  - Flagged over 120 (bad GPS): {df['flagged_over120'].sum():,}")
    print(f"  - Speeding over 120 (good GPS): {df['speeding_over120'].sum():,}")
    print(f"  - Excluded from metrics: {df['exclude_from_metrics'].sum():,}")


    # ## 6. Trip Segmentation (Hybrid)

    # In[29]:


    # Initialize trip_id - VECTORIZED APPROACH
    df['trip_id'] = 0
    
    # Fill NaN values in engine_on to avoid comparison issues
    df['engine_on'] = df['engine_on'].fillna(False)
    
    # Vectorized trip segmentation per vehicle
    g = df.groupby('vehicle_id', sort=False)
    
    # Detect ignition change: False -> True
    engine_prev = g['engine_on'].shift(1).fillna(False)
    ignition_start = (~engine_prev) & df['engine_on']
    
    # Detect time gap >= 15 minutes
    ts_prev = g['timestamp_utc'].shift(1)
    time_gap_min = (df['timestamp_utc'] - ts_prev).dt.total_seconds() / 60
    time_gap_start = time_gap_min >= 15
    
    # Mark new trip starts
    new_trip = ignition_start | time_gap_start
    
    # Assign trip_id as cumulative sum within each vehicle
    df['trip_id'] = g.apply(lambda x: new_trip.loc[x.index].cumsum()).values

    print(f"Trip segmentation completed")
    print(f"Total trips identified: {df['trip_id'].nunique():,}")


    # In[30]:


    # Create unique trip identifier combining vehicle and trip
    df['trip_key'] = df['vehicle_id'].astype(str) + '_' + df['trip_id'].astype(str)

    print(f"Unique trip keys: {df['trip_key'].nunique():,}")
    print(f"Trips per vehicle statistics:")
    trips_per_vehicle = df.groupby('vehicle_id')['trip_id'].nunique()
    print(trips_per_vehicle.describe())


    # ## 7. Trip-Level Feature Calculation

    # In[31]:


    # Calculate time delta in minutes for idle/moving calculations
    df['time_delta_min'] = df['time_delta_hours'] * 60

    # Mark idle periods (engine on, speed < 1 km/h)
    df['is_idle'] = (df['engine_on'] == True) & (df['speed_smoothed'] < 1)

    # Mark moving periods (engine on, speed >= 1 km/h)
    df['is_moving'] = (df['engine_on'] == True) & (df['speed_smoothed'] >= 1)

    print("Idle and moving periods identified")
    print(f"Idle periods: {df['is_idle'].sum():,}")
    print(f"Moving periods: {df['is_moving'].sum():,}")


    # In[ ]:


    # Aggregate trip-level features - OPTIMIZED WITH GROUPBY
    
    # Prepare valid data mask
    valid_mask = ~df['exclude_from_metrics']
    df_valid = df[valid_mask].copy()
    
    # Group by vehicle and trip
    trip_grp = df_valid.groupby(['vehicle_id', 'trip_id'], sort=False)
    
    # Calculate aggregations
    trip_agg = trip_grp.agg(
        Trip_Start_UTC=('timestamp_utc', 'min'),
        Trip_End_UTC=('timestamp_utc', 'max'),
        Trip_Distance_km=('fused_step_km', 'sum')
    ).reset_index()
    
    # Calculate idle and moving time using custom aggregations
    idle_times = df_valid[df_valid['is_idle']].groupby(['vehicle_id', 'trip_id'])['time_delta_min'].sum()
    moving_times = df_valid[df_valid['is_moving']].groupby(['vehicle_id', 'trip_id'])['time_delta_min'].sum()
    
    trip_agg = trip_agg.merge(
        idle_times.rename('Idle_Time_min').reset_index(),
        on=['vehicle_id', 'trip_id'],
        how='left'
    )
    trip_agg = trip_agg.merge(
        moving_times.rename('Moving_Time_min').reset_index(),
        on=['vehicle_id', 'trip_id'],
        how='left'
    )
    
    # Fill NaN for trips with no idle/moving time
    trip_agg['Idle_Time_min'] = trip_agg['Idle_Time_min'].fillna(0)
    trip_agg['Moving_Time_min'] = trip_agg['Moving_Time_min'].fillna(0)
    
    # Calculate derived metrics
    trip_agg['Trip_Duration_min'] = (trip_agg['Trip_End_UTC'] - trip_agg['Trip_Start_UTC']).dt.total_seconds() / 60
    trip_agg['Avg_Speed_kmh'] = np.where(
        trip_agg['Trip_Duration_min'] > 0,
        trip_agg['Trip_Distance_km'] / (trip_agg['Trip_Duration_min'] / 60),
        0
    )
    trip_agg['Idle_Percentage'] = np.where(
        trip_agg['Trip_Duration_min'] > 0,
        (trip_agg['Idle_Time_min'] / trip_agg['Trip_Duration_min']) * 100,
        0
    )
    
    # Add speeding and flagged counts from full dataset (not just valid)
    counts = df.groupby(['vehicle_id', 'trip_id'], sort=False).agg(
        Speeding_Count=('speeding_over120', 'sum'),
        Flagged_Count=('flagged_over120', 'sum')
    ).reset_index()
    
    trip_agg = trip_agg.merge(counts, on=['vehicle_id', 'trip_id'], how='left')
    trip_agg['Speeding_Count'] = trip_agg['Speeding_Count'].fillna(0).astype(int)
    trip_agg['Flagged_Count'] = trip_agg['Flagged_Count'].fillna(0).astype(int)
    
    # Create trip_key
    trip_agg['trip_key'] = trip_agg['vehicle_id'].astype(str) + '_' + trip_agg['trip_id'].astype(str)
    
    df_trips = trip_agg

    print(f"Trip features calculated for {len(df_trips):,} trips")
    print(f"\nTrip statistics:")
    print(df_trips[['Trip_Duration_min', 'Trip_Distance_km', 'Avg_Speed_kmh', 'Idle_Percentage']].describe())


    # In[ ]:


    # Filter trips by validity criteria
    # Keep trips with duration >= 5 min AND distance >= 1 km
    df_trips_valid = df_trips[
        (df_trips['Trip_Duration_min'] >= 5) & 
        (df_trips['Trip_Distance_km'] >= 1)
    ].copy()

    print(f"Valid trips after filtering: {len(df_trips_valid):,} (removed {len(df_trips) - len(df_trips_valid):,})")
    print(f"\nValid trip statistics:")
    print(df_trips_valid[['Trip_Duration_min', 'Trip_Distance_km', 'Avg_Speed_kmh', 'Idle_Percentage']].describe())

    df_trips_valid.head(10)


    # ## 8. Daily Aggregation (MYT Timezone)

    # In[ ]:


    # Convert to MYT timezone (UTC+8)
    df_trips_valid['Trip_Start_MYT'] = df_trips_valid['Trip_Start_UTC'].dt.tz_convert('Asia/Kuala_Lumpur')
    df_trips_valid['date_myt'] = df_trips_valid['Trip_Start_MYT'].dt.date

    print(f"Date range (MYT): {df_trips_valid['date_myt'].min()} to {df_trips_valid['date_myt'].max()}")
    print(f"Number of unique dates: {df_trips_valid['date_myt'].nunique()}")


    # In[ ]:


    # Aggregate daily metrics per vehicle-day
    daily_agg = df_trips_valid.groupby(['vehicle_id', 'date_myt'], sort=False).agg({
        'Trip_Distance_km': 'sum',
        'Trip_Duration_min': 'sum',
        'Idle_Time_min': 'sum',
        'Moving_Time_min': 'sum',
        'trip_id': 'count',  # Trip count
        'Speeding_Count': 'sum',
        'Flagged_Count': 'sum'
    }).reset_index()

    # Rename columns
    daily_agg.rename(columns={
        'Trip_Distance_km': 'Total_Distance_km',
        'Trip_Duration_min': 'Total_Duration_min',
        'Idle_Time_min': 'Total_Idle_min',
        'Moving_Time_min': 'Total_Moving_min',
        'trip_id': 'Trip_Count'
    }, inplace=True)

    # Calculate daily average speed and idle percentage
    daily_agg['Daily_Avg_Speed_kmh'] = np.where(
        daily_agg['Total_Duration_min'] > 0,
        daily_agg['Total_Distance_km'] / (daily_agg['Total_Duration_min'] / 60),
        0
    )
    
    daily_agg['Daily_Idle_Percentage'] = np.where(
        daily_agg['Total_Duration_min'] > 0,
        (daily_agg['Total_Idle_min'] / daily_agg['Total_Duration_min']) * 100,
        0
    )

    df_daily = daily_agg.copy()

    print(f"Daily aggregates created for {len(df_daily):,} vehicle-days")
    print(f"\nDaily statistics:")
    print(df_daily[['Total_Distance_km', 'Total_Duration_min', 'Daily_Avg_Speed_kmh', 'Daily_Idle_Percentage', 'Trip_Count']].describe())

    df_daily.head(10)


    # ## 9. Validation and Quality Checks

    # In[ ]:


    # Consistency check: trip distance vs raw fused distance
    total_trip_distance = df_trips_valid['Trip_Distance_km'].sum()
    total_fused_distance = df[~df['exclude_from_metrics']]['fused_step_km'].sum()

    print("=== CONSISTENCY CHECKS ===")
    print(f"Total trip distance: {total_trip_distance:,.2f} km")
    print(f"Total fused distance (valid points): {total_fused_distance:,.2f} km")
    print(f"Difference: {abs(total_trip_distance - total_fused_distance):,.2f} km")
    print(f"Difference percentage: {abs(total_trip_distance - total_fused_distance) / total_fused_distance * 100:.2f}%")
    print()


    # In[ ]:


    # Sanity band checks
    print("=== SANITY CHECKS ===")
    print(f"\nAverage Speed Distribution:")
    print(f"  Typical range (20-90 km/h): {((df_trips_valid['Avg_Speed_kmh'] >= 20) & (df_trips_valid['Avg_Speed_kmh'] <= 90)).sum()} trips ({((df_trips_valid['Avg_Speed_kmh'] >= 20) & (df_trips_valid['Avg_Speed_kmh'] <= 90)).sum() / len(df_trips_valid) * 100:.1f}%)")
    print(f"  Below 20 km/h: {(df_trips_valid['Avg_Speed_kmh'] < 20).sum()} trips")
    print(f"  Above 90 km/h: {(df_trips_valid['Avg_Speed_kmh'] > 90).sum()} trips")

    print(f"\nIdle Percentage Distribution:")
    print(f"  Typical range (0-60%): {((df_trips_valid['Idle_Percentage'] >= 0) & (df_trips_valid['Idle_Percentage'] <= 60)).sum()} trips ({((df_trips_valid['Idle_Percentage'] >= 0) & (df_trips_valid['Idle_Percentage'] <= 60)).sum() / len(df_trips_valid) * 100:.1f}%)")
    print(f"  High idle (>60%): {(df_trips_valid['Idle_Percentage'] > 60).sum()} trips")
    print(f"  Very high idle (>80%): {(df_trips_valid['Idle_Percentage'] > 80).sum()} trips")

    print(f"\nTrips requiring investigation (Avg_Speed > 90 or Idle > 80%):")
    investigate = df_trips_valid[(df_trips_valid['Avg_Speed_kmh'] > 90) | (df_trips_valid['Idle_Percentage'] > 80)]
    print(f"  Total: {len(investigate)} trips")
    if len(investigate) > 0:
        print(investigate[['vehicle_id', 'Trip_Distance_km', 'Trip_Duration_min', 'Avg_Speed_kmh', 'Idle_Percentage']].head())


    # ## 10. Prepare and Save Consolidated Output

    # In[ ]:


    # Create consolidated cleaned dataset with all relevant columns
    df_cleaned = df[[
        'timestamp_utc', 'vehicle_id', 'engine_on', 
        'speed_kmh', 'speed_smoothed', 'odometer_km', 
        'lat', 'lon', 'gps_ok',
        'gps_step_km', 'time_delta_hours', 'time_delta_min',
        'odo_step_km', 'invalid_odo_step', 'scale_factor',
        'fused_step_km', 'distance_missing',
        'implied_speed_kmh', 'flagged_over120', 'speeding_over120', 'exclude_from_metrics',
        'trip_id', 'trip_key', 'is_idle', 'is_moving'
    ]].copy()

    # Save the consolidated cleaned dataset
    output_path = f'combined_data_cleaned_pass1.csv'
    df_cleaned.to_csv(output_path, index=False)

    print(f"✓ Saved {output_path}")
    print(f"  Total rows: {len(df_cleaned):,}")
    print(f"  Total columns: {len(df_cleaned.columns)}")
    print(f"  File size: ~{len(df_cleaned) * len(df_cleaned.columns) * 8 / 1024 / 1024:.1f} MB (estimated)")

    # Also save the aggregated dataframes for convenience
    df_trips_valid.to_csv('df_trips.csv', index=False)
    df_daily.to_csv('df_daily.csv', index=False)

    print(f"\n✓ Also saved supplementary files:")
    print(f"  - data/df_trips.csv: {len(df_trips_valid):,} trips")
    print(f"  - data/df_daily.csv: {len(df_daily):,} vehicle-days")

    print("\nAll outputs saved successfully!")


    # In[ ]:


    # Display structure of consolidated cleaned data
    print("=" * 60)
    print("CONSOLIDATED CLEANED DATA STRUCTURE")
    print("=" * 60)

    print(f"\nColumns in combined_data_cleaned_pass1.csv:")
    print(f"\nRaw/Original columns:")
    print("  - timestamp_utc: UTC timestamp")
    print("  - vehicle_id: Vehicle identifier")
    print("  - engine_on: Engine status (boolean)")
    print("  - speed_kmh: Raw speed (km/h)")
    print("  - odometer_km: Raw odometer reading (km)")
    print("  - lat, lon: GPS coordinates")
    print("  - gps_ok: GPS quality indicator")

    print(f"\nDerived/Engineered columns:")
    print("  - speed_smoothed: Rolling median smoothed speed")
    print("  - gps_step_km: Haversine distance between GPS points")
    print("  - time_delta_hours/min: Time between consecutive points")
    print("  - odo_step_km: Odometer delta")
    print("  - invalid_odo_step: Odometer anomaly flag")
    print("  - scale_factor: GPS-corrected odometer calibration")
    print("  - fused_step_km: Fused distance (GPS-corrected odo + GPS fallback)")
    print("  - distance_missing: Flag for missing distance data")
    print("  - implied_speed_kmh: Speed calculated from distance/time")
    print("  - flagged_over120: Invalid speeding (bad GPS)")
    print("  - speeding_over120: Valid speeding (good GPS)")
    print("  - exclude_from_metrics: Exclusion flag for calculations")
    print("  - trip_id, trip_key: Trip segmentation identifiers")
    print("  - is_idle, is_moving: Idle/moving status flags")

    print(f"\nSample of cleaned data:")
    df_cleaned.head()


    # ## 11. Summary Statistics

    # In[ ]:


    print("=" * 60)
    print("SAFETRUCK DATA CLEANING SUMMARY")
    print("=" * 60)

    print(f"\nRAW DATA:")
    print(f"  Total rows loaded: {len(df):,}")
    print(f"  Number of vehicles: {df['vehicle_id'].nunique()}")
    print(f"  Date range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")

    print(f"\nDATA QUALITY:")
    print(f"  GPS teleports filtered: {teleport_count:,}")
    print(f"  Invalid odometer steps: {df['invalid_odo_step'].sum():,}")
    print(f"  Flagged over 120 (bad GPS): {df['flagged_over120'].sum():,}")
    print(f"  Speeding over 120 (good GPS): {df['speeding_over120'].sum():,}")
    print(f"  Excluded from metrics: {df['exclude_from_metrics'].sum():,}")

    print(f"\nTRIP SEGMENTATION:")
    print(f"  Total trips identified: {len(df_trips):,}")
    print(f"  Valid trips (≥5 min, ≥1 km): {len(df_trips_valid):,}")
    print(f"  Trips per vehicle (mean): {len(df_trips_valid) / df['vehicle_id'].nunique():.1f}")

    print(f"\nDISTANCE FUSION:")
    print(f"  Total fused distance: {total_fused_distance:,.2f} km")
    print(f"  From odometer (scaled): {valid_odo_mask.sum():,} segments")
    print(f"  From GPS: {valid_gps_mask.sum():,} segments")

    print(f"\nTRIP STATISTICS:")
    print(f"  Mean duration: {df_trips_valid['Trip_Duration_min'].mean():.1f} min")
    print(f"  Mean distance: {df_trips_valid['Trip_Distance_km'].mean():.2f} km")
    print(f"  Mean avg speed: {df_trips_valid['Avg_Speed_kmh'].mean():.1f} km/h")
    print(f"  Mean idle %: {df_trips_valid['Idle_Percentage'].mean():.1f}%")

    print(f"\nDAILY AGGREGATES:")
    print(f"  Vehicle-days: {len(df_daily):,}")
    print(f"  Mean trips/day: {df_daily['Trip_Count'].mean():.1f}")
    print(f"  Mean distance/day: {df_daily['Total_Distance_km'].mean():.1f} km")
    print(f"  Mean duration/day: {df_daily['Total_Duration_min'].mean():.1f} min")

    print(f"\nOUTPUT FILES:")
    print(f"  combined_data_cleaned_pass1.csv: {len(df_cleaned):,} rows × {len(df_cleaned.columns)} columns")
    print(f"  df_trips.csv: {len(df_trips_valid):,} trips")
    print(f"  df_daily.csv: {len(df_daily):,} vehicle-days")

    print("\n" + "=" * 60)


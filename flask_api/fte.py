#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering Pipeline - SafeTruck Hackathon
# 
# This notebook implements the feature engineering pipeline as specified in `_plan_.md`:
# - Trip segmentation (Engine ON → OFF)
# - Distance, duration, speed metrics
# - Idle time analysis
# - Efficiency submetrics (Distance, Time, Idle, Signal Reliability)
# - Age of Truck estimation
# - Idempotent processing with robust error handling

# ## 1. Import Libraries and Setup

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# For haversine distance calculation
from math import radians, cos, sin, asin, sqrt


# ## 2. Helper Functions

# In[2]:


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth (in km).
    Vectorized for pandas Series.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of earth in kilometers
    r = 6371.0
    return c * r


def column_exists(df, col_name):
    """Check if column exists (case-insensitive)."""
    return col_name.lower() in [c.lower() for c in df.columns]


def safe_divide(numerator, denominator, default=np.nan):
    """Safely divide two arrays, handling zero division."""
    result = np.full_like(numerator, default, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def fte(input_file="combined_data.csv", trip_output_file="safetruck_data_iter1_trips.csv", row_output_file="safetruck_data_iter1_rows.csv"):
    """
    Feature Engineering Pipeline for SafeTruck Hackathon
    
    Args:
        input_file: Path to input CSV file (default: "combined_data.csv")
        trip_output_file: Path to output trip-level CSV file (default: "safetruck_data_iter1_trips.csv")
        row_output_file: Path to output row-level CSV file (default: "safetruck_data_iter1_rows.csv")
    
    Returns:
        tuple: (trip_output, row_output) - two pandas DataFrames
    """
    
    print("Libraries imported successfully!")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    print("Helper functions defined!")
    
    # ## 3. Load and Normalize Data
    
    # In[3]:
    
    # Column mapping from combined_data.csv
    COLUMN_MAPPING = {
        'timestamp': 'Timestamp',
        'vehicle_id': 'CarNumberPlate',
        'lat': 'Latitude',
        'lon': 'Longitude',
        'speed_kmh': 'Speed',
        'engine_status': 'EngineStatus',
        'odometer_km': 'Odometer',
        'fuel_level': 'FuelLevelPercentage',
        'gps_valid': 'GPSLocated'
    }
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumns: {list(df.columns)}")
    
    
    # In[4]:
    
    
    # Normalize data
    print("\nNormalizing data...")
    
    # Parse timestamp to UTC
    df['timestamp'] = pd.to_datetime(df[COLUMN_MAPPING['timestamp']], utc=True, errors='coerce')
    
    # Vehicle ID
    df['vehicle_id'] = df[COLUMN_MAPPING['vehicle_id']].astype(str)
    
    # Lat/Lon as float
    df['lat'] = pd.to_numeric(df[COLUMN_MAPPING['lat']], errors='coerce')
    df['lon'] = pd.to_numeric(df[COLUMN_MAPPING['lon']], errors='coerce')
    
    # Speed in km/h
    df['speed_kmh'] = pd.to_numeric(df[COLUMN_MAPPING['speed_kmh']], errors='coerce')
    
    # Engine status: convert boolean or string to boolean
    # EngineStatus is already boolean (True/False) in the data
    df['engine_on'] = df[COLUMN_MAPPING['engine_status']].astype(str).str.upper().isin(['TRUE', '1', 'ON'])
    
    # Odometer in km
    df['odometer_km'] = pd.to_numeric(df[COLUMN_MAPPING['odometer_km']], errors='coerce')
    
    # Fuel level (optional, for later)
    df['fuel_level'] = pd.to_numeric(df[COLUMN_MAPPING['fuel_level']], errors='coerce')
    
    # GPS validity
    df['gps_valid'] = df[COLUMN_MAPPING['gps_valid']].astype(str).str.upper().isin(['TRUE', '1'])
    
    # Drop rows with missing critical fields
    df = df.dropna(subset=['timestamp', 'vehicle_id'])
    
    # Sort by vehicle and timestamp
    df = df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
    
    print(f"After normalization: {len(df):,} rows")
    print(f"Unique vehicles: {df['vehicle_id'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    
    # ## 4. Trip Segmentation (Engine ON → OFF)
    
    # In[5]:
    
    
    print("Segmenting trips...")
    
    # Remove duplicates per vehicle
    df = df.drop_duplicates(subset=['vehicle_id', 'timestamp'])
    
    # Initialize trip_id column
    df['trip_id'] = 0
    
    # Process each vehicle
    trip_counter = 0
    for vehicle in df['vehicle_id'].unique():
        mask = df['vehicle_id'] == vehicle
        vehicle_df = df.loc[mask].copy()
    
        # Detect transitions: OFF→ON starts a trip, ON→OFF ends it
        engine_diff = vehicle_df['engine_on'].astype(int).diff()
    
        # Trip starts: engine goes from OFF to ON (diff == 1) or first row is ON
        trip_starts = (engine_diff == 1) | ((engine_diff.isna()) & (vehicle_df['engine_on']))
    
        # Assign cumulative trip IDs
        trip_ids = trip_starts.cumsum() + trip_counter
        df.loc[mask, 'trip_id'] = trip_ids.values
    
        trip_counter = df.loc[mask, 'trip_id'].max() + 1
    
    # Filter to only rows where engine is ON (trips only)
    df_trips = df[df['engine_on']].copy()
    
    print(f"Total trips identified: {df_trips['trip_id'].nunique():,}")
    print(f"Rows in trips (engine ON): {len(df_trips):,}")
    print(f"Average rows per trip: {len(df_trips) / df_trips['trip_id'].nunique():.1f}")
    
    
    # ## 5. Calculate Time Deltas and State Windows
    
    # In[6]:
    
    
    print("Calculating time deltas and states...")
    
    # Calculate time delta to next row (within same trip)
    df_trips['time_to_next'] = df_trips.groupby('trip_id')['timestamp'].diff(-1).abs()
    df_trips['delta_t_sec'] = df_trips['time_to_next'].dt.total_seconds()
    
    # Cap delta_t to 300 seconds (5 minutes) to avoid inflated durations
    MAX_GAP_SEC = 300
    df_trips['delta_t_sec'] = df_trips['delta_t_sec'].clip(upper=MAX_GAP_SEC)
    
    # Fill NaN for last row of each trip with 0
    df_trips['delta_t_sec'] = df_trips['delta_t_sec'].fillna(0)
    
    # Define idle and moving flags
    IDLE_THRESHOLD_KMH = 1.0
    df_trips['is_idle'] = (df_trips['engine_on']) & (df_trips['speed_kmh'] <= IDLE_THRESHOLD_KMH)
    df_trips['is_moving'] = (df_trips['engine_on']) & (df_trips['speed_kmh'] > IDLE_THRESHOLD_KMH)
    
    print(f"Idle rows: {df_trips['is_idle'].sum():,} ({df_trips['is_idle'].sum() / len(df_trips) * 100:.1f}%)")
    print(f"Moving rows: {df_trips['is_moving'].sum():,} ({df_trips['is_moving'].sum() / len(df_trips) * 100:.1f}%)")
    
    
    # ## 6. Compute Trip-Level Metrics
    
    # In[7]:
    
    
    print("Computing trip-level metrics...")
    
    # Aggregate by trip
    trip_agg = df_trips.groupby(['vehicle_id', 'trip_id']).agg({
        'timestamp': ['min', 'max'],
        'lat': ['first', 'last'],
        'lon': ['first', 'last'],
        'odometer_km': ['first', 'last'],
        'delta_t_sec': 'sum',
        'speed_kmh': 'mean',
        'gps_valid': 'sum'  # Count of valid GPS records
    }).reset_index()
    
    # Flatten column names
    trip_agg.columns = ['_'.join(col).strip('_') for col in trip_agg.columns.values]
    trip_agg.columns = ['vehicle_id', 'trip_id', 'timestamp_start', 'timestamp_end',
                        'lat_start', 'lat_end', 'lon_start', 'lon_end',
                        'odometer_start', 'odometer_end', 'total_time_sec',
                        'avg_speed_raw', 'gps_valid_count']
    
    # Calculate trip distance from odometer
    trip_agg['Trip_Distance_km'] = trip_agg['odometer_end'] - trip_agg['odometer_start']
    
    # Handle negative distances (odometer reset or error)
    trip_agg.loc[trip_agg['Trip_Distance_km'] < 0, 'Trip_Distance_km'] = np.nan
    
    # Calculate trip duration in minutes
    trip_agg['Trip_Duration_min'] = (trip_agg['timestamp_end'] - trip_agg['timestamp_start']).dt.total_seconds() / 60
    
    # Calculate average speed (km/h)
    trip_agg['Avg_Speed'] = safe_divide(
        trip_agg['Trip_Distance_km'].values,
        (trip_agg['Trip_Duration_min'] / 60).values,
        default=np.nan
    )
    
    print(f"Trip-level aggregation complete: {len(trip_agg):,} trips")
    print(f"Trips with valid distance: {trip_agg['Trip_Distance_km'].notna().sum():,}")
    print(f"\nTrip distance stats:\n{trip_agg['Trip_Distance_km'].describe()}")
    
    
    # In[8]:
    
    
    print("Calculating idle and moving times...")
    
    # Aggregate idle and moving times per trip
    time_states = df_trips.groupby('trip_id').apply(
        lambda g: pd.Series({
            'Idle_Time_min': (g.loc[g['is_idle'], 'delta_t_sec'].sum()) / 60,
            'Moving_Time_min': (g.loc[g['is_moving'], 'delta_t_sec'].sum()) / 60,
            'total_rows': len(g)
        })
    ).reset_index()
    
    # Merge with trip_agg
    trip_agg = trip_agg.merge(time_states, on='trip_id', how='left')
    
    # Fill NaN with 0 for idle/moving times
    trip_agg['Idle_Time_min'] = trip_agg['Idle_Time_min'].fillna(0)
    trip_agg['Moving_Time_min'] = trip_agg['Moving_Time_min'].fillna(0)
    
    # Calculate idle percentage
    trip_agg['Idle_Percentage'] = safe_divide(
        trip_agg['Idle_Time_min'].values,
        trip_agg['Trip_Duration_min'].values,
        default=0
    )
    trip_agg['Idle_Percentage'] = trip_agg['Idle_Percentage'].clip(0, 1)
    
    print(f"Idle/moving time calculation complete")
    print(f"\nIdle percentage stats:\n{trip_agg['Idle_Percentage'].describe()}")
    
    
    # ## 7. Calculate Efficiency Submetrics
    
    # In[9]:
    
    
    print("Calculating efficiency submetrics...")
    
    # 1. Distance Efficiency = Optimal / Actual
    # Optimal = haversine distance between start and end points
    trip_agg['Optimal_Distance_km'] = haversine_distance(
        trip_agg['lat_start'],
        trip_agg['lon_start'],
        trip_agg['lat_end'],
        trip_agg['lon_end']
    )
    
    trip_agg['Distance_Efficiency'] = safe_divide(
        trip_agg['Optimal_Distance_km'].values,
        trip_agg['Trip_Distance_km'].values,
        default=np.nan
    )
    # Cap to reasonable range (0 to 1.5) - efficiency can't be > 1.5x
    trip_agg['Distance_Efficiency'] = trip_agg['Distance_Efficiency'].clip(0, 1.5)
    
    # 2. Time Efficiency = Expected / Actual
    # Expected time = Optimal distance / 80 km/h (baseline speed)
    BASELINE_SPEED_KMH = 80
    trip_agg['Expected_Time_hr'] = trip_agg['Optimal_Distance_km'] / BASELINE_SPEED_KMH
    trip_agg['Actual_Time_hr'] = trip_agg['Trip_Duration_min'] / 60
    
    trip_agg['Time_Efficiency'] = safe_divide(
        trip_agg['Expected_Time_hr'].values,
        trip_agg['Actual_Time_hr'].values,
        default=np.nan
    )
    # Cap to reasonable range
    trip_agg['Time_Efficiency'] = trip_agg['Time_Efficiency'].clip(0, 2.0)
    
    # 3. Idle Efficiency = 1 - (Idle_Time / Trip_Duration)
    trip_agg['Idle_Efficiency'] = 1 - trip_agg['Idle_Percentage']
    
    # 4. Signal Reliability = Located / Total
    trip_agg['Signal_Reliability'] = safe_divide(
        trip_agg['gps_valid_count'].values,
        trip_agg['total_rows'].values,
        default=np.nan
    )
    
    print("Efficiency metrics calculated!")
    print(f"\nDistance Efficiency stats:\n{trip_agg['Distance_Efficiency'].describe()}")
    print(f"\nTime Efficiency stats:\n{trip_agg['Time_Efficiency'].describe()}")
    print(f"\nIdle Efficiency stats:\n{trip_agg['Idle_Efficiency'].describe()}")
    print(f"\nSignal Reliability stats:\n{trip_agg['Signal_Reliability'].describe()}")
    
    
    # ## 8. Calculate Age of Truck (months)
    
    # In[10]:
    
    
    print("Calculating Age of Truck...")
    
    def calculate_truck_age(vehicle_df, window_days=60):
        """
        Calculate truck age based on odometer readings over the last N days.
        Age = Total Odometer / Monthly Average Distance
        """
        # Get the most recent date for this vehicle
        max_date = vehicle_df['timestamp'].max()
        cutoff_date = max_date - pd.Timedelta(days=window_days)
    
        # Filter to last N days
        recent_df = vehicle_df[vehicle_df['timestamp'] >= cutoff_date].copy()
    
        if len(recent_df) == 0:
            return np.nan
    
        # Extract date only
        recent_df['date'] = recent_df['timestamp'].dt.date
    
        # For each day, get first and last odometer reading
        daily_odo = recent_df.groupby('date')['odometer_km'].agg(['first', 'last'])
        daily_odo['daily_distance'] = daily_odo['last'] - daily_odo['first']
    
        # Filter out negative or anomalous days
        daily_odo = daily_odo[daily_odo['daily_distance'] >= 0]
    
        # Sum valid daily distances
        S = daily_odo['daily_distance'].sum()
    
        # Get final odometer reading
        O_final = vehicle_df['odometer_km'].max()
    
        # Calculate age in months
        if S > 0:
            # Normalize S to 30-day month
            normalized_monthly_distance = S * (30 / window_days)
            age_months = O_final / normalized_monthly_distance
            return age_months
        else:
            return np.nan
    
    # Calculate age for each vehicle
    vehicle_ages = []
    for vehicle_id in df['vehicle_id'].unique():
        vehicle_df = df[df['vehicle_id'] == vehicle_id]
        age = calculate_truck_age(vehicle_df, window_days=60)
        vehicle_ages.append({'vehicle_id': vehicle_id, 'Age_of_Truck_months': age})
    
    vehicle_age_df = pd.DataFrame(vehicle_ages)
    
    # Merge with trip data
    trip_agg = trip_agg.merge(vehicle_age_df, on='vehicle_id', how='left')
    
    print(f"Age of Truck calculated for {vehicle_age_df['Age_of_Truck_months'].notna().sum()} vehicles")
    print(f"\nAge statistics:\n{vehicle_age_df['Age_of_Truck_months'].describe()}")
    
    
    # ## 9. Prepare Output Tables
    
    # In[11]:
    
    
    print("Preparing output tables...")
    
    # 1. Trip-level table
    trip_output = trip_agg[[
        'vehicle_id', 'trip_id',
        'timestamp_start', 'timestamp_end',
        'lat_start', 'lon_start', 'lat_end', 'lon_end',
        'Trip_Distance_km', 'Trip_Duration_min', 'Avg_Speed',
        'Idle_Time_min', 'Moving_Time_min', 'Idle_Percentage',
        'Distance_Efficiency', 'Time_Efficiency', 'Idle_Efficiency', 'Signal_Reliability',
        'Age_of_Truck_months',
        'total_rows'
    ]].copy()
    
    # Add QA flags
    trip_output['QA_negative_distance'] = trip_output['Trip_Distance_km'].isna()
    trip_output['QA_zero_duration'] = trip_output['Trip_Duration_min'] <= 0
    
    print(f"\nTrip-level table: {len(trip_output):,} rows, {len(trip_output.columns)} columns")
    
    # 2. Row-level table (slim version)
    row_output = df_trips[[
        'vehicle_id', 'trip_id', 'timestamp',
        'lat', 'lon', 'speed_kmh', 'engine_on', 'odometer_km',
        'is_idle', 'is_moving', 'gps_valid'
    ]].copy()
    
    print(f"Row-level table: {len(row_output):,} rows, {len(row_output.columns)} columns")
    
    
    # ## 10. Save Output Files
    
    # In[12]:
    
    
    # Output filenames
    
    print("Saving output files...")
    
    # Save trip-level table
    trip_output.to_csv(trip_output_file, index=False)
    print(f"✓ Trip-level table saved: {trip_output_file}")
    print(f"  Size: {os.path.getsize(trip_output_file) / 1024**2:.2f} MB")
    
    # Save row-level table
    row_output.to_csv(row_output_file, index=False)
    print(f"✓ Row-level table saved: {row_output_file}")
    print(f"  Size: {os.path.getsize(row_output_file) / 1024**2:.2f} MB")
    
    print("\n✅ Feature engineering pipeline complete!")
    
    return trip_output, row_output


# Example usage:
# if __name__ == "__main__":
#     trip_output, row_output = fte(
#         input_file="combined_data.csv",
#         trip_output_file="safetruck_data_iter1_trips.csv",
#         row_output_file="safetruck_data_iter1_rows.csv"
#     )


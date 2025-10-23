#!/usr/bin/env python
# coding: utf-8

# # Team 3 Hackathon EDA
# 
# This notebook conducts a thorough exploratory data analysis (EDA) on the Safetruck dataset, including:
# 
# - Data loading and merging from three CSV sources
# - Feature engineering for route efficiency (target variable)
# - Univariate analysis per dataset
# - Correlation analysis on merged data
# - Target mapping (route_eff vs. features)
# - Geospatial visualization
# 
# **Plan:** Based on `_plan_.md`, we use lat/lon, speed, and direction to engineer a normalized route efficiency metric, then systematically explore relationships with other features.

# Imports and Configuration
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kruskal
from pathlib import Path
import warnings
import os
import re

def eda(uuid):
    """
    Perform comprehensive Exploratory Data Analysis on Safetruck dataset.
    
    This function conducts a thorough EDA including:
    - Data loading and merging from three CSV sources
    - Feature engineering for route efficiency (target variable)
    - Univariate analysis per dataset
    - Correlation analysis on merged data
    - Target mapping (route_eff vs. features)
    - Geospatial visualization
    """
    warnings.filterwarnings('ignore')
    
    # Plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    # os.chdir('data')
    # os.chdir(uuid)
    # Paths
    DATA_DIR = Path('../collected')
    CACHE_DIR = Path('../cache')
    OUTPUT_DIR = Path('deliverables')
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create separate directories for each dataset
    for dataset in ['rows', 'trips', 'combined', 'merged', 'trip_level']:
        Path(f'{OUTPUT_DIR}/{dataset}').mkdir(exist_ok=True)
        Path(f'{OUTPUT_DIR}/{dataset}/univariate').mkdir(exist_ok=True)
        Path(f'{OUTPUT_DIR}/{dataset}/maps').mkdir(exist_ok=True)
        Path(f'{OUTPUT_DIR}/{dataset}/correlations').mkdir(exist_ok=True)
        Path(f'{OUTPUT_DIR}/{dataset}/target').mkdir(exist_ok=True)
        Path(f'{OUTPUT_DIR}/{dataset}/summaries').mkdir(exist_ok=True)
    
    # ============================================================================
    # CONFIGURATION: Data Cleaning and Feature Selection
    # ============================================================================
    
    # Columns to drop completely (IDs, PII, high-cardinality noise)
    DROP_COLUMNS = [
        'trip_id', 'vehicle_id', 'vehicle_id_trip',  # IDs
        'number_plate', 'vin', 'license',  # PII (if present)
        'filename', 'file_name', 'file_path',  # File metadata
    ]
    
    # Columns to exclude from correlation/visualization (target leakage + ingredients)
    EXCLUDE_FROM_ANALYSIS = [
        # Target ingredients (keep for validation only)
        'forward_progress', 'speed_kmh', 'speed_mps', 'movement_speed', 
        'implied_speed', 'speed_ratio', 'dist_to_prev', 'time_delta',
        'bearing_track',
        # Raw geospatial (use only for mapping)
        'lat', 'lon', 'lat_prev', 'lon_prev', 
        'lat_start', 'lon_start', 'lat_end', 'lon_end',
        # Timestamps (use for temporal aggregation only)
        'timestamp', 'timestamp_start', 'timestamp_end',
        # Low-signal columns
        'total_rows', 'avg_speed',  # avg_speed is broken in data
    ]
    
    # Outlier handling configuration
    OUTLIER_CONFIG = {
        'global_quantile_clip': [0.005, 0.995],  # Clip all numerics to [0.5th, 99.5th] percentile
        'domain_caps': {
            'speed_mps': [0, 45],  # 0-162 km/h
            'implied_speed': [0, 60],  # m/s
            'movement_speed': [0, 60],  # m/s
            'time_delta': [0, 600],  # seconds
            'dist_to_prev': [0, 10000],  # meters
            'age_of_truck_months': [0, 240],  # 0-20 years
            'trip_distance_km': [0, 5000],  # reasonable trip max
            'trip_duration_min': [0, 1440],  # 24 hours max
            'idle_time_min': [0, 1440],
            'moving_time_min': [0, 1440],
        }
    }
    
    # Analysis configuration
    AGG_LEVEL = 'trip'  # 'row' or 'trip' - analyze at trip level for actionable insights
    TARGETS = ['route_eff', 'idle_efficiency', 'distance_efficiency', 'time_efficiency']
    PLOT_TOP_K = 10  # Only plot top K features per target
    SAMPLE_SIZE_PLOT = 20_000  # Max points for scatter plots
    SAMPLE_SIZE_UNIVARIATE = 50_000  # Max for histograms
    
    # Constants
    SPEED_THRESHOLD_MPS = 0.5  # Stationary threshold in m/s
    OUTLIER_SPEED_MPS = 60.0  # GPS jump detection threshold
    HIGH_CARDINALITY_THRESHOLD = 100
    UNIQUENESS_RATIO_THRESHOLD = 0.5
    
    # Plotting mode configuration
    EDA_PLOT_MODE = 'per_feature'  # 'per_feature' or 'grid'
    DATASET_NAME = 'trip_level'  # Dataset identifier for output paths
    
    print("✓ Setup complete")
    
    
    # ## 1. Data Loading
    # 
    # Load the three CSV files with memory-efficient dtypes.
    
    # Load CSVs
    print("Loading data...")
    
    # Trips data
    df_trips = pd.read_csv(
        f'{DATA_DIR}/safetruck_data_iter1_trips.csv',
        parse_dates=['timestamp_start', 'timestamp_end']
    )
    
    # Rows data (telemetry)
    df_rows = pd.read_csv(
        f'{DATA_DIR}/safetruck_data_iter1_rows.csv',
        parse_dates=['timestamp']
    )
    
    # Combined data
    df_combined = pd.read_csv(
        DATA_DIR / 'combined_data.csv',
        parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(DATA_DIR / 'combined_data.csv', nrows=0).columns else None
    )
    
    print(f"df_rows shape: {df_rows.shape}")
    print(f"df_trips shape: {df_trips.shape}")
    print(f"df_combined shape: {df_combined.shape}")
    print("\ndf_rows columns:", list(df_rows.columns))
    print("\ndf_trips columns:", list(df_trips.columns))
    print("\ndf_combined columns:", list(df_combined.columns))
    
    
    # ## 2. Data Cleaning and Validation
    # 
    # Standardize column names, validate coordinates and speeds, check for missing values.
    
    # Data Cleaning
    print("Cleaning and validating data...")
    
    # Standardize column names (lowercase with underscores)
    for df in [df_rows, df_trips, df_combined]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Check coordinate ranges for df_rows
    if 'lat' in df_rows.columns and 'lon' in df_rows.columns:
        lat_valid = df_rows['lat'].between(-90, 90)
        lon_valid = df_rows['lon'].between(-180, 180)
        print(f"Invalid lat: {(~lat_valid).sum()}, Invalid lon: {(~lon_valid).sum()}")
        df_rows = df_rows[lat_valid & lon_valid].copy()
    
    # Check speed ranges (assuming speed column exists)
    if 'speed' in df_rows.columns:
        speed_valid = df_rows['speed'] >= 0
        print(f"Invalid speed (negative): {(~speed_valid).sum()}")
        df_rows = df_rows[speed_valid].copy()
    
    # Display missing values
    print("\nMissing values in df_rows:")
    print(df_rows.isnull().sum()[df_rows.isnull().sum() > 0])
    print("\nMissing values in df_trips:")
    print(df_trips.isnull().sum()[df_trips.isnull().sum() > 0])
    print("\nMissing values in df_combined:")
    print(df_combined.isnull().sum()[df_combined.isnull().sum() > 0])
    
    print(f"\nCleaned df_rows shape: {df_rows.shape}")
    
    
    # ## 3. Data Merging
    # 
    # Merge rows with trips on `trip_id`, then optionally join with combined_data.
    
    # Merge rows with trips
    print("Merging data...")
    
    # Left join rows with trips on trip_id
    df_merged = df_rows.merge(df_trips, on='trip_id', how='left', suffixes=('', '_trip'))
    
    print(f"df_merged shape after rows+trips join: {df_merged.shape}")
    print(f"Match rate: {df_merged['vehicle_id_trip'].notna().mean():.2%}")
    
    # Check if combined_data can be joined
    if 'trip_id' in df_combined.columns:
        # Determine if combined_data is row-level or trip-level
        combined_has_timestamp = 'timestamp' in df_combined.columns
    
        if combined_has_timestamp:
            # Try to join on trip_id and timestamp
            print("Attempting to join combined_data on trip_id and timestamp...")
            df_merged = df_merged.merge(
                df_combined, 
                on=['trip_id', 'timestamp'], 
                how='left', 
                suffixes=('', '_combined')
            )
        else:
            # Join on trip_id only (trip-level data)
            print("Joining combined_data on trip_id only...")
            df_merged = df_merged.merge(
                df_combined, 
                on='trip_id', 
                how='left', 
                suffixes=('', '_combined')
            )
    
        print(f"df_merged shape after combined join: {df_merged.shape}")
    else:
        print("combined_data does not have trip_id column, skipping merge.")
    
    print(f"\nFinal df_merged shape: {df_merged.shape}")
    print(f"Columns: {len(df_merged.columns)}")
    
    
    # ## 4. Feature Engineering: Route Efficiency
    # 
    # Compute route efficiency from lat/lon and speed. Since direction/heading sensor data is not available, we use GPS-derived movement speed as a proxy for forward progress.
    
    # Helper functions for route efficiency
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance in meters between two lat/lon points."""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
    
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        return R * c
    
    def initial_bearing(lat1, lon1, lat2, lon2):
        """Calculate initial bearing in degrees [0, 360) from point 1 to point 2."""
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dlambda = np.radians(lon2 - lon1)
    
        x = np.sin(dlambda) * np.cos(phi2)
        y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    
        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360
    
    def angle_difference(angle1, angle2):
        """Calculate smallest angle difference between two angles in degrees."""
        diff = np.abs(angle1 - angle2)
        diff = np.where(diff > 180, 360 - diff, diff)
        return diff
    
    def clip_outliers(df, config):
        """Apply outlier clipping based on configuration."""
        df_clean = df.copy()
        
        # Get numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Apply global quantile clipping
        q_low, q_high = config['global_quantile_clip']
        
        for col in numeric_cols:
            if col in df_clean.columns and df_clean[col].notna().sum() > 0:
                # Apply domain caps first if specified
                if col in config['domain_caps']:
                    min_cap, max_cap = config['domain_caps'][col]
                    df_clean[col] = df_clean[col].clip(lower=min_cap, upper=max_cap)
                
                # Then apply quantile clipping
                q_vals = df_clean[col].quantile([q_low, q_high])
                df_clean[col] = df_clean[col].clip(lower=q_vals.iloc[0], upper=q_vals.iloc[1])
        
        return df_clean
    
    print("✓ Helper functions defined")
    
    
    # Compute route efficiency
    print("Computing route efficiency...")
    
    # Work on df_merged (or df_rows if needed)
    df_work = df_merged.copy()
    
    # Ensure required columns exist
    required_cols = ['lat', 'lon', 'speed_kmh', 'trip_id', 'timestamp']
    missing_cols = [col for col in required_cols if col not in df_work.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns {missing_cols}")
        print("Available columns:", list(df_work.columns[:20]))
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✓ All required columns present: {required_cols}")
    
    # Sort by trip_id and timestamp
    df_work = df_work.sort_values(['trip_id', 'timestamp']).reset_index(drop=True)
    
    # Convert speed from km/h to m/s
    df_work['speed_mps'] = df_work['speed_kmh'] / 3.6
    
    # Compute per-trip sequential bearing (track bearing)
    df_work['lat_prev'] = df_work.groupby('trip_id')['lat'].shift(1)
    df_work['lon_prev'] = df_work.groupby('trip_id')['lon'].shift(1)
    
    # Calculate distance and bearing to previous point
    df_work['dist_to_prev'] = haversine_distance(
        df_work['lat_prev'], df_work['lon_prev'],
        df_work['lat'], df_work['lon']
    )
    
    df_work['bearing_track'] = initial_bearing(
        df_work['lat_prev'], df_work['lon_prev'],
        df_work['lat'], df_work['lon']
    )
    
    # Calculate time delta for implied speed check
    df_work['time_delta'] = df_work.groupby('trip_id')['timestamp'].diff().dt.total_seconds()
    df_work['implied_speed'] = df_work['dist_to_prev'] / df_work['time_delta'].replace(0, np.nan)
    
    # Mark unreliable GPS (implied speed > 60 m/s)
    unreliable_count = (df_work['implied_speed'] > OUTLIER_SPEED_MPS).sum()
    print(f"Unreliable GPS points detected (implied speed > {OUTLIER_SPEED_MPS} m/s): {unreliable_count}")
    df_work.loc[df_work['implied_speed'] > OUTLIER_SPEED_MPS, 'bearing_track'] = np.nan
    
    # Since we don't have sensor heading/direction, we'll use a simplified route efficiency:
    # Route efficiency = forward speed normalized by max speed
    # For points with valid previous location, we use actual movement speed
    # This measures "productive movement" vs "stationary/idle time"
    
    # Calculate actual movement speed (distance / time)
    df_work['movement_speed'] = df_work['dist_to_prev'] / df_work['time_delta'].replace(0, np.nan)
    
    # Use the reported speed_mps as the baseline, but validate against movement_speed
    # If they're very different, something is wrong (GPS jump, stationary with speed sensor error, etc.)
    df_work['speed_ratio'] = df_work['movement_speed'] / df_work['speed_mps'].replace(0, np.nan)
    
    # Forward progress approximation: use minimum of reported speed and movement speed
    # This is conservative and handles sensor errors
    df_work['forward_progress'] = np.minimum(df_work['speed_mps'], df_work['movement_speed'].fillna(df_work['speed_mps']))
    
    # Set forward progress to 0 if speed below threshold
    df_work.loc[df_work['speed_mps'] < SPEED_THRESHOLD_MPS, 'forward_progress'] = 0
    
    # Set forward progress to 0 for unreliable points
    df_work.loc[df_work['bearing_track'].isna(), 'forward_progress'] = np.nan
    
    # Normalize by 95th percentile of speed_mps
    scale = df_work['speed_mps'].quantile(0.95)
    print(f"Speed 95th percentile: {scale:.2f} m/s ({scale*3.6:.2f} km/h)")
    
    df_work['route_eff'] = np.clip(df_work['forward_progress'] / scale, 0, 1)
    
    # Update df_merged
    df_merged = df_work.copy()
    
    # Display summary
    print(f"\nRoute efficiency computed for {df_merged.shape[0]} rows")
    print(f"route_eff summary:\n{df_merged['route_eff'].describe()}")
    print(f"NaN values in route_eff: {df_merged['route_eff'].isna().sum()} ({df_merged['route_eff'].isna().mean():.2%})")
    
    # Additional diagnostics
    print(f"\nDiagnostics:")
    print(f"  Stationary points (speed < {SPEED_THRESHOLD_MPS} m/s): {(df_merged['speed_mps'] < SPEED_THRESHOLD_MPS).sum():,}")
    print(f"  Moving points: {(df_merged['speed_mps'] >= SPEED_THRESHOLD_MPS).sum():,}")
    print(f"  Points with route_eff = 0: {(df_merged['route_eff'] == 0).sum():,}")
    print(f"  Points with route_eff > 0.5: {(df_merged['route_eff'] > 0.5).sum():,}")
    
    
    # Visualize route efficiency distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df_merged['route_eff'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Route Efficiency')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Route Efficiency Distribution')
    axes[0].axvline(df_merged['route_eff'].median(), color='red', linestyle='--', label=f'Median: {df_merged["route_eff"].median():.3f}')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(df_merged['route_eff'].dropna(), vert=True)
    axes[1].set_ylabel('Route Efficiency')
    axes[1].set_title('Route Efficiency Box Plot')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{DATASET_NAME}/target/route_eff_distribution.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print("✓ Route efficiency distribution plotted")
    
    
    # ## 4.5. Trip-Level Aggregation
    # 
    # Aggregate row-level telemetry to trip-level features for actionable insights.
    
    print("\n" + "="*80)
    print("TRIP-LEVEL FEATURE ENGINEERING")
    print("="*80)
    
    def aggregate_to_trips(df_rows):
        """Aggregate row-level telemetry to trip-level features."""
        print(f"Aggregating {len(df_rows)} rows to trip level...")
        
        # Sort by trip and timestamp
        df_rows = df_rows.sort_values(['trip_id', 'timestamp'])
        
        # Compute trip-level aggregates
        trip_aggs = df_rows.groupby('trip_id').agg({
            # Basic trip info
            'timestamp': ['min', 'max'],
            'lat': ['first', 'last'],
            'lon': ['first', 'last'],
            
            # Route efficiency metrics
            'route_eff': ['mean', 'median', 'std', lambda x: (x > 0.5).mean()],
            
            # Speed metrics
            'speed_mps': ['mean', 'std', 'max', lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)],
            
            # Idle and movement
            'idle_time_min': 'sum',
            'moving_time_min': 'sum',
            'idle_percentage': 'mean',
            'idle_efficiency': 'mean',
            
            # Distance and duration
            'trip_distance_km': 'first',  # Already aggregated
            'trip_duration_min': 'first',
            
            # Efficiency metrics
            'distance_efficiency': 'mean',
            'time_efficiency': 'mean',
            
            # Data quality
            'signal_reliability': 'mean',
            
            # Count of records
            'speed_kmh': 'count',  # Number of telemetry points
        }).reset_index()
        
        # Flatten column names
        trip_aggs.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in trip_aggs.columns.values]
        
        # Rename columns for clarity
        rename_map = {
            'timestamp_min': 'trip_start',
            'timestamp_max': 'trip_end',
            'lat_first': 'lat_start',
            'lat_last': 'lat_end',
            'lon_first': 'lon_start',
            'lon_last': 'lon_end',
            'route_eff_mean': 'route_eff_avg',
            'route_eff_median': 'route_eff_med',
            'route_eff_std': 'route_eff_std',
            'route_eff_<lambda>': 'route_eff_high_pct',
            'speed_mps_mean': 'speed_avg',
            'speed_mps_std': 'speed_std',
            'speed_mps_max': 'speed_max',
            'speed_mps_<lambda_0>': 'speed_p10',
            'speed_mps_<lambda_1>': 'speed_p90',
            'speed_kmh_count': 'n_telemetry_points',
            'idle_time_min_sum': 'idle_time_total',
            'moving_time_min_sum': 'moving_time_total',
            'trip_distance_km_first': 'trip_distance_km',
            'trip_duration_min_first': 'trip_duration_min',
        }
        trip_aggs = trip_aggs.rename(columns=rename_map)
        
        # Engineer additional features
        print("Engineering trip-level features...")
        
        # Route directness (great circle / actual distance)
        from math import radians, cos, sin, asin, sqrt
        def haversine_vec(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            return 6371 * c  # Earth radius in km
        
        trip_aggs['great_circle_km'] = haversine_vec(
            trip_aggs['lat_start'], trip_aggs['lon_start'],
            trip_aggs['lat_end'], trip_aggs['lon_end']
        )
        trip_aggs['route_directness'] = (
            trip_aggs['great_circle_km'] / 
            trip_aggs['trip_distance_km'].replace(0, np.nan)
        ).clip(0, 1)
        
        # Stops per km (proxy: count of low-speed points)
        # We'll compute this from the raw data
        def count_stops(group):
            # Detect transitions from moving to idle
            is_moving = group['speed_mps'] > SPEED_THRESHOLD_MPS
            transitions = (is_moving != is_moving.shift(1)).sum()
            return transitions / 2  # Divide by 2 to get stop count (enter+exit = 1 stop)
        
        stops = df_rows.groupby('trip_id').apply(count_stops).reset_index(name='n_stops')
        trip_aggs = trip_aggs.merge(stops, on='trip_id', how='left')
        trip_aggs['stops_per_km'] = (
            trip_aggs['n_stops'] / 
            trip_aggs['trip_distance_km'].replace(0, np.nan)
        )
        
        # Idle ratio
        trip_aggs['idle_ratio'] = (
            trip_aggs['idle_time_total'] / 
            (trip_aggs['idle_time_total'] + trip_aggs['moving_time_total']).replace(0, np.nan)
        )
        
        # Speed variability
        trip_aggs['speed_cv'] = (
            trip_aggs['speed_std'] / 
            trip_aggs['speed_avg'].replace(0, np.nan)
        )
        
        # Temporal features
        trip_aggs['hour_of_day'] = pd.to_datetime(trip_aggs['trip_start']).dt.hour
        trip_aggs['day_of_week'] = pd.to_datetime(trip_aggs['trip_start']).dt.dayofweek
        trip_aggs['is_weekend'] = trip_aggs['day_of_week'].isin([5, 6]).astype(int)
        trip_aggs['is_peak_hour'] = trip_aggs['hour_of_day'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Distance and time normalized metrics
        trip_aggs['km_per_hour'] = (
            trip_aggs['trip_distance_km'] / 
            (trip_aggs['trip_duration_min'] / 60).replace(0, np.nan)
        )
        
        print(f"✓ Created {len(trip_aggs)} trip-level records with {len(trip_aggs.columns)} features")
        return trip_aggs
    
    # Aggregate to trip level
    if AGG_LEVEL == 'trip':
        df_trips_agg = aggregate_to_trips(df_merged)
        df_analysis = df_trips_agg.copy()
        print(f"\nAnalysis dataset: {df_analysis.shape}")
        print(f"Columns: {list(df_analysis.columns[:20])}...")
    else:
        df_analysis = df_merged.copy()
        print(f"\nUsing row-level data for analysis: {df_analysis.shape}")
    
    
    # Helper function for safe filenames
    def _safe_name(name: str) -> str:
        """Sanitize a string for filesystem-safe filenames."""
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_')
    
    
    # ## 5. Column Typing and Classification
    # 
    # Automatically classify columns as numeric, categorical, datetime, or geospatial.
    
    def classify_columns(df, overrides=None):
        """
        Classify columns into: numeric, categorical, datetime, geospatial, id_like.
    
        Args:
            df: DataFrame to classify
            overrides: dict of column_name -> type for manual overrides
    
        Returns:
            dict with keys: 'numeric', 'categorical', 'datetime', 'geospatial', 'id_like'
        """
        overrides = overrides or {}
    
        classification = {
            'numeric': [],
            'categorical': [],
            'datetime': [],
            'geospatial': [],
            'id_like': []
        }
    
        for col in df.columns:
            # Apply override if exists
            if col in overrides:
                classification[overrides[col]].append(col)
                continue
    
            # Datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                classification['datetime'].append(col)
                continue
    
            # Geospatial
            if col.lower() in ['lat', 'latitude', 'lon', 'longitude', 'lat_prev', 'lon_prev']:
                classification['geospatial'].append(col)
                continue
    
            # Boolean columns should be categorical
            if pd.api.types.is_bool_dtype(df[col]) or df[col].dtype == bool:
                classification['categorical'].append(col)
                continue
    
            # ID-like: high uniqueness or specific names
            uniqueness_ratio = df[col].nunique() / len(df)
            is_id_name = any(x in col.lower() for x in ['id', 'uuid', 'vin', 'key'])
    
            if uniqueness_ratio > 0.9 or (is_id_name and uniqueness_ratio > 0.5):
                classification['id_like'].append(col)
                continue
    
            # Numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's really categorical (low cardinality numeric or binary)
                n_unique = df[col].nunique()
                if n_unique <= 10 and not is_id_name:  # Changed from 50 to 10 for stricter categorization
                    classification['categorical'].append(col)
                else:
                    classification['numeric'].append(col)
                continue
    
            # Categorical (object/category types)
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                classification['categorical'].append(col)
                continue
    
        return classification
    
    # Classify columns for each dataset
    print("Classifying columns...")
    
    cols_rows = classify_columns(df_rows)
    cols_trips = classify_columns(df_trips)
    cols_combined = classify_columns(df_combined)
    cols_merged = classify_columns(df_merged)
    
    print("\ndf_merged column classification:")
    for ctype, cols in cols_merged.items():
        print(f"  {ctype}: {len(cols)} columns")
        if len(cols) <= 10:
            print(f"    {cols}")
        else:
            print(f"    {cols[:10]} ... ({len(cols)-10} more)")
    
    
    # ## 6. Univariate EDA - Numeric Features (Top Features Only)
    # 
    # Histograms and summary statistics for top-ranked numeric columns.
    
    def plot_numeric_univariate(df, cols, dataset_name='data', max_plots=15, mode='per_feature'):
        """Plot histograms for numeric columns with improved visualization."""
        cols = [c for c in cols if c in df.columns][:max_plots]
    
        if not cols:
            print(f"No numeric columns to plot for {dataset_name}")
            return
    
        if mode == 'per_feature':
            # Save one plot per feature
            out_dir = Path(f"{OUTPUT_DIR}/{dataset_name}/univariate")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for col in cols:
                data = df[col].dropna()
                fig, ax = plt.subplots(figsize=(6, 4))
    
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(col)
                else:
                    # Sample if too large
                    if len(data) > SAMPLE_SIZE_UNIVARIATE:
                        data = data.sample(SAMPLE_SIZE_UNIVARIATE, random_state=42)
    
                    # Use histogram with KDE overlay for better visualization
                    ax.hist(data, bins=50, edgecolor='black', alpha=0.6, density=True, label='Histogram')
                    
                    # Add KDE if reasonable
                    try:
                        from scipy.stats import gaussian_kde
                        if len(data.unique()) > 10:  # Only if not too discrete
                            kde = gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 100)
                            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                    except:
                        pass
                    
                    ax.set_xlabel(col)
                    ax.set_ylabel('Density')
                    ax.set_title(f'{col}\n(μ={data.mean():.2f}, σ={data.std():.2f}, med={data.median():.2f})')
                    ax.grid(alpha=0.3)
                    ax.legend()
    
                plt.tight_layout()
                fname = out_dir / f"{dataset_name}_numeric__{_safe_name(col)}.svg"
                plt.savefig(fname, format='svg', bbox_inches='tight')
                plt.close(fig)
            
            print(f"✓ Saved {len(cols)} numeric univariate plots to {out_dir}")
        
        else:
            # Original grid mode
            n_cols = min(len(cols), 4)
            n_rows = (len(cols) + n_cols - 1) // n_cols
    
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]
    
            for idx, col in enumerate(cols):
                ax = axes[idx]
                data = df[col].dropna()
    
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(col)
                    continue
    
                # Sample if too large
                if len(data) > SAMPLE_SIZE_PLOT:
                    data = data.sample(SAMPLE_SIZE_PLOT, random_state=42)
    
                ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{col}\n(mean={data.mean():.2f}, std={data.std():.2f})')
                ax.grid(alpha=0.3)
    
            # Hide unused subplots
            for idx in range(len(cols), len(axes)):
                axes[idx].axis('off')
    
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/univariate/{dataset_name}_numeric.svg', format='svg', bbox_inches='tight')
            plt.close()
            print(f"✓ Saved numeric univariate grid to {OUTPUT_DIR}/univariate/{dataset_name}_numeric.svg")
    
    # Initialize variables for later use
    feature_rankings = {}
    primary_target = None
    cols_analysis = classify_columns(df_analysis)
    
    # Skip plotting for now - will do after feature ranking
    print("Deferring univariate plots until after feature ranking...")
    
    
    # ## 7. Univariate EDA - Categorical Features
    # 
    # Bar charts for categorical columns (top 20 categories for high-cardinality).
    
    def plot_categorical_univariate(df, cols, dataset_name='data', max_plots=15, mode='per_feature'):
        """Plot bar charts for categorical columns."""
        cols = [c for c in cols if c in df.columns][:max_plots]
    
        if not cols:
            print(f"No categorical columns to plot for {dataset_name}")
            return
    
        if mode == 'per_feature':
            # Save one plot per feature
            out_dir = Path(f"{OUTPUT_DIR}/{dataset_name}/univariate")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for col in cols:
                # Get value counts (top 20 for high cardinality)
                value_counts = df[col].value_counts().head(20)
                fig, ax = plt.subplots(figsize=(7, 4))
    
                if len(value_counts) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(col)
                else:
                    value_counts.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.7)
                    ax.set_xlabel(col)
                    ax.set_ylabel('Count')
                    ax.set_title(f'{col} (n_unique={df[col].nunique()})')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(alpha=0.3, axis='y')
    
                plt.tight_layout()
                fname = out_dir / f"{dataset_name}_categorical__{_safe_name(col)}.svg"
                plt.savefig(fname, format='svg', bbox_inches='tight')
                plt.close(fig)
            
            print(f"✓ Saved {len(cols)} categorical univariate plots to {out_dir}")
        
        else:
            # Original grid mode
            n_cols = min(len(cols), 3)
            n_rows = (len(cols) + n_cols - 1) // n_cols
    
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]
    
            for idx, col in enumerate(cols):
                ax = axes[idx]
    
                # Get value counts (top 20 for high cardinality)
                value_counts = df[col].value_counts().head(20)
    
                if len(value_counts) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(col)
                    continue
    
                value_counts.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.7)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.set_title(f'{col} (n_unique={df[col].nunique()})')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3, axis='y')
    
            # Hide unused subplots
            for idx in range(len(cols), len(axes)):
                axes[idx].axis('off')
    
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/univariate/{dataset_name}_categorical.svg', format='svg', bbox_inches='tight')
            plt.close()
            print(f"✓ Saved categorical univariate grid to {OUTPUT_DIR}/univariate/{dataset_name}_categorical.svg")
    
    # Identify categorical features for later
    cat_features = [c for c in cols_analysis['categorical'] if c not in EXCLUDE_FROM_ANALYSIS][:8]
    print(f"Identified {len(cat_features)} categorical features for analysis")
    
    
    # ## 7.5. Apply Outlier Clipping
    # 
    # Clean outliers before visualization and correlation analysis.
    
    print("\n" + "="*80)
    print("OUTLIER CLIPPING")
    print("="*80)
    
    print("Applying outlier clipping...")
    df_analysis_raw = df_analysis.copy()
    df_analysis = clip_outliers(df_analysis, OUTLIER_CONFIG)
    
    # Report clipping statistics
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
    print(f"\nClipped {len(numeric_cols)} numeric columns")
    print("Sample before/after ranges:")
    for col in list(numeric_cols[:5]):
        if col in df_analysis_raw.columns:
            raw_range = f"[{df_analysis_raw[col].min():.2f}, {df_analysis_raw[col].max():.2f}]"
            clean_range = f"[{df_analysis[col].min():.2f}, {df_analysis[col].max():.2f}]"
            print(f"  {col}: {raw_range} -> {clean_range}")
    
    print("✓ Outlier clipping complete")
    
    
    # ## 8. Feature Selection and Ranking
    # 
    # Rank features by correlation with targets, excluding leakage and noise.
    
    print("\n" + "="*80)
    print("FEATURE SELECTION AND RANKING")
    print("="*80)
    
    # Drop unwanted columns
    cols_to_drop = [col for col in DROP_COLUMNS if col in df_analysis.columns]
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} ID/PII columns: {cols_to_drop}")
        df_analysis = df_analysis.drop(columns=cols_to_drop)
    
    # Get available targets
    available_targets = [t for t in TARGETS if t in df_analysis.columns]
    if not available_targets:
        print(f"WARNING: No targets found in dataset. Available columns: {list(df_analysis.columns[:20])}")
        # Use route_eff_avg if available (trip-level)
        if 'route_eff_avg' in df_analysis.columns:
            available_targets = ['route_eff_avg']
            print(f"Using trip-level target: route_eff_avg")
    
    print(f"Targets for analysis: {available_targets}")
    
    # Set primary target
    primary_target = available_targets[0] if available_targets else None
    
    # Select features for correlation (exclude leakage and noise)
    feature_candidates = [
        c for c in cols_analysis['numeric'] 
        if c not in EXCLUDE_FROM_ANALYSIS and c not in available_targets
    ]
    
    print(f"\nFeature candidates: {len(feature_candidates)}")
    print(f"Excluded from analysis: {len([c for c in df_analysis.columns if c in EXCLUDE_FROM_ANALYSIS])}")
    
    # Rank features by correlation with each target
    feature_rankings = {}
    
    for target in available_targets:
        if target not in df_analysis.columns:
            continue
            
        print(f"\n--- Ranking features for {target} ---")
        
        # Compute correlations
        correlations = []
        for feat in feature_candidates:
            if feat in df_analysis.columns:
                valid_data = df_analysis[[feat, target]].dropna()
                if len(valid_data) > 10:
                    try:
                        spearman_r, _ = stats.spearmanr(valid_data[feat], valid_data[target])
                        pearson_r, _ = stats.pearsonr(valid_data[feat], valid_data[target])
                        correlations.append({
                            'feature': feat,
                            'spearman_r': spearman_r,
                            'pearson_r': pearson_r,
                            'abs_spearman': abs(spearman_r),
                            'abs_pearson': abs(pearson_r),
                        })
                    except:
                        pass
        
        if correlations:
            corr_df = pd.DataFrame(correlations).sort_values('abs_spearman', ascending=False)
            feature_rankings[target] = corr_df
            
            print(f"\nTop {min(15, len(corr_df))} features by absolute Spearman correlation:")
            print(corr_df.head(15)[['feature', 'spearman_r', 'pearson_r']].to_string(index=False))
            
            # Save rankings
            out_path = f'{OUTPUT_DIR}/{DATASET_NAME}/summaries/{target}_feature_rankings.csv'
            corr_df.to_csv(out_path, index=False)
            print(f"✓ Saved rankings to {out_path}")
    
    # Now plot univariate for top features
    if feature_rankings and primary_target and primary_target in feature_rankings:
        print("\n" + "="*80)
        print("UNIVARIATE PLOTS FOR TOP FEATURES")
        print("="*80)
        
        top_features_to_plot = feature_rankings[primary_target].head(PLOT_TOP_K)['feature'].tolist()
        print(f"\nPlotting top {len(top_features_to_plot)} numeric features...")
        plot_numeric_univariate(df_analysis, top_features_to_plot, DATASET_NAME, mode=EDA_PLOT_MODE)
        
        # Plot categorical features
        if cat_features:
            print(f"\nPlotting {len(cat_features)} categorical features...")
            plot_categorical_univariate(df_analysis, cat_features, DATASET_NAME, mode=EDA_PLOT_MODE)
    
    
    # ## 8.5. Correlation Analysis
    # 
    # Compute Pearson and Spearman correlations for top features only.
    
    # Select numeric columns for correlation (exclude geospatial and some IDs)
    corr_cols = [c for c in feature_candidates if c in df_analysis.columns][:30]  # Limit to top 30
    
    # Add primary target to correlation matrix (already set earlier)
    if primary_target and primary_target not in corr_cols:
        corr_cols.append(primary_target)
    
    print(f"\nComputing correlation matrix for {len(corr_cols)} features...")
    
    # Select data and drop rows with all NaN
    df_corr = df_analysis[corr_cols].dropna(how='all')
    
    # Pearson correlation
    corr_pearson = df_corr.corr(method='pearson')
    
    # Spearman correlation
    corr_spearman = df_corr.corr(method='spearman')
    
    print(f"Correlation matrix shape: {corr_pearson.shape}")
    if primary_target and primary_target in corr_pearson.columns:
        print(f"\nTop correlations with {primary_target} (Pearson):")
        route_corr = corr_pearson[primary_target].sort_values(ascending=False)
        print(route_corr.head(10))
    
    
    # Plot Pearson correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_pearson, dtype=bool))
    
    sns.heatmap(
        corr_pearson, 
        mask=mask, 
        cmap='coolwarm', 
        center=0, 
        vmin=-1, 
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False,  # Too many features for annotation
        ax=ax
    )
    
    ax.set_title('Pearson Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{DATASET_NAME}/correlations/pearson_correlation.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print("✓ Pearson correlation heatmap plotted")
    
    
    # Plot Spearman correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_spearman, dtype=bool))
    
    sns.heatmap(
        corr_spearman, 
        mask=mask, 
        cmap='coolwarm', 
        center=0, 
        vmin=-1, 
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=False,
        ax=ax
    )
    
    ax.set_title('Spearman Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{DATASET_NAME}/correlations/spearman_correlation.svg', format='svg', bbox_inches='tight')
    plt.close()
    
    print("✓ Spearman correlation heatmap plotted")
    
    
    # Save correlation summary for primary target
    if primary_target and primary_target in corr_pearson.columns:
        corr_summary = pd.DataFrame({
            'feature': corr_pearson.index,
            'pearson_r': corr_pearson[primary_target].values,
            'spearman_r': corr_spearman[primary_target].values
        })
    
        corr_summary = corr_summary.sort_values('pearson_r', ascending=False, key=abs)
        corr_summary.to_csv(f'{OUTPUT_DIR}/{DATASET_NAME}/summaries/{primary_target}_correlations.csv', index=False)
    
        print(f"\nTop 15 features by absolute Pearson correlation with {primary_target}:")
        print(corr_summary.head(15).to_string(index=False))
    else:
        print(f"Target {primary_target} not found in correlation matrix")
    
    
    # ## 9. Target Mapping - Numeric vs Target (Top Features Only)
    # 
    # Enhanced scatter plots with hexbin for top features vs targets.
    
    def plot_numeric_vs_target(df, feature_cols, target='route_eff', dataset_name='data', max_plots=10, mode='per_feature'):
        """Plot hexbin/scatter plots of numeric features vs target with correlation stats."""
    
        if target not in df.columns:
            print(f"Target {target} not found in dataframe")
            return
    
        # Select top correlated features
        feature_cols = [c for c in feature_cols if c in df.columns and c != target]
    
        if not feature_cols:
            print("No features to plot")
            return
    
        # Limit to max_plots
        feature_cols = feature_cols[:max_plots]
    
        if mode == 'per_feature':
            # Save one plot per feature
            out_dir = Path(f"{OUTPUT_DIR}/{dataset_name}/target")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for col in feature_cols:
                # Get clean data
                plot_data = df[[col, target]].dropna()
                fig, ax = plt.subplots(figsize=(6.5, 5))
    
                if len(plot_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(col)
                else:
                    # Compute correlations first
                    pearson_r, pearson_p = stats.pearsonr(plot_data[col], plot_data[target])
                    spearman_r, spearman_p = stats.spearmanr(plot_data[col], plot_data[target])
                    
                    # Use hexbin for large datasets, scatter for small
                    if len(plot_data) > 5000:
                        # Hexbin plot for large datasets
                        hexbin = ax.hexbin(plot_data[col], plot_data[target], 
                                          gridsize=30, cmap='YlOrRd', mincnt=1, alpha=0.8)
                        plt.colorbar(hexbin, ax=ax, label='Count')
                    else:
                        # Regular scatter for smaller datasets
                        ax.scatter(plot_data[col], plot_data[target], alpha=0.4, s=20, edgecolors='k', linewidth=0.5)
    
                    # Add LOWESS smoothing line
                    try:
                        from statsmodels.nonparametric.smoothers_lowess import lowess
                        if len(plot_data) > 100:
                            # Sample for lowess if too large
                            sample_size = min(5000, len(plot_data))
                            sample_data = plot_data.sample(sample_size, random_state=42)
                            smoothed = lowess(sample_data[target], sample_data[col], frac=0.2)
                            ax.plot(smoothed[:, 0], smoothed[:, 1], 'b-', linewidth=3, label='LOWESS', alpha=0.8)
                    except:
                        # Fallback to linear regression
                        try:
                            z = np.polyfit(plot_data[col], plot_data[target], 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(plot_data[col].min(), plot_data[col].max(), 100)
                            ax.plot(x_line, p(x_line), "b-", alpha=0.8, linewidth=2, label='Linear fit')
                        except:
                            pass
    
                    ax.set_xlabel(col, fontsize=11, fontweight='bold')
                    ax.set_ylabel(target, fontsize=11, fontweight='bold')
                    ax.set_title(f'{col} vs {target}\nSpearman ρ={spearman_r:.3f}, Pearson r={pearson_r:.3f}', 
                                fontsize=10)
                    ax.grid(alpha=0.3)
                    
                    # Check if we added a legend-worthy line
                    handles, labels = ax.get_legend_handles_labels()
                    if labels:  # If there are any labels, show legend
                        ax.legend()
    
                plt.tight_layout()
                fname = out_dir / f"{dataset_name}_numeric_vs_{_safe_name(target)}__{_safe_name(col)}.svg"
                plt.savefig(fname, format='svg', bbox_inches='tight')
                plt.close(fig)
            
            print(f"✓ Saved {len(feature_cols)} numeric vs {target} plots to {out_dir}")
        
        else:
            # Original grid mode
            n_cols = 3
            n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]
    
            for idx, col in enumerate(feature_cols):
                ax = axes[idx]
    
                # Get clean data
                plot_data = df[[col, target]].dropna()
    
                if len(plot_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_title(col)
                    continue
    
                # Sample if too large
                if len(plot_data) > SAMPLE_SIZE_PLOT:
                    plot_data = plot_data.sample(SAMPLE_SIZE_PLOT, random_state=42)
    
                # Scatter plot
                ax.scatter(plot_data[col], plot_data[target], alpha=0.3, s=10)
    
                # Compute correlations
                pearson_r, pearson_p = stats.pearsonr(plot_data[col], plot_data[target])
                spearman_r, spearman_p = stats.spearmanr(plot_data[col], plot_data[target])
    
                # Add regression line
                try:
                    z = np.polyfit(plot_data[col], plot_data[target], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_data[col].min(), plot_data[col].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                except:
                    pass
    
                ax.set_xlabel(col)
                ax.set_ylabel(target)
                ax.set_title(f'{col} vs {target}\nPearson r={pearson_r:.3f} (p={pearson_p:.1e})')
                ax.grid(alpha=0.3)
    
            # Hide unused subplots
            for idx in range(len(feature_cols), len(axes)):
                axes[idx].axis('off')
    
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/target/numeric_vs_{target}.svg', format='svg', bbox_inches='tight')
            plt.close()
            print(f"✓ Saved numeric vs {target} grid to {OUTPUT_DIR}/target/numeric_vs_{target}.svg")
    
    # Plot top features vs each target
    for target in available_targets[:2]:  # Limit to top 2 targets
        if target in feature_rankings:
            top_features = feature_rankings[target].head(PLOT_TOP_K)['feature'].tolist()
            top_features = [f for f in top_features if f != target and f in df_analysis.columns]
        
            print(f"\nPlotting top {len(top_features)} features vs {target}...")
            plot_numeric_vs_target(df_analysis, top_features, target, dataset_name=DATASET_NAME, mode=EDA_PLOT_MODE)
        else:
            print(f"No feature rankings for {target}, skipping target mapping")
    
    
    # ## 10. Target Mapping - Categorical vs Route Efficiency
    # 
    # Box plots for categorical features vs route_eff with Kruskal-Wallis test.
    
    def plot_categorical_vs_target(df, feature_cols, target='route_eff', dataset_name='data', max_plots=9, mode='per_feature'):
        """Plot box plots of categorical features vs target with Kruskal-Wallis test."""
    
        if target not in df.columns:
            print(f"Target {target} not found in dataframe")
            return
    
        feature_cols = [c for c in feature_cols if c in df.columns][:max_plots]
    
        if not feature_cols:
            print("No categorical features to plot")
            return
    
        kw_results = []
        
        if mode == 'per_feature':
            # Save one plot per feature
            out_dir = Path(f"{OUTPUT_DIR}/{dataset_name}/target")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for col in feature_cols:
                # Get clean data
                plot_data = df[[col, target]].dropna()
                fig, ax = plt.subplots(figsize=(7.5, 5))
    
                if len(plot_data) == 0 or plot_data[col].nunique() < 2:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                    ax.set_title(col)
                else:
                    # Limit to top categories if too many
                    top_cats = plot_data[col].value_counts().head(15).index
                    plot_data = plot_data[plot_data[col].isin(top_cats)]
    
                    # Sample if too large
                    if len(plot_data) > SAMPLE_SIZE_PLOT:
                        plot_data = plot_data.sample(SAMPLE_SIZE_PLOT, random_state=42)
    
                    # Box plot
                    plot_data.boxplot(column=target, by=col, ax=ax)
                    ax.set_xlabel(col)
                    ax.set_ylabel(target)
                    ax.get_figure().suptitle('')  # Remove automatic title
    
                    # Kruskal-Wallis test
                    groups = [group[target].values for name, group in plot_data.groupby(col)]
                    if len(groups) >= 2:
                        try:
                            kw_stat, kw_p = kruskal(*groups)
                            ax.set_title(f'{col}\nKruskal-Wallis p={kw_p:.1e}')
                            kw_results.append({'feature': col, 'kw_statistic': kw_stat, 'kw_pvalue': kw_p})
                        except:
                            ax.set_title(col)
                    else:
                        ax.set_title(col)
    
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(alpha=0.3, axis='y')
    
                plt.tight_layout()
                fname = out_dir / f"{dataset_name}_categorical_vs_{_safe_name(target)}__{_safe_name(col)}.svg"
                plt.savefig(fname, format='svg', bbox_inches='tight')
                plt.close(fig)
            
            print(f"✓ Saved {len(feature_cols)} categorical vs {target} plots to {out_dir}")
        
        else:
            # Original grid mode
            n_cols = 3
            n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]
    
            for idx, col in enumerate(feature_cols):
                ax = axes[idx]
    
                # Get clean data
                plot_data = df[[col, target]].dropna()
    
                if len(plot_data) == 0 or plot_data[col].nunique() < 2:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                    ax.set_title(col)
                    continue
    
                # Limit to top categories if too many
                top_cats = plot_data[col].value_counts().head(15).index
                plot_data = plot_data[plot_data[col].isin(top_cats)]
    
                # Sample if too large
                if len(plot_data) > SAMPLE_SIZE_PLOT:
                    plot_data = plot_data.sample(SAMPLE_SIZE_PLOT, random_state=42)
    
                # Box plot
                plot_data.boxplot(column=target, by=col, ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel(target)
                ax.get_figure().suptitle('')  # Remove automatic title
    
                # Kruskal-Wallis test
                groups = [group[target].values for name, group in plot_data.groupby(col)]
                if len(groups) >= 2:
                    try:
                        kw_stat, kw_p = kruskal(*groups)
                        ax.set_title(f'{col}\nKruskal-Wallis p={kw_p:.1e}')
                        kw_results.append({'feature': col, 'kw_statistic': kw_stat, 'kw_pvalue': kw_p})
                    except:
                        ax.set_title(col)
                else:
                    ax.set_title(col)
    
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3, axis='y')
    
            # Hide unused subplots
            for idx in range(len(feature_cols), len(axes)):
                axes[idx].axis('off')
    
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/target/categorical_vs_{target}.svg', format='svg', bbox_inches='tight')
            plt.close()
            print(f"✓ Saved categorical vs {target} grid to {OUTPUT_DIR}/target/categorical_vs_{target}.svg")
    
        # Save Kruskal-Wallis results
        if kw_results:
            kw_df = pd.DataFrame(kw_results).sort_values('kw_pvalue')
            summary_dir = Path(f"{OUTPUT_DIR}/{dataset_name}/summaries")
            summary_dir.mkdir(parents=True, exist_ok=True)
            kw_df.to_csv(summary_dir / f'categorical_vs_{target}_kruskal.csv', index=False)
            print(f"\nKruskal-Wallis test results saved. Top results:")
            print(kw_df.head(10).to_string(index=False))
    
    # Plot categorical features vs primary target
    if cat_features and primary_target:
        print(f"\nPlotting categorical features vs {primary_target}...")
        plot_categorical_vs_target(df_analysis, cat_features[:6], primary_target, dataset_name=DATASET_NAME, mode=EDA_PLOT_MODE)
    else:
        print("No categorical features or target not available")
    
    
    # ## 11. Geospatial Visualization
    # 
    # Map visualization of GPS points to validate spatial coverage.
    
    # Geospatial map using Plotly
    # For trip-level, map start and end points
    if AGG_LEVEL == 'trip' and 'lat_start' in df_analysis.columns:
        print("Creating geospatial map (trip start/end points)...")
    
        # Prepare mapping data
        map_sample_size = min(5_000, len(df_analysis))
        df_map_sample = df_analysis.sample(map_sample_size, random_state=42)
        
        # Create routes (start -> end)
        df_map = pd.DataFrame({
            'lat': df_map_sample['lat_start'].tolist() + df_map_sample['lat_end'].tolist(),
            'lon': df_map_sample['lon_start'].tolist() + df_map_sample['lon_end'].tolist(),
            'route_eff': df_map_sample[primary_target].tolist() + df_map_sample[primary_target].tolist(),
            'type': ['start'] * len(df_map_sample) + ['end'] * len(df_map_sample)
        }).dropna()
        
    elif 'lat' in df_merged.columns and 'lon' in df_merged.columns:
        print("Creating geospatial map (row-level GPS points)...")
    
        # Sample data for mapping
        map_sample_size = min(50_000, len(df_merged))
        df_map = df_merged[['lat', 'lon', 'route_eff']].dropna().sample(map_sample_size, random_state=42)
    else:
        df_map = None
    
    if df_map is not None and len(df_map) > 0:
    
        # Create scatter map
        color_col = 'route_eff' if 'route_eff' in df_map.columns else primary_target
        fig = px.scatter_mapbox(
            df_map,
            lat='lat',
            lon='lon',
            color=color_col,
            color_continuous_scale='RdYlGn',
            zoom=8,
            height=600,
            title=f'GPS Points colored by {color_col} (n={len(df_map):,})',
            labels={color_col: color_col.replace('_', ' ').title()},
            hover_data=['type'] if 'type' in df_map.columns else None
        )
    
        # Use open street map (no token needed)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    
        # Save as HTML (PDF export requires kaleido which may not be installed)
        map_path = f'{OUTPUT_DIR}/{DATASET_NAME}/maps/gps_{primary_target}_map.html'
        fig.write_html(map_path)
        try:
            fig.show()
        except:
            pass  # Don't fail if can't display
    
        print(f"✓ Map created with {len(df_map):,} points")
        print(f"  Map saved to: {map_path}")
    else:
        print("Latitude/Longitude columns not found for mapping")
    
    
    # ## 12. Temporal Analysis
    # 
    # Analyze route efficiency by time of day and day of week (if timestamp available).
    
    # Temporal analysis if temporal features exist
    has_temporal = 'hour_of_day' in df_analysis.columns or 'timestamp' in df_merged.columns
    
    if has_temporal and primary_target in df_analysis.columns:
        print("Performing temporal analysis...")
    
        # Use existing temporal features or create them
        if 'hour_of_day' in df_analysis.columns:
            df_temp = df_analysis.copy()
        else:
            df_temp = df_merged.copy()
            df_temp['hour_of_day'] = df_temp['timestamp'].dt.hour
            df_temp['day_of_week'] = df_temp['timestamp'].dt.dayofweek
            df_temp['day_name'] = df_temp['timestamp'].dt.day_name()
    
        # Target by hour of day
        hourly_eff = df_temp.groupby('hour_of_day')[primary_target].agg(['mean', 'median', 'std', 'count']).reset_index()
        hourly_eff = hourly_eff.rename(columns={'hour_of_day': 'hour'})
    
        # Target by day of week
        if 'day_name' in df_temp.columns:
            daily_eff = df_temp.groupby(['day_of_week', 'day_name'])[primary_target].agg(['mean', 'median', 'std', 'count']).reset_index()
        else:
            daily_eff = None
    
        # Plot
        n_plots = 2 if daily_eff is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(16 if n_plots == 2 else 10, 5))
        if n_plots == 1:
            axes = [axes]
    
        # Hour of day
        axes[0].plot(hourly_eff['hour'], hourly_eff['mean'], marker='o', linewidth=2, label='Mean', color='steelblue')
        axes[0].fill_between(
            hourly_eff['hour'], 
            hourly_eff['mean'] - hourly_eff['std'], 
            hourly_eff['mean'] + hourly_eff['std'],
            alpha=0.3, color='steelblue'
        )
        axes[0].set_xlabel('Hour of Day', fontweight='bold')
        axes[0].set_ylabel(primary_target.replace('_', ' ').title(), fontweight='bold')
        axes[0].set_title(f'{primary_target.replace("_", " ").title()} by Hour of Day')
        axes[0].grid(alpha=0.3)
        axes[0].set_xticks(range(0, 24, 2))
        axes[0].legend()
    
        # Day of week
        if daily_eff is not None:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_eff['day_name'] = pd.Categorical(daily_eff['day_name'], categories=day_order, ordered=True)
            daily_eff = daily_eff.sort_values('day_name')
            
            axes[1].bar(daily_eff['day_name'], daily_eff['mean'], edgecolor='black', alpha=0.7, color='coral')
            axes[1].errorbar(daily_eff['day_name'], daily_eff['mean'], yerr=daily_eff['std'], 
                           fmt='none', color='black', capsize=5)
            axes[1].set_xlabel('Day of Week', fontweight='bold')
            axes[1].set_ylabel(f'{primary_target.replace("_", " ").title()} (Mean)', fontweight='bold')
            axes[1].set_title(f'{primary_target.replace("_", " ").title()} by Day of Week')
            axes[1].grid(alpha=0.3, axis='y')
            axes[1].tick_params(axis='x', rotation=45)
    
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{DATASET_NAME}/target/{primary_target}_temporal.svg', format='svg', bbox_inches='tight')
        plt.close()
    
        # Save temporal summary
        summary_parts = [hourly_eff.assign(grouping='hour_of_day').rename(columns={'hour': 'group_value'})]
        if daily_eff is not None:
            summary_parts.append(daily_eff.assign(grouping='day_of_week').rename(columns={'day_name': 'group_value'}))
        
        temporal_summary = pd.concat(summary_parts, ignore_index=True)
        temporal_summary.to_csv(f'{OUTPUT_DIR}/{DATASET_NAME}/summaries/{primary_target}_temporal.csv', index=False)
    
        print("✓ Temporal analysis complete")
    else:
        print("Temporal features or target not available for temporal analysis")
    
    
    # ## 13. Summary Statistics
    # 
    # Generate comprehensive summary statistics for all datasets.
    
    # Summary statistics for analysis dataset
    print("\n" + "=" * 80)
    print(f"SUMMARY STATISTICS FOR {DATASET_NAME.upper()} DATASET")
    print("=" * 80)
    
    print(f"\nDataset shape: {df_analysis.shape}")
    print(f"Memory usage: {df_analysis.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    print("\n--- Numeric Features Summary (Top Features) ---")
    if feature_rankings and primary_target:
        top_numeric = feature_rankings[primary_target].head(15)['feature'].tolist()
        top_numeric = [c for c in top_numeric if c in df_analysis.columns]
        if top_numeric:
            numeric_summary = df_analysis[top_numeric].describe().T
            print(numeric_summary.to_string())
    
    if cat_features:
        print("\n--- Categorical Features Summary ---")
        for col in cat_features[:5]:  # Limit to first 5
            if col in df_analysis.columns:
                n_unique = df_analysis[col].nunique()
                top_val = df_analysis[col].value_counts().head(3)
                print(f"\n{col}: {n_unique} unique values")
                print(f"  Top 3: {top_val.to_dict()}")
    
    print(f"\n--- Target Variables Summary ---")
    for target in available_targets:
        if target in df_analysis.columns:
            print(f"\n{target}:")
            target_summary = df_analysis[target].describe()
            print(target_summary)
            print(f"Percentiles:")
            for p in [10, 25, 50, 75, 90, 95, 99]:
                print(f"  {p}th: {df_analysis[target].quantile(p/100):.4f}")
    
    # Save summary to file
    if feature_rankings and primary_target:
        numeric_summary.to_csv(f'{OUTPUT_DIR}/{DATASET_NAME}/summaries/numeric_summary_top_features.csv')
        print(f"\n✓ Summary statistics saved to {OUTPUT_DIR}/{DATASET_NAME}/summaries/numeric_summary_top_features.csv")
    
    
    # ## 14. Cache Cleaned Data
    # 
    # Save processed data to Parquet for faster re-runs.
    
    # In[32]:
    
    
    # Save cleaned and processed data to Parquet
    print("\nSaving processed data to Parquet...")
    
    df_rows.to_parquet(CACHE_DIR / 'rows.parquet', index=False)
    df_trips.to_parquet(CACHE_DIR / 'trips.parquet', index=False)
    df_combined.to_parquet(CACHE_DIR / 'combined.parquet', index=False)
    df_merged.to_parquet(CACHE_DIR / 'merged.parquet', index=False)
    
    if AGG_LEVEL == 'trip':
        df_analysis.to_parquet(CACHE_DIR / 'trip_level_analysis.parquet', index=False)
        print(f"  trip_level_analysis.parquet: {(CACHE_DIR / 'trip_level_analysis.parquet').stat().st_size / 1e6:.2f} MB")
    
    print(f"✓ Data cached to {CACHE_DIR}/")
    print(f"  rows.parquet: {(CACHE_DIR / 'rows.parquet').stat().st_size / 1e6:.2f} MB")
    print(f"  trips.parquet: {(CACHE_DIR / 'trips.parquet').stat().st_size / 1e6:.2f} MB")
    print(f"  combined.parquet: {(CACHE_DIR / 'combined.parquet').stat().st_size / 1e6:.2f} MB")
    print(f"  merged.parquet: {(CACHE_DIR / 'merged.parquet').stat().st_size / 1e6:.2f} MB")
    
    
    # ## 15. Actionable Insights and Key Findings
    # 
    # Generate actionable insights from feature rankings and patterns.
    
    print("\n" + "=" * 80)
    print("ACTIONABLE INSIGHTS")
    print("=" * 80)
    
    insights = []
    
    if feature_rankings and primary_target:
        top_features_list = feature_rankings[primary_target].head(10)
        
        print(f"\nTop 10 features impacting {primary_target}:")
        for idx, row in top_features_list.iterrows():
            feat = row['feature']
            corr = row['spearman_r']
            direction = "positively" if corr > 0 else "negatively"
            print(f"  {idx+1}. {feat}: ρ={corr:.3f} ({direction} correlated)")
        
        # Generate specific insights based on top features
        print("\n--- Operational Recommendations ---")
        
        # Check for idle-related features
        idle_features = [f for f in top_features_list['feature'].tolist() 
                        if 'idle' in f.lower() or 'stops' in f.lower()]
        if idle_features:
            print("\n• IDLE TIME & STOPS:")
            print("  - High idle time/stops correlate with lower efficiency")
            print("  - ACTION: Optimize route planning to minimize stops")
            print("  - ACTION: Review delivery schedules to reduce waiting time")
            print("  - ACTION: Implement driver training on minimizing unnecessary stops")
            insights.append({
                'category': 'Idle Management',
                'finding': 'High idle time correlates with lower efficiency',
                'action': 'Optimize routes, schedules, and driver behavior to minimize stops'
            })
        
        # Check for speed-related features
        speed_features = [f for f in top_features_list['feature'].tolist() 
                         if 'speed' in f.lower() and f not in EXCLUDE_FROM_ANALYSIS]
        if speed_features:
            print("\n• SPEED PATTERNS:")
            print("  - Speed variability impacts efficiency")
            print("  - ACTION: Encourage smooth acceleration/deceleration")
            print("  - ACTION: Use adaptive cruise control where available")
            print("  - ACTION: Identify and avoid congested route segments")
            insights.append({
                'category': 'Speed Management',
                'finding': 'Speed variability impacts efficiency',
                'action': 'Driver coaching, cruise control, route optimization'
            })
        
        # Check for temporal patterns
        if 'is_peak_hour' in top_features_list['feature'].tolist() or \
           'hour_of_day' in top_features_list['feature'].tolist():
            print("\n• TEMPORAL PATTERNS:")
            print("  - Time of day significantly affects efficiency")
            print("  - ACTION: Shift departure times to avoid peak congestion")
            print("  - ACTION: Use dynamic routing based on time-of-day traffic")
            print("  - ACTION: Prioritize off-peak deliveries for efficiency-sensitive loads")
            insights.append({
                'category': 'Scheduling',
                'finding': 'Peak hours reduce efficiency',
                'action': 'Time-shift operations and use dynamic routing'
            })
        
        # Check for route directness
        if 'route_directness' in top_features_list['feature'].tolist():
            directness_corr = top_features_list[top_features_list['feature'] == 'route_directness']['spearman_r'].values[0]
            if directness_corr > 0:
                print("\n• ROUTE OPTIMIZATION:")
                print("  - More direct routes improve efficiency")
                print("  - ACTION: Review routing algorithms for suboptimal paths")
                print("  - ACTION: Update map data for truck-specific restrictions")
                print("  - ACTION: Consider real-time traffic integration")
                insights.append({
                    'category': 'Route Planning',
                    'finding': 'Indirect routes reduce efficiency',
                    'action': 'Improve routing algorithms and map data quality'
                })
        
        # Check for signal reliability
        if 'signal_reliability' in top_features_list['feature'].tolist():
            print("\n• DATA QUALITY:")
            print("  - GPS signal quality affects measurement accuracy")
            print("  - ACTION: Audit vehicles with low signal reliability")
            print("  - ACTION: Check antenna placement and hardware condition")
            print("  - ACTION: Exclude low-reliability periods from KPI calculations")
            insights.append({
                'category': 'Data Quality',
                'finding': 'GPS signal issues affect measurement',
                'action': 'Hardware maintenance and data filtering'
            })
        
        print("\n" + "-" * 80)
        
        # Save insights to CSV
        if insights:
            insights_df = pd.DataFrame(insights)
            insights_df.to_csv(f'{OUTPUT_DIR}/{DATASET_NAME}/summaries/actionable_insights.csv', index=False)
            print(f"✓ Actionable insights saved to {OUTPUT_DIR}/{DATASET_NAME}/summaries/actionable_insights.csv")
    
    
    # ## 16. Summary
    # 
    # Print final summary of outputs.
    
    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)
    
    print(f"\n✓ Analysis level: {AGG_LEVEL}")
    print(f"✓ Primary target: {primary_target if primary_target else 'N/A'}")
    print(f"✓ Top features identified: {PLOT_TOP_K}")
    print(f"✓ Outliers handled: {OUTLIER_CONFIG['global_quantile_clip']}")
    
    print(f"\nAll outputs saved to `{OUTPUT_DIR}/{DATASET_NAME}/` directory:")
    print(f"  - Feature rankings: summaries/*_feature_rankings.csv")
    print(f"  - Univariate plots: univariate/*.svg")
    print(f"  - Correlation heatmaps: correlations/*.svg")
    print(f"  - Target mapping plots: target/*.svg")
    print(f"  - Geospatial maps: maps/*.html")
    print(f"  - Actionable insights: summaries/actionable_insights.csv")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    eda('extracted_data')


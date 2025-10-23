import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time as time_module

# Load model artifacts
try:
    artifacts = joblib.load('model_artifacts.joblib')
    print(f"✓ Model loaded: {artifacts['model_version']}")
    print(f"  Test MAE: {artifacts['test_mae']:.4f}")
    print(f"  Test R²: {artifacts['test_r2']:.4f}")
    print(f"  Features: {artifacts['feature_count']}")
except FileNotFoundError:
    print("ERROR: model_artifacts.joblib not found. Please train the model first.")
    artifacts = None

class TripRequest(BaseModel):
    """
    Request payload for route efficiency prediction.
    All fields represent a completed trip (post-trip scoring mode).
    """
    vehicle_id: str = Field(..., description="Vehicle identifier (e.g., 'ABA0048')")
    timestamp_start: str = Field(..., description="Trip start time in ISO format (e.g., '2025-08-01T08:01:19Z')")
    lat_start: float = Field(..., description="Starting latitude", ge=-90, le=90)
    lon_start: float = Field(..., description="Starting longitude", ge=-180, le=180)
    lat_end: float = Field(..., description="Ending latitude", ge=-90, le=90)
    lon_end: float = Field(..., description="Ending longitude", ge=-180, le=180)
    Trip_Distance_km: float = Field(..., description="Total trip distance in km", gt=0)
    Trip_Duration_min: float = Field(..., description="Total trip duration in minutes", gt=0)
    Idle_Time_min: float = Field(..., description="Idle time in minutes", ge=0)
    Moving_Time_min: float = Field(..., description="Moving time in minutes", ge=0)
    Avg_Speed: float = Field(..., description="Average speed in km/h", ge=0)
    Idle_Percentage: float = Field(..., description="Idle time percentage (0-100)", ge=0, le=100)
    Age_of_Truck_months: float = Field(..., description="Age of the truck in months", ge=0)
    
    # Optional telemetry aggregates (for richer predictions)
    speed_mean: Optional[float] = Field(None, description="Mean speed from telemetry")
    speed_median: Optional[float] = Field(None, description="Median speed from telemetry")
    speed_p25: Optional[float] = Field(None, description="25th percentile speed")
    speed_p75: Optional[float] = Field(None, description="75th percentile speed")
    speed_p90: Optional[float] = Field(None, description="90th percentile speed")
    speed_std: Optional[float] = Field(None, description="Speed standard deviation")
    stop_count: Optional[int] = Field(None, description="Number of stops (speed < 3 km/h)")
    overspeed_count: Optional[int] = Field(None, description="Overspeeding events (speed > 90 km/h)")
    idle_event_count: Optional[int] = Field(None, description="Idle event count")
    moving_event_count: Optional[int] = Field(None, description="Moving event count")

class PredictionResponse(BaseModel):
    predicted_route_efficiency: float = Field(..., description="Predicted route efficiency score (0-1)")
    model_version: str = Field(..., description="Model version used for prediction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    vehicle_id: str = Field(..., description="Vehicle ID from request")

app = FastAPI(
    title='Route Efficiency Prediction API',
    version='2.0',
    description='Predicts route efficiency for completed truck trips using LightGBM model'
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in km"""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))

def engineer_features(request: TripRequest, artifacts: dict) -> pd.DataFrame:
    """
    Reproduce the exact feature engineering pipeline from training.
    Returns a DataFrame with features in the exact order expected by the model.
    """
    eps = artifacts['epsilon']
    
    # Parse timestamp
    ts = pd.to_datetime(request.timestamp_start)
    hour_of_day = ts.hour
    day_of_week = ts.dayofweek
    
    # Geographic features
    haversine_km = haversine(request.lat_start, request.lon_start, 
                             request.lat_end, request.lon_end)
    bearing = calculate_bearing(request.lat_start, request.lon_start,
                               request.lat_end, request.lon_end)
    delta_lat = request.lat_end - request.lat_start
    delta_lon = request.lon_end - request.lon_start
    
    # Time features (cyclical)
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Distance bucket
    distance_bucket = pd.cut([request.Trip_Distance_km], 
                            bins=artifacts['bucket_bins'], 
                            labels=artifacts['bucket_labels'])[0]
    
    # Vehicle encoding
    vehicle_means = artifacts['vehicle_means']
    global_mean = artifacts['global_mean']
    vehicle_id_encoded = vehicle_means.get(request.vehicle_id, global_mean)
    
    # Zone clustering
    zone_start_id = artifacts['kmeans_start'].predict([[request.lat_start, request.lon_start]])[0]
    zone_end_id = artifacts['kmeans_end'].predict([[request.lat_end, request.lon_end]])[0]
    
    # Telemetry aggregates (use provided values or compute defaults)
    if request.speed_mean is not None:
        speed_mean = request.speed_mean
        speed_median = request.speed_median or speed_mean
        speed_p25 = request.speed_p25 or speed_mean * 0.8
        speed_p75 = request.speed_p75 or speed_mean * 1.2
        speed_p90 = request.speed_p90 or speed_mean * 1.4
        speed_std = request.speed_std or speed_mean * 0.2
    else:
        # Fallback: use trip averages
        speed_mean = request.Avg_Speed
        speed_median = request.Avg_Speed
        speed_p25 = request.Avg_Speed * 0.8
        speed_p75 = request.Avg_Speed * 1.2
        speed_p90 = request.Avg_Speed * 1.4
        speed_std = request.Avg_Speed * 0.2
    
    speed_iqr = speed_p75 - speed_p25
    speed_cv = speed_std / (speed_mean + eps)
    
    # Operational ratios (from original trips-only model)
    detour_ratio = request.Trip_Distance_km / (haversine_km + eps)
    idle_ratio = request.Idle_Time_min / (request.Trip_Duration_min + eps)
    moving_ratio = request.Moving_Time_min / (request.Trip_Duration_min + eps)
    idle_per_km = request.Idle_Time_min / (request.Trip_Distance_km + eps)
    
    # Stop/idle/overspeed counts (use provided or estimate)
    stop_count = request.stop_count if request.stop_count is not None else 0
    overspeed_count = request.overspeed_count if request.overspeed_count is not None else 0
    idle_event_count = request.idle_event_count if request.idle_event_count is not None else 0
    moving_event_count = request.moving_event_count if request.moving_event_count is not None else 0
    total_rows = max(stop_count + moving_event_count, 1)
    
    # Normalized rates
    stop_count_per_100km = stop_count / (request.Trip_Distance_km / 100 + eps)
    overspeed_count_per_100km = overspeed_count / (request.Trip_Distance_km / 100 + eps)
    idle_events_per_100km = idle_event_count / (request.Trip_Distance_km / 100 + eps)
    stop_count_per_hour = stop_count / (request.Trip_Duration_min / 60 + eps)
    overspeed_count_per_hour = overspeed_count / (request.Trip_Duration_min / 60 + eps)
    
    idle_percentage_rows = idle_event_count / (total_rows + eps) if total_rows > 0 else 0
    moving_percentage_rows = moving_event_count / (total_rows + eps) if total_rows > 0 else 1
    
    # Build numeric features dict (must match training order)
    feature_dict = {
        'Trip_Distance_km': request.Trip_Distance_km,
        'Trip_Duration_min': request.Trip_Duration_min,
        'Avg_Speed': request.Avg_Speed,
        'Idle_Time_min': request.Idle_Time_min,
        'Moving_Time_min': request.Moving_Time_min,
        'Idle_Percentage': request.Idle_Percentage,
        'haversine_km': haversine_km,
        'detour_ratio': detour_ratio,
        'bearing': bearing,
        'delta_lat': delta_lat,
        'delta_lon': delta_lon,
        'lat_start': request.lat_start,
        'lon_start': request.lon_start,
        'lat_end': request.lat_end,
        'lon_end': request.lon_end,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'idle_ratio': idle_ratio,
        'moving_ratio': moving_ratio,
        'idle_per_km': idle_per_km,
        'Age_of_Truck_months': request.Age_of_Truck_months,
        'vehicle_id_encoded': vehicle_id_encoded,
        'speed_mean': speed_mean,
        'speed_median': speed_median,
        'speed_p25': speed_p25,
        'speed_p75': speed_p75,
        'speed_p90': speed_p90,
        'speed_std': speed_std,
        'speed_min': speed_p25 * 0.5,  # estimate
        'speed_max': speed_p90 * 1.2,  # estimate
        'speed_iqr': speed_iqr,
        'speed_cv': speed_cv,
        'stop_count': stop_count,
        'total_rows': total_rows,
        'overspeed_count': overspeed_count,
        'idle_event_count': idle_event_count,
        'idle_percentage_rows': idle_percentage_rows,
        'moving_event_count': moving_event_count,
        'moving_percentage_rows': moving_percentage_rows,
        'stop_count_per_100km': stop_count_per_100km,
        'overspeed_count_per_100km': overspeed_count_per_100km,
        'idle_events_per_100km': idle_events_per_100km,
        'stop_count_per_hour': stop_count_per_hour,
        'overspeed_count_per_hour': overspeed_count_per_hour,
    }
    
    # One-hot encode categorical features
    categorical_dict = {
        'zone_start_id': zone_start_id,
        'zone_end_id': zone_end_id,
        'distance_bucket': distance_bucket
    }
    
    # Create temp dataframe for get_dummies
    temp_df = pd.DataFrame([categorical_dict])
    encoded = pd.get_dummies(temp_df, columns=['zone_start_id', 'zone_end_id', 'distance_bucket'],
                            prefix=['zone_start_id', 'zone_end_id', 'distance_bucket'])
    
    # Align to training dummy columns
    dummy_columns = artifacts['dummy_columns']
    for col in dummy_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[dummy_columns]
    
    # Combine numeric + categorical in exact training order
    numeric_features = artifacts['numeric_features']
    numeric_df = pd.DataFrame([feature_dict])[numeric_features]
    X = pd.concat([numeric_df.reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    
    return X

@app.get('/')
def root():
    if artifacts is None:
        return {'status': 'error', 'message': 'Model not loaded'}
    return {
        'status': 'ok',
        'model_version': artifacts['model_version'],
        'usage': 'POST /predict',
        'test_mae': artifacts['test_mae'],
        'test_r2': artifacts['test_r2']
    }

@app.post('/predict', response_model=PredictionResponse)
def predict(request: TripRequest):
    """
    Predict route efficiency for a completed trip.
    """
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time_module.time()
    
    try:
        # Engineer features
        X = engineer_features(request, artifacts)
        
        # Predict
        model = artifacts['model']
        prediction = model.predict(X)[0]
        
        # Clip to valid range
        prediction = float(np.clip(prediction, 0, 1))
        
        inference_time_ms = (time_module.time() - start_time) * 1000
        
        return PredictionResponse(
            predicted_route_efficiency=prediction,
            model_version=artifacts['model_version'],
            inference_time_ms=inference_time_ms,
            vehicle_id=request.vehicle_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get('/health')
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy' if artifacts is not None else 'unhealthy',
        'model_loaded': artifacts is not None
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("Starting Route Efficiency Prediction API")
    print("="*80)
    print(f"Server will run at: http://localhost:8000")
    print(f"Interactive docs: http://localhost:8000/docs")
    print(f"Alternative docs: http://localhost:8000/redoc")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

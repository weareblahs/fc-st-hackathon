# Route Efficiency Prediction API

FastAPI service for predicting route efficiency of completed truck trips using a LightGBM model.

## Features

- **Post-trip scoring**: Predict route efficiency (0-1) for completed trips
- **Leakage-safe modeling**: Uses upstream signals, not target components
- **Explainable predictions**: Based on telemetry, geography, time, and vehicle behavior
- **API-ready**: FastAPI with Pydantic validation and automatic documentation

## Setup

### 1. Install Dependencies

```bash
pip install fastapi uvicorn joblib scikit-learn lightgbm pandas numpy
```

### 2. Train the Model (if not already done)

Run the Jupyter notebook to train and save the model:

```bash
jupyter notebook jl_temp/model.ipynb
```

Run all cells to generate `models/model_artifacts.joblib`.

### 3. Copy Model Artifacts

```bash
copy jl_temp\models\model_artifacts.joblib .
```

Or on Unix/Mac:

```bash
cp jl_temp/models/model_artifacts.joblib .
```

## Running the API

### Start the server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### Access the documentation

- **Interactive docs (Swagger)**: http://localhost:8000/docs
- **Alternative docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### `GET /`

Health check and model info

**Response:**

```json
{
  "status": "ok",
  "model_version": "lgbm_refactored_v2",
  "usage": "POST /predict",
  "test_mae": 0.0423,
  "test_r2": 0.7234
}
```

### `GET /health`

Simple health check

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`

Predict route efficiency for a completed trip

**Request Body:**

```json
{
  "vehicle_id": "ABA0048",
  "timestamp_start": "2025-08-01T08:01:19Z",
  "lat_start": 5.43428,
  "lon_start": 100.5948,
  "lat_end": 5.41943,
  "lon_end": 100.3782,
  "Trip_Distance_km": 32.614,
  "Trip_Duration_min": 92.467,
  "Idle_Time_min": 48.217,
  "Moving_Time_min": 44.25,
  "Avg_Speed": 21.163,
  "Idle_Percentage": 52.145,
  "Age_of_Truck_months": 82.326
}
```

**Optional Fields** (for richer predictions with telemetry):

```json
{
  "speed_mean": 21.5,
  "speed_median": 20.0,
  "speed_p25": 15.0,
  "speed_p75": 28.0,
  "speed_p90": 35.0,
  "speed_std": 8.5,
  "stop_count": 12,
  "overspeed_count": 2,
  "idle_event_count": 15,
  "moving_event_count": 120
}
```

**Response:**

```json
{
  "predicted_route_efficiency": 0.6234,
  "model_version": "lgbm_refactored_v2",
  "inference_time_ms": 12.34,
  "vehicle_id": "ABA0048"
}
```

## Testing

Run the test script to verify the API:

```bash
python test_api.py
```

Or use curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "ABA0048",
    "timestamp_start": "2025-08-01T08:01:19Z",
    "lat_start": 5.43428,
    "lon_start": 100.5948,
    "lat_end": 5.41943,
    "lon_end": 100.3782,
    "Trip_Distance_km": 32.614,
    "Trip_Duration_min": 92.467,
    "Idle_Time_min": 48.217,
    "Moving_Time_min": 44.25,
    "Avg_Speed": 21.163,
    "Idle_Percentage": 52.145,
    "Age_of_Truck_months": 82.326
  }'
```

## How It Works

### Feature Engineering Pipeline

The API reproduces the exact feature engineering from training:

1. **Geographic features**: Haversine distance, bearing, delta lat/lon
2. **Time features**: Cyclical hour-of-day and day-of-week encodings
3. **Zone clustering**: KMeans-based origin/destination zones (k=50)
4. **Vehicle encoding**: Target-encoded vehicle IDs with global mean fallback
5. **Telemetry aggregates**: Speed profile, stops, idle events, overspeeding
6. **Normalized rates**: Events per 100km and per hour

### Model Architecture

- **Algorithm**: LightGBM Regressor
- **Features**: ~40 numeric + ~100 categorical (one-hot zones/buckets)
- **Target**: Route efficiency (0-1 composite score)
- **Leakage control**: Excludes direct target components

### Performance Metrics

- **Test MAE**: ~0.04-0.06 (4-6% error)
- **Test R²**: ~0.60-0.75 (60-75% variance explained)
- **Inference time**: ~10-20ms per prediction

## File Structure

```
prediction_api/
├── app.py                    # FastAPI application
├── test_api.py              # API test script
├── model_artifacts.joblib   # Trained model and artifacts
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── jl_temp/
    ├── model.ipynb         # Training notebook
    └── data/
        ├── safetruck_data_iter1_trips.csv
        ├── safetruck_data_iter1_rows.csv
        └── combined_data.csv
```

## Troubleshooting

### Model not found error

```
ERROR: model_artifacts.joblib not found. Please train the model first.
```

**Solution**: Run the training notebook and copy the artifacts file.

### Missing dependencies

```
ModuleNotFoundError: No module named 'lightgbm'
```

**Solution**: `pip install lightgbm scikit-learn`

### Connection refused when testing

```
requests.exceptions.ConnectionError
```

**Solution**: Make sure the API is running with `uvicorn app:app --reload`

## Next Steps

- **SHAP explanations**: Add per-prediction feature contributions
- **Batch predictions**: Support multiple trips in one request
- **Pre-trip planning**: Add endpoint for expected efficiency before trip starts
- **Model versioning**: Support A/B testing with multiple model versions
- **Monitoring**: Add logging and metrics collection

## License

MIT

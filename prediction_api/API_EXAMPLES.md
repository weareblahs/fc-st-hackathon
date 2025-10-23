# Sample API Test Commands

## 1. Check API Status

```bash
curl http://localhost:8000/
```

## 2. Health Check

```bash
curl http://localhost:8000/health
```

## 3. Basic Prediction (Minimal Fields)

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

## 4. Prediction with Telemetry (Rich Data)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_id": "ABA0048",
    "timestamp_start": "2025-08-01T11:52:12Z",
    "lat_start": 5.41943,
    "lon_start": 100.3781,
    "lat_end": 5.4001,
    "lon_end": 100.5934,
    "Trip_Distance_km": 37.978,
    "Trip_Duration_min": 198.65,
    "Idle_Time_min": 144.65,
    "Moving_Time_min": 54.0,
    "Avg_Speed": 11.471,
    "Idle_Percentage": 72.817,
    "Age_of_Truck_months": 82.326,
    "speed_mean": 12.5,
    "speed_median": 10.0,
    "speed_p25": 5.0,
    "speed_p75": 18.0,
    "speed_p90": 25.0,
    "speed_std": 7.2,
    "stop_count": 25,
    "overspeed_count": 0,
    "idle_event_count": 30,
    "moving_event_count": 150
  }'
```

## PowerShell Equivalent

### Basic Prediction

```powershell
$body = @{
    vehicle_id = "ABA0048"
    timestamp_start = "2025-08-01T08:01:19Z"
    lat_start = 5.43428
    lon_start = 100.5948
    lat_end = 5.41943
    lon_end = 100.3782
    Trip_Distance_km = 32.614
    Trip_Duration_min = 92.467
    Idle_Time_min = 48.217
    Moving_Time_min = 44.25
    Avg_Speed = 21.163
    Idle_Percentage = 52.145
    Age_of_Truck_months = 82.326
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```

## Python Equivalent

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
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
)

print(response.json())
# Output:
# {
#   "predicted_route_efficiency": 0.6234,
#   "model_version": "lgbm_refactored_v2",
#   "inference_time_ms": 12.34,
#   "vehicle_id": "ABA0048"
# }
```

## Expected Response Format

```json
{
  "predicted_route_efficiency": 0.6234,
  "model_version": "lgbm_refactored_v2",
  "inference_time_ms": 12.34,
  "vehicle_id": "ABA0048"
}
```

## Notes

- Route efficiency is a score between 0 and 1 (higher is better)
- All coordinate fields must be valid lat/lon values
- Optional telemetry fields will improve prediction accuracy
- Unknown vehicles will use global mean encoding (fallback)

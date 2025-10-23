"""
Test script for Route Efficiency Prediction API
Demonstrates how to call the /predict endpoint with sample data
"""

import requests
import json

# API endpoint (adjust if running on different host/port)
BASE_URL = "http://localhost:8000"

# Sample trip data (from test set)
sample_trip = {
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

# Optional: Include telemetry aggregates for richer predictions
sample_trip_with_telemetry = {
    **sample_trip,
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

def test_root():
    """Test the root endpoint"""
    print("=" * 80)
    print("Testing GET /")
    print("=" * 80)
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_health():
    """Test the health endpoint"""
    print("=" * 80)
    print("Testing GET /health")
    print("=" * 80)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict_basic():
    """Test prediction with basic trip data (no telemetry)"""
    print("=" * 80)
    print("Testing POST /predict (Basic Trip Data)")
    print("=" * 80)
    response = requests.post(f"{BASE_URL}/predict", json=sample_trip)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"\n✓ Predicted Route Efficiency: {result['predicted_route_efficiency']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_predict_with_telemetry():
    """Test prediction with telemetry aggregates"""
    print("=" * 80)
    print("Testing POST /predict (With Telemetry Aggregates)")
    print("=" * 80)
    response = requests.post(f"{BASE_URL}/predict", json=sample_trip_with_telemetry)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"\n✓ Predicted Route Efficiency: {result['predicted_route_efficiency']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_multiple_trips():
    """Test predictions for multiple trips"""
    print("=" * 80)
    print("Testing Multiple Trip Predictions")
    print("=" * 80)
    
    trips = [
        sample_trip,
        {**sample_trip, "Trip_Distance_km": 15.0, "Idle_Time_min": 10.0},
        {**sample_trip, "Trip_Distance_km": 100.0, "Idle_Time_min": 5.0},
    ]
    
    for i, trip in enumerate(trips, 1):
        response = requests.post(f"{BASE_URL}/predict", json=trip)
        if response.status_code == 200:
            result = response.json()
            print(f"Trip {i}: Distance={trip['Trip_Distance_km']}km, "
                  f"Efficiency={result['predicted_route_efficiency']:.4f}, "
                  f"Time={result['inference_time_ms']:.2f}ms")
        else:
            print(f"Trip {i}: Error - {response.status_code}")
    print()

if __name__ == "__main__":
    print("\n🚀 Starting API Tests\n")
    
    try:
        test_root()
        test_health()
        test_predict_basic()
        test_predict_with_telemetry()
        test_multiple_trips()
        
        print("=" * 80)
        print("✅ All tests completed!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running with:")
        print("  uvicorn app:app --reload")
        print()

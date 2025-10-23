"""
Quick test to verify the API is working
Run this AFTER starting the server with: python app.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("\n" + "="*80)
print("Testing Route Efficiency API")
print("="*80)

# Test 1: Root endpoint
print("\n1. Testing GET / (Model Info)")
print("-" * 40)
try:
    response = requests.get(f"{BASE_URL}/", timeout=5)
    print(f"✓ Status: {response.status_code}")
    data = response.json()
    print(f"✓ Model Version: {data.get('model_version')}")
    print(f"✓ Test MAE: {data.get('test_mae')}")
    print(f"✓ Test R²: {data.get('test_r2')}")
except requests.exceptions.ConnectionError:
    print("✗ ERROR: Could not connect to API")
    print("  Make sure the server is running with: python app.py")
    exit(1)
except Exception as e:
    print(f"✗ ERROR: {e}")
    exit(1)

# Test 2: Prediction
print("\n2. Testing POST /predict")
print("-" * 40)

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

try:
    response = requests.post(f"{BASE_URL}/predict", json=sample_trip, timeout=5)
    print(f"✓ Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Predicted Route Efficiency: {result['predicted_route_efficiency']:.4f}")
        print(f"✓ Model Version: {result['model_version']}")
        print(f"✓ Inference Time: {result['inference_time_ms']:.2f}ms")
        print(f"✓ Vehicle ID: {result['vehicle_id']}")
    else:
        print(f"✗ ERROR: {response.status_code}")
        print(f"  {response.text}")
except Exception as e:
    print(f"✗ ERROR: {e}")

# Test 3: Health check
print("\n3. Testing GET /health")
print("-" * 40)
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"✓ Status: {response.status_code}")
    data = response.json()
    print(f"✓ Health Status: {data.get('status')}")
    print(f"✓ Model Loaded: {data.get('model_loaded')}")
except Exception as e:
    print(f"✗ ERROR: {e}")

print("\n" + "="*80)
print("✅ All tests completed!")
print("="*80)
print("\nTo view interactive API docs, open:")
print("  http://localhost:8000/docs")
print("="*80 + "\n")

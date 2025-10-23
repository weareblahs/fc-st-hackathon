from flask import Flask, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from combine import combine
from cleaning import cleaning
import os
app = Flask(__name__)

# HELPER FUNCTIONS START
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
# HELPER FUNCTIONS END

@app.route('/')
def home():
    """Root endpoint that returns an example message."""
    return jsonify({
        "message": "Hello from Flask API!",
        "status": "success"
    })

@app.route('/generate_eda')
def eda_gen():
    """Cleans the data, ."""
    # get uuid
    uuid = 'extracted_data'
    warnings.filterwarnings('ignore')
    from math import radians, cos, sin, asin, sqrt

    print("Libraries imported successfully!")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    # STEP 1: combine
    # Get the absolute path to the data directory relative to this file
    combine(uuid)
    cleaning(uuid)
    return jsonify({
        "message": "Hello from Flask API!",
        "status": "success"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

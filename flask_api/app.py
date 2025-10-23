from flask import Flask, jsonify, redirect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from combine import combine
from cleaning import cleaning
import os
from pathlib import Path
from flask_cors import CORS
from eda import eda
from fte import fte
app = Flask(__name__, static_folder='data')
CORS(app)
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
    fte()
    eda(uuid)
    
    # http://localhost:5000/data/extracted_data/deliverables/trip_level/correlations/pearson_correlation.svg
    # recursive search under the static dir
    path = Path(f'data/{uuid}')
    return jsonify({
        "message": "Successful data retrieval!",
        "images": [str(p).replace('\\','/') for p in list(path.rglob("*.svg"))]
    })
    # return redirect('https://i.pinimg.com/736x/80/66/9e/80669efdaef7dead858fe7ce1d12ba9f.jpg')

@app.route('/<uuid>/avg_speed')
def check_fe(uuid):
    data = pd.read_csv(f'data/{uuid}/safetruck_data_iter1_trips.csv')
    print(data.columns)
    exp_data = pd.DataFrame({
        'date': [i[0].split("-")[2] for i in data['timestamp_start'].str.split(' ')],
        'avg_speed': data[
            'Avg_Speed'
        ]
    })
    d2 = exp_data.groupby('date')['avg_speed'].mean().round(2).reset_index()
    d2.columns = ['time', 'value']
    return d2.to_json(orient='records', date_format='iso')

@app.route('/<uuid>/top_drivers')
def top10(uuid):
    data = pd.read_csv(f'data/{uuid}/df_daily.csv')
    d2 = data.groupby('vehicle_id')['Total_Moving_min'].sum().reset_index().sort_values(by='Total_Moving_min', ascending=False).reset_index().reset_index()[['level_0', 'vehicle_id', 'Total_Moving_min']]
    d2.columns = ['rank', 'vehicle', 'time']
    return d2.to_json(orient='records')

    # return "test"

def categorize_direction(degree):
    if 1 <= degree <= 90:
        return 'north'
    elif 91 <= degree <= 180:
        return 'east'
    elif 181 <= degree <= 270:
        return 'south'
    elif 271 <= degree <= 360:
        return 'west'
    return None

@app.route('/<uuid>/direction_analysis')
def direction(uuid):
    data = pd.read_csv(f'data/{uuid}/combined_data.csv')
    d2 = data.groupby('Direction').count()['Timestamp'].reset_index()
    d2 = d2[d2['Direction'] != 0]
    d2['Cardinal'] = d2['Direction'].apply(categorize_direction)
    d3 = d2.groupby('Cardinal')['Timestamp'].sum().reset_index()
    d3.columns = ['position', 'data']
    d3 = d3.set_index('position').reindex(['north', 'east', 'south', 'west']).reset_index()
    d3_json = d3.to_dict(orient='records')
    d3_json[0]['fill'] = "var(--color-n)"
    d3_json[1]['fill'] = "var(--color-s)"
    d3_json[2]['fill'] = "var(--color-e)"
    d3_json[3]['fill'] = "var(--color-w)"
    return jsonify(d3_json)

@app.route('/<uuid>/list_of_trucks')
def list_of_trucks(uuid):
    data = pd.read_csv(f'data/{uuid}/df_daily.csv')
    print()
    return list((data['vehicle_id'].unique()))

@app.route('/<uuid>/<truck>/pos')
def get_pos(uuid, truck):
    d = pd.read_csv(f'data/{uuid}/combined_data.csv', usecols=['Latitude', 'Longitude', 'CarNumberPlate'])
    data = d[d['CarNumberPlate'] == truck][['Latitude', 'Longitude']].drop_duplicates()
    
    # Filter out points that are too close to the previous point using vectorized operations
    threshold = 0.005  # Approximately 50 meters
    
    if len(data) == 0:
        return []
    
    # Convert to numpy array for faster operations
    coords = data.values
    
    # Calculate differences between consecutive points
    diff = np.abs(np.diff(coords, axis=0))
    
    # Keep points where either lat or lon difference exceeds threshold
    mask = np.any(diff > threshold, axis=1)
    
    # Always include the first point, then filter based on mask
    indices = np.concatenate([[0], np.where(mask)[0] + 1])
    
    filtered_data = coords[indices].tolist()
    
    return filtered_data

@app.route('/<uuid>/<truck>/pos_idle')
def get_pos_idle(uuid, truck):
    d = pd.read_csv(f'data/{uuid}/safetruck_data_iter1_rows.csv', usecols=['lat', 'lon', 'is_idle', 'vehicle_id'])
    data = d[(d['vehicle_id'] == truck) & (d['is_idle'] == True)][['lat', 'lon']].drop_duplicates()
    print(data)
    # Filter out points that are too close to the previous point using vectorized operations
    threshold = 0.001  # Approximately 111 meters
    
    if len(data) == 0:
        return []
    
    # Convert to numpy array for faster operations
    coords = data.values
    
    # Calculate differences between consecutive points
    diff = np.abs(np.diff(coords, axis=0))
    
    # Keep points where either lat or lon difference exceeds threshold
    mask = np.any(diff > threshold, axis=1)
    
    # Always include the first point, then filter based on mask
    indices = np.concatenate([[0], np.where(mask)[0] + 1])
    
    filtered_data = coords[indices].tolist()
    
    return filtered_data

@app.route('/<uuid>/<truck>/stats')
def get_stats(uuid, truck):
    d = pd.read_csv(f'data/{uuid}/df_daily.csv', usecols=['vehicle_id', 'Total_Distance_km', 'Total_Idle_min'])
    # DATA SOURCE: https://www.webfleet.com/en_gb/webfleet/industries/transport/fuel-efficiency/
    # assuming 100km = 35l, 1l = 2.86km (static data!)
    # diesel price source: https://data.gov.my/data-catalogue/fuelprice = 2.89/litre
    return jsonify(
        {
            'month_total_distance': d[d['vehicle_id'] == truck]['Total_Distance_km'].sum().round(2),
            'fuel_liters': ((35 * (d[d['vehicle_id'] == truck]['Total_Distance_km'].sum().round(2) / 100)).round(2)),
            'total_wasted': (((35 * (d[d['vehicle_id'] == truck]['Total_Distance_km'].sum().round(2) / 100)).round(2)) * 2.89).round(2),
            'day_total_distance_avg': d[(d['vehicle_id'] == truck) & (d['Total_Distance_km'] != 0.0)]['Total_Distance_km'].mean().round(2),
            'total_idle_min': d[(d['vehicle_id'] == truck) & (d['Total_Idle_min'] != 0.0)]['Total_Idle_min'].sum().round(0)
        }
    )

@app.route('/<uuid>/<truck>/data_for_predict')
def get_predict_data(uuid, truck):
    d = pd.read_csv(f'data/{uuid}/safetruck_data_iter1_trips.csv', usecols=['vehicle_id', 'timestamp_start', 'lat_start', 'lon_start', 'lat_end', 'lon_end', 'Trip_Distance_km', 'Trip_Duration_min', 'Idle_Time_min', 'Moving_Time_min', 'Avg_Speed', 'Idle_Percentage', 'Age_of_Truck_months'])
    d_nonzero = d[(d['vehicle_id'] == truck) & (d['lat_start'] != 0)& (d['lat_end'] != 0)& (d['lon_start'] != 0)& (d['lon_end'] != 0)]
    return jsonify({
    "vehicle_id": truck,
    "timestamp_start": d_nonzero.tail(1)['timestamp_start'].values[0].replace(" ","T").replace("+00:00","Z"),
    "lat_start": d_nonzero.tail(1)['lat_start'].values[0],
    "lon_start": d_nonzero.tail(1)['lon_start'].values[0], 
    "lat_end": d_nonzero.tail(1)['lat_end'].values[0], 
    "lon_end": d_nonzero.tail(1)['lon_end'].values[0], 
    "Trip_Distance_km": d_nonzero['Trip_Distance_km'].mean(), 
    "Trip_Duration_min": d_nonzero['Trip_Duration_min'].mean(), 
    "Idle_Time_min": d_nonzero['Idle_Time_min'].mean(), 
    "Moving_Time_min": d_nonzero['Moving_Time_min'].mean(), 
    "Avg_Speed": d_nonzero['Avg_Speed'].mean(),
    "Idle_Percentage": d_nonzero['Idle_Percentage'].mean(),
    "Age_of_Truck_months": d_nonzero['Trip_Distance_km'].mean()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

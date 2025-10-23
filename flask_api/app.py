from flask import Flask, jsonify, redirect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from combine import combine
from cleaning import cleaning
import os
from pathlib import Path
from eda import eda
app = Flask(__name__, static_folder='data')

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
    # combine(uuid)
    # cleaning(uuid)
    # eda(uuid)
    
    # http://localhost:5000/data/extracted_data/deliverables/trip_level/correlations/pearson_correlation.svg
    # recursive search under the static dir
    path = Path(f'data/{uuid}/deliverables')
    return jsonify({
        "message": "Successful data retrieval!",
        "images": [str(p).replace('\\','/') for p in list(path.rglob("*.svg"))]
    })
    # return redirect('https://i.pinimg.com/736x/80/66/9e/80669efdaef7dead858fe7ce1d12ba9f.jpg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

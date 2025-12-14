"""
Flask Blueprint for Energy Consumption Forecasting
Uses LSTM model trained on UCI Household Electric Power Consumption dataset
"""

from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
import os
import gc
import threading
import time
import psutil
from datetime import datetime, timedelta

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

energy_bp = Blueprint('energy', __name__, url_prefix='/api/energy')

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Global variables for lazy loading
_model = None
_dataset = None
_scaler = None
_last_used = None
_cleanup_timer = None
_lock = threading.Lock()

# Model configuration
WINDOW_SIZE = 60  # 60 hours of historical data
HORIZON = 1  # Predict 1 hour ahead
TARGET_COLUMN = 'Global_active_power'

# Cleanup timeout (10 minutes)
CLEANUP_TIMEOUT = 600

def cleanup_resources():
    """Unload model and dataset from memory after timeout"""
    global _model, _dataset, _scaler, _cleanup_timer
    with _lock:
        if _model is not None and time.time() - _last_used > CLEANUP_TIMEOUT:
            print("Energy forecasting model inactive for 10 minutes, cleaning up...")
            _model = None
            _dataset = None
            _scaler = None
            gc.collect()
            print("Energy forecasting resources unloaded from memory")
        _cleanup_timer = None

def schedule_cleanup():
    """Schedule resource cleanup after timeout"""
    global _cleanup_timer
    if _cleanup_timer:
        _cleanup_timer.cancel()
    _cleanup_timer = threading.Timer(CLEANUP_TIMEOUT, cleanup_resources)
    _cleanup_timer.daemon = True
    _cleanup_timer.start()

def load_and_prepare_dataset():
    """Load UCI dataset and prepare it for predictions (matching notebook exactly)"""
    print("Loading UCI Household Electric Power Consumption dataset...")

    # Fetch dataset from UCI repository
    dataset = fetch_ucirepo(id=235)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)

    print(f"Dataset loaded: {len(df)} rows (minute-level)")

    # Clean dataset (matching notebook exactly) - only numeric columns
    df_clean = df.copy()
    df_clean = df_clean.replace("?", np.nan)

    # Only convert numeric columns (not Date/Time)
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    print(f"Missing values in {TARGET_COLUMN} before interpolation: {df_clean[TARGET_COLUMN].isna().sum()}")

    # Linear interpolation for missing values (matching notebook)
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')

    print(f"Missing values in {TARGET_COLUMN} after interpolation: {df_clean[TARGET_COLUMN].isna().sum()}")

    # Drop rows where target column is still NaN
    df_clean = df_clean.dropna(subset=[TARGET_COLUMN])

    print(f"Shape after cleaning: {df_clean.shape}")

    # Apply winsorization (matching notebook)
    lower = df_clean[TARGET_COLUMN].quantile(0.01)
    upper = df_clean[TARGET_COLUMN].quantile(0.99)
    df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].clip(lower=lower, upper=upper)

    print(f"Winsorization applied: min={df_clean[TARGET_COLUMN].min():.3f}, max={df_clean[TARGET_COLUMN].max():.3f}")

    # Create datetime index AFTER cleaning
    # Original data is minute-level from Dec 2006 - Nov 2010
    start_date = pd.Timestamp('2006-12-16 17:24:00')
    df_clean['datetime'] = pd.date_range(start=start_date, periods=len(df_clean), freq='min')
    df_clean = df_clean.set_index('datetime')

    print(f"Datetime index created. Range: {df_clean.index.min()} to {df_clean.index.max()}")

    # CRITICAL: Resample to hourly means (matching notebook)
    df_hourly = df_clean[TARGET_COLUMN].resample('h').mean()
    df_hourly = df_hourly.dropna()

    print(f"Resampled to hourly: {len(df_hourly)} rows")
    print(f"Hourly date range: {df_hourly.index.min()} to {df_hourly.index.max()}")

    # Convert to DataFrame for consistency
    df_hourly = df_hourly.to_frame()

    return df_hourly

def fit_scaler_on_training_data(dataset):
    """Fit scaler on training portion of dataset (matching notebook exactly)"""
    # Use first 70% as training data (matching notebook's 70/15/15 split)
    train_size = int(len(dataset) * 0.70)
    train_data = dataset.iloc[:train_size]

    # Fit scaler on training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_values = train_data.values.reshape(-1, 1)
    scaler.fit(train_values)

    print(f"Scaler fit on {train_size} training samples (70% of data)")
    return scaler

def get_resources():
    """Lazy load model, dataset, and scaler"""
    global _model, _dataset, _scaler, _last_used

    with _lock:
        if _model is None or _dataset is None:
            mem_before = get_memory_usage()
            print(f"Memory before loading: {mem_before:.2f} MB")

            # Load model with custom objects for compatibility
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, 'best_model.h5')
            print(f"Loading LSTM model from {model_path}...")

            # Load with compile=False to avoid metric deserialization issues
            _model = keras.models.load_model(model_path, compile=False)

            # Recompile with standard metrics
            _model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print("✓ Model loaded successfully")

            # Load and prepare dataset
            _dataset = load_and_prepare_dataset()

            # Debug: Verify dataset has valid datetime index
            print(f"DEBUG: Dataset type: {type(_dataset)}")
            print(f"DEBUG: Dataset shape: {_dataset.shape}")
            print(f"DEBUG: Dataset index type: {type(_dataset.index)}")
            print(f"DEBUG: Dataset index min: {_dataset.index.min()}")
            print(f"DEBUG: Dataset index max: {_dataset.index.max()}")
            print(f"DEBUG: First few timestamps: {_dataset.index[:3].tolist()}")

            # Fit scaler on training data (same as notebook training)
            _scaler = fit_scaler_on_training_data(_dataset)

            mem_after = get_memory_usage()
            mem_used = mem_after - mem_before
            print(f"Memory after loading: {mem_after:.2f} MB")
            print(f"Resources use: {mem_used:.2f} MB")

        _last_used = time.time()
        schedule_cleanup()

        # Debug: Verify dataset before returning
        print(f"DEBUG: About to return dataset with index range: {_dataset.index.min()} to {_dataset.index.max()}")

        return _model, _dataset, _scaler

def predict_energy(prediction_start_date, hours_ahead):
    """
    Predict energy consumption using autoregressive forecasting

    Args:
        prediction_start_date: datetime object for when predictions start
        hours_ahead: number of hours to predict (1-24)

    Returns:
        dict with historical data, predictions, and actual values
    """
    model, dataset, scaler = get_resources()

    # Debug: Check what we received
    print(f"DEBUG predict_energy: Dataset type: {type(dataset)}")
    print(f"DEBUG predict_energy: Dataset shape: {dataset.shape}")
    print(f"DEBUG predict_energy: Dataset index type: {type(dataset.index)}")
    print(f"DEBUG predict_energy: Dataset index range: {dataset.index.min()} to {dataset.index.max()}")

    # Get the index of the prediction start
    if prediction_start_date not in dataset.index:
        # Find nearest datetime
        nearest_idx = dataset.index.get_indexer([prediction_start_date], method='nearest')[0]

        # Check if valid index was found
        if nearest_idx == -1 or nearest_idx >= len(dataset):
            raise ValueError(f"Selected date {prediction_start_date} is outside dataset range. "
                           f"Valid range: {dataset.index.min()} to {dataset.index.max()}")

        prediction_start_date = dataset.index[nearest_idx]

    start_idx = dataset.index.get_loc(prediction_start_date)

    # Verify we have enough data
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after preprocessing")

    if start_idx < 0 or start_idx >= len(dataset):
        raise ValueError(f"Invalid start index: {start_idx}")

    # Need 60 hours before prediction start
    if start_idx < WINDOW_SIZE:
        raise ValueError(f"Not enough historical data. Need at least {WINDOW_SIZE} hours before {prediction_start_date}")

    # Get historical data (60 hours before prediction start)
    historical_start_idx = start_idx - WINDOW_SIZE
    historical_data = dataset.iloc[historical_start_idx:start_idx]

    # Verify we got data
    if len(historical_data) == 0:
        raise ValueError(f"No historical data found for the selected date range")

    if len(historical_data) != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} hours of historical data, got {len(historical_data)}")

    # Get actual future data for comparison
    end_idx = start_idx + hours_ahead
    if end_idx > len(dataset):
        end_idx = len(dataset)
        hours_ahead = end_idx - start_idx

    actual_future = dataset.iloc[start_idx:end_idx]

    if len(actual_future) == 0:
        raise ValueError("No future data available for comparison at selected date")

    # Prepare data for prediction
    # Extract target column and scale using pre-fitted scaler
    # Handle both DataFrame and Series
    if isinstance(historical_data, pd.DataFrame):
        historical_values = historical_data[TARGET_COLUMN].values.reshape(-1, 1)
    else:
        historical_values = historical_data.values.reshape(-1, 1)

    # Use the scaler that was fit on training data (NOT refitting!)
    historical_scaled = scaler.transform(historical_values)

    # Autoregressive prediction
    predictions = []
    current_window = historical_scaled.copy()

    for _ in range(hours_ahead):
        # Reshape for model input (1, 60, 1)
        input_seq = current_window[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)

        # Predict next hour
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred_scaled)

        # Update window for next prediction (rolling forecast)
        current_window = np.append(current_window, [[pred_scaled]], axis=0)

    # Inverse transform predictions to original scale
    predictions_array = np.array(predictions).reshape(-1, 1)
    predictions_unscaled = scaler.inverse_transform(predictions_array).flatten()

    # Prepare response data
    # Handle both DataFrame and Series for response
    if isinstance(historical_data, pd.DataFrame):
        historical_values_list = historical_data[TARGET_COLUMN].tolist()
    else:
        historical_values_list = historical_data.tolist()

    if isinstance(actual_future, pd.DataFrame):
        actual_values_list = actual_future[TARGET_COLUMN].tolist()
    else:
        actual_values_list = actual_future.tolist()

    result = {
        'prediction_start': prediction_start_date.isoformat(),
        'hours_predicted': hours_ahead,
        'historical': {
            'timestamps': historical_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'values': historical_values_list
        },
        'predictions': {
            'timestamps': actual_future.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'values': predictions_unscaled.tolist()
        },
        'actual': {
            'timestamps': actual_future.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'values': actual_values_list
        }
    }

    return result

# API Routes

@energy_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': _model is not None,
        'dataset_loaded': _dataset is not None
    })

@energy_bp.route('/memory', methods=['GET'])
def memory_stats():
    """Get current memory usage statistics"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return jsonify({
        'total_memory_mb': mem_info.rss / 1024 / 1024,
        'model_loaded': _model is not None,
        'dataset_loaded': _dataset is not None
    })

@energy_bp.route('/dataset-info', methods=['GET'])
def dataset_info():
    """Get dataset date range information"""
    try:
        model, dataset, scaler = get_resources()

        return jsonify({
            'start_date': dataset.index.min().isoformat(),
            'end_date': dataset.index.max().isoformat(),
            'total_hours': len(dataset),
            'min_prediction_start': (dataset.index.min() + timedelta(hours=WINDOW_SIZE)).isoformat(),
            'max_prediction_start': (dataset.index.max() - timedelta(hours=24)).isoformat()
        })
    except Exception as e:
        print(f"Error in dataset_info: {str(e)}")
        return jsonify({'error': f'Failed to load dataset: {str(e)}'}), 500

@energy_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict energy consumption

    Request body:
        {
            "prediction_start": "2007-03-01T00:00:00",
            "hours_ahead": 24
        }

    Response:
        {
            "prediction_start": "2007-03-01T00:00:00",
            "hours_predicted": 24,
            "historical": {
                "timestamps": [...],
                "values": [...]
            },
            "predictions": {
                "timestamps": [...],
                "values": [...]
            },
            "actual": {
                "timestamps": [...],
                "values": [...]
            }
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if 'prediction_start' not in data:
            return jsonify({'error': 'Missing "prediction_start" field'}), 400

        # Parse prediction start date
        prediction_start = pd.Timestamp(data['prediction_start'])
        hours_ahead = data.get('hours_ahead', 24)

        # Validate hours_ahead
        if hours_ahead < 1 or hours_ahead > 24:
            return jsonify({'error': 'hours_ahead must be between 1 and 24'}), 400

        # Make prediction
        result = predict_energy(prediction_start, hours_ahead)

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
import tensorflow as tf
import json

# Set the path to config.json based on the current file's location
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

def build_lstm(sequence_length, feature_count):
    """Builds and compiles an LSTM model based on config settings."""
    lstm_units = config['lstm']['units']
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, feature_count)),
        tf.keras.layers.LSTM(lstm_units, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=config['lstm']['optimizer'], loss=config['lstm']['loss'])
    return model

def build_random_forest():
    """Initializes RandomForestRegressor based on config settings."""
    params = config['model_params']['RandomForestRegressor']
    return RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'])

def build_gradient_boosting():
    """Initializes GradientBoostingRegressor based on config settings."""
    params = config['model_params']['GradientBoostingRegressor']
    return GradientBoostingRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'])

def build_xgboost():
    """Initializes XGBRegressor based on config settings."""
    params = config['model_params']['XGBRegressor']
    return XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=params['learning_rate'])

def build_extra_trees():
    """Initializes ExtraTreesRegressor based on config settings."""
    params = config['model_params']['ExtraTreesRegressor']
    return ExtraTreesRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'])

def build_catboost():
    """Initializes CatBoostRegressor based on config settings."""
    params = config['model_params']['CatBoostRegressor']
    return CatBoostRegressor(iterations=params['iterations'], depth=params['depth'], learning_rate=params['learning_rate'], verbose=0)

def build_ridge():
    """Initializes Ridge meta-model for stacking based on config settings."""
    alpha = config['model_params']['Ridge']['alpha']
    return Ridge(alpha=alpha)

import joblib
from keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Check if root_mean_squared_error is available in the current version of scikit-learn
try:
    from sklearn.metrics import root_mean_squared_error  # For future versions (scikit-learn 1.6 and beyond)
except ImportError:
    # Fallback for scikit-learn versions <1.6
    root_mean_squared_error = lambda y_true, predictions: mean_squared_error(y_true, predictions, squared=False)

def save_model(model, model_name):
    joblib.dump(model, f'{model_name}.pkl')
    print(f"Model {model_name} saved to disk.")

def save_keras_model(model, model_name):
    model.save(f'{model_name}.h5')
    print(f"Keras model {model_name} saved to disk.")

def load_model_sklearn(model_name):
    try:
        model = joblib.load(f'{model_name}.pkl')
        print(f"Model {model_name} loaded from disk.")
        return model
    except FileNotFoundError:
        print(f"No saved model found for {model_name}, training from scratch.")
        return None

def load_keras_model(model_name):
    try:
        model = load_model(f'{model_name}.h5')
        print(f"Keras model {model_name} loaded from disk.")
        return model
    except FileNotFoundError:
        print(f"No saved Keras model found for {model_name}, training from scratch.")
        return None

def evaluate_model(y_true, predictions):
    mae = mean_absolute_error(y_true, predictions)
    rmse = root_mean_squared_error(y_true, predictions) # Using the new function or fallback
    return mae, rmse

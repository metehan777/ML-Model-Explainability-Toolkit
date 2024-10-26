import pandas as pd
import numpy as np
from sklearn.base import is_classifier, is_regressor
import joblib

def load_model_and_data(model_file, data_file):
    """Load model and dataset from uploaded files."""
    try:
        model = joblib.load(model_file)
        data = pd.read_csv(data_file)
        return model, data
    except Exception as e:
        raise ValueError(f"Error loading model or data: {str(e)}")

def validate_model_and_data(model, data):
    """Validate that model and data are compatible."""
    if not (is_classifier(model) or is_regressor(model)):
        raise ValueError("Uploaded model must be a scikit-learn classifier or regressor")
    
    required_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if required_features is not None:
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
    
    return True

def get_feature_names(model, data):
    """Get feature names from model or data."""
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    return data.columns.tolist()

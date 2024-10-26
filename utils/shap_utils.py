import shap
import numpy as np

def get_shap_values(model, data, background_samples=100):
    """Calculate SHAP values for the model and data."""
    try:
        # Create background distribution
        background = shap.sample(data, background_samples)
        
        # Choose explainer based on model type
        if hasattr(model, "predict_proba"):
            explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(model.predict, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(data)
        
        return explainer, shap_values
    except Exception as e:
        raise ValueError(f"Error calculating SHAP values: {str(e)}")

def get_local_explanation(explainer, data_point):
    """Get SHAP values for a single instance."""
    return explainer.shap_values(data_point)

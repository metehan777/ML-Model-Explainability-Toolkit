import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

def get_lime_explainer(model, data, mode="classification"):
    """Create and return a LIME explainer for the model and data."""
    try:
        # Create LIME explainer
        feature_names = data.columns.tolist()
        categorical_features = data.select_dtypes(include=['object', 'category']).columns
        categorical_names = {i: data[col].unique().tolist() 
                           for i, col in enumerate(data.columns) 
                           if col in categorical_features}
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            data.values,
            mode=mode,
            feature_names=feature_names,
            categorical_features=[data.columns.get_loc(col) for col in categorical_features],
            categorical_names=categorical_names,
            class_names=None,  # Will be set based on model type
            random_state=42
        )
        
        return explainer
    except Exception as e:
        raise ValueError(f"Error creating LIME explainer: {str(e)}")

def explain_instance(explainer, model, instance, num_features=10):
    """Generate LIME explanation for a single instance."""
    try:
        # Determine if model is a classifier or regressor
        if hasattr(model, 'predict_proba'):
            exp = explainer.explain_instance(
                instance.values,
                model.predict_proba,
                num_features=num_features
            )
        else:
            exp = explainer.explain_instance(
                instance.values,
                model.predict,
                num_features=num_features
            )
        return exp
    except Exception as e:
        raise ValueError(f"Error explaining instance: {str(e)}")

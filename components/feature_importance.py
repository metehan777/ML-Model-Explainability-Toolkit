import streamlit as st
import numpy as np
from utils.plot_utils import create_feature_importance_plot

def render_feature_importance(model, data):
    """Render feature importance section."""
    st.header("Feature Importance Analysis")
    
    with st.expander("ℹ️ About Feature Importance", expanded=False):
        st.markdown("""
        Feature importance shows how much each feature contributes to the model's predictions.
        Higher values indicate more important features.
        """)
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            st.warning("This model type doesn't support direct feature importance calculation")
            return
        
        feature_names = data.columns.tolist()
        fig = create_feature_importance_plot(importances, feature_names)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error calculating feature importance: {str(e)}")

import streamlit as st
import numpy as np
from utils.shap_utils import get_shap_values, get_local_explanation
import plotly.graph_objects as go

def render_shap_section(model, data):
    """Render SHAP analysis section."""
    st.header("SHAP Value Analysis")
    
    with st.expander("ℹ️ About SHAP Values", expanded=False):
        st.markdown("""
        SHAP (SHapley Additive exPlanations) values explain how much each feature contributes to individual predictions.
        - Global explanations show overall feature impact
        - Local explanations show feature impact for specific instances
        """)
    
    try:
        with st.spinner("Calculating SHAP values..."):
            explainer, shap_values = get_shap_values(model, data)
            
        # Global SHAP values
        st.subheader("Global SHAP Values")
        if isinstance(shap_values, list):
            shap_means = np.abs(shap_values[0]).mean(0)
        else:
            shap_means = np.abs(shap_values).mean(0)
            
        fig = go.Figure([
            go.Bar(
                x=shap_means,
                y=data.columns,
                orientation='h'
            )
        ])
        fig.update_layout(
            title="Global SHAP Values",
            xaxis_title="mean(|SHAP value|)",
            yaxis_title="Features"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Local explanation for a single instance
        st.subheader("Local Explanation")
        selected_idx = st.slider("Select instance index", 0, len(data)-1, 0)
        local_shap = get_local_explanation(explainer, data.iloc[selected_idx:selected_idx+1])
        
        if isinstance(local_shap, list):
            local_values = local_shap[0][0]
        else:
            local_values = local_shap[0]
            
        fig = go.Figure([
            go.Bar(
                x=local_values,
                y=data.columns,
                orientation='h'
            )
        ])
        fig.update_layout(
            title="Local SHAP Values for Selected Instance",
            xaxis_title="SHAP value",
            yaxis_title="Features"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in SHAP analysis: {str(e)}")

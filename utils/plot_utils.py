import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.inspection import partial_dependence

def create_feature_importance_plot(importance_values, feature_names):
    """Create feature importance bar plot."""
    fig = go.Figure([
        go.Bar(
            x=importance_values,
            y=feature_names,
            orientation='h'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Features",
        height=400 + len(feature_names) * 20
    )
    return fig

def create_pdp_plot(model, data, feature, feature_name):
    """Create partial dependence plot."""
    pdp_results = partial_dependence(
        model, data, [feature], percentiles=(0.05, 0.95), grid_resolution=20
    )
    
    fig = go.Figure([
        go.Scatter(
            x=pdp_results[1][0],
            y=pdp_results[0][0],
            mode='lines',
            name='Partial dependence'
        )
    ])
    
    fig.update_layout(
        title=f"Partial Dependence Plot for {feature_name}",
        xaxis_title=feature_name,
        yaxis_title="Partial dependence",
        height=400
    )
    return fig

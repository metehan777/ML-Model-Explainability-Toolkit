import streamlit as st
import plotly.graph_objects as go
from utils.lime_utils import get_lime_explainer, explain_instance

def render_lime_section(model, data):
    """Render LIME analysis section."""
    st.header("LIME Explanations")
    
    with st.expander("ℹ️ About LIME", expanded=False):
        st.markdown("""
        LIME (Local Interpretable Model-agnostic Explanations) helps understand model predictions by:
        - Creating a simple, interpretable model around each prediction
        - Showing which features contribute positively or negatively to the prediction
        - Providing local explanations that are more intuitive than global explanations
        """)
    
    try:
        # Create LIME explainer
        mode = "classification" if hasattr(model, 'predict_proba') else "regression"
        explainer = get_lime_explainer(model, data, mode=mode)
        
        # Select instance to explain
        st.subheader("Select Instance to Explain")
        selected_idx = st.slider(
            "Choose an instance from your dataset",
            0, len(data)-1, 0,
            help="Select a data point to generate LIME explanation"
        )
        
        selected_instance = data.iloc[selected_idx]
        
        # Show selected instance details
        st.write("Selected Instance Features:")
        st.dataframe(selected_instance.to_frame().T)
        
        # Number of features to show in explanation
        num_features = st.slider(
            "Number of features to show in explanation",
            min_value=1,
            max_value=len(data.columns),
            value=min(6, len(data.columns))
        )
        
        # Generate and display explanation
        with st.spinner("Generating LIME explanation..."):
            exp = explain_instance(explainer, model, selected_instance, num_features)
            
            # Extract explanation data
            feature_weights = exp.as_list()
            features, weights = zip(*feature_weights)
            
            # Create visualization
            colors = ['#ff4b4b' if w < 0 else '#00cc96' for w in weights]
            
            fig = go.Figure([
                go.Bar(
                    x=weights,
                    y=features,
                    orientation='h',
                    marker_color=colors
                )
            ])
            
            fig.update_layout(
                title="Feature Contributions to Prediction",
                xaxis_title="Impact on prediction",
                yaxis_title="Features",
                height=400 + len(features) * 20
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show prediction
            prediction = model.predict([selected_instance])[0]
            st.write(f"Model prediction for this instance: {prediction}")
            
    except Exception as e:
        st.error(f"Error in LIME analysis: {str(e)}")

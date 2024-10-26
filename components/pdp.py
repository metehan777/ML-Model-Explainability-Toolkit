import streamlit as st
from utils.plot_utils import create_pdp_plot

def render_pdp_section(model, data):
    """Render Partial Dependence Plot section."""
    st.header("Partial Dependence Plots")
    
    with st.expander("ℹ️ About Partial Dependence Plots", expanded=False):
        st.markdown("""
        Partial Dependence Plots (PDPs) show how a feature affects predictions while accounting for all other features.
        The y-axis shows the change in the predicted target as the feature value changes.
        """)
    
    try:
        numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numerical_features:
            st.warning("No numerical features found in the dataset")
            return
        
        selected_feature = st.selectbox(
            "Select feature for PDP analysis",
            numerical_features
        )
        
        if selected_feature:
            feature_idx = data.columns.get_loc(selected_feature)
            fig = create_pdp_plot(model, data, feature_idx, selected_feature)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error creating PDP: {str(e)}")

import streamlit as st
from components.upload import render_upload_section
from components.feature_importance import render_feature_importance
from components.pdp import render_pdp_section
from components.lime_analysis import render_lime_section
from components.shap_analysis import render_shap_section

def main():
    st.set_page_config(
        page_title="ML Model Explainability Toolkit",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 ML Model Explainability Toolkit")
    
    st.markdown("""
    This toolkit helps you understand your machine learning models through:
    - Feature Importance Analysis
    - Partial Dependence Plots (PDPs)
    - LIME Explanations
    - SHAP Values
    """)
    
    # Upload section
    model, data = render_upload_section()
    
    if model is not None and data is not None:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Feature Importance",
            "Partial Dependence Plots",
            "LIME Explanations",
            "SHAP Analysis"
        ])
        
        with tab1:
            render_feature_importance(model, data)
            
        with tab2:
            render_pdp_section(model, data)
            
        with tab3:
            render_lime_section(model, data)
            
        with tab4:
            render_shap_analysis(model, data)

if __name__ == "__main__":
    main()

import streamlit as st
from utils.model_utils import load_model_and_data, validate_model_and_data

def render_upload_section():
    """Render the model and data upload section."""
    st.header("Upload Model and Data")
    
    with st.expander("📌 Upload Instructions", expanded=True):
        st.markdown("""
        1. Upload your trained scikit-learn model (joblib format)
        2. Upload your dataset (CSV format)
        3. Ensure the dataset contains all features used during model training
        """)
    
    model_file = st.file_uploader("Upload Model (joblib)", type=['joblib', 'pkl'])
    data_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if model_file and data_file:
        try:
            model, data = load_model_and_data(model_file, data_file)
            validate_model_and_data(model, data)
            st.success("Model and data loaded successfully!")
            return model, data
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None, None
    return None, None

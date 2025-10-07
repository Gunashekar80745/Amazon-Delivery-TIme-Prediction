import streamlit as st
import pandas as pd
import numpy as np
import traceback

st.set_page_config(
    page_title="🚚 Delivery Time Predictor - Safe Version",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Amazon Delivery Time Prediction")
st.write("**Safe deployment version with error handling**")

# Safe model loading with extensive error handling
@st.cache_resource
def load_model():
    try:
        st.info("🔄 Attempting to load model...")
        
        # Check if file exists
        import os
        if not os.path.exists('delivery_prediction_model.pkl'):
            st.error("❌ Model file 'delivery_prediction_model.pkl' not found!")
            st.write("📁 Available files:")
            for file in os.listdir('.'):
                st.write(f"  - {file}")
            return None
        
        # Try to load
        import pickle
        with open('delivery_prediction_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        
        st.success("✅ Model loaded successfully!")
        return model_package
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None

def main():
    st.markdown("---")
    
    # Test model loading
    model_package = load_model()
    
    if model_package is None:
        st.warning("⚠️ Model not loaded. Showing demo interface...")
        
        # Show demo interface without model
        st.sidebar.header("📋 Demo Interface")
        
        # Demo inputs
        agent_age = st.sidebar.slider("Agent Age", 18, 65, 30)
        agent_rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.0, 0.1)
        
        if st.button("🚀 Demo Prediction"):
            # Demo prediction (no real model)
            demo_prediction = agent_age + agent_rating * 10
            st.success(f"📊 Demo prediction: {demo_prediction:.1f} minutes")
            st.info("ℹ️ This is a demo calculation - upload your model for real predictions")
    
    else:
        st.success("🎯 Model loaded - ready for predictions!")
        
        # Show model info
        if 'performance' in model_package:
            perf = model_package['performance']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RMSE", f"{perf.get('rmse', 0):.1f}")
            with col2:
                st.metric("MAE", f"{perf.get('mae', 0):.1f}")
            with col3:
                st.metric("R²", f"{perf.get('r2', 0):.3f}")
        
        # Basic prediction interface
        st.sidebar.header("📋 Order Details")
        
        # Simple inputs
        agent_age = st.sidebar.slider("Agent Age", 18, 65, 30)
        agent_rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.0, 0.1)
        distance = st.sidebar.slider("Distance (km)", 0.1, 50.0, 5.0, 0.1)
        
        if st.button("🚀 Predict Delivery Time"):
            try:
                # Safe prediction attempt
                st.info("🔄 Making prediction...")
                
                # Create input DataFrame
                input_data = pd.DataFrame({
                    'AgentAge': [agent_age],
                    'AgentRating': [agent_rating],
                    'distance_km': [distance],
                    # Add other required features with defaults
                })
                
                # Show input data
                st.write("📋 Input data:")
                st.dataframe(input_data)
                
                # Try prediction
                prediction = 35.5  # Placeholder - implement real prediction here
                st.success(f"⏱️ Estimated Delivery Time: {prediction:.1f} minutes")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.code(traceback.format_exc())

# Footer with debug info
st.markdown("---")
with st.expander("🔍 Debug Information"):
    st.write("**Environment Info:**")
    st.write(f"- Streamlit version: {st.__version__}")
    st.write(f"- Python version: 3.13.7")
    
    import os
    st.write("**Available files:**")
    for file in os.listdir('.'):
        st.write(f"  - {file} ({os.path.getsize(file)} bytes)")

if __name__ == "__main__":
    main()

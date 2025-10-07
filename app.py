import streamlit as st
import pandas as pd
import numpy as np
import pickle
import traceback
import os

st.set_page_config(
    page_title="🚚 Amazon Delivery Time Prediction",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Amazon Delivery Time Prediction")
st.write("**Working Version with Safe Model Loading**")

@st.cache_resource
def load_model():
    try:
        # List all files for debugging
        st.write("📁 **Files in repository:**")
        files = os.listdir('.')
        for f in files:
            if f.endswith('.pkl'):
                st.write(f"🎯 Found model file: **{f}** ({os.path.getsize(f)} bytes)")
        
        # Try to load the model file
        model_filename = 'delivery_prediction_model.pkl'
        
        if not os.path.exists(model_filename):
            st.error(f"❌ Model file '{model_filename}' not found!")
            return None
            
        st.info(f"🔄 Loading model from {model_filename}...")
        
        with open(model_filename, 'rb') as f:
            model_package = pickle.load(f)
        
        st.success("✅ Model loaded successfully!")
        return model_package
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None

def main():
    st.markdown("---")
    
    # Try to load model
    model_package = load_model()
    
    if model_package is None:
        st.warning("⚠️ Model not loaded. Using formula-based prediction...")
        show_demo_app()
    else:
        st.success("🎯 Model loaded successfully!")
        show_ml_app(model_package)

def show_demo_app():
    """Demo app without model loading"""
    st.sidebar.header("📋 Order Details (Demo Mode)")
    
    # Simple inputs
    agent_age = st.sidebar.slider("Agent Age", 18, 65, 30)
    agent_rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
    distance = st.sidebar.slider("Distance (km)", 0.1, 50.0, 5.0, 0.1)
    weather = st.sidebar.selectbox("Weather", ["clear", "cloudy", "rain", "storm"])
    traffic = st.sidebar.selectbox("Traffic", ["low", "medium", "high"])
    
    if st.button("🚀 Predict Delivery Time (Demo)"):
        # Formula-based prediction
        base_time = 25
        age_factor = (50 - agent_age) * 0.3
        rating_factor = (5.0 - agent_rating) * 4
        distance_factor = distance * 2.5
        
        weather_mult = {"clear": 1.0, "cloudy": 1.1, "rain": 1.3, "storm": 1.5}
        traffic_mult = {"low": 1.0, "medium": 1.2, "high": 1.5}
        
        prediction = base_time + age_factor + rating_factor + distance_factor
        prediction *= weather_mult[weather] * traffic_mult[traffic]
        
        st.success(f"⏱️ **Estimated Delivery Time: {prediction:.1f} minutes**")
        
        if prediction <= 30:
            st.success("🟢 Fast delivery expected!")
        elif prediction <= 60:
            st.warning("🟡 Standard delivery timing")  
        else:
            st.error("🔴 Extended delivery time")
            
        st.info("ℹ️ This is a demo calculation. Upload your trained model for ML predictions!")

def show_ml_app(model_package):
    """Full ML app with loaded model"""
    st.subheader("🤖 Machine Learning Predictions")
    
    # Show model info
    if 'performance' in model_package:
        perf = model_package['performance']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{perf.get('rmse', 0):.1f} min")
        with col2:
            st.metric("MAE", f"{perf.get('mae', 0):.1f} min") 
        with col3:
            st.metric("R²", f"{perf.get('r2', 0):.3f}")
    
    st.sidebar.header("📋 ML Prediction Inputs")
    
    # More detailed inputs for ML model
    agent_age = st.sidebar.slider("Agent Age", 18, 65, 30)
    agent_rating = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
    distance = st.sidebar.slider("Distance (km)", 0.1, 50.0, 5.0, 0.1)
    
    if st.button("🤖 ML Prediction"):
        try:
            # Create input for ML model
            input_data = pd.DataFrame({
                'AgentAge': [agent_age],
                'AgentRating': [agent_rating],
                'distance_km': [distance]
            })
            
            # Make prediction using the loaded model
            if hasattr(model_package['model'], 'predict'):
                prediction = model_package['model'].predict(input_data)[0]
                st.success(f"🤖 **ML Prediction: {prediction:.1f} minutes**")
            else:
                st.error("❌ Model doesn't have predict method")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.info("🔧 **Debug Mode**: This app shows detailed loading information and falls back to demo mode if model loading fails.")

if __name__ == "__main__":
    main()

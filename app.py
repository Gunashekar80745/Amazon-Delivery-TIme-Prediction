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

@st.cache_resource
def load_model():
    try:
        with open('delivery_prediction_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        st.success("✅ Model loaded successfully!")
        return model_package
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def create_model_input(model_package, user_inputs):
    """Create properly formatted input with all required features"""
    try:
        # Get the required feature columns from the model
        if 'feature_columns' in model_package:
            required_features = model_package['feature_columns']
        else:
            # Fallback - create expected features based on common training setup
            required_features = [
                'AgentAge', 'AgentRating', 'distance_km',
                'order_hour', 'order_wday', 'is_weekend', 'order_month', 'order_year',
                'order_to_pickup_min'
            ]
        
        st.write(f"🔍 **Model expects {len(required_features)} features:**")
        st.write(required_features)
        
        # Create input DataFrame with all required features
        input_data = {}
        
        # Fill with user inputs where available
        input_data.update(user_inputs)
        
        # Fill missing features with defaults
        defaults = {
            'order_hour': 12.0,
            'order_wday': 2.0,  # Wednesday
            'is_weekend': 0,
            'order_month': 6.0,  # June
            'order_year': 2024.0,
            'order_to_pickup_min': 10.0
        }
        
        for feature in required_features:
            if feature not in input_data:
                if feature in defaults:
                    input_data[feature] = defaults[feature]
                elif 'stats' in model_package and feature in model_package['stats']:
                    input_data[feature] = model_package['stats'][feature]
                else:
                    input_data[feature] = 0  # Last resort default
        
        # Create DataFrame with correct order
        df_input = pd.DataFrame([input_data])
        df_input = df_input[required_features]  # Ensure correct order
        
        st.write("📊 **Input data shape:**", df_input.shape)
        st.dataframe(df_input)
        
        return df_input
        
    except Exception as e:
        st.error(f"❌ Error creating input: {str(e)}")
        st.code(traceback.format_exc())
        return None

def main():
    model_package = load_model()
    
    if model_package is None:
        st.error("❌ Could not load model. Please check the model file.")
        return
    
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
    
    st.markdown("---")
    
    # User inputs
    st.sidebar.header("📋 Order Details")
    
    # Collect all necessary inputs
    user_inputs = {}
    
    # Agent details
    user_inputs['AgentAge'] = st.sidebar.slider("Agent Age", 18, 65, 30)
    user_inputs['AgentRating'] = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.5, 0.1)
    
    # Order details
    user_inputs['distance_km'] = st.sidebar.slider("Distance (km)", 0.1, 50.0, 5.0, 0.1)
    
    # Time details
    user_inputs['order_hour'] = st.sidebar.slider("Order Hour (0-23)", 0, 23, 12)
    
    order_day = st.sidebar.selectbox("Day of Week", 
        options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    user_inputs['order_wday'] = float(day_mapping[order_day])
    user_inputs['is_weekend'] = 1 if user_inputs['order_wday'] >= 5 else 0
    
    user_inputs['order_month'] = st.sidebar.slider("Month", 1, 12, 6)
    user_inputs['order_year'] = st.sidebar.slider("Year", 2022, 2025, 2024)
    user_inputs['order_to_pickup_min'] = st.sidebar.slider("Order to Pickup (min)", 1, 60, 10)
    
    # Categorical inputs (if your model needs them)
    weather = st.sidebar.selectbox("Weather", ["clear", "cloudy", "rain", "storm"])
    traffic = st.sidebar.selectbox("Traffic", ["low", "medium", "high"])
    vehicle = st.sidebar.selectbox("Vehicle", ["motorcycle", "scooter", "bicycle", "car"])
    area = st.sidebar.selectbox("Area", ["Urban", "Suburban", "Rural", "Metropolitian"])
    category = st.sidebar.selectbox("Category", ["Electronics", "Food", "Clothing", "Books"])
    
    # Add categorical variables to inputs
    user_inputs.update({
        'Weather': weather,
        'Traffic': traffic,
        'Vehicle': vehicle,
        'Area': area,
        'Category': category
    })
    
    # Convert floats for consistency
    for key, value in user_inputs.items():
        if isinstance(value, int) and key not in ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']:
            user_inputs[key] = float(value)
    
    if st.button("🚀 Predict Delivery Time", type="primary"):
        try:
            # Create properly formatted input
            model_input = create_model_input(model_package, user_inputs)
            
            if model_input is not None:
                # Make prediction
                if 'preprocessor' in model_package:
                    # If model has preprocessor, use it
                    processed_input = model_package['preprocessor'].transform(model_input)
                    prediction = model_package['model'].predict(processed_input)[0]
                else:
                    # Direct prediction if no preprocessor
                    prediction = model_package['model'].predict(model_input)[0]
                
                # Display result
                st.success(f"⏱️ **Estimated Delivery Time: {prediction:.1f} minutes**")
                
                # Time breakdown
                hours = int(prediction // 60)
                mins = int(prediction % 60)
                if hours > 0:
                    st.info(f"🕐 That's approximately **{hours}h {mins}m**")
                
                # Speed categorization
                if prediction <= 30:
                    st.success("🟢 **Fast Delivery** - Great!")
                    st.balloons()
                elif prediction <= 60:
                    st.warning("🟡 **Standard Delivery** - Normal timing")
                else:
                    st.error("🔴 **Extended Delivery** - Consider factors")
                    
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

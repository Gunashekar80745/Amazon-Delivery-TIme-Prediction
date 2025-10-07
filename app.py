import streamlit as st
import pandas as pd
import numpy as np
import pickle
import traceback

st.set_page_config(
    page_title="🚚 Delivery Time Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        with open('streamlit_model.pkl', 'rb') as file:  # ← CORRECTED FILENAME
            return pickle.load(file)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_input(input_data, model_package):
    """Preprocess input data using the saved preprocessor"""
    try:
        processed_data = input_data.copy()
        
        for col in model_package['numeric_columns']:
            if col in processed_data.columns and processed_data[col].isnull().any():
                fill_val = model_package['feature_stats']['numeric_medians'][col]
                processed_data[col] = processed_data[col].fillna(fill_val)
        
        for col in model_package['categorical_columns']:
            if col in processed_data.columns and processed_data[col].isnull().any():
                fill_val = model_package['feature_stats']['categorical_modes'][col]
                processed_data[col] = processed_data[col].fillna(fill_val)
        
        transformed_data = model_package['preprocessor'].transform(processed_data)
        return transformed_data
        
    except Exception as e:
        st.error(f"❌ Preprocessing error: {str(e)}")
        return None

def main():
    st.title("🚚 Delivery Time Predictor")
    st.markdown("**Predict delivery times using machine learning**")
    st.markdown("---")
    
    # Load model
    model_package = load_model()
    if model_package is None:
        st.error("❌ Could not load the model. Please check if 'streamlit_model.pkl' is uploaded.")
        st.stop()
    
    # Display model info
    st.success(f"✅ {model_package['model_name']} model loaded successfully!")
    
    # Model performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Model", model_package['model_name'])
    with col2:
        st.metric("📊 RMSE", f"{model_package['performance']['rmse']:.1f} min")
    with col3:
        st.metric("📈 R² Score", f"{model_package['performance']['r2']:.3f}")
    with col4:
        st.metric("🔧 Features", len(model_package['feature_columns']))
    
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("📋 Enter Order Details")
    
    # Create input form
    input_data = {}
    feature_cols = model_package['feature_columns']
    
    # Agent Information
    if any(col in feature_cols for col in ['Agent_Age', 'Agent_Rating']):
        st.sidebar.subheader("👨‍💼 Agent Information")
        
        if 'Agent_Age' in feature_cols:
            input_data['Agent_Age'] = st.sidebar.slider("Agent Age", 18, 65, 30)
        
        if 'Agent_Rating' in feature_cols:
            input_data['Agent_Rating'] = st.sidebar.slider("Agent Rating", 1.0, 5.0, 4.0, 0.1)
    
    # Order Details
    st.sidebar.subheader("📦 Order Information")
    
    if 'distance_km' in feature_cols:
        input_data['distance_km'] = st.sidebar.slider("Distance (km)", 0.1, 50.0, 5.0, 0.1)
    
    if 'Category' in feature_cols:
        input_data['Category'] = st.sidebar.selectbox("Order Category", 
            ["Food", "Electronics", "Clothing", "Books", "Sports", "Cosmetics", "Toys"])
    
    # Time Information
    if any(col in feature_cols for col in ['order_hour', 'order_wday', 'order_month', 'order_year']):
        st.sidebar.subheader("⏰ Time Information")
        
        if 'order_hour' in feature_cols:
            input_data['order_hour'] = st.sidebar.slider("Order Hour", 0.0, 23.0, 12.0, 1.0)
        
        if 'order_wday' in feature_cols:
            input_data['order_wday'] = st.sidebar.selectbox("Day of Week", 
                options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(x)])
        
        if 'order_month' in feature_cols:
            input_data['order_month'] = st.sidebar.slider("Month", 1.0, 12.0, 6.0, 1.0)
        
        if 'order_year' in feature_cols:
            input_data['order_year'] = st.sidebar.slider("Year", 2022.0, 2025.0, 2024.0, 1.0)
        
        if 'order_to_pickup_min' in feature_cols:
            input_data['order_to_pickup_min'] = st.sidebar.slider("Order to Pickup (min)", 1.0, 60.0, 10.0, 1.0)
    
    # Weekend calculation
    if 'is_weekend' in feature_cols and 'order_wday' in input_data:
        input_data['is_weekend'] = 1 if input_data['order_wday'] >= 5.0 else 0
    
    # Conditions
    st.sidebar.subheader("🌦️ Delivery Conditions")
    
    if 'Weather' in feature_cols:
        input_data['Weather'] = st.sidebar.selectbox("Weather", 
            ["clear", "cloudy", "rain", "storm", "other"])
    
    if 'Traffic' in feature_cols:
        input_data['Traffic'] = st.sidebar.selectbox("Traffic", 
            ["low", "medium", "high", "other"])
    
    if 'Vehicle' in feature_cols:
        input_data['Vehicle'] = st.sidebar.selectbox("Vehicle Type", 
            ["motorcycle", "scooter", "bicycle", "car"])
    
    if 'Area' in feature_cols:
        input_data['Area'] = st.sidebar.selectbox("Area Type", 
            ["Urban", "Suburban", "Rural", "Metropolitian"])
    
    # Main prediction area
    st.subheader("🎯 Delivery Time Prediction")
    
    if st.button("🚀 Predict Delivery Time", type="primary", use_container_width=True):
        try:
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required features are present
            for col in model_package['feature_columns']:
                if col not in input_df.columns:
                    if col in model_package['numeric_columns']:
                        default_val = model_package['feature_stats']['numeric_medians'].get(col, 0)
                        input_df[col] = default_val
                    else:
                        default_val = model_package['feature_stats']['categorical_modes'].get(col, 'unknown')
                        input_df[col] = default_val
            
            # Reorder columns
            input_df = input_df[model_package['feature_columns']]
            
            # Preprocess data
            processed_data = preprocess_input(input_df, model_package)
            
            if processed_data is not None:
                # Make prediction
                prediction = model_package['model'].predict(processed_data)[0]
                
                # Display result
                st.success(f"⏱️ **Estimated Delivery Time: {prediction:.1f} minutes**")
                
                # Convert to hours and minutes
                hours = int(prediction // 60)
                mins = int(prediction % 60)
                
                if hours > 0:
                    st.info(f"🕐 That's approximately **{hours}h {mins}m**")
                else:
                    st.info(f"🕐 That's approximately **{mins} minutes**")
                
                # Speed categorization
                if prediction <= 30:
                    st.success("🟢 **Fast Delivery** - Great choice!")
                    st.balloons()
                elif prediction <= 60:
                    st.warning("🟡 **Standard Delivery** - Normal timing")
                else:
                    st.error("🔴 **Extended Delivery** - Consider peak hours")
                    
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.error("Please check all inputs and try again.")

if __name__ == "__main__":
    main()

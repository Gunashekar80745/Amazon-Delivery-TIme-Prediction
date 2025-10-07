import streamlit as st
import traceback

st.title("🚚 Debugging Version - Step by Step")

try:
    st.write("✅ Step 1: Basic imports working")
    
    # Test pandas import
    import pandas as pd
    st.write("✅ Step 2: Pandas imported successfully")
    
    # Test numpy import  
    import numpy as np
    st.write("✅ Step 3: NumPy imported successfully")
    
    # Test pickle import
    import pickle
    st.write("✅ Step 4: Pickle imported successfully")
    
    # Test file existence (THIS IS LIKELY WHERE IT HANGS)
    import os
    st.write("✅ Step 5: Checking files...")
    
    files_in_repo = os.listdir('.')
    st.write(f"📁 Files found: {files_in_repo}")
    
    # Check if model file exists
    model_exists = os.path.exists('delivery_model_compatible.pkl')
    st.write(f"🔍 Model file exists: {model_exists}")
    
    if model_exists:
        st.write("✅ Step 6: Model file found, attempting to load...")
        
        # THIS IS LIKELY WHERE THE INFINITE LOOP HAPPENS
        with open('delivery_model_compatible.pkl', 'rb') as f:
            model_package = pickle.load(f)
        
        st.write("✅ Step 7: Model loaded successfully!")
        st.json({"model_type": str(type(model_package))})
    else:
        st.warning("⚠️ Model file not found - that's okay for testing")
    
    st.success("🎉 All steps completed - no infinite loops!")
    
except Exception as e:
    st.error(f"❌ Error at step: {str(e)}")
    st.code(traceback.format_exc())

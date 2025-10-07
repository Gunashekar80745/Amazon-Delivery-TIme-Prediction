import streamlit as st
import pandas as pd
import numpy as np

st.title("🚚 Test App")
st.write("If you see this, the deployment works!")

# Test basic functionality
df = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': [4, 5, 6]
})

st.write("Test dataframe:")
st.dataframe(df)

st.success("✅ App is working!")

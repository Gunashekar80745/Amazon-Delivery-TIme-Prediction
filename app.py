import streamlit as st

st.title("🚚 Ultra-Simple Test")
st.write("If you see this, your deployment works!")
st.success("✅ No infinite loops here!")

# Simple input
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello {name}!")

st.balloons()

# widget_state1.py
import streamlit as st
if 'celsius' not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.celsius = 50.0

st.slider(
    'Temperature in Celsius',
    min_value=-100.0,
    max_value=100.0,
    key='celsius')

st.write(st.session_state.celsius)
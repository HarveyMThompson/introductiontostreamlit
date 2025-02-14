# button3.py
"""
Created on Tue Jun  4 10:04:23 2024

@author: harveythompson
"""

import streamlit as st

if 'button' not in st.session_state:
    st.session_state.button = False

def click_button():
    st.session_state.button = not st.session_state.button

st.button('Click me', on_click=click_button)

st.slider('Select a value', disabled=st.session_state.button)
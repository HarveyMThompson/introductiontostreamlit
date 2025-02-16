# enumexample.py
"""
Created on Thu Jun  6 10:03:48 2024

@author: harveythompson
"""

from enum import Enum
import streamlit as st

# class syntax
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

selected_colors = set(st.multiselect("Pick colors", options=Color))

if selected_colors == {Color.RED, Color.GREEN}:
    st.write("Hooray, you found the color YELLOW!")
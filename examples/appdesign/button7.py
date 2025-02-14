# button7.py
"""
Created on Wed Jun  5 09:32:38 2024

@author: harveythompson
"""

import streamlit as st

if st.button('Button 1'):
    st.write('Button 1 was clicked')
    if st.button('Button 2'):
        # This will never be executed.
        st.write('Button 2 was clicked')
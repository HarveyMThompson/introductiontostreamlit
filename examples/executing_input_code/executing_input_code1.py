# executing_input_code1.py
# Prgoram to input text box and then run this as code
"""
Created on Fri Aug  2 14:00:58 2024

@author: harveythompson
"""
import streamlit as st
import sys

# Add heading and introductory text
st.title("Program for executing input code")
st.write("this application allows user to input a text box of python code and then run  it")
st.markdown("---")

user_code = st.text_area("Input code to be run", value="Hello world",height=200)
# Execute the user's code
try:
    exec(user_code)
except Exception as e:
    st.error(f"An error occured running the code: {str(e)}")
    st.stop()
    
""" Example of user_code that could be input into the st.text_area
import numpy as np
for i in range(10):
    print(f"i={i}")

print(np.sqrt(100))
"""
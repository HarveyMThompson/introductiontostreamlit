# -*- coding: utf-8 -*- caching4.py
"""
Created on Thu May  9 10:01:07 2024

@author: harveythompson
"""
import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data   # turn off copying behaviour
def load_data(url):
    df = pd.read_csv(url)   # download the data
    return df

df = load_data("https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv")
st.dataframe(df)

df.drop(columns=['Lat'], inplace=True)    # mutate the dataframe inplace

st.button("Rerun")
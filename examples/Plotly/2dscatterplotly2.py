# 2dscatterplotly.py

import streamlit as st
import pandas as pd

# x and y given as array_like objects
import plotly.express as px

# Need to create a pandas dataframe of data for use in plotly
x=[0, 1, 2, 3, 4]
y=[0, 1, 4, 9, 16]
df = pd.DataFrame(
    {'x': x,
     'y': y,
    })

fig = px.scatter(df, x='x', y='y')
fig2 = px.line(df, x='x', y='y')
st.plotly_chart(fig)
st.plotly_chart(fig2)
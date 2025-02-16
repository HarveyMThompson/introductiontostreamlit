# multipleaxes.py
import streamlit as st
import numpy as np
import math #needed for definition of pi
import plotly.graph_objects as go

x = np.arange(1,11)
y1 = np.exp(x)
y2 = np.log(x)
trace1 = go.Scatter(
   x = x,
   y = y1,
   name = 'exp'
)
trace2 = go.Scatter(
   x = x,
   y = y2,
   name = 'log',
   yaxis = 'y2'
)
data = [trace1, trace2]
layout = go.Layout(
   title = 'Double Y Axis Example',
   yaxis = dict(
      title = 'exp',zeroline=True,
      showline = True
   ),
   yaxis2 = dict(
      title = 'log',
      zeroline = True,
      showline = True,
      overlaying = 'y',
      side = 'right'
   )
)
fig = go.Figure(data=data, layout=layout)
st.plotly_chart(fig)
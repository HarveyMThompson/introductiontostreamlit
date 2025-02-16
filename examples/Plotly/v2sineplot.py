# v2sineplot.py
import streamlit as st
import numpy as np
import math #needed for definition of pi
import plotly.graph_objects as go

xpoints = np.arange(0, math.pi*2, 0.05)
y1 = np.sin(xpoints)
y2 = np.cos(xpoints)
trace0 = go.Scatter(
   x = xpoints,
   y = y1,
   name='Sine'
)
trace1 = go.Scatter(
   x = xpoints,
   y = y2,
   name = 'cos'
)
data = [trace0, trace1]
layout = go.Layout(title = "Sine and cos", xaxis = {'title':'angle'}, yaxis = {'title':'value'})
fig = go.Figure(data = data, layout = layout)
st.plotly_chart(fig)

layout2 = go.Layout(
   title = "Sine and cos",
   xaxis = dict(
      title = 'angle',
      showgrid = True,
      zeroline = True,
      showline = True,
      showticklabels = True,
      gridwidth = 1
   ),
   yaxis = dict(
      showgrid = True,
      zeroline = True,
      showline = True,
      gridcolor = '#bdbdbd',
      gridwidth = 2,
      zerolinecolor = '#969696',
      zerolinewidth = 2,
      linecolor = '#636363',
      linewidth = 2,
      title = 'VALUE',
      titlefont = dict(
         family = 'Arial, sans-serif',
         size = 18,
         color = 'lightgrey'
      ),
      showticklabels = True,
      tickangle = 45,
      tickfont = dict(
      family = 'Old Standard TT, serif',
      size = 14,
      color = 'black'
      ),
      tickmode = 'linear',
      tick0 = 0.0,
      dtick = 0.25
   )
)

fig2 = go.Figure(data = data, layout = layout2)
st.plotly_chart(fig2)
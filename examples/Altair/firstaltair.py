# firstaltair.py

import altair as alt
import streamlit as st
from vega_datasets import data

iris = data.iris()

alt_chart = (
    alt.Chart(iris).mark_point().encode(
    x='petalLength',
    y='petalWidth',
    color='species',
    )
    .interactive()
)
st.altair_chart(alt_chart, use_container_width=True)
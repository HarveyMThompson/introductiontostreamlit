# fifth2Daltair.py
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

ad_type = ["SEO", "Email", "PPC",  "PR", "Direct Mail", "OMB"]
roi = [68.9, 56.7, 52.4, 48.5, 37.4,  19.9]

data = {"adtype":ad_type, "roi":roi}
df = pd.DataFrame(data)

roi_bars = alt.Chart(df).mark_bar().encode(
    x = alt.X("adtype:O", title="Type of Advertising"),
    y = alt.Y("roi:Q", title="Return on Investment [%]"),
    color= alt.condition(
        alt.datum.adtype == "Email", 
        alt.value("darkred"),
        alt.value("lightgrey"))
).properties(
    width=300,
    height=200
)

roi_text = roi_bars.mark_text(
    align='center',
    baseline='middle',
    dy=-10,  # Nudges text up so it doesn't appear on top of the bar
    color="darkred"
).encode(
    text="roi:Q",
    opacity= alt.condition(
        alt.datum.adtype == "Email", 
        alt.value(1.0),
        alt.value(0.0))
)

st.altair_chart(roi_bars+roi_text)
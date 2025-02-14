# third2Daltair.py
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

ad_type = ["SEO", "Email", "PPC",  "PR", "Direct Mail", "OMB"]
roi = [68.9, 56.7, 52.4, 48.5, 37.4,  19.9]

data = {"adtype":ad_type, "roi":roi}
df = pd.DataFrame(data)
df

c = (
     alt.Chart(df).mark_bar().encode(
    x = "adtype:O",
    y = "roi:Q",
    )
)

st.altair_chart(c, use_container_width=True)
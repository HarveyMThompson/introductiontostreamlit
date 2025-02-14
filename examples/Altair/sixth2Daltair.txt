# sixth2Daltair.py
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

ad_type = ["SEO", "Email", "PPC",  "PR", "Direct Mail", "OMB"]
roi = [68.9, 56.7, 52.4, 48.5, 37.4,  19.9]

data = {"adtype":ad_type, "roi":roi}
df = pd.DataFrame(data)

# define a selector of type "single" (only a single element active at a time)
# and define what behavious it should do if nothing is clicked:
# if "all" is selected, all bars will be dark red by default until you click on one
# if "none" is selected, all bars will be lightgrey until you click on one
selector = alt.selection(type="single", empty='none')

roi_bars = alt.Chart(df).mark_bar().encode(
    x = alt.X("adtype:O", title="Type of Advertising",
               sort = alt.EncodingSortField(
                    field="roi",
                    order="descending"
        )
),
    y = alt.Y("roi:Q", 
              title="Return on Investment [%]"
),
    color= alt.condition(
        selector, # replace the previous condition with the selector
        alt.value("darkred"),
        alt.value("lightgrey"))
).add_selection(
    selector  # Add the selector here so that the chart knows to use it
).properties(
    width=300,
    height=200
)

roi_bars
st.altair_chart(roi_bars)
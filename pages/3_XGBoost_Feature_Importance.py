import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="YBRFSS Question XGBoost Visualization", layout="wide")
st.title("XGBoost Feature Importance Ranking")

option = st.selectbox(
    "What Model You want to predict the suicide rate?",
    ("XGBoost", "Linear Regression", "Decision Tree"),
)

st.write("You selected:", option)
import streamlit as st
import pandas as pd

st.set_page_config(page_title="YBRFSS Analysis", layout="wide")

st.title("Youth Behavioral Risk Factor Surveillance System (YBRFSS)")

st.write("""
### Project Overview
This project analyzes data from the Youth Behavioral Risk Factor Surveillance System (YBRFSS), specifically focusing on adolescent suicide risk factors. 

Upload your dataset here once to start exploring the data across multiple pages.
""")


@st.cache_data
def load_full_dataset(file):
    return pd.read_csv(file)

df_full = load_full_dataset('SADCQ_2023.csv')
st.session_state['df_full'] = df_full
st.success("Data loaded successfully!")

st.write("### Dataset Preview")
st.dataframe(df_full.head())

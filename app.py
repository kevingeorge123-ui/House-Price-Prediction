import streamlit as st
import pandas as pd

st.title("Kevin C George - House Price Prediction")

st.write("Welcome to my first Machine Learning Streamlit project!")

# Load the dataset
df = pd.read_csv("train.csv")

st.subheader("House Price Dataset")

st.write("Number of rows:", df.shape[0])
st.write("Number of columns:", df.shape[1])

st.dataframe(df.head())

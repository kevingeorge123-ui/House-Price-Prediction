import streamlit as st
import pandas as pd

st.title("Kevin C George - House Price Prediction")

st.write("Welcome to my first Machine Learning Streamlit project!")

# Load the dataset
df = pd.read_csv("train.csv")

st.subheader("House Price Dataset")

st.write("Number of rows:", df.shape[0])
st.write("Number of columns:", df.shape[1])



X = df.drop("SalePrice", axis=1)

y = df["SalePrice"]

st.subheader("Machine Learning Data")

st.write("Features (X):")
st.write(X.shape)

st.write("Target (y):")
st.write(y.shape)
st.subheader("Missing Values")

missing_values = df.isnull().sum()

st.write(missing_values)

numeric_columns = X.select_dtypes(include=["number"]).columns
categorical_columns = X.select_dtypes(include=["object"]).columns

st.subheader("Data Types")

st.write("Numerical columns:", len(numeric_columns))
st.write("Categorical columns:", len(categorical_columns))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

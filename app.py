import streamlit as st
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# -----------------------------
# Streamlit Title
# -----------------------------

st.title("Kevin C George - House Price Prediction")

st.write("Welcome to my first Machine Learning Streamlit project!")


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("train.csv")


st.subheader("House Price Dataset")

st.write("Number of rows:", df.shape[0])
st.write("Number of columns:", df.shape[1])


# -----------------------------
# Features and Target
# -----------------------------

X = df.drop("SalePrice", axis=1)

y = df["SalePrice"]


st.subheader("Machine Learning Data")

st.write("Features (X):")
st.write(X.shape)

st.write("Target (y):")
st.write(y.shape)


# -----------------------------
# Missing Values
# -----------------------------

st.subheader("Missing Values")

missing_values = df.isnull().sum()

st.write(missing_values)


# -----------------------------
# Data Types
# -----------------------------

numeric_columns = X.select_dtypes(include=["number"]).columns

categorical_columns = X.select_dtypes(include=["object"]).columns


st.subheader("Data Types")

st.write("Numerical columns:", len(numeric_columns))

st.write("Categorical columns:", len(categorical_columns))


# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


st.subheader("Train/Test Split")

st.write("Training rows:", X_train.shape[0])

st.write("Testing rows:", X_test.shape[0])


# -----------------------------
# Imputers
# -----------------------------

numeric_imputer = SimpleImputer(strategy="median")

categorical_imputer = SimpleImputer(strategy="most_frequent")

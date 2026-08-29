# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv("train.csv")


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("SalePrice", axis=1)

y = df["SalePrice"]


# ==========================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL
# ==========================================

numeric_columns = X.select_dtypes(include=["number"]).columns

categorical_columns = X.select_dtypes(include=["object"]).columns


# ==========================================
# 5. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 6. HANDLE MISSING VALUES + CATEGORIES
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ==========================================
# 7. COMBINE THE PROCESSING
# ==========================================

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# ==========================================
# 8. CREATE THE MODEL
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])


# ==========================================
# 9. TRAIN THE MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 11. EVALUATE THE MODEL
# ==========================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)


# ==========================================
# 12. STREAMLIT APP
# ==========================================

st.title("Kevin C George - House Price Prediction")

st.write("House Price Prediction using Machine Learning")



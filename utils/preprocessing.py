import pandas as pd
import numpy as np

def drop_redundant_columns(df):
    df = df.copy()
    return df.drop(
        columns=[
            "ID", "Case Number", "Block",
            "X Coordinate", "Y Coordinate",
            "Location", "Updated On"
        ],
        errors="ignore"
    )

def extract_datetime_features(df, date_col="Date"):
    df = df.copy()

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"   # ignore invalid format
    )

    df["Hour"] = df[date_col].dt.hour
    df["Day"] = df[date_col].dt.day
    df["Month"] = df[date_col].dt.month
    df["DayOfWeek"] = df[date_col].dt.dayofweek

    return df.drop(columns=[date_col])


def fill_missing_values(df):
    df = df.copy()

    for col in df.select_dtypes(include=["int64", "float64"]):
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def remove_outliers_iqr(df, cols):
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

def preprocess_user_input(raw_dict):
    df_new = pd.DataFrame([raw_dict])

    # One-hot encode same way
    df_new = pd.get_dummies(df_new, drop_first=True)

    # Add missing columns
    for col in model_cols:
        if col not in df_new.columns:
            df_new[col] = 0

    # Keep only training columns order
    df_new = df_new[model_cols]

    # Scale numeric cols (same list you used)
    num_features = ["Latitude","Longitude","Year","Month","Day","Hour","DayOfWeek"]
    df_new[num_features] = scaler.transform(df_new[num_features])

    return df_new
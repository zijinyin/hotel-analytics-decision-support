import pickle
import numpy as np
import pandas as pd


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "BookingsCheckedIn" in df.columns:
        df = df[df["BookingsCheckedIn"] > 0].copy()

    drop_cols = ["ID", "NameHash", "DocIDHash"]
    drop_now = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_now)

    if "Age" in df.columns:
        df.loc[df["Age"] < 18, "Age"] = np.nan
        df.loc[df["Age"] > 100, "Age"] = np.nan
        df["Age"] = df["Age"].fillna(df["Age"].median())

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    leakage_cols = ["LodgingRevenue", "OtherRevenue", "PersonsNights", "RoomNights", "is_high_value"]
    drop_now = [c for c in leakage_cols if c in df.columns]
    df = df.drop(columns=drop_now)

    if "Nationality" in df.columns:
        top_nations = df["Nationality"].value_counts().head(10).index.tolist()
        df["Nationality"] = df["Nationality"].apply(lambda x: x if x in top_nations else "Other")

    sr_cols = [c for c in df.columns if c.startswith("SR")]
    if sr_cols:
        df["total_special_requests"] = df[sr_cols].sum(axis=1)
    else:
        df["total_special_requests"] = 0

    if "BookingsCanceled" in df.columns:
        df["has_canceled"] = (df["BookingsCanceled"] > 0).astype(int)

    if "BookingsNoShowed" in df.columns:
        df["has_noshowed"] = (df["BookingsNoShowed"] > 0).astype(int)

    if "BookingsCheckedIn" in df.columns:
        df["is_repeat"] = (df["BookingsCheckedIn"] > 1).astype(int)

    if {"DaysSinceFirstStay", "DaysSinceLastStay"}.issubset(df.columns):
        df["tenure_days"] = df["DaysSinceFirstStay"] - df["DaysSinceLastStay"]
        df["tenure_days"] = df["tenure_days"].clip(lower=0)

    if "AverageLeadTime" in df.columns:
        df["AverageLeadTime_log"] = np.log1p(df["AverageLeadTime"].clip(lower=0))

    if "DaysSinceCreation" in df.columns:
        df["DaysSinceCreation_log"] = np.log1p(df["DaysSinceCreation"].clip(lower=0))

    return df


def align_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    df = df.copy()

    missing_cols = [c for c in feature_list if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    extra_cols = [c for c in df.columns if c not in feature_list]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    df = df[feature_list]
    return df


def predict_customer_value(input_df: pd.DataFrame, model_path: str, feature_list_path: str):
    model = load_pickle(model_path)
    feature_list = load_pickle(feature_list_path)

    df_model = clean_data(input_df)
    df_model = feature_engineering(df_model)
    X = align_features(df_model, feature_list)

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    result = df_model.copy()
    result["prediction_label"] = pred
    result["prediction_probability"] = proba
    result["value_level"] = result["prediction_label"].map({1: "High", 0: "Low"})

    return result
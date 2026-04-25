import argparse
import pickle

import numpy as np
import pandas as pd


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_input(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError("Only .csv and .xlsx input files are supported.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    exclude = [
        "reservation_status",
        "reservation_status_date",
        "arrival_date_year",
        "assigned_room_type",
        "booking_changes",
        "days_in_waiting_list",
        "country",
    ]
    drop_now = [c for c in exclude if c in df.columns]
    df = df.drop(columns=drop_now)

    if "meal" in df.columns:
        df["meal"] = df["meal"].replace("Undefined", "SC")

    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)

    if "agent" in df.columns:
        df["agent"] = df["agent"].fillna(0)

    if "company" in df.columns:
        df["company"] = df["company"].fillna(0)

    needed_guest_cols = {"adults", "children", "babies"}
    if needed_guest_cols.issubset(df.columns):
        df = df[(df["adults"] + df["children"] + df["babies"]) > 0].copy()

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_agent"] = (df["agent"] > 0).astype(int)
    df["has_company"] = (df["company"] > 0).astype(int)
    df["has_special_requests"] = (df["total_of_special_requests"] > 0).astype(int)

    for col in ["lead_time", "adr", "agent", "company"]:
        df[f"{col}_log"] = np.log1p(df[col])

    total_prev = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["cancel_rate_history"] = np.where(
        total_prev > 0,
        df["previous_cancellations"] / total_prev,
        0,
    )

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


def main(args):
    model = load_pickle(args.model_path)
    feature_list = load_pickle(args.feature_list_path)

    df_input_raw = load_input(args.input_path)

    # keep original rows for final output
    df_output = df_input_raw.copy()

    # apply same transformations as training
    df_model = clean_data(df_input_raw)
    df_model = feature_engineering(df_model)
    X = align_features(df_model, feature_list)

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    # use cleaned/valid rows for output so row counts match prediction rows
    result = df_model.copy()
    result["prediction_label"] = pred
    result["prediction_probability"] = proba

    result.to_csv(args.output_path, index=False)
    print(f"Saved predictions -> {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="xgb_best_model.pkl")
    parser.add_argument("--feature_list_path", type=str, default="feature_list.pkl")
    parser.add_argument("--output_path", type=str, default="cancellation_predictions.csv")
    args = parser.parse_args()
    main(args)
import argparse
import pickle

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


def align_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    df = df.copy()

    missing_cols = [c for c in feature_list if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in feature_list]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if extra_cols:
        df = df.drop(columns=extra_cols)

    df = df[feature_list]
    return df


def main(args):
    model = load_pickle(args.model_path)
    feature_list = load_pickle(args.feature_list_path)

    df_input = load_input(args.input_path)
    X = align_features(df_input, feature_list)

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    result = df_input.copy()
    result["prediction_label"] = pred
    result["prediction_probability"] = proba

    result.to_csv(args.output_path, index=False)
    print(f"Saved predictions -> {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="xgb_customer_value_model.pkl")
    parser.add_argument("--feature_list_path", type=str, default="customer_feature_list.pkl")
    parser.add_argument("--output_path", type=str, default="customer_value_predictions.csv")
    args = parser.parse_args()
    main(args)
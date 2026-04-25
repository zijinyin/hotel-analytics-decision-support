import argparse
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError("Only .csv and .xlsx input files are supported.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["BookingsCheckedIn"] > 0].copy()

    drop_cols = ["ID", "NameHash", "DocIDHash"]
    df = df.drop(columns=drop_cols)

    df.loc[df["Age"] < 18, "Age"] = np.nan
    df.loc[df["Age"] > 100, "Age"] = np.nan
    df["Age"] = df["Age"].fillna(df["Age"].median())

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    revenue_median = df["LodgingRevenue"].median()
    df["is_high_value"] = (df["LodgingRevenue"] > revenue_median).astype(int)
    return df


def feature_engineering(df: pd.DataFrame):
    df = df.copy()
    target = "is_high_value"

    leakage_cols = ["LodgingRevenue", "OtherRevenue", "PersonsNights", "RoomNights"]
    df_model = df.drop(columns=leakage_cols + [target]).copy()

    top_nations = df_model["Nationality"].value_counts().head(10).index.tolist()
    df_model["Nationality"] = df_model["Nationality"].apply(
        lambda x: x if x in top_nations else "Other"
    )

    sr_cols = [c for c in df_model.columns if c.startswith("SR")]
    df_model["total_special_requests"] = df_model[sr_cols].sum(axis=1)

    df_model["has_canceled"] = (df_model["BookingsCanceled"] > 0).astype(int)
    df_model["has_noshowed"] = (df_model["BookingsNoShowed"] > 0).astype(int)
    df_model["is_repeat"] = (df_model["BookingsCheckedIn"] > 1).astype(int)

    df_model["tenure_days"] = df_model["DaysSinceFirstStay"] - df_model["DaysSinceLastStay"]
    df_model["tenure_days"] = df_model["tenure_days"].clip(lower=0)

    for col in ["AverageLeadTime", "DaysSinceCreation"]:
        df_model[f"{col}_log"] = np.log1p(df_model[col].clip(lower=0))

    y = df[target].copy()
    X = df_model.copy()

    return X, y


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    return preprocessor, num_cols, cat_cols


def main(args):
    print("Loading data...")
    df = load_data(args.input_path)

    print("Cleaning data...")
    df = clean_data(df)

    print("Creating target...")
    df = create_target(df)

    print("Creating features...")
    X, y = feature_engineering(df)

    print(f"Final training matrix: {X.shape}")
    print(f"Target positive rate: {y.mean():.4f}")

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    xgb_pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training XGBoost pipeline...")
    xgb_pipe.fit(X_train, y_train)

    pred = xgb_pipe.predict(X_test)
    proba = xgb_pipe.predict_proba(X_test)[:, 1]

    print("\n=== FINAL TEST METRICS ===")
    print(f"Test F1  : {f1_score(y_test, pred):.4f}")
    print(f"Test AUC : {roc_auc_score(y_test, proba):.4f}")

    print("\nSaving model artifacts...")
    with open(args.model_output, "wb") as f:
        pickle.dump(xgb_pipe, f)

    with open(args.feature_output, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Saved model   -> {args.model_output}")
    print(f"Saved features-> {args.feature_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="HotelCustomersDataset.xlsx")
    parser.add_argument("--model_output", type=str, default="xgb_customer_value_model.pkl")
    parser.add_argument("--feature_output", type=str, default="customer_feature_list.pkl")
    args = parser.parse_args()
    main(args)
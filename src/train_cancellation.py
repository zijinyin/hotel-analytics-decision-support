import argparse
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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
    df = df.drop(columns=exclude)

    df["meal"] = df["meal"].replace("Undefined", "SC")
    df["children"] = df["children"].fillna(0)
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)

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


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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

    print("Creating features...")
    df = feature_engineering(df)

    target = "is_canceled"
    X = df.drop(columns=[target])
    y = df[target]

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

    base_xgb_pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                XGBClassifier(
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.05, 0.1],
        "clf__n_estimators": [200, 300],
        "clf__subsample": [0.8],
    }

    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=base_xgb_pipe,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_pipe = grid_search.best_estimator_
    best_pred = best_pipe.predict(X_test)
    best_proba = best_pipe.predict_proba(X_test)[:, 1]

    print("\n=== FINAL TEST METRICS ===")
    print("Best params:", grid_search.best_params_)
    print(f"Test F1  : {f1_score(y_test, best_pred):.4f}")
    print(f"Test AUC : {roc_auc_score(y_test, best_proba):.4f}")

    print("\nSaving model artifacts...")
    with open(args.model_output, "wb") as f:
        pickle.dump(best_pipe, f)

    with open(args.feature_output, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Saved model   -> {args.model_output}")
    print(f"Saved features-> {args.feature_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="hotel_bookings.csv")
    parser.add_argument("--model_output", type=str, default="xgb_best_model.pkl")
    parser.add_argument("--feature_output", type=str, default="feature_list.pkl")
    args = parser.parse_args()
    main(args)
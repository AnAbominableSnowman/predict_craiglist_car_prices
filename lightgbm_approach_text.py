import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error
from numpy import ndarray
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    cars = pd.read_parquet(filepath)
    categorical_columns = cars.select_dtypes(include=["object"]).columns.tolist()
    existing_categorical_columns = [
        col for col in categorical_columns if col in cars.columns
    ]
    if existing_categorical_columns:
        cars[existing_categorical_columns] = cars[existing_categorical_columns].astype(
            "category"
        )
    else:
        print("No categorical columns to convert.")
    return cars, existing_categorical_columns


def split_data(cars: pd.DataFrame, target_column: str):
    y = cars.pop(target_column).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        cars, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_lightgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: ndarray,
    y_test: ndarray,
    categorical_columns: list,
    hyperparameter_tuning: bool = False,
):
    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_columns
    )
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    if hyperparameter_tuning:

        def objective(params):
            params = {
                "objective": "regression",
                "metric": "mean_squared_error",
                "boosting_type": "gbdt",
                "verbose": -1,
                **params,
            }

            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
            )
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            rmse = root_mean_squared_error(y_test, y_pred)
            return {"loss": rmse, "status": STATUS_OK}

        # Define the search space
        space = {
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
            "num_leaves": hp.quniform("num_leaves", 20, 150, 1),
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "min_child_samples": hp.quniform("min_child_samples", 1, 100, 1),
            "subsample": hp.uniform("subsample", 0.1, 1.0),
        }

        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
        print("Best hyperparameters:", best)

        # Train the model with the best parameters
        model = lgb.train(
            {
                **best,
                "objective": "regression",
                "metric": "mean_squared_error",
                "boosting_type": "gbdt",
            },
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )

    else:
        params = {
            "objective": "regression",
            "metric": "mean_squared_error",
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "verbose": -1,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return model, y_pred


def evaluate_model(y_test, y_pred):
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")
    return rmse, r2


def plot_results(y_test, y_pred, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, edgecolor="k", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, edgecolor="k", alpha=0.7)
    plt.axhline(y=0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")

    plt.tight_layout()

    directory = os.path.dirname(f"{save_path}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{save_path}/predicted_vs_actual.png")


def train_fit_score_light_gbm(input_path: str, hyperparameter_tuning: bool = False):
    cars, categorical_columns = load_and_prepare_data(f"output/{input_path}.parquet")
    X_train, X_test, y_train, y_test = split_data(cars, "price")
    model, y_pred = train_lightgbm(
        X_train, X_test, y_train, y_test, categorical_columns, hyperparameter_tuning
    )
    evaluate_model(y_test, y_pred)

    model_name = "LightGBM"
    if X_train.columns.str.startswith("tfidf").any():
        model_name += "_with_words"
    if hyperparameter_tuning:
        model_name += "_and_hyperparameter_tuning"

    model_name += "/"
    plot_results(y_test, y_pred, model_name)

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error
from numpy import ndarray
import os
from hyperopt import hp, fmin, tpe


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
    return cars


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
    params: dict,
):
    categorical_columns = X_train.select_dtypes(include=["category"]).columns.tolist()

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_columns
    )
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    evals_result = {}

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5_000,
        valid_sets=[train_data, test_data],
        valid_names=["training", "validation"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, min_delta=10.0),
            lgb.record_evaluation(evals_result),
        ],
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return model, y_pred, evals_result


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
    plt.close()


def objective(params, cars, target_column):
    num_word_cols = int(params["num_word_cols"])
    tfidf_cols = [col for col in cars.columns if col.startswith("tfidf_")]
    selected_tfidf_cols = tfidf_cols[:num_word_cols]

    selected_features = selected_tfidf_cols + [
        col
        for col in cars.columns
        if col not in selected_tfidf_cols and col != target_column
    ]

    X = cars[selected_features]
    y = cars[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2018
    )

    # Set the parameters for LightGBM
    lightgbm_params = {
        "objective": "regression",
        "metric": "mean_squared_error",
        "boosting_type": "gbdt",
        "learning_rate": params["learning_rate"],
        "max_depth": int(params["max_depth"]),
        "min_data_in_leaf": 5000,  # Fixed value
        "verbose": -1,
    }

    model, y_pred, eval_results = train_lightgbm(
        X_train, X_test, y_train, y_test, lightgbm_params
    )

    rmse, _ = evaluate_model(y_test, y_pred)
    return rmse


def train_fit_score_light_gbm(input_path: str):
    cars = load_and_prepare_data(f"output/{input_path}.parquet")

    # Define the search space for Hyperopt
    space = {
        "num_word_cols": hp.quniform("num_word_cols", 50, 500, 25),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "max_depth": hp.quniform("max_depth", 4, 10, 1),
    }

    best_params = fmin(
        fn=lambda params: objective(params, cars, "price"),
        space=space,
        algo=tpe.suggest,
        max_evals=10,
    )

    best_num_word_cols = int(best_params["num_word_cols"])

    tfidf_cols = [col for col in cars.columns if col.startswith("tfidf_")]
    selected_tfidf_cols = tfidf_cols[:best_num_word_cols]

    selected_features = selected_tfidf_cols + [
        col for col in cars.columns if col not in selected_tfidf_cols
    ]

    X_train, X_test, y_train, y_test = split_data(cars[selected_features], "price")

    # Prepare params for the final model
    final_params = {
        "objective": "regression",
        "metric": "mean_squared_error",
        "boosting_type": "gbdt",
        "min_data_in_leaf": 1000,
        "learning_rate": best_params["learning_rate"],
        "max_depth": int(best_params["max_depth"]),
        "verbose": -1,
    }

    tfidf_columns = [col for col in X_train.columns if col.startswith("tfidf_")]
    print(f"Count of columns that start with 'tfidf_': {len(tfidf_columns)}")
    model, y_pred, evals_result = train_lightgbm(
        X_train, X_test, y_train, y_test, final_params
    )

    evaluate_model(y_test, y_pred)

    model_name = "LightGBM"
    if X_train.columns.str.startswith("tfidf").any():
        model_name += "_with_words"
    model_name += "/"

    plot_results(y_test, y_pred, model_name)
    plot_rmse_over_rounds(evals_result, model_name)

    import pickle

    with open(f"{model_name}/best_lightgbm_model.pkl", "wb") as file:
        pickle.dump(model, file)
    with open(f"{model_name}/final_params.pkl", "wb") as file:
        pickle.dump(final_params, file)


def plot_rmse_over_rounds(evals_result, save_path):
    plt.figure(figsize=(10, 5))
    train_rmse = evals_result["training"]["l2"]
    valid_rmse = evals_result["validation"]["l2"]

    plt.plot(train_rmse, label="Training RMSE", color="blue")
    plt.plot(valid_rmse, label="Validation RMSE", color="orange")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("RMSE")
    plt.title("RMSE Across Boosting Rounds")
    plt.legend()

    directory = os.path.dirname(f"{save_path}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f"{save_path}/rmse_over_rounds.png")
    plt.close()


# Example usage
# train_fit_score_light_gbm("your_file_name")

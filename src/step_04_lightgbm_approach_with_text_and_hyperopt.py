from __future__ import annotations
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from numpy import ndarray
import os
from hyperopt import hp, fmin, tpe
import pickle
import json
from pathlib import Path
from step_05_shap_analysis import plot_shap_summary


def fit_model(model_type: str, hyper_parm_tune: bool = False) -> None:
    # Define basic model parameters and columns
    # we could also save this off as pickle, but not today
    basic_lightgbm_params = {
        "objective": "regression",
        "metric": "root_mean_squared_error",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "max_depth": 6,
        "verbose": -1,
        "lambda_l1": 0,
        "lambda_l2": 0,
        "random_seed": 2018,
    }

    basic_cols = [
        "region",
        "price",
        "year",
        "model",
        "condition",
        "cylinders",
        "fuel",
        "odometer",
        "title_status",
        "transmission",
        "drive",
        "type",
        "paint_color",
        "state",
        "lat",
        "long",
        "manufacturer",
    ]

    if model_type == "basic":
        # Fit model three logic
        params = basic_lightgbm_params
        output_path = "results/light_gbm_basic/"
        col_subset = basic_cols

    elif model_type == "hyperopt":
        output_path = "results/light_gbm__hyperopt_and_feature_engineering/"
        col_subset = None  # Use all columns by default

        if hyper_parm_tune:
            prompt_confirmation()
            params = None  # Let hyperparameter tuning define params
        else:
            # Load the saved hyperparameters
            with open(f"{output_path}final_params.pkl", "rb") as file:
                hyperparams = pickle.load(file)

            # Convert to dictionary if loaded as a JSON string
            if isinstance(hyperparams, str):
                hyperparams = json.loads(hyperparams)

            # Calculate num_leaves based on max_depth
            hyperparams["num_leaves"] = int(2 ** hyperparams["max_depth"] * 0.65)
            params = hyperparams

    else:
        raise ValueError("Invalid model_type. Choose 'basic' or 'hyperopt'.")

    # Train model
    train_fit_light_gbm(
        input_path="cleaned_edited_feature_engineered_input",
        params=params,
        output_path=output_path,
        col_subset=col_subset,
    )

    # Score model on test data
    score_model_on_test(
        parquet_file="intermediate_data/test_data.parquet",
        model_file=f"{output_path}best_lightgbm_model.pkl",
        model_name=output_path,
        col_subset=col_subset,
    )

    # Plot SHAP summary
    plot_shap_summary(
        model_path=f"{output_path}best_lightgbm_model.pkl",
        data_path="intermediate_data/cleaned_edited_feature_engineered_input.parquet",
        output_dir=output_path,
        col_subset=col_subset,
    )


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    cars = pd.read_parquet(filepath)
    categorical_columns = cars.select_dtypes(include="object").columns.tolist()
    cars[categorical_columns] = cars[categorical_columns].astype("category")
    return cars


def split_data(
    cars: pd.DataFrame, target_column: str
) -> list[pd.DataFrame, pd.DataFrame, ndarray, ndarray]:
    y = cars.pop(target_column).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        cars, y, test_size=0.2, random_state=2018
    )
    return X_train, X_val, y_train, y_val


def train_lightgbm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: ndarray,
    y_val: ndarray,
    params: dict,
) -> list[lgb.booster, ndarray, dict]:
    categorical_columns = X_train.select_dtypes(include=["category"]).columns.tolist()

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=categorical_columns
    )
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    evals_result = {}

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5_000,
        valid_sets=[train_data, val_data],
        valid_names=["training", "validation"],
        callbacks=[
            # 10 dollars feels like a meaningful level of imporement but also small.
            # this could be hyper parm tuned but i dont have compute for it.
            lgb.early_stopping(stopping_rounds=10, min_delta=10.0),
            lgb.record_evaluation(evals_result),
        ],
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    return model, y_pred, evals_result


def evaluate_model(y_actual, y_pred, model_path) -> list[float, float]:
    rmse = root_mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")
    # Save results to a JSON file
    results = {"RMSE": rmse, "R-squared": r2}

    # Define the model path
    model_path = Path(model_path)

    # Create the directory if it does not exist
    model_path.mkdir(parents=True, exist_ok=True)

    # Save results to JSON file
    with open(model_path / "evaluation_results.json", "w") as json_file:
        json.dump(results, json_file)
    return rmse, r2


def plot_results(y_actual, y_pred, save_path) -> None:
    plt.figure()
    residuals = y_actual - y_pred
    plt.scatter(y_pred, residuals, edgecolor="k", alpha=0.4)
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


def objective(params, cars, target_column) -> None:
    selected_features = [col for col in cars.columns if col != target_column]

    X = cars[selected_features]
    y = cars[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=2018
    )

    # Set the parameters for LightGBM
    lightgbm_params = {
        "objective": "regression",
        "metric": "root_mean_squared_error",
        "learning_rate": params["learning_rate"],
        "max_depth": int(params["max_depth"]),
        "min_data_in_leaf": 2000,  # Fixed value
        "num_leaves": int(2 ** int(params["max_depth"]) * 0.65),
        "lambda_l1": params["lambda_l1"],
        "lambda_l2": params["lambda_l2"],
        "verbose": -1,
        "random_seed": 2018,
    }

    model, y_pred, eval_results = train_lightgbm(
        X_train, X_val, y_train, y_val, lightgbm_params
    )

    rmse = root_mean_squared_error(y_val, y_pred)
    return rmse


def train_fit_light_gbm(
    input_path: str, params: dict, output_path: str, col_subset: list[str]
) -> None:
    cars = load_and_prepare_data(f"intermediate_data/{input_path}.parquet")
    if col_subset is not None:
        if "price" not in col_subset:
            col_subset.append("price")
        cars = cars[col_subset]

    model_name = "results/LightGBM"

    # If no params are passed, run the optimization using Hyperopt's fmin
    if params is None:
        # Define the search space for Hyperopt (if not using predefined params)
        space = {
            # using log uniform here, we can get more sampling in the lower smaller regions.
            # e^-75 = .006, e^-1.7 - .18
            "learning_rate": hp.loguniform("learning_rate", -7, -1),
            "max_depth": hp.quniform("max_depth", 4, 10, 1),
            "lambda_l1": hp.uniform(
                "lambda_l1", 0.0, 100.0
            ),  # 0 to 100 suggested by https://www.kaggle.com/code/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-
            "lambda_l2": hp.uniform(
                "lambda_l2", 0.0, 100.0
            ),  # 0 to 100 suggested by https://www.kaggle.com/code/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-understanding
        }
        best_params = fmin(
            fn=lambda params: objective(params, cars, "price"),
            space=space,
            algo=tpe.suggest,
            max_evals=50,
        )

        best_params["max_depth"] = int(best_params["max_depth"])
        model_name = model_name + "Hyperopt"
    else:
        # Use the provided params directly
        best_params = params

    X_train, X_val, y_train, y_val = split_data(cars, "price")

    # Prepare params for the final model
    final_params = {
        "objective": "regression",
        "metric": "root_mean_squared_error",
        "min_data_in_leaf": 2000,
        "learning_rate": best_params["learning_rate"],
        "max_depth": best_params["max_depth"],
        "num_leaves": int(2 ** int(best_params["max_depth"]) * 0.65),
        "lambda_l1": best_params["lambda_l1"],
        "lambda_l2": best_params["lambda_l2"],
        "verbose": -1,
        "random_seed": 2018,
    }

    model, y_pred, evals_result = train_lightgbm(
        X_train, X_val, y_train, y_val, final_params
    )
    model_name = model_name + "/"

    if output_path is not None:
        model_name = output_path

    plot_rmse_over_rounds(evals_result, model_name)

    # Save the model and parameters
    with open(f"{model_name}/best_lightgbm_model.pkl", "wb") as file:
        pickle.dump(model, file)
    with open(f"{model_name}/final_params.pkl", "wb") as file:
        pickle.dump(final_params, file)


def plot_rmse_over_rounds(evals_result, save_path) -> None:
    # light gbm has a built in method to do this BUT
    # it wasn't as good as I'd like.
    plt.figure(figsize=(10, 5))
    train_rmse = evals_result["training"]["rmse"]
    valid_rmse = evals_result["validation"]["rmse"]

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


def score_model_on_test(
    parquet_file: str, model_file: str, model_name: str, col_subset
) -> None:
    test_data = load_and_prepare_data(parquet_file)
    if col_subset:
        test_data = test_data[col_subset]
    y_test = test_data.pop("price").to_numpy()

    # Load the trained model
    with open(model_file, "rb") as file:
        model = pickle.load(file)

    # Make predictions
    y_pred = model.predict(test_data)

    evaluate_model(y_test, y_pred, model_name)
    plot_results(y_test, y_pred, model_name)


def prompt_confirmation() -> None:
    # Prompt the user for confirmation
    response = (
        input(
            """You are about to manually tune a Light GBM model. 
            This takes about 3 hours on my modest machine. 
            Are you sure you wish to proceed? (yes/no): """
        )
        .strip()
        .lower()
    )

    # Check the response
    if response in ["yes", "y"]:
        print(
            """User confirmed to proceed! You can cancel at any time. 
            My hyper optimized parameters are stored in a different folder 
            and can be used if you over write them here."""
        )
    elif response in ["no", "n"]:
        print("User canceled the action.")
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

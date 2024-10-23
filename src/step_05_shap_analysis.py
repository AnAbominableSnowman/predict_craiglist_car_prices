import os
import pickle
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def load_model(model_path: str):
    """Load a pickled model."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def load_data(data_path: str, col_subset):
    """Load dataset and process categorical columns."""
    cars = pd.read_parquet(data_path)
    if col_subset is not None:
        cars = cars[col_subset]
    cars = cars.sample(80_000)

    categorical_columns = cars.select_dtypes(include=["object"]).columns.tolist()
    existing_categorical_columns = [
        col for col in categorical_columns if col in cars.columns
    ]
    if existing_categorical_columns:
        cars[existing_categorical_columns] = cars[existing_categorical_columns].astype(
            "category"
        )
    cars.pop("price")
    return cars


def shap_analysis(model, X: pd.DataFrame):
    """Perform SHAP analysis and return SHAP values."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


def save_shap_to_json(
    model_path: str, data_path: str, output_dir: str, col_subset: list
):
    """Save SHAP values to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path)
    data = load_data(data_path, col_subset)

    shap_values = shap_analysis(model, data)

    # Save SHAP values to JSON
    shap_json_path = os.path.join(output_dir, "shap_values.json")
    with open(shap_json_path, "w") as f:
        json.dump(shap_values.values.tolist(), f)
    print(f"SHAP values saved to {shap_json_path}")


def plot_shap_dependence_for_categoricals(
    shap_json_path: str, data_path: str, output_dir: str, col_subset: list
):
    """Plot SHAP summary and dependence plots for all categorical variables."""

    os.makedirs(output_dir, exist_ok=True)

    # Load SHAP values from JSON
    with open(shap_json_path, "r") as f:
        shap_values = np.array(json.load(f))

    data = load_data(data_path, col_subset)

    # Plot SHAP summary plot
    print("Creating SHAP summary plot...")
    shap.summary_plot(shap_values, data, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()

    # Plot SHAP dependence plots for categorical variables
    categorical_columns = X.select_dtypes(include=["category"]).columns.tolist()

    for cat_col in categorical_columns:
        print(f"Creating SHAP dependence plot for {cat_col}...")
        shap.dependence_plot(cat_col, shap_values, data, show=False)
        plt.savefig(os.path.join(output_dir, f"shap_dependence_{cat_col}.png"))
        plt.close()

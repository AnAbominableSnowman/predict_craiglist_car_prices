import os
import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt


def load_model(model_path: str):
    """Load a pickled model."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def load_data(data_path: str, col_subset: list = None):
    """Load dataset and process categorical columns."""
    cars = pd.read_parquet(data_path)
    if col_subset is not None:
        cars = cars[col_subset]
    cars = cars.sample(200_000, random_state=2018)
    categorical_columns = cars.select_dtypes(
        include=[
            "object",
        ]
    ).columns.tolist()
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


def plot_shap_dependence_for_categoricals(
    model_path: str, data_path: str, output_dir: str, col_subset: list
):
    """Plot SHAP summary and dependence plots for all categorical variables."""
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path)
    data = load_data(data_path, col_subset)

    shap_values = shap_analysis(model, data)

    # Plot SHAP summary plot
    print("Creating SHAP summary plot...")
    shap.summary_plot(shap_values.values, data, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()

    # # Plot SHAP dependence plots for categorical variables
    # # categorical_columns = data.select_dtypes(include=["category"]).columns.tolist()
    # from numpy import number

    # # Plot SHAP dependence plots for numeric variables only
    # numeric_columns = data.select_dtypes(include=[number]).columns.tolist()
    # print(numeric_columns)
    # for num_col in numeric_columns:
    #     print(f"Creating SHAP dependence plot for {num_col}...")
    #     shap.dependence_plot(num_col, shap_values.values, data, show=False)
    #     plt.savefig(os.path.join(output_dir, f"shap_dependence_{num_col}.png"))
    #     plt.close()

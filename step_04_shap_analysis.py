import pickle
import shap
import pandas as pd
import numpy as np


def load_model(model_path: str):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def load_data(data_path: str):
    # Adjust this function to load your data
    # For example, if your data is in a CSV file:
    data = pd.read_parquet(data_path)
    return data


def shap_analysis(model, X: pd.DataFrame):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    return shap_values


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame):
    shap.summary_plot(shap_values, X)


def plot_shap_interaction(shap_values: np.ndarray, X: pd.DataFrame):
    # Select two features for interaction
    feature1 = X.columns[0]  # Change to a specific feature if needed
    feature2 = X.columns[1]  # Change to a specific feature if needed

    shap.dependence_plot(feature1, shap_values.values, X, interaction_index=feature2)


def main(model_path: str, data_path: str):
    model = load_model(model_path)
    data = load_data(data_path)

    # Separate features from the target variable if necessary
    X = data.drop(
        columns=["price"]
    )  # Replace "target" with your actual target column name

    shap_values = shap_analysis(model, X)

    # Plot SHAP summary
    plot_shap_summary(shap_values, X)

    # Plot SHAP interaction
    plot_shap_interaction(shap_values, X)


if __name__ == "__main__":
    model_path = "best_lightgbm_model.pkl"  # Path to your pickled model
    data_path = "data.csv"  # Path to your data file

    main(model_path, data_path)

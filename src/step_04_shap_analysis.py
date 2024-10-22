import os
import pickle
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_model(model_path: str):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def load_data(data_path: str):
    cars = pd.read_parquet(data_path)
    cars = cars.sample(n=10_000, random_state=2018)

    categorical_columns = cars.select_dtypes(include=["object"]).columns.tolist()

    existing_categorical_columns = [
        col for col in categorical_columns if col in cars.columns
    ]
    if existing_categorical_columns:
        cars[existing_categorical_columns] = cars[existing_categorical_columns].astype(
            "category"
        )

    return cars


def shap_analysis(model, X: pd.DataFrame):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    return shap_values


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, output_dir: str):
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.close()


def plot_shap_interaction(shap_values: np.ndarray, X: pd.DataFrame, output_dir: str):
    # Select two features for interaction
    feature1 = X.columns[0]  # Change to a specific feature if needed
    feature2 = X.columns[1]  # Change to a specific feature if needed

    shap.dependence_plot(
        feature1, shap_values.values, X, interaction_index=feature2, show=False
    )
    plt.savefig(os.path.join(output_dir, "shap_interaction_plot.png"))
    plt.close()


def main(model_path: str, data_path: str, output_dir: str):
    os.makedirs(
        output_dir, exist_ok=True
    )  # Create the output directory if it doesn't exist
    model = load_model(model_path)
    data = load_data(data_path)

    # Separate features from the target variable if necessary
    X = data.drop(
        columns=["price"]
    )  # Replace "price" with your actual target column name

    shap_values = shap_analysis(model, X)

    # Plot SHAP summary
    plot_shap_summary(shap_values, X, output_dir)

    # Plot SHAP interaction
    plot_shap_interaction(shap_values, X, output_dir)


model_path = "LightGBM_with_words/best_lightgbm_model.pkl"  # Path to your pickled model
data_path = (
    "output/cleaned_edited_feature_engineered_input.parquet"  # Path to your data file
)
output_dir = "LightGBM_with_words/"  # Directory to save the plots

main(model_path, data_path, output_dir)

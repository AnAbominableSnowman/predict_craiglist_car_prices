import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
import statsmodels.api as sm
import os
from numpy import ndarray, log
import polars as pl
from sklearn.impute import SimpleImputer


def fit_model_one(path: str = "intermediate_data/cleaned_and_edited_input.parquet"):
    cars = pl.read_parquet(path)
    # linear regression won't handle this missing vals nicely. So here I,
    # encode them with mean or mode. This isn't as elegant as possible.
    # But sufficent for a first pass with linear regression.
    cars_imputed_missing_for_lin_regrs = fill_missing_values_column_level(
        cars,
        ["odometer"],
    )
    cars_imputed_missing_for_lin_regrs = cars_imputed_missing_for_lin_regrs.to_pandas()
    y = cars_imputed_missing_for_lin_regrs.pop("price").to_numpy()
    X = cars_imputed_missing_for_lin_regrs

    train_fit_score_linear_regression(X["odometer"], y, log=False, one_hot_encode=False)
    # x.to_numpy()


def fit_model_two(path: str = "intermediate_data/cleaned_and_edited_input.parquet"):
    cars = pl.read_parquet(path)
    covariates = [
        "odometer",
        "year",
        "manufacturer",
        "state",
        "title_status",
        "paint_color",
    ]
    # linear regression won't handle these missing vals nicely. So here I,
    # encode them with mean or mode. This isn't as elegant as possible.
    # But sufficent for a first pass with linear regression.
    cars_imputed_missing_for_lin_regrs = fill_missing_values_column_level(
        cars,
        covariates,
    )
    cars_imputed_missing_for_lin_regrs = cars_imputed_missing_for_lin_regrs.to_pandas()
    y = cars_imputed_missing_for_lin_regrs.pop("price").to_numpy()
    X = cars_imputed_missing_for_lin_regrs

    train_fit_score_linear_regression(
        X[covariates], log(y), log=True, one_hot_encode=True
    )


def train_fit_score_linear_regression(
    X: pd.DataFrame, y: ndarray, log: bool, one_hot_encode: bool
):
    if one_hot_encode:
        X = one_hot_columns(X)
    # Add a constant term for the intercept (as statsmodels does not include it by default)
    X = sm.add_constant(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2018
    )

    # Create and fit the linear regression model using statsmodels
    model = sm.OLS(y_train, X_train).fit()

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    if log:
        y_pred = np.exp(y_pred)
        y_test = np.exp(y_test)
        model_name = "Log price Linear Regression/"
    else:
        model_name = "Simple Linear Regression of Price by Odometer/"

    save_path = "results/"
    # Ensure the directory exists
    directory = os.path.dirname(f"{save_path}{model_name}")
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_results(y_test, y_pred, log, directory)
    print_results(y_test, y_pred, model, directory)
    return model


def plot_results(y_test: ndarray, y_pred: ndarray, log: bool, directory) -> None:
    # Plot predicted vs actual values
    plt.figure(figsize=(10, 5))

    # Plot 1: Predicted vs Actual values
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, edgecolor="k", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")

    # Plot 2: Residuals plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, edgecolor="k", alpha=0.7)
    plt.axhline(y=0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")

    plt.tight_layout()

    plt.savefig(f"{directory}/predicted_vs_actual.png")  # Save Predicted vs Actual plot
    plt.figure(figsize=(5, 5))

    # Residuals plot
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, edgecolor="k", alpha=0.7)
    plt.axhline(y=0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")

    plt.tight_layout()

    # Save the residuals plot
    plt.savefig(f"{directory}/residuals.png")


def print_results(y_test: ndarray, y_pred: ndarray, model, directory) -> None:
    # Calculate RMSE and R2 score
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")

    # Print the summary of the regression model
    print("\nModel Summary:")
    print(model.summary())
    with open(f"{directory}/ols_summary.txt", "w") as f:
        f.write(model.summary().as_text())
        f.write(f"rsme is: {rmse}")


def one_hot_columns(cars: pd.DataFrame) -> pd.DataFrame:
    # Create an empty list to store processed columns
    processed_columns = []

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(drop="first", sparse_output=False)

    # Step 1: Process each column based on its dtype
    for col in cars.columns:
        if cars[col].dtype == "bool":  # Convert boolean columns to integers
            cars[col] = cars[col].astype(int)
            processed_columns.append(cars[[col]])  # Append processed column

        elif cars[col].dtype == "object":  # One-hot encode string/categorical columns
            encoded = encoder.fit_transform(cars[[col]])
            encoded_cars = pd.DataFrame(
                encoded, columns=encoder.get_feature_names_out([col])
            )
            processed_columns.append(encoded_cars)  # Append one-hot encoded columns

        elif col != "price":  # Include numeric columns as they are
            processed_columns.append(cars[[col]])

    X = pd.concat(processed_columns, axis=1)
    return X


def fill_missing_values_column_level(
    df: pl.DataFrame, columns: list[str]
) -> pl.DataFrame:
    # not sure if this is needed?
    updated_df = df.clone()

    # doing it col by col is inefficent,
    # but this is only done for a few columns and is a pretty cheap
    # operation so I wont optimize.
    for col in columns:
        if col in df.columns:
            col_type = df.schema[col]
            # Reshape for the imputer
            data = updated_df[col].to_numpy().reshape(-1, 1)
            # Create the appropriate imputer
            if col_type == pl.Int64 or col_type == pl.UInt32:
                imputer = SimpleImputer(strategy="mean")
            elif col_type == pl.Utf8:
                imputer = SimpleImputer(strategy="most_frequent")
            else:
                continue

            imputed_data = imputer.fit_transform(data)

            updated_df = updated_df.with_columns(
                pl.Series(name=col, values=imputed_data.flatten())
            )

    return updated_df

import polars as pl
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np


def generate_profiling_report(
    data: pl.DataFrame, output_path: str, subsample_percent: float
) -> None:
    cars_pandas = data.to_pandas()
    cars_pandas = cars_pandas.sample(frac=subsample_percent)

    profile = ProfileReport(cars_pandas, title="Pandas Profiling Report")
    profile.to_file(output_path)


import polars as pl
from pathlib import Path


def plot_histogram(data: pl.DataFrame, column: str, bins: int = 100) -> None:
    data = data.filter(pl.col(column).is_not_null())
    values = data[column].to_list()

    # Create a KDE object
    kde = gaussian_kde(values)

    # Generate a range of values for the x-axis
    x = np.linspace(min(values), max(values), 1000)
    y = kde(x)

    # Plotting the KDE
    plt.figure(figsize=(10, 6))
    plt.fill_between(x, y, color="#2f2e65", alpha=0.7)
    if column.lower() == "price":
        plt.xlabel(f"{column.capitalize()} in US Dollars")
    elif column.lower() == "odometer":
        plt.xlabel(f"{column.capitalize()} in Miles")
    plt.ylabel("Density")
    plt.title(f"Histogram of {column.capitalize()}")
    plt.grid(axis="y")

    output_path = Path(f"results/visuals/histogram_of_{column}.png")

    # Create the directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the histogram as a PNG file
    plt.savefig(output_path)
    plt.close()


def kde_of_category_by_value(
    cars: pl.DataFrame, category_column: str, value_column: str
) -> None:
    # Verify if the category and value columns exist
    if category_column in cars.columns and value_column in cars.columns:
        unique_categories = cars[category_column].unique().to_list()

        plt.figure(figsize=(12, 8))

        # Loop through each unique category to plot their value distribution
        for category_value in unique_categories:
            filtered_values = cars.filter(pl.col(category_column) == category_value)[
                value_column
            ].to_list()

            if len(filtered_values) > 1:  # At least two values needed for KDE
                # Calculate KDE
                kde = gaussian_kde(filtered_values)
                x = np.linspace(
                    min(filtered_values), max(filtered_values), 100
                )  # Create a range for the x-axis
                plt.plot(x, kde(x), label=category_value)  # Plot the KDE
            else:
                print(f"Not enough data for {category_column}: {category_value}")

        if value_column.lower() == "price":
            plt.xlabel(f"{value_column.capitalize()} in US Dollars")
        elif value_column.lower() == "odometer":
            plt.xlabel(f"{value_column.capitalize()} in Miles")
        plt.ylabel("Density")
        plt.title(
            f"Histograms of {value_column.capitalize()} by {category_column.capitalize()}"
        )
        plt.legend(title=f"{category_column.capitalize()}")
        output_path = Path(
            f"results/visuals/histogram_of_{value_column}_by_{category_column}.png"
        )

        # Create the directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the histogram as a PNG file
        plt.savefig(output_path)

    else:
        print(
            f"The '{value_column}' or '{category_column}' column does not exist in the DataFrame."
        )


def percent_below_threshold(cars: pl.DataFrame, threshold: float, col_to_check: str):
    count_below_threshold = cars.filter(pl.col(col_to_check) < threshold).height
    total_rows = cars.height
    percentage_below_threshold = (
        (count_below_threshold / total_rows) * 100 if total_rows > 0 else 0
    )

    print(f"Percentage of rows below {threshold}: {percentage_below_threshold:.2f}%")

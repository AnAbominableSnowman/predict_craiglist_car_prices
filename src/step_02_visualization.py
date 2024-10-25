import polars as pl
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from step_00_load_and_clean_input import replace_rare_and_null_manufacturer
from pathlib import Path


def generate_profiling_report(
    data: pl.DataFrame, output_path: str, subsample_percent: float
) -> None:
    cars_pandas = data.to_pandas()
    cars_pandas = cars_pandas.sample(frac=subsample_percent)

    profile = ProfileReport(cars_pandas, title="Pandas Profiling Report")
    profile.to_file(output_path)


def visuals_for_report_hist_and_first_kde(
    path: str = r"intermediate_data/cleaned_and_edited_input.parquet",
):
    cars = pl.read_parquet(path)
    cars = replace_rare_and_null_manufacturer(cars, 3, "Other")

    plot_histogram(cars, "price")
    plot_histogram(cars, "odometer")
    kde_of_category_by_value(cars, "manufacturer", "price")


def visuals_for_report_second_kde_and_data_dict(
    path: str = r"intermediate_data/cleaned_and_edited_input.parquet",
):
    cars = pl.read_parquet(path)
    cars = replace_rare_and_null_manufacturer(cars, 3, "Other")

    kde_of_category_by_value(cars, "manufacturer", "odometer")
    column_statistics(cars)
    count_empty_description_rows(cars)


def generate_ydata_eda(raw_or_clean: str):
    if raw_or_clean.lower == "raw":
        raw_cars = pl.read_parquet("intermediate_data/raw_input.parquet")
        generate_profiling_report(
            # emojis in description kill y_data
            raw_cars.drop("description"),
            "results/data_profile_raw_subsampled_to_three_percent.html",
            0.03,
        )
    elif raw_or_clean.lower == "clean":
        clean_cars = pl.read_parquet(
            "intermediate_data/cleaned_and_edited_input.parquet"
        )
        # Select columns that don't start with 'TF_IDF'
        non_tf_idf_columns = [
            col for col in clean_cars.columns if not col.startswith("tf")
        ]

        # Select the first 10 columns that start with 'TF_IDF'
        tf_idf_columns = [col for col in clean_cars.columns if col.startswith("tf")][
            :10
        ]

        # Combine the two selections
        selected_columns = list(set(non_tf_idf_columns + tf_idf_columns))
        generate_profiling_report(
            # emojis in description kill y_data
            clean_cars.select(selected_columns),
            "results/data_profile_cleaned_subsampled_to_three_percent.html",
            0.03,
        )


def column_statistics(df: pl.DataFrame) -> None:
    row_count = df.height
    print(f"Total Row Count: {row_count}")

    for col in df.columns:
        total_count = row_count
        missing_count = df[col].null_count()
        zero_count = (
            df.filter(pl.col(col) == 0).height
            if df.schema[col] in [pl.Int64, pl.Float64]
            else 0
        )
        distinct_count = df[col].n_unique()
        print(
            f"Column Name{ col} % Missing{ round((missing_count / total_count) * 100,1)}% Zeros {round((zero_count / total_count) * 100,1)}% Distinct{ round((distinct_count / total_count) * 100,1)}"
        )


def count_empty_description_rows(df: pl.DataFrame) -> None:
    empty_string_count = df.filter(
        (pl.col("description") == "") | pl.col("description").is_null()
    ).height
    print(
        f"Number of rows where 'description' is an empty string: {empty_string_count}"
    )


def plot_histogram(data: pl.DataFrame, column: str) -> None:
    # Filter out null values
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

    # Set x-axis label based on column type
    if column.lower() == "price":
        plt.xlabel(f"{column.capitalize()} in US Dollars", fontsize=16)
    elif column.lower() == "odometer":
        plt.xlabel(f"{column.capitalize()} in Miles", fontsize=16)
    else:
        plt.xlabel(column.capitalize(), fontsize=16)

    plt.ylabel("Density", fontsize=16)
    plt.title(f"Histograms of {column.capitalize()}", fontsize=18)
    plt.grid(axis="y")
    plt.tick_params(axis="both", labelsize=14)
    # Specify output path for saving the plot
    output_path = Path(f"results/visuals/histogram_of_{column}.png")

    # Create the directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the KDE plot as a PNG file
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
            plt.xlabel(f"{value_column.capitalize()} in US Dollars", fontsize=18)
        elif value_column.lower() == "odometer":
            plt.xlabel(f"{value_column.capitalize()} in Miles", fontsize=18)

        plt.ylabel("Density", fontsize=18)
        plt.title(
            f"Histograms of {value_column.capitalize()} by {category_column.capitalize()}",
            fontsize=18,
        )
        plt.tick_params(axis="both", labelsize=14)
        plt.legend(
            title=f"{category_column.capitalize()}", fontsize=16, title_fontsize=18
        )

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

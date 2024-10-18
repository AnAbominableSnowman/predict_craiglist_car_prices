import polars as pl
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import polars as pl

def generate_profiling_report(data: pl.DataFrame, output_path: str) -> None:

    cars_pandas = data.to_pandas()

    profile = ProfileReport(cars_pandas, title="Pandas Profiling Report")
    profile.to_file(output_path)

def plot_histogram(data: pl.DataFrame, column: str, bin_width: int = 500) -> None:

    values = data[column].to_list()
    
    # Calculate the number of bins based on the range of values
    bins = range(int(min(values)), int(max(values)) + bin_width, bin_width)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel(column.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column.capitalize()}")
    plt.grid(axis='y')
    plt.show()


def kde_of_col_1_by_col2(cars:pl.DataFrame,col_1:str,col_2:str):
    # Verify if the manufacturer column exists
    if col_1 in cars.columns and col_2 in cars.columns:
        values_of_col_1 = cars[col_1].unique().to_list()

        plt.figure(figsize=(12, 8))

        # Loop through each col_1 to plot their price distribution
        for filter_value in values_of_col_1:
            col_2s = cars.filter(pl.col(col_1) == filter_value)[col_2].to_list()
            
            if len(col_2s) > 1:  # At least two prices needed for KDE
                # Calculate KDE
                kde = gaussian_kde(col_2s)
                x = np.linspace(min(col_2s), max(col_2s), 100)  # Create a range for the x-axis
                plt.plot(x, kde(x), label=col_1)  # Plot the KDE
            else:
                print(f"Not enough data for {col_1}: {filter_value}")

        plt.xlabel(col_2)
        plt.ylabel("Density")
        plt.title(f"KDE of {col_2} by {col_1}")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print(f"The '{col_2}' or '{col_1}' column does not exist in the DataFrame.")


def percent_below_threshold(cars: pl.DataFrame,threshold: float, col_to_check:str):

    count_below_threshold = cars.filter(pl.col(col_to_check) < threshold).height
    total_rows = cars.height
    percentage_below_threshold = (count_below_threshold / total_rows) * 100 if total_rows > 0 else 0

    print(f"Percentage of rows below {threshold}: {percentage_below_threshold:.2f}%")
    



cars_raw = pl.read_parquet("output/raw_input.parquet")
cars_cleaned = pl.read_parquet("output/cleaned_engineered_input.parquet")

generate_profiling_report(cars_raw.limit(25_000), output_path="output/raw_sub_sampled_data_profiling_report.html", display_in_notebook=True)
generate_profiling_report(cars_cleaned.limit(25_000), output_path="output/cleaned_sub_sampled_data_profiling_report.html", display_in_notebook=True)

plot_histogram(cars_raw, column='price')
plot_histogram(cars_cleaned, column='price') 

plot_histogram(cars_raw, column='odometer') 
plot_histogram(cars_cleaned, column='odometer')  

kde_of_col_1_by_col2(cars_raw,"manufacturer","price")
kde_of_col_1_by_col2(cars_cleaned,"manufacturer","price")
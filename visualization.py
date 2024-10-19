import polars as pl
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np



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
    plt.close()


def kde_of_category_by_value(cars: pl.DataFrame, category_column: str, value_column: str) -> None:
    # Verify if the category and value columns exist
    if category_column in cars.columns and value_column in cars.columns:
        unique_categories = cars[category_column].unique().to_list()

        plt.figure(figsize=(12, 8))

        # Loop through each unique category to plot their value distribution
        for category_value in unique_categories:
            filtered_values = cars.filter(pl.col(category_column) == category_value)[value_column].to_list()
            
            if len(filtered_values) > 1:  # At least two values needed for KDE
                # Calculate KDE
                kde = gaussian_kde(filtered_values)
                x = np.linspace(min(filtered_values), max(filtered_values), 100)  # Create a range for the x-axis
                plt.plot(x, kde(x), label=category_value)  # Plot the KDE
            else:
                print(f"Not enough data for {category_column}: {category_value}")

        plt.xlabel(value_column.capitalize())
        plt.ylabel("Density")
        plt.title(f"KDE of {value_column.capitalize()} by {category_column.capitalize()}")
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()
    else:
        print(f"The '{value_column}' or '{category_column}' column does not exist in the DataFrame.")


def percent_below_threshold(cars: pl.DataFrame,threshold: float, col_to_check:str):

    count_below_threshold = cars.filter(pl.col(col_to_check) < threshold).height
    total_rows = cars.height
    percentage_below_threshold = (count_below_threshold / total_rows) * 100 if total_rows > 0 else 0

    print(f"Percentage of rows below {threshold}: {percentage_below_threshold:.2f}%")
    



# cars_raw = pl.read_parquet("output/raw_input.parquet")
cars_cleaned = pl.read_parquet("output/cleaned_engineered_input.parquet")

# # Get the list of columns that start with 'tfidf_'
# tfidf_columns = [col for col in cars_cleaned.columns if col.startswith("tfidf_")][-10:]

# # Get all columns that don't start with 'tfidf_'
# non_tfidf_columns = [col for col in cars_cleaned.columns if not col.startswith("tfidf_")]


# # Generate profiling report on the last 10 tfidf_ columns and limit to 5000 rows
# generate_profiling_report(
#     cars_raw.limit(5_000), 
#     output_path="output/raw_sub_sampled_data_profiling_report.html"
# )
# generate_profiling_report(cars_cleaned.select(non_tfidf_columns+tfidf_columns).limit(5_000), output_path="output/cleaned_sub_sampled_data_profiling_report.html",)

# plot_histogram(cars_raw.limit(5_000), column='price')
# plot_histogram(cars_cleaned, column='price') 

# plot_histogram(cars_raw, column='odometer') 
# plot_histogram(cars_cleaned, column='odometer')  

# kde_of_category_by_value(cars_raw,"manufacturer","price")
kde_of_category_by_value(cars_cleaned,"manufacturer","price")
kde_of_category_by_value(cars_cleaned,"manufacturer","odometer")
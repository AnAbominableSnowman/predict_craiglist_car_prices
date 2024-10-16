import polars as pl
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


# Write the DataFrame to a Parquet file
cars = pl.read_parquet("output/cleaned_engineered_input.parquet").limit(50_000)
print(cars.columns)
# Convert Polars DataFrame to Pandas DataFrame
# cars = cars.to_pandas()

# # Run profiling
# profile = ProfileReport(cars, title="Pandas Profiling Report")

# # Save the report to an HTML file
# profile.to_file("output/profiling_report.html")

# Optionally, display the report in a Jupyter notebook
# profile.to_notebook_iframe()


prices = cars['price'].to_list()
bin_width = 500

# Calculate the number of bins based on the range of prices
bins = range(int(min(prices)), int(max(prices)) + bin_width, bin_width)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(prices, bins=bins, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Histogram of Prices")
plt.grid(axis='y')
plt.show()


# Verify if the manufacturer column exists
if 'manufacturer' in cars.columns and 'price' in cars.columns:
    manufacturers = cars['manufacturer'].unique().to_list()

    plt.figure(figsize=(12, 8))

    # Loop through each manufacturer to plot their price distribution
    for manufacturer in manufacturers:
        prices = cars.filter(pl.col("manufacturer") == manufacturer)['price'].to_list()
        
        if len(prices) > 1:  # At least two prices needed for KDE
            # Calculate KDE
            kde = gaussian_kde(prices)
            x = np.linspace(min(prices), max(prices), 100)  # Create a range for the x-axis
            plt.plot(x, kde(x), label=manufacturer)  # Plot the KDE
        else:
            print(f"Not enough data for manufacturer: {manufacturer}")

    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.title("KDE of Prices by Manufacturer")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("The 'manufacturer' or 'price' column does not exist in the DataFrame.")
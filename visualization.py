import polars as pl
import pandas as pd
from ydata_profiling import ProfileReport

# Write the DataFrame to a Parquet file
claims = pl.read_parquet("cleaned_input.parquet").limit(50_000)
# Convert Polars DataFrame to Pandas DataFrame
claims = claims.to_pandas()

# Run profiling
profile = ProfileReport(claims, title="Pandas Profiling Report")

# Save the report to an HTML file
profile.to_file("profiling_report.html")

# Optionally, display the report in a Jupyter notebook
profile.to_notebook_iframe()

import polars as pl
import pandas as pd
from ydata_profiling import ProfileReport



cars = pl.read_parquet("output/cleaned_input.parquet")
total_rows = cars.height

cars = cars.with_columns(pl.col("manufacturer").alias("org_manuf")).drop("manufacturer")
# Group by the 'manufacturer' column and count occurrences
grouped_df = (cars
    .group_by("org_manuf")
    .agg(pl.count().alias("count"))
    .with_columns((pl.col("count") / total_rows * 100).alias("percent_of_total"))
)

print(grouped_df)
# Replace manufacturers with less than 3% of total with "Other"
grouped_df = grouped_df.with_columns(
    pl.when(pl.col("percent_of_total") < 3)
    .then(pl.lit("Other"))
    .when(pl.col("org_manuf").is_null())
    .then(pl.lit("Other"))
    .otherwise(pl.col("org_manuf"))
    .alias("manufacturer")
).select("manufacturer").unique()
print(grouped_df)

joined_df = cars.join(grouped_df, left_on="org_manuf", right_on="manufacturer", how="left", coalesce=False)

print(joined_df)

joined_df.write_parquet("output/cleaned_engineered_input.parquet")
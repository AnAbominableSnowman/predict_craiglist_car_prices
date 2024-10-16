import polars as pl
from rich import print
import polars.selectors as cs
import polars as pl
from sklearn.impute import SimpleImputer
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
import numpy as np

cars = pl.read_csv(r"inputs\vehicles.csv")
# print(claims.dtypes)
# print(claims)
# Define a function to check if a column has "Yes" and "No" values and convert to Boolean
# utlra_cars = cars.filter(pl.col("price")>950_000)
# utlra_cars.write_excel()
cars = cars.with_columns(pl.col('cylinders').str.replace_all(r'\D', '').alias('cylinders'))
cars = (cars
        .drop(
            # not real variables
            "id",
            "url",
            "region_url","VIN","image_url","county","posting_date",
            # size is too missing 75%
            "size"
            ))
def null_out_impossible_values(cars,col,upper_col_limit:int)->pl.DataFrame:
    cars = cars.with_columns(pl.col(col).cast(pl.Int64))
    rows_to_nullify = cars.filter(pl.col(col) > upper_col_limit).height

    cars = cars.with_columns(
    pl.when(pl.col(col).cast(pl.Int64) > upper_col_limit)
    .then(None)
    .otherwise(pl.col(col)))

    # Log the filter and number of rows affected
    print(f"Filter applied: {col} > {upper_col_limit}")
    print(f"Rows set to none: {rows_to_nullify}")
    return cars

def drop_out_impossible_values(cars, col,upper_col_limit):
    cars = cars.with_columns(pl.col(col).cast(pl.Int64))
    rows_to_nullify = cars.filter(pl.col(col) < upper_col_limit).height

    cars = cars.filter(pl.col(col) > upper_col_limit)
    # Log the filter and number of rows affected
    print(f"Filter applied: {col} > {upper_col_limit}")
    print(f"Rows removed: {rows_to_nullify}")
    return cars



def fill_missing_values_column_level(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    # Prepare a copy of the DataFrame to avoid modifying the original
    updated_df = df.clone()

    for col in columns:
        if col in df.columns:
            # Determine the data type of the column
            col_type = df.schema[col]
            
            # Convert the selected column to a NumPy array
            data = updated_df[col].to_numpy().reshape(-1, 1)  # Reshape for the imputer
            
            # Create the appropriate imputer
            if col_type == pl.Int64 or col_type == pl.UInt32:
                imputer = SimpleImputer(strategy='mean')
            elif col_type == pl.Utf8:
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                continue  # Skip unsupported types
            
            # Impute missing values
            imputed_data = imputer.fit_transform(data)
            
            # Update the DataFrame with the imputed values
            updated_df = updated_df.with_columns(pl.Series(name=col, values=imputed_data.flatten()))
    
    return updated_df

cars = null_out_impossible_values(cars,"odometer",300_000)
# cars = null_out_impossible_values(cars,"price",250_000)
cars = drop_out_impossible_values(cars,"price",250_000)
cars.write_parquet("cleaned_input.parquet")
cars = pl.read_parquet("cleaned_input.parquet")
cars = fill_missing_values_column_level(cars,["odometer","year","region"])
# Write the DataFrame to a Parquet file
cars.write_parquet("cleaned_input.parquet")
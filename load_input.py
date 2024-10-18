import polars as pl
from rich import print
import polars.selectors as cs
import polars as pl
from sklearn.impute import SimpleImputer
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
import numpy as np
import zipfile
from pathlib import Path


def unzip_and_load_csv(zip_file_path: str, output_directory: str) -> pl.DataFrame:
    zip_file_path = Path(zip_file_path)
    output_directory = Path(output_directory)

    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

    print(f"Files extracted to {output_directory}")

    # Load the CSV file into a Polars DataFrame
    csv_file_path = output_directory / "vehicles.csv"
    return pl.read_csv(csv_file_path)

# Example usage
cars = unzip_and_load_csv(r"inputs\vehicles.csv.zip", r"inputs\vehicles_unzipped")


def clean_cylinders_column(cars: pl.DataFrame) -> pl.DataFrame:
    cars = cars.with_columns(
        pl.when(pl.col('cylinders').str.replace_all(r'\D', '') != '')
        .then(pl.col('cylinders').str.replace_all(r'\D', ''))
        .otherwise(None)
        .alias('cylinders')
    )
    return(cars)
# Example usage
# Assuming `cars` is already defined as a Polars DataFrame
cars = clean_cylinders_column(cars)


def drop_unnecessary_columns(cars: pl.DataFrame) -> pl.DataFrame:
    """Drop unnecessary columns from the Polars DataFrame.

    Args:
        cars (pl.DataFrame): The input Polars DataFrame containing the car data.

    Returns:
        pl.DataFrame: The DataFrame with specified columns dropped.
    """
    return cars.drop(
        "id",
        "url",
        "region_url",
        "VIN",
        "image_url",
        "county",
        "posting_date",
        "size"
    )

# Example usage
# Assuming `cars` is already defined as a Polars DataFrame
cars = drop_unnecessary_columns(cars)

def null_out_impossible_values(cars,col,upper_col_limit:int)->pl.DataFrame:
    cars = cars.with_columns(pl.col(col).cast(pl.Int64))
    rows_to_nullify = cars.filter(pl.col(col) > upper_col_limit).height

    cars = cars.with_columns(
    pl.when(pl.col(col) > upper_col_limit)
    .then(None)
    .otherwise(pl.col(col)).alias(col))

    # Log the filter and number of rows affected
    print(f"Filter applied: {col} > {upper_col_limit}")
    print(f"Rows set to none: {rows_to_nullify}")
    return cars

def drop_out_impossible_values(cars, col,col_limit,upper: True):
    cars = cars.with_columns(pl.col(col).cast(pl.Int64))
    if upper:
        rows_to_nullify = cars.filter(pl.col(col) > col_limit).height

        cars = cars.filter(pl.col(col) < col_limit)
        # Log the filter and number of rows affected
        print(f"Filter applied: {col} >  {col_limit}")
    else:
        rows_to_nullify = cars.filter(pl.col(col) < col_limit).height

        cars = cars.filter(pl.col(col) > col_limit)
        # Log the filter and number of rows affected
        print(f"Filter applied: {col} <  {col_limit}")
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

cars = drop_out_impossible_values(cars,"odometer",300_000,True)
# cars = null_out_impossible_values(cars,"price",250_000)
cars = drop_out_impossible_values(cars,"price",125_000,True)
cars = drop_out_impossible_values(cars,"price",2_000,False)

cars = fill_missing_values_column_level(cars,[
                                              "odometer",
                                              "year",
                                            #   "region",
                                              "manufacturer",
                                            #   "model",
                                              "state",
                                              "title_status",
                                              "paint_color",
                                            #   "drive",
                                            #   "fuel"
                                              ])

# Assuming 'cars' is your DataFrame and 'price' is the column you're interested in
# Define the threshold value
threshold = 2000

# Calculate the number of rows below the threshold
count_below_threshold = cars.filter(pl.col('price') < threshold).height

# Calculate the total number of rows
total_rows = cars.height

# Calculate the percentage
percentage_below_threshold = (count_below_threshold / total_rows) * 100 if total_rows > 0 else 0

# Print the result
print(f"Percentage of rows below {threshold}: {percentage_below_threshold:.2f}%")

# Write the DataFrame to a Parquet file
cars.write_parquet("output/cleaned_input.parquet")
from __future__ import annotations
import polars as pl
from sklearn.impute import SimpleImputer
import zipfile
from pathlib import Path


def unzip_and_load_csv(zip_file_path: str, output_directory: str) -> pl.DataFrame:
    zip_file_path = Path(zip_file_path)
    output_directory = Path(output_directory)

    # Unzip the file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_directory)

    print(f"Files extracted to {output_directory}")

    # Load the CSV file into a Polars DataFrame
    csv_file_path = output_directory / "vehicles.csv"
    return pl.read_csv(csv_file_path)


def clean_cylinders_column(cars: pl.DataFrame) -> pl.DataFrame:
    cars = cars.with_columns(
        pl.when(pl.col("cylinders").str.replace_all(r"\D", "") != "")
        .then(pl.col("cylinders").str.replace_all(r"\D", ""))
        .otherwise(None)
        .alias("cylinders")
    )
    return cars


def drop_unnecessary_columns(cars: pl.DataFrame) -> pl.DataFrame:
    return cars.drop(
        "id", "url", "region_url", "VIN", "image_url", "county", "posting_date", "size"
    )


def null_out_impossible_values(cars, col, upper_col_limit: int) -> pl.DataFrame:
    cars = cars.with_columns(pl.col(col).cast(pl.Int64))
    rows_to_nullify = cars.filter(pl.col(col) > upper_col_limit).height

    cars = cars.with_columns(
        pl.when(pl.col(col) > upper_col_limit)
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
    )

    # Log the filter and number of rows affected
    print(f"Filter applied: {col} > {upper_col_limit}")
    print(f"Rows set to none: {rows_to_nullify}")
    return cars


def drop_out_impossible_values(cars, col, col_limit, upper: True) -> pl.DataFrame:
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


def fill_missing_values_column_level(
    df: pl.DataFrame, columns: list[str]
) -> pl.DataFrame:
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
                imputer = SimpleImputer(strategy="mean")
            elif col_type == pl.Utf8:
                imputer = SimpleImputer(strategy="most_frequent")
            else:
                continue  # Skip unsupported types

            # Impute missing values
            imputed_data = imputer.fit_transform(data)

            # Update the DataFrame with the imputed values
            updated_df = updated_df.with_columns(
                pl.Series(name=col, values=imputed_data.flatten())
            )

    return updated_df


def switch_condition_to_ordinal(cars: pl.DataFrame):
    # this is subjective and open to SME.
    ordinal_mapping = {
        "salvage": -3,
        "fair": -2,
        "good": -1,
        "excellent": 1,
        "new": 2,
        "like new": 2,
    }
    cars = cars.with_columns(
        pl.col("condition").replace(ordinal_mapping).alias("condition")
    )
    return cars


# # Example usage
# cars = unzip_and_load_csv(r"inputs\vehicles.csv.zip", r"inputs\vehicles_unzipped")

# cars = drop_unnecessary_columns(cars)
# cars.write_parquet("output/raw_input.parquet")
# cars = clean_cylinders_column(cars)
# cars = switch_condition_to_ordinal(cars)
# cars = drop_out_impossible_values(cars,"odometer",300_000,True)
# cars = drop_out_impossible_values(cars,"price",125_000,True)
# cars = drop_out_impossible_values(cars,"price",2_000,False)
# cars = fill_missing_values_column_level(cars,[
#                                               "odometer",
#                                               "year",
#                                               "manufacturer",
#                                               "state",
#                                               "title_status",
#                                               "paint_color",
#                                               ])

# # Write the DataFrame to a Parquet file
# cars.write_parquet("output/cleaned_input.parquet")

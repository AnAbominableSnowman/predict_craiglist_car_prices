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

    csv_file_path = output_directory / "vehicles.csv"
    return pl.read_csv(csv_file_path)


def drop_unnecessary_columns(cars: pl.DataFrame) -> pl.DataFrame:
    cars = cars.drop(
        # these columns are mostly constants, corrupted or too missing to be useful
        "id",
        "url",
        "region_url",
        "VIN",
        "image_url",
        "county",
        "posting_date",
        "size",
    )
    return cars


def detect_if_description_exists(cars: pl.DataFrame) -> pl.DataFrame:
    if "description" not in cars.columns:
        raise ValueError("The column 'description' does not exist in the DataFrame.")

    # why? I do this since we are going to turn DEscription into 500 TF_IDF cols,
    # even if a description doesnt show up in the tf_idf cols, i want to keep the face it had a
    # description
    cars = cars.with_columns(
        (pl.col("description").is_not_null() & (pl.col("description") != "")).alias(
            "description_exists"
        )
    )
    return cars


def delete_description_if_caravana(cars: pl.DataFrame) -> pl.DataFrame:
    # delete out descriptions with carvana ads to stop boilerplate
    # overloading tf_idf
    cars = cars.with_columns(
        pl.when(pl.col("carvana_ad"))
        .then(pl.lit(""))
        .otherwise(pl.col("description"))
        .alias("description")
    )
    return cars


def detect_if_carvana_ad(cars: pl.DataFrame) -> pl.DataFrame:
    cars = cars.with_columns(
        (
            pl.col("description")
            .fill_null("")
            .str.to_lowercase()
            # almost all carvana ads start with this
            .str.contains("carvana is the safer way to buy a car")
        ).alias("carvana_ad")
    )
    return cars


def null_out_impossible_values(
    cars: pl.DataFrame, col: str, upper_col_limit: int
) -> pl.DataFrame:
    cars = cars.with_columns(pl.col(col).cast(pl.Int64))
    rows_to_nullify = cars.filter(pl.col(col) > upper_col_limit).height

    cars = cars.with_columns(
        pl.when(pl.col(col) > upper_col_limit)
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
    )

    # print the filter and number of rows affected
    print(f"Filter applied: {col} > {upper_col_limit}")
    print(f"Rows set to none: {rows_to_nullify}")
    return cars


def drop_out_impossible_values(
    cars: pl.DataFrame, col: str, col_limit: float, upper: True
) -> pl.DataFrame:
    cars = cars.with_columns(pl.col(col).cast(pl.Int64))
    # use upper to allow users to specify a lower or upper bound on a column.
    # Just flips the sign for all comparision.
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


def remove_duplicate_rows(cars: pl.DataFrame) -> pl.DataFrame:
    cars = cars.unique()
    return cars


def fill_missing_values_column_level(
    df: pl.DataFrame, columns: list[str]
) -> pl.DataFrame:
    # not sure if this is needed?
    updated_df = df.clone()

    # doing it col by col is inefficent,
    # but this is only done for a few columns and is a pretty cheap
    # operation so I wont optimize.
    for col in columns:
        if col in df.columns:
            col_type = df.schema[col]
            # Reshape for the imputer
            data = updated_df[col].to_numpy().reshape(-1, 1)
            # Create the appropriate imputer
            if col_type == pl.Int64 or col_type == pl.UInt32:
                imputer = SimpleImputer(strategy="mean")
            elif col_type == pl.Utf8:
                imputer = SimpleImputer(strategy="most_frequent")
            else:
                continue

            imputed_data = imputer.fit_transform(data)

            updated_df = updated_df.with_columns(
                pl.Series(name=col, values=imputed_data.flatten())
            )

    return updated_df


def switch_condition_to_ordinal(cars: pl.DataFrame) -> pl.DataFrame:
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

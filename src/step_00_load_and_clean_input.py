from __future__ import annotations
import polars as pl
import zipfile
from pathlib import Path


def unzip_and_clean_data():
    # pull in and unzip the zip from kaggle
    cars = unzip_and_load_csv(r"inputs/vehicles.csv.zip", r"inputs/vehicles_unzipped")

    # these are mostly non informative columns like URL, or constant values, or columns that
    # the author mentioned were corrupted.
    cars = drop_unnecessary_columns(cars)

    # we'll pump these out as a record, of what the data looked pre processing.
    # This will mostly be used in Ydata-profiling to give an idea of how data cleaning
    # affected the data.
    cars.write_parquet("intermediate_data/raw_input.parquet")
    print(f"starting rows{cars.height}")

    # create boolean for descriptions
    cars = detect_if_description_exists(cars)
    print(
        f"number of ad's with description: {cars.filter(pl.col('description_exists')).height}"
    )

    # about %10 of data are carvana ads
    cars = detect_if_carvana_ad(cars)
    print(
        f"number of carvana ads and corresponding descriptions deleted: {cars.filter(pl.col('carvana_ad')).height}"
    )

    cars = delete_description_if_caravana(cars)

    # condition has a natural ranking so I encode that. IE. like new is better then fair
    cars = switch_condition_to_ordinal(cars)
    print(
        f"number of conditions switched to ordinal: {cars.filter(pl.col('condition').is_not_null()).height}"
    )

    # These values are incredibly rare and most of these values
    # are misstypes, people avoiding sharing info, and the rare spam ad.
    # Alot of this is called price anchoring.
    cars = drop_out_impossible_values(cars, "odometer", 300_000, True)
    cars = drop_out_impossible_values(cars, "price", 125_000, True)
    cars = drop_out_impossible_values(cars, "price", 2_000, False)

    # we seem to have about 45,000 duplicate recrods.
    # its unlikely to have two cars selling in the same location,
    # with the same price, mileage and color, etc. So ill drop them.
    num_rows_before_deduping = cars.height
    cars = remove_duplicate_rows(cars)
    print(f"rows lost due to deduping: {num_rows_before_deduping- cars.height}")

    cars.write_parquet("intermediate_data/cleaned_and_edited_input.parquet")


def get_zip_file_path(default_path: str = "inputs/vehicles.csv.zip") -> Path:
    zip_file_path = Path(default_path)

    # Check if the default path exists
    if zip_file_path.exists() and zip_file_path.is_file():
        return zip_file_path

    # Prompt the user to enter a new path
    user_input = input(
        f"The default file does not exist: {zip_file_path}. Please enter the path to the zip file: "
    ).strip()

    # Use user input if provided, otherwise raise an error
    if user_input:
        zip_file_path = Path(user_input)
        if not zip_file_path.exists() or not zip_file_path.is_file():
            raise FileNotFoundError(
                f"The specified path '{zip_file_path}' does not exist or is not a valid file."
            )
    else:
        raise ValueError("No path provided and default file does not exist.")

    return zip_file_path


def unzip_and_load_csv(
    zip_file_path: str = "inputs/vehicles.csv.zip",
    output_directory: str = "intermediate_data",
) -> pl.DataFrame:
    zip_file_path = get_zip_file_path(zip_file_path)

    output_directory = Path(output_directory)
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(output_directory)

    print(f"Files extracted to {output_directory}")

    # Load the extracted CSV file into a Polars DataFrame
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


def replace_rare_and_null_manufacturer(
    cars: pl.DataFrame, percent_needed: float, replacement_value: str
) -> pl.DataFrame:
    # we need this later to take only mfgers with more then 3 percent in the histograms by mfger.
    total_rows = cars.height
    cars = cars.with_columns(pl.col("manufacturer").alias("org_manuf")).drop(
        "manufacturer"
    )
    # Group by the 'manufacturer' column and count occurrences
    grouped_df = (
        cars.group_by("org_manuf")
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / total_rows * 100).alias("percent_of_total"))
    )

    # Replace manufacturers with less than 3% of total with "Other"
    grouped_df = (
        grouped_df.with_columns(
            pl.when(pl.col("percent_of_total") < percent_needed)
            .then(pl.lit(replacement_value))
            .when(pl.col("org_manuf").is_null())
            .then(pl.lit(replacement_value))
            .otherwise(pl.col("org_manuf"))
            .alias("manufacturer")
        )
        .select("manufacturer")
        .unique()
    )

    joined_df = cars.join(
        grouped_df,
        left_on="org_manuf",
        right_on="manufacturer",
        how="left",
        coalesce=False,
    )
    return joined_df


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
            # almost all carvana ads start with this. We might get false positives.
            # but i can live with that.
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

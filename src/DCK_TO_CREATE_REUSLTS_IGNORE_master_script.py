from step_00_load_and_clean_input import (
    unzip_and_load_csv,
    drop_unnecessary_columns,
    detect_if_carvana_ad,
    switch_condition_to_ordinal,
    drop_out_impossible_values,
    fill_missing_values_column_level,
    delete_description_if_caravana,
    remove_duplicate_rows,
    detect_if_description_exists,
)
from step_01_feature_engineering import (
    remove_punc_short_words_lower_case,
    create_tf_idf_cols,
    replace_rare_and_null_manufacturer,
)
from step_03_linear_regression_approach import train_fit_score_linear_regression
from step_02_visualization import generate_profiling_report
from step_04_lightgbm_approach_with_text_and_hyperopt import (
    train_fit_score_light_gbm,
    #     # plot_results
)
import polars as pl
import pandas as pd
from numpy import log

# # pull in and unzip the zip from kaggle
cars = unzip_and_load_csv(r"inputs\vehicles.csv.zip", r"inputs\vehicles_unzipped")

# these are mostly non informative columns like URL, or constant values, or columns that
# the author mentioned were corrupted.
cars = drop_unnecessary_columns(cars)
# we'll pump these out as a record, of what the data looked pre processing.
# This will mostly be used in Ydata-profiling to give an idea of how data cleaning
# affected the data.
cars.write_parquet("output/raw_input.parquet")
generate_profiling_report(
    cars, "output/data_profile_raw_subsampled_to_one_percent", 0.01
)

# create
cars = detect_if_description_exists(cars)
# # ## about %10 of data are carvana ads
cars = detect_if_carvana_ad(cars)
cars = delete_description_if_caravana(cars)

# # condition has a natural ranking so I encode that. IE. like new is better then fair
cars = switch_condition_to_ordinal(cars)

# These values are incredibly rare and most of these values
# are misstypes, people avoiding sharing info, and the rare spam ad.
# Alof of this is called price anchoring.
cars = drop_out_impossible_values(cars, "odometer", 300_000, True)
cars = drop_out_impossible_values(cars, "price", 125_000, True)
cars = drop_out_impossible_values(cars, "price", 2_000, False)

# manufacturer is a huge source of cardinality here. With one of mfgers, and
# mispellings like Forde. By setting all rare manufacturers to other,
# I can reduce the problem.
cars = replace_rare_and_null_manufacturer(cars, 3, "Other")

# we seem to have about 45,000 duplicate recrods.
# its unlikely to have two cars selling in the same location,
# with the same price, mileage and color, etc. So ill drop them.
cars = remove_duplicate_rows(cars)

# create a nice report on data and its quality.
generate_profiling_report(
    cars, "output/data_profile_cleaned_subsampled_to_one_percent", 0.01
)
cars.write_parquet("output/cleaned_and_edited_input.parquet")


# lightGBM takes care of null and missing values nicely. But
# linear regression won't handle this nicely. So here I,
# encode them with mean or mode. This isn't as elegant as possible.
# But sufficent for a first pass with linear regression.
cars_imputed_missing_for_lin_regrs = fill_missing_values_column_level(
    cars,
    [
        "odometer",
        "year",
        "manufacturer",
        "state",
        "title_status",
        "paint_color",
    ],
)


def write_df_to_parquet(df: pl.DataFrame, file_path: str) -> None:
    try:
        # Optionally, perform sanity checks on the DataFrame before writing
        if df.is_empty():
            raise ValueError("DataFrame is empty, nothing to write.")

        # Check for any null or invalid data in the columns
        null_counts = df.null_count().to_dict()
        print(f"Null counts in columns: {null_counts}")

        # Write the DataFrame to Parquet
        df.write_parquet(file_path)
        print("Data successfully written to Parquet.")
    except Exception as e:
        # Handle the error with more information
        raise RuntimeError(f"Failed to write DataFrame to Parquet: {e}") from e


def check_mixed_types_all(df: pl.DataFrame) -> None:
    for col_name in df.columns:
        unique_types = (
            df.select(
                pl.col(col_name).map_elements(lambda x: type(x).__name__).unique()
            )
            .to_series(0)
            .to_list()
        )

        if len(unique_types) > 1:
            print(f"Column '{col_name}' contains mixed types: {unique_types}")
        else:
            print(f"Column '{col_name}' has a consistent type: {unique_types[0]}")


# Example usage
check_mixed_types_all(cars_imputed_missing_for_lin_regrs)
# Call the function
write_df_to_parquet(
    cars_imputed_missing_for_lin_regrs,
    "output/cleaned_input_with_imputed_missing_values_for_linr_regrsn.parquet",
)
print("safe")

cars_imputed_missing_for_lin_regrs = pd.read_parquet(
    "output/cleaned_input_with_imputed_missing_values_for_linr_regrsn.parquet"
)

y = cars_imputed_missing_for_lin_regrs.pop("price").to_numpy()
X = cars_imputed_missing_for_lin_regrs

train_fit_score_linear_regression(X["odometer"], y, log=False, one_hot_encode=False)


explanatory_variables = [
    "year",
    "manufacturer",
    "odometer",
    "paint_color",
    "state",
    "title_status",
]


train_fit_score_linear_regression(
    X[explanatory_variables], log(y), log=True, one_hot_encode=True
)

cars = pl.read_parquet("output/cleaned_and_edited_input.parquet")

# Description is a huge potential source of info. So I'll use Tf_Idf
# to try to squeeze some knowledge out.

# Preprocess the cars DataFrame
cars = remove_punc_short_words_lower_case(cars)
cars = create_tf_idf_cols(cars, 500)
cars.write_parquet("output/cleaned_edited_feature_engineered_input.parquet")


lightgbm_params = {
    "objective": "regression",
    "metric": "mean_squared_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "max_depth": 6,
    # "min_data_in_leaf": 5000,  # Fixed value
    "verbose": -1,
}


# Calculate num_leaves based on max_depth
# lightgbm_params["num_leaves"] = int(2 ** lightgbm_params["max_depth"] * 0.65)

basic_cols = [
    "region",
    "price",
    "year",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state",
    "lat",
    "long",
    "manufacturer",
]
print("start fitting Light GBM")

train_fit_score_light_gbm(
    input_path="cleaned_edited_feature_engineered_input",
    params=lightgbm_params,
    output_path="results/light_gbm_basic/",
    col_subset=basic_cols,
)

lightgbm_params = {
    "objective": "regression",
    "metric": "mean_squared_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.05032013271321068,
    "max_depth": 8,
    "min_data_in_leaf": 5000,  # Fixed value
    "verbose": -1,
}

# Calculate num_leaves based on max_depth
lightgbm_params["num_leaves"] = int(2 ** lightgbm_params["max_depth"] * 0.65)


print("start fitting Light GBM")
# train_fit_score_light_gbm("cleaned_edited_feature_engineered_input")
train_fit_score_light_gbm(
    input_path="cleaned_edited_feature_engineered_input",
    params=lightgbm_params,
    output_path="results/light_gbm__hyperopt_and_feature_engineering/",
    col_subset=None,
)

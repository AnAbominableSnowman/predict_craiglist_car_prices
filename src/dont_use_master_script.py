from step_00_load_and_clean_input import (
    unzip_and_load_csv,
    drop_unnecessary_columns,
    detect_if_carvana_ad,
    switch_condition_to_ordinal,
    drop_out_impossible_values,
    delete_description_if_caravana,
    remove_duplicate_rows,
    detect_if_description_exists,
    replace_rare_and_null_manufacturer,
)
from step_01_feature_engineering import (
    remove_punc_short_words_lower_case,
    create_tf_idf_cols,
   
)
from step_03_linear_regression_approach import train_fit_linear_regression, fill_missing_values_column_level,
from step_04_lightgbm_approach_with_text_and_hyperopt import (
    train_fit_light_gbm,
)
import polars as pl
from numpy import log
import pickle
import json

# # pull in and unzip the zip from kaggle
cars = unzip_and_load_csv(r"inputs/vehicles.csv.zip", r"inputs\vehicles_unzipped")

# these are mostly non informative columns like URL, or constant values, or columns that
# the author mentioned were corrupted.
cars = drop_unnecessary_columns(cars)

# we'll pump these out as a record, of what the data looked pre processing.
# This will mostly be used in Ydata-profiling to give an idea of how data cleaning
# affected the data.
cars.write_parquet("intermediate_data/raw_input.parquet")

cars = detect_if_description_exists(cars)
# # ## about %10 of data are carvana ads
cars = detect_if_carvana_ad(cars)
# We delete out the caravana adds because the boiler plate drowns out all other words
cars = delete_description_if_caravana(cars)

# # condition has a natural ranking so I encode that. IE. like new is better then fair
cars = switch_condition_to_ordinal(cars)

# These values are incredibly rare and most of these values
# are misstypes, people avoiding sharing info, and the rare spam ad.
# Alof of this is called price anchoring.
cars = drop_out_impossible_values(cars, "odometer", 300_000, True)
cars = drop_out_impossible_values(cars, "price", 125_000, True)
cars = drop_out_impossible_values(cars, "price", 2_000, False)

# for some reason, we have 45k duplicate rows which feel impossible.
cars = remove_duplicate_rows(cars)
cars.write_parquet("intermediate_data/cleaned_and_edited_input.parquet")

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


y = cars_imputed_missing_for_lin_regrs.pop("price").to_numpy()
X = cars_imputed_missing_for_lin_regrs

# fit model 1
train_fit_linear_regression(X["odometer"], y, log=False, one_hot_encode=False)


# fit model 2
explanatory_variables = [
    "year",
    "manufacturer",
    "odometer",
    "paint_color",
    "state",
    "title_status",
]

train_fit_linear_regression(
    X[explanatory_variables], log(y), log=True, one_hot_encode=True
)

cars = pl.read_parquet("intermediate_data/cleaned_and_edited_input.parquet")


# Description is a huge potential source of info. So I'll use Tf_Idf
# to try to squeeze some knowledge out.
cars = remove_punc_short_words_lower_case(cars)
cars = create_tf_idf_cols(cars, 500)
cars.write_parquet("intermediate_data/cleaned_edited_feature_engineered_input.parquet")

# basic params
lightgbm_params = {
    "objective": "regression",
    "metric": "mean_squared_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "max_depth": 6,
    "verbose": -1,
}

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


train_fit_light_gbm(
    input_path="cleaned_edited_feature_engineered_input",
    params=lightgbm_params,
    output_path="results_from_master/light_gbm_basic/",
    col_subset=basic_cols,
)

 # Load the pickled JSON file
with open(
    r"results/light_gbm__hyperopt_and_feature_engineering/final_params.pkl",
    "rb",
) as file:
    hyperparams = pickle.load(file)

# If the data inside the pickle file is JSON, convert it to a dictionary
if isinstance(hyperparams, str):  # In case it's a JSON string
    hyperparams = json.loads(hyperparams)

# Calculate num_leaves based on max_depth
hyperparams["num_leaves"] = int(2 ** hyperparams["max_depth"] * 0.65)


print("start fitting Light GBM")
# train_fit_score_light_gbm("cleaned_edited_feature_engineered_input")
train_fit_light_gbm(
    input_path="cleaned_edited_feature_engineered_input",
    params=hyperparams,
    output_path="results_from_master/light_gbm__hyperopt_and_feature_engineering/",
    col_subset=None,
)

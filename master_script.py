from feature_engineering import (
    create_tf_idf_cols,
    replace_rare_and_null_manufacturer,
    remove_punc_short_words_lower_case,
)

from load_and_clean_input import (
    drop_unnecessary_columns,
    unzip_and_load_csv,
    clean_cylinders_column,
    switch_condition_to_ordinal,
    drop_out_impossible_values,
    fill_missing_values_column_level,
)

# pull in and unzip the zip from kaggle
cars = unzip_and_load_csv(r"inputs\vehicles.csv.zip", r"inputs\vehicles_unzipped")
# these are mostly non informative columns like URL, or constant values, or columns that
# the author mentioned were corrupted.
cars = drop_unnecessary_columns(cars)
# we'll pump these out as a record, of what the data looked pre processing.
# This will mostly be used in Ydata-profiling to give an idea of how data cleaning
# affected the data.
cars.write_parquet("output/raw_input.parquet")

# cylinders can be ints but aren't so I clean them to int.
cars = clean_cylinders_column(cars)
# condition has a natural ranking so I encode that. IE. like new is better then fair
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

# Write the DataFrame to a Parquet file
cars_imputed_missing_for_lin_regrs.write_parquet(
    "output/cleaned_input_with_imputed_missing_values_for_linr_regrsn.parquet"
)


cars_imputed_missing_for_lin_regrs = pd.read_parquet(
    "output/cleaned_input_with_imputed_missing_values_for_linr_regrsn.parquet"
)

y = cars_imputed_missing_for_lin_regrs.pop("price").to_numpy()


explanatory_variables = [
    "year",
    "manufacturer",
    "odometer",
    "paint_color",
    "state",
    "title_status",
]

print(cars.columns)
print(cars.dtypes)
X = one_hot_columns(cars[explanatory_variables])
y_log = np.log(y)
print(X)
train_fit_score_model(X, y_log, log=True)
# Description is a huge potential source of info. So I'll use Tf_Idf
# to try to squeeze some knowledge out.
cars = remove_punc_short_words_lower_case(cars)
cars = create_tf_idf_cols(cars, num_features=500)

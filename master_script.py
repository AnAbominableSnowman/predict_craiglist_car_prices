from lightgbm_approach_text import train_fit_score_light_gbm

# # # pull in and unzip the zip from kaggle
# cars = unzip_and_load_csv(r"inputs\vehicles.csv.zip", r"inputs\vehicles_unzipped")

# # these are mostly non informative columns like URL, or constant values, or columns that
# # the author mentioned were corrupted.
# cars = drop_unnecessary_columns(cars)
# # we'll pump these out as a record, of what the data looked pre processing.
# # This will mostly be used in Ydata-profiling to give an idea of how data cleaning
# # affected the data.
# cars.write_parquet("output/raw_input.parquet")

# # cylinders can be ints but aren't so I clean them to int.
# cars = clean_cylinders_column(cars)
# # condition has a natural ranking so I encode that. IE. like new is better then fair
# cars = switch_condition_to_ordinal(cars)

# # These values are incredibly rare and most of these values
# # are misstypes, people avoiding sharing info, and the rare spam ad.
# # Alof of this is called price anchoring.
# cars = drop_out_impossible_values(cars, "odometer", 300_000, True)
# cars = drop_out_impossible_values(cars, "price", 125_000, True)
# cars = drop_out_impossible_values(cars, "price", 2_000, False)

# # manufacturer is a huge source of cardinality here. With one of mfgers, and
# # mispellings like Forde. By setting all rare manufacturers to other,
# # I can reduce the problem.
# cars = replace_rare_and_null_manufacturer(cars, 3, "Other")

# cars.write_parquet("output/cleaned_and_edited_input.parquet")

# # lightGBM takes care of null and missing values nicely. But
# # linear regression won't handle this nicely. So here I,
# # encode them with mean or mode. This isn't as elegant as possible.
# # But sufficent for a first pass with linear regression.
# cars_imputed_missing_for_lin_regrs = fill_missing_values_column_level(
#     cars,
#     [
#         "odometer",
#         "year",
#         "manufacturer",
#         "state",
#         "title_status",
#         "paint_color",
#     ],
# )

# # # Write the DataFrame to a Parquet file
# cars_imputed_missing_for_lin_regrs.write_parquet(
#     "output/cleaned_input_with_imputed_missing_values_for_linr_regrsn.parquet"
# )


# cars_imputed_missing_for_lin_regrs = pd.read_parquet(
#     "output/cleaned_input_with_imputed_missing_values_for_linr_regrsn.parquet"
# )

# y = cars_imputed_missing_for_lin_regrs.pop("price").to_numpy()
# X = cars_imputed_missing_for_lin_regrs

# train_fit_score_linear_regression(X["odometer"], y, log=False, one_hot_encode=False)


# explanatory_variables = [
#     "year",
#     "manufacturer",
#     "odometer",
#     "paint_color",
#     "state",
#     "title_status",
# ]


# train_fit_score_linear_regression(
#     X[explanatory_variables], log(y), log=True, one_hot_encode=True
# )

# cars = pl.read_parquet("output/cleaned_and_edited_input.parquet")

# # Description is a huge potential source of info. So I'll use Tf_Idf
# # to try to squeeze some knowledge out.

# # Preprocess the cars DataFrame
# cars = remove_punc_short_words_lower_case(cars)
# cars = create_tf_idf_cols(cars, 500)
# cars.write_parquet("output/cleaned_edited_feature_engineered_input.parquet")

# train_fit_score_light_gbm("cleaned_edited_feature_engineered_input")
train_fit_score_light_gbm(
    input_path="cleaned_edited_feature_engineered_input", hyperparameter_tuning=False
)
# cars = pl.read_parquet("output/cleaned_input.parquet")

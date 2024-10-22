# from step_03_lightgbm_approach_with_text_and_hyperopt import train_fit_score_light_gbm
import pickle

from step_03_lightgbm_approach_with_text_and_hyperopt import train_fit_score_light_gbm

with open("LightGBM_with_words/final_params_og_model.pkl", "rb") as file:
    best_params = pickle.load(file)
print(best_params)

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


# print("start fitting Light GBM")
# train_fit_score_light_gbm("cleaned_edited_feature_engineered_input")
train_fit_score_light_gbm(
    input_path="cleaned_edited_feature_engineered_input", params=lightgbm_params
)

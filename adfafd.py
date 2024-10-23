import pickle

# Replace 'your_pickle_file.pkl' with the path to your pickle file
# pickle_file_path = r"C:\git_repos\video_game_sales\video_game_sales_predictions\results - Copy_final\light_gbm__hyperopt_and_feature_engineering_final\final_parms.pkl"

# Load the pickle file
with open(
    r"results - Copy_final\light_gbm_basic\final_params.pkl",
    "rb",
) as file:
    data = pickle.load(file)

# Print the contents of the pickle file
print(data)

# C:\git_repos\video_game_sales\video_game_sales_predictions\results - Copy_final\light_gbm__hyperopt_and_feature_engineering_final

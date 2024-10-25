import pickle
import json

# Load the pickled JSON file
with open(
    r"results/light_gbm__hyperopt_and_feature_engineering/final_params.pkl",
    "rb",
) as file:
    hyperparams = pickle.load(file)

# If the data inside the pickle file is JSON, convert it to a dictionary
if isinstance(hyperparams, str):  # In case it's a JSON string
    hyperparams = json.loads(hyperparams)

# Now you can use the hyperparameters in your model or function
print(hyperparams)

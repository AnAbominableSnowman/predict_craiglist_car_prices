import lightgbm as lgb
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset using Polars
cars = pl.read_parquet("output/cleaned_engineered_input.parquet")

# Identify categorical columns and cast them to 'categorical'
categorical_columns = [col for col in cars.columns if cars[col].dtype == pl.Utf8]
cars = cars.with_columns([pl.col(c).cast(pl.Categorical) for c in categorical_columns])

# Split the data into features and target
y = cars.select("price").to_numpy().flatten()
X = cars.drop("price")

# Convert Polars DataFrame to NumPy for compatibility with LightGBM
X_np = X.to_pandas()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_np, y, test_size=0.2, random_state=42)

# Define the objective function for Hyperopt
def objective(params):
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': params['learning_rate'],
        'num_leaves': int(params['num_leaves']),
        'max_depth': int(params['max_depth']) if params['max_depth'] > 0 else -1,
        'min_child_samples': int(params['min_child_samples']),
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'verbose': -1
    }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_columns)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train the model without early stopping
    model = lgb.train(model_params, train_data, num_boost_round=1000, valid_sets=[test_data], verbose_eval=False)
    
    # Predict on the test set
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return {'loss': rmse, 'status': STATUS_OK}

# Define the search space for Hyperopt
space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'max_depth': hp.quniform('max_depth', -1, 20, 1),
    'min_child_samples': hp.quniform('min_child_samples', 5, 100, 1),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
}

# Run the optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# Train the model with the best parameters
best_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': best['learning_rate'],
    'num_leaves': int(best['num_leaves']),
    'max_depth': int(best['max_depth']) if best['max_depth'] > 0 else -1,
    'min_child_samples': int(best['min_child_samples']),
    'subsample': best['subsample'],
    'colsample_bytree': best['colsample_bytree'],
    'verbose': -1
}

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_columns)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train the model with optimal hyperparameters
model = lgb.train(best_params, train_data, num_boost_round=1000, valid_sets=[test_data])

# Make predictions
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# Calculate RMSE and R2 score
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Best Hyperparameters: {best}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Plot predicted vs actual values
plt.figure(figsize=(10, 5))

# Plot 1: Predicted vs Actual values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")

# Plot 2: Residuals plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, edgecolor='k', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals Plot")

plt.tight_layout()
plt.show()
plt.close()

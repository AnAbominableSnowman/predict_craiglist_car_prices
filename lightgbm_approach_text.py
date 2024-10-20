import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score


cars = pd.read_parquet("output/cleaned_engineered_input.parquet")

categorical_columns = cars.select_dtypes(include=["object"]).columns.tolist()

existing_categorical_columns = [
    col for col in categorical_columns if col in cars.columns
]

if existing_categorical_columns:  # Check if there are any valid columns
    cars[existing_categorical_columns] = cars[existing_categorical_columns].astype(
        "category"
    )
else:
    print("No categorical columns to convert.")

# Split the data into features and target
y = cars.pop("price").to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    cars, y, test_size=0.2, random_state=42
)

# Create a LightGBM dataset
train_data = lgb.Dataset(
    X_train, label=y_train, categorical_feature=categorical_columns
)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# # Set parameters for the LightGBM model
params = {
    "objective": "regression",
    "metric": "mean_squared_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbose": -1,
}

# # Train the LightGBM model
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10)],
)

# Make predictions
y_pred = model.predict(X_test, num_iteration=model.best_iteration)


# Calculate RMSE and R2 score
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Plot predicted vs actual values
plt.figure(figsize=(10, 5))

# Plot 1: Predicted vs Actual values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, edgecolor="k", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")

# Plot 2: Residuals plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, edgecolor="k", alpha=0.7)
plt.axhline(y=0, color="r", linestyle="--", lw=2)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals Plot")

plt.tight_layout()
plt.show()
plt.close()

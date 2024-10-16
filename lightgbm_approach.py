import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


cars = pd.read_parquet("output/cleaned_engineered_input.parquet")
cars = pd.DataFrame(cars)
categorical_columns = cars.select_dtypes(include=['object']).columns.tolist()
# region_code, segment, model, fuel_type, max_torque, max_power, engine_type: object, rear_brakes_type: object, transmission_type: object, steering_type
cars[categorical_columns] = cars[categorical_columns].astype('category')
print(cars.dtypes)
# Split the data into features and target
y = cars.pop("price").to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(cars, y, test_size=0.2, random_state=42)



# print(categorical_columns)
# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature= categorical_columns)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# # Set parameters for the LightGBM model
params = {
    'objective': 'regression',
    'metric': 'mean_squared_error',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1
}

# # Train the LightGBM model
model = lgb.train(params, train_data,  num_boost_round=100,valid_sets=[test_data], 
                  #callbacks=[lgb.early_stopping(stopping_rounds=10)]
                  )

# Make predictions
y_pred = model.predict(X_test, num_iteration=model.best_iteration)


# Calculate RMSE and R2 score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Print the summary of the regression model (including beta coefficients, p-values, F-statistic, etc.)
print("\nModel Summary:")
# print(model.summary())

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
# # Convert probabilities to binary outcomes
# y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

# Generate the confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred_binary)

# # Display the confusion matrix
# print("Confusion Matrix:")
# print(conf_matrix)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.show()

# # Optional: Display classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_binary))

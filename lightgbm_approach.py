import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

claims = pd.read_parquet("cleaned_input.parquet")
claims = pd.DataFrame(claims)
categorical_columns = claims.select_dtypes(include=['object']).columns.tolist()
# region_code, segment, model, fuel_type, max_torque, max_power, engine_type: object, rear_brakes_type: object, transmission_type: object, steering_type
claims[categorical_columns] = claims[categorical_columns].astype('category')
print(claims.dtypes)
# Split the data into features and target
y = claims.pop("claim_status").to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(claims, y, test_size=0.2, random_state=42)



# print(categorical_columns)
# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature= categorical_columns)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# # Set parameters for the LightGBM model
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
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

# Convert probabilities to binary outcomes
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Optional: Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

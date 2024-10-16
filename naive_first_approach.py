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

def train_fit_score_model(X, y, log:bool):
    # Add a constant term for the intercept (as statsmodels does not include it by default)
    X = sm.add_constant(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the linear regression model using statsmodels
    model = sm.OLS(y_train, X_train).fit()

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    if log:
        y_pred = np.exp(y_pred)
        y_test = np.exp(y_test)
        
    # Calculate RMSE and R2 score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")

    # Print the summary of the regression model (including beta coefficients, p-values, F-statistic, etc.)
    print("\nModel Summary:")
    print(model.summary())

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

    return model


# Initialize OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Create an empty list to store processed columns
processed_columns = []

claims = pd.read_parquet("output/cleaned_engineered_input.parquet")

y = claims.pop("price").to_numpy()

def one_hot_columns(claims):
    # Step 1: Process each column based on its dtype
    for col in claims.columns:
        if claims[col].dtype == 'bool':  # Convert boolean columns to integers
            claims[col] = claims[col].astype(int)
            processed_columns.append(claims[[col]])  # Append processed column

        elif claims[col].dtype == 'object':  # One-hot encode string/categorical columns
            encoded = encoder.fit_transform(claims[[col]])
            encoded_claims = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
            processed_columns.append(encoded_claims)  # Append one-hot encoded columns

        elif col != 'target':  # Include numeric columns as they are
            processed_columns.append(claims[[col]])

    X = pd.concat(processed_columns, axis=1)
    return(X)


explanatory_variables = [
                        #  "odometer",
                         "year",
                        #  "region",
                        #  "manufacturer",
                        #  "model",
                        #  "state",
                        # 'region', 
                        # 'year', 
                        'manufacturer',
                        # 'model', 
                        # 'condition', 
                        # 'cylinders', 
                        # 'fuel', 
                        'odometer', 
                        # 'title_status',
                        # 'transmission', 
                        # 'drive', 
                        # 'size', 
                        # 'type', 
                        'paint_color',
                        'state',
                        "title_status",
                        # "paint_color","drive","fuel"
       ]

print(claims.columns)
print(claims.dtypes)
X = one_hot_columns(claims[explanatory_variables])
y_log = np.log(y)
print(X)
train_fit_score_model(X,y_log,log=True)
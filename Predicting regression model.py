import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Loading the sample dataset
file_path = "housing_sample.csv"  # Make sure the CSV file is in the same directory
df = pd.read_csv(file_path)


print(" Dataset Loaded:")
print(df.head(), "\n")


features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = df[features]
y = df[target]

# Train-test spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Trainning of Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predicting and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(" Model Trained Successfully!")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Model Accuracy: {r2*100:.2f}%")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# User Prediction
print("\n--- Predict a House Price ---")
try:
    user_area = float(input("Enter living area (GrLivArea in sq ft): "))
    user_bedrooms = int(input("Enter number of bedrooms: "))
    user_bathrooms = int(input("Enter number of full bathrooms: "))

    user_input = pd.DataFrame({
        'GrLivArea': [user_area],
        'BedroomAbvGr': [user_bedrooms],
        'FullBath': [user_bathrooms]
    })

    user_input_scaled = scaler.transform(user_input)
    predicted_price = model.predict(user_input_scaled)
    print(f"\n Predicted House Price: ₹{predicted_price[0]:,.2f}")

except ValueError:
    print(" Invalid input !. Please enter numeric values.")

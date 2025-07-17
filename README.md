# SCT_TrackCode_TaskNumber

# House Price Prediction using Regression

This project demonstrates how to build a **House Price Prediction model** using **Linear Regression** in Python. It uses a sample housing dataset with features such as **living area**, **number of bedrooms**, and **number of bathrooms** to predict house prices.

Let us walk through the task 
---

## Features of the Project !!

* Reads data from a CSV file (`housing_sample.csv`)
* Preprocesses the data (handles missing values and scales features)
* Splits data into **training** and **testing** sets
* Trains a **Linear Regression model** using scikit-learn
* Evaluates the model using:

  * Mean Squared Error (MSE)
  * R² Score
  * Model Accuracy
* Visualizes **Actual vs Predicted Prices** using a scatter plot
* Allows **user input** to predict house price interactively

---

## 📂 Project Structure

```
house-price-prediction/
│
├── housing_sample.csv         # Sample dataset
├── predicting_regression model.py   # Main Python script
└── README.md                  # Documentation
```

---

## Sample Dataset (`housing_sample.csv`)

The dataset contains:

* `GrLivArea` → Living area (sq ft)
* `BedroomAbvGr` → Number of bedrooms
* `FullBath` → Number of full bathrooms
* `SalePrice` → House price (target variable)



## ▶ How to Run

1. Clone or download this project folder.
2. Ensure you have Python 3 installed with these libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
3. Run the script:

   ```bash
   python predicting_regression model.py
   ```

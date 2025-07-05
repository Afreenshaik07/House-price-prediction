# 🏠 House Price Prediction using Machine Learning

## 📌 What is House Price Prediction?

*House Price Prediction* is the process of estimating the market value of a residential property based on its features such as location, number of bedrooms, square footage, garage availability, furnishing status, and more.

In this project, we use *machine learning algorithms* to build models that learn patterns from historical housing data and accurately predict the selling price of new, unseen houses.

---

## 💡 Why is House Price Prediction Important?

- Helps *buyers* and *sellers* make informed decisions
- Assists *real estate agents* in pricing strategy
- Can be used in *property valuation systems* and *banking* (for home loans)
- Useful in *automated real estate platforms*

---

## 🎯 Project Objective

The main goal of this project is to:
- Preprocess and clean housing data
- Apply various machine learning models
- Use scaling to improve performance
- Evaluate and compare models
- (Optional) Use logistic regression for binary classification (expensive or not)

---

## 🧰 Technologies Used

- *Python 3.x*
- *Pandas, NumPy* – Data processing
- *Matplotlib, Seaborn* – Data visualization
- *Scikit-learn* – ML models and preprocessing
- *XGBoost* – Advanced boosting algorithm
- *Jupyter Notebook* – Code development and testing

---

## ⚙ ML Algorithms Used

### 1. *Linear Regression*
- Predicts continuous values like house prices
- Assumes a linear relationship between input features and output

### 2. *Random Forest Regressor*
- An ensemble of decision trees
- Handles nonlinear relationships better and reduces overfitting

### 3. *XGBoost Regressor*
- Gradient boosting method that builds trees sequentially
- More accurate and optimized for speed and performance

### 4. *Logistic Regression* (optional)
- Used to classify houses as *Expensive (1)* or *Affordable (0)*
- Converts regression problem into binary classification using median price

---

## 🔄 Preprocessing Steps

- Removed duplicates and missing values
- Separated *categorical* and *numerical* features
- Applied:
  - *OneHotEncoder* to categorical columns
  - *StandardScaler* to numerical columns
- Combined preprocessing and model using *Pipeline*

---

## 🔍 Evaluation Metrics

### For Regression:
- *Mean Squared Error (MSE)*
- *Root Mean Squared Error (RMSE)*
- *Mean Absolute Error (MAE)*
- *R² Score*

### For Classification (Logistic Regression):
- *Accuracy*
- *Precision*
- *Recall*
- *F1 Score*
- *Confusion Matrix*

---

## 📈 Project Workflow

1. Load dataset House Price Prediction Dataset.csv
2. Clean the data
3. Preprocess using OneHotEncoder + StandardScaler
4. Split dataset (80% train, 20% test)
5. Train:
   - Linear Regression
   - Random Forest Regressor
   - XGBoost Regressor
6. Optional: Logistic Regression for binary classification
7. Evaluate and compare models
8. Interpret results

---


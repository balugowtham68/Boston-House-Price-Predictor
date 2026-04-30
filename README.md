# Boston House Price Prediction

A beginner-friendly machine learning project that predicts Boston house prices using regression models. The project includes data loading, exploratory data analysis, preprocessing, feature selection, model training, model comparison, hyperparameter tuning, visualization, and custom house price prediction.

## Project Overview

This project uses the Boston Housing dataset to predict the median value of houses based on features such as crime rate, number of rooms, property tax rate, pupil-teacher ratio, pollution level, and other neighborhood-related variables.

The main goal is to build a complete machine learning regression workflow that helps understand how different features affect house prices and how models can be trained to make accurate predictions.

## Dataset

The dataset contains housing information with features such as:

- `CRIM` - Crime rate by town
- `ZN` - Residential land zoning proportion
- `INDUS` - Non-retail business acres
- `CHAS` - Charles River dummy variable
- `NOX` - Nitric oxide concentration
- `RM` - Average number of rooms
- `AGE` - Age of owner-occupied units
- `DIS` - Distance to employment centers
- `RAD` - Highway accessibility index
- `TAX` - Property tax rate
- `PTRATIO` - Pupil-teacher ratio
- `B` - Demographic-related dataset feature
- `LSTAT` - Lower status population percentage
- `MEDV` - Median house value, used as the target variable

## Project Workflow

1. Import required Python libraries
2. Load the Boston Housing dataset
3. Explore the dataset using EDA
4. Check missing values and basic statistics
5. Visualize target distribution and feature relationships
6. Handle missing values using median imputation
7. Detect and treat outliers using the IQR method
8. Select important features using correlation
9. Split data into training and testing sets
10. Scale features using StandardScaler
11. Train multiple regression models
12. Compare model performance
13. Tune Random Forest using GridSearchCV
14. Visualize actual vs predicted prices
15. Analyze feature importance and residuals
16. Predict house prices using custom input values

## Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation Metrics

The models are evaluated using:

- R2 Score
- RMSE
- MAE
- Cross-validation score

## Visualizations

The project creates several useful charts:

- Target price distribution
- Correlation heatmap
- Scatter plots of important features
- Outlier box plots
- Model performance comparison
- Actual vs predicted price plots
- Feature importance chart
- Residual analysis plot

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/boston-house-price-prediction.git
```

2. Open the project folder:

```bash
cd boston-house-price-prediction
```

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the notebook:

```bash
jupyter notebook Boston_House_Price_Prediction.ipynb
```

Or run the Python script:

```bash
python boston_house_price_prediction.py
```

## Output

The project trains regression models and predicts house prices in thousands of dollars. It also saves result charts such as model comparison, feature importance, actual vs predicted prices, and residual analysis.

## Key Learning Outcomes

- Understanding regression problems
- Performing exploratory data analysis
- Cleaning and preprocessing data
- Handling missing values and outliers
- Training and comparing machine learning models
- Using hyperparameter tuning
- Evaluating model performance
- Making predictions from custom input data

## Repository Topic

```text
boston-house-price-prediction
```

## Conclusion

This project demonstrates a complete machine learning pipeline for predicting house prices. It is suitable for beginners who want to learn practical regression modeling, data preprocessing, visualization, and model evaluation using Python.

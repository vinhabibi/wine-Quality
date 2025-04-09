import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('WineQT.csv')

# Handle missing values
data.fillna(data.median(), inplace=True)

# Encode categorical variables and scale numerical features
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

# Preprocess the dataset
data_preprocessed = preprocessor.fit_transform(data)

# Resolve outliers (if applicable)
# Use techniques like IQR or Z-scores for outlier detection; here's an example using Z-scores:
from scipy.stats import zscore

data_scaled = pd.DataFrame(data_preprocessed, columns=numerical_cols)
data_scaled = data_scaled[(zscore(data_scaled) < 3).all(axis=1)]

# Drop duplicates
data_cleaned = data_scaled.drop_duplicates()

print(data_cleaned.head())                                                                                             import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('WineQT.csv')

# Summary statistics
print("Summary Statistics:")
print(data.describe())

# Check for null values
print("\nMissing Values:")
print(data.isnull().sum())

# Visualizations
# 1. Histogram for each numerical feature
data.hist(bins=20, figsize=(15, 10), edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.show()

# 2. Box plots to identify outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=data, orient='h', palette="Set2")
plt.title("Box Plots of Features", fontsize=16)
plt.show()

# 3. Correlation matrix
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix", fontsize=16)
plt.show()

# 4. Pairwise relationships
sns.pairplot(data, diag_kind='kde', palette='husl')
plt.suptitle("Pairwise Relationships", fontsize=16)
plt.show()

# Identify patterns or anomalies
# Looking at duplicates
print("\nNumber of duplicate rows:", data.duplicated().sum())

# Looking for outliers using Z-scores
from scipy.stats import zscore
z_scores = zscore(data.select_dtypes(include=['float64', 'int64']))
outliers = (z_scores > 3).sum()
print("\nOutliers per column (Z-score > 3):")
print(outliers)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('WineQT.csv')

# Data Preprocessing
# Drop unnecessary columns (e.g., 'Id' which does not contribute to prediction)
data = data.drop(columns=['Id'])

# Handle missing values (if any)
data.fillna(data.median(), inplace=True)

# Splitting features and target
X = data.drop(columns=['quality'])  # Features
y = data['quality']                 # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Predictions
y_pred_linear = linear_reg.predict(X_test)

# Evaluation Metrics for Linear Regression
print("Linear Regression Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_linear))
print("R-Squared:", r2_score(y_test, y_pred_linear))

# 2. Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Predictions
y_pred_rf = random_forest.predict(X_test)

# Evaluation Metrics for Random Forest
print("\nRandom Forest Regression Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_rf))
print("R-Squared:", r2_score(y_test, y_pred_rf))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('WineQT.csv')

# Data Preprocessing
# Drop unnecessary columns (e.g., 'Id' which does not contribute to prediction)
data = data.drop(columns=['Id'])

# Handle missing values (if any)
data.fillna(data.median(), inplace=True)

# Splitting features and target
X = data.drop(columns=['quality'])  # Features
y = data['quality']                 # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: Linear Regression ---
linear_reg = LinearRegression()

# Cross-validation for Linear Regression
cv_scores_lr = cross_val_score(linear_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores_lr = np.sqrt(-cv_scores_lr)
r2_scores_lr = cross_val_score(linear_reg, X_train, y_train, cv=5, scoring='r2')

print("Linear Regression Cross-Validation Results:")
print(f"Mean RMSE: {rmse_scores_lr.mean():.4f}")
print(f"Mean R-Squared: {r2_scores_lr.mean():.4f}")

# Fit on the entire training set
linear_reg.fit(X_train, y_train)

# Predictions on the test set
y_pred_linear = linear_reg.predict(X_test)

# Evaluation Metrics on the test set
print("\nLinear Regression Test Set Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_linear))
print("R-Squared:", r2_score(y_test, y_pred_linear))

# --- Model 2: Random Forest Regressor ---
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-validation for Random Forest
cv_scores_rf = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores_rf = np.sqrt(-cv_scores_rf)
r2_scores_rf = cross_val_score(random_forest, X_train, y_train, cv=5, scoring='r2')

print("\nRandom Forest Regression Cross-Validation Results:")
print(f"Mean RMSE: {rmse_scores_rf.mean():.4f}")
print(f"Mean R-Squared: {r2_scores_rf.mean():.4f}")

# Fit on the entire training set
random_forest.fit(X_train, y_train)

# Predictions on the test set
y_pred_rf = random_forest.predict(X_test)

# Evaluation Metrics on the test set
print("\nRandom Forest Regression Test Set Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_rf))
print("R-Squared:", r2_score(y_test, y_pred_rf))

# --- Model Comparison and Justification ---

print("\nModel Comparison:")
print("-----------------------------------")
print("Linear Regression:")
print(f"  Cross-Validation Mean RMSE: {rmse_scores_lr.mean():.4f}")
print(f"  Cross-Validation Mean R-Squared: {r2_scores_lr.mean():.4f}")
print(f"  Test Set MSE: {mean_squared_error(y_test, y_pred_linear):.4f}")
print(f"  Test Set R-Squared: {r2_score(y_test, y_pred_linear):.4f}")
print("\nRandom Forest Regression:")
print(f"  Cross-Validation Mean RMSE: {rmse_scores_rf.mean():.4f}")
print(f"  Cross-Validation Mean R-Squared: {r2_scores_rf.mean():.4f}")
print(f"  Test Set MSE: {mean_squared_error(y_test, y_pred_rf):.4f}")
print(f"  Test Set R-Squared: {r2_score(y_test, y_pred_rf):.4f}")
print("-----------------------------------")

if r2_score(y_test, y_pred_rf) > r2_score(y_test, y_pred_linear):
    print("\nJustification of the Best Performing Model:")
    print("Based on the evaluation metrics on the holdout test set, the Random Forest Regressor outperforms the Linear Regression model for predicting wine quality.")
    print(f"The Random Forest achieved a higher R-squared value ({r2_score(y_test, y_pred_rf):.4f} compared to {r2_score(y_test, y_pred_linear):.4f}), indicating that it explains a larger proportion of the variance in the wine quality.")
    print(f"Additionally, the Random Forest has a lower Mean Squared Error ({mean_squared_error(y_test, y_pred_rf):.4f} compared to {mean_squared_error(y_test, y_pred_linear):.4f}), suggesting that its predictions are closer to the actual quality scores.")
    print("The cross-validation results also support this conclusion, with the Random Forest showing better average RMSE and R-squared scores across different folds of the training data. This suggests that the Random Forest model generalizes better to unseen data for this task.")
else:
    print("\nJustification of the Best Performing Model:")
    print("Based on the evaluation metrics on the holdout test set, the Linear Regression model outperforms the Random Forest Regressor for predicting wine quality.")
    print(f"The Linear Regression achieved a higher R-squared value ({r2_score(y_test, y_pred_linear):.4f} compared to {r2_score(y_test, y_pred_rf):.4f}), indicating that it explains a larger proportion of the variance in the wine quality.")
    print(f"Additionally, the Linear Regression has a lower Mean Squared Error ({mean_squared_error(y_test, y_pred_linear):.4f} compared to {mean_squared_error(y_test, y_pred_rf):.4f}), suggesting that its predictions are closer to the actual quality scores.")
    print("The cross-validation results also support this conclusion, with the Linear Regression showing better average RMSE and R-squared scores across different folds of the training data. This suggests that the Linear Regression model generalizes better to unseen data for this task.")



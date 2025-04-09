# wine-Quality
The project aims to assess wine quality through statistical analysis and predictive modeling while improving data integrity through preprocessing steps like handling missing values, outliers, and duplicates.                          Objective: The project is designed to preprocess data, explore patterns, detect anomalies, and build regression models to predict wine quality.                                                                                                            
Key Steps:
1.Loading the dataset:Reading the dataset into a pandas Frame.

2.Preprocessing:
   .Handling missing values with imputation (median or most frequent).
   .Encoding categorical variables using OneHotEncoder.
   .Scaling numerical features using Standard Scaler.
   
3.Outlier Detection: Identifying and resolving outliers using Z-scores or Interquartile Range (IQR) methods.

4.Exploratory Data Analysis (EDA):
   .Visualizing data with histograms, boxplots, and a correlation matrix.
   .Investigating pairwise relationships to spot trends or anomalies.
   .Summarising statistics and identifying missing values or duplicate rows.
   
5.Modeling:
  .Creating prediction models for wine quality using:
    .Lnear Regression for simple predictive modeling.
    .Random Forest Regression for potentially more robust, ensemble-based predictions.
    .Evaluating modell metrics using Mean Squared Error(MSE) and R-Squared.
    Tools and Libraries:
.Python Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, and Scipy.
.Machine Learning Methods:Regression modelling(Linear Regression and Random forest Regressor)   


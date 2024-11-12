# Project 1: Predict the Housing Prices in Ames

Author: Chaojie Wang (netID: 656449601) UIUC MCS Online Fall 2024

## Technical Details

### Data Pre-processing

1. Data Loading and Cleaning:
   - We loaded the training data from CSV files.
   - Outliers were filtered out based on specific thresholds for several features:
     - Lot_Frontage < 250
     - Lot_Area < 100000
     - Mas_Vnr_Area < 1400
     - BsmtFin_SF_2 < 1200
     - Bsmt_Unf_SF < 2300
     - First_Flr_SF < 3500
     - Garage_Yr_Blt < 2010

2. Target Variable Transformation:
   - The 'Sale_Price' was log-transformed to handle skewness.

3. Feature Selection:
   - We selected 31 numerical features and 34 categorical features for the model.
   - Numerical features: Lot_Frontage, Lot_Area, Year_Built, Year_Remod_Add, Mas_Vnr_Area, BsmtFin_SF_1, BsmtFin_SF_2, Bsmt_Unf_SF, Total_Bsmt_SF, First_Flr_SF, Second_Flr_SF, Gr_Liv_Area, Bsmt_Full_Bath, Bsmt_Half_Bath, Full_Bath, Half_Bath, Bedroom_AbvGr, Kitchen_AbvGr, TotRms_AbvGrd, Fireplaces, Garage_Yr_Blt, Garage_Cars, Garage_Area, Wood_Deck_SF, Open_Porch_SF, Enclosed_Porch, Three_season_porch, Screen_Porch, Misc_Val, Mo_Sold, Year_Sold.
   - Categorical features: Lot_Config, Neighborhood, Condition_1, Bldg_Type, House_Style, Roof_Style, Exterior_1st, Exterior_2nd, Mas_Vnr_Type, Foundation, Electrical, Sale_Type, MS_Zoning, Sale_Condition, Garage_Type, Alley, Lot_Shape, Land_Contour, Land_Slope, Bsmt_Qual, BsmtFin_Type_1, Central_Air, Functional, Fence, Fireplace_Qu, Garage_Finish, Garage_Qual, Paved_Drive, Exter_Cond, Kitchen_Qual, Bsmt_Exposure, Heating_QC, Exter_Qual, Bsmt_Cond.

4. Feature Encoding:
   - Numerical features were standardized using StandardScaler after imputing missing values with the mean.
   - Categorical features were one-hot encoded using OneHotEncoder after imputing missing values with the most frequent value.

5. Feature Transformation Pipeline:
   - We used scikit-learn's ColumnTransformer to apply different preprocessing steps to numerical and categorical columns.

### Model Implementation

1. XGBoost Regressor:
   - We used XGBRegressor from the xgboost library.
   - Hyperparameters:
     - learning_rate: 0.05
     - max_depth: 5
     - n_estimators: 5000
     - subsample: 0.7

2. Lasso-Ridge Combination:
   - We first used LassoCV for feature selection:
     - cv=5 for cross-validation
   - Features were selected using SelectFromModel with a 'median' threshold
     - This method selects features based on their importance scores
     - The 'median' threshold means features with importance >= median are kept
   - RidgeCV was then applied to the selected features:
     - alphas tested: [0.1, 1.0, 10.0]
     - cv=5 for cross-validation

3. Model Evaluation:
   - We used Root Mean Squared Error (RMSE) on log-transformed Sale_Price as the evaluation metric.
   - A threshold of 0.125 was used for folds 1-5, and 0.135 for folds 6-10 to determine model performance.

4. Prediction:
   - Both models (XGBoost and Lasso-Ridge) were used to make predictions on the test set.
   - The predictions were exponentiated to reverse the log transformation before saving to CSV files.

## Performance Metrics

The code was run on a Ubuntu desktop (CPU: AMD Ryzen 9 7950X3D 16-Core Processor - 5.7GHz; Memory: 64GB). The execution times and RMSE scores (calculated on the log-transformed Sale_Price) for each fold were as follows:

1. Fold 1: 14.56 seconds
   - XGBoost RMSE: 0.11449730891369764
   - Lasso-Ridge RMSE: 0.11985791524327039
2. Fold 2: 11.91 seconds
   - XGBoost RMSE: 0.12345802469285123
   - Lasso-Ridge RMSE: 0.12232506575693047
3. Fold 3: 21.83 seconds
   - XGBoost RMSE: 0.11692322440230722
   - Lasso-Ridge RMSE: 0.11570179515977012
4. Fold 4: 12.17 seconds
   - XGBoost RMSE: 0.11427620584766865
   - Lasso-Ridge RMSE: 0.11133156065012521
5. Fold 5: 14.27 seconds
   - XGBoost RMSE: 0.1113757695065617
   - Lasso-Ridge RMSE: 0.11080252535272476
6. Fold 6: 12.90 seconds
   - XGBoost RMSE: 0.1243605018151954
   - Lasso-Ridge RMSE: 0.12986748645895513
7. Fold 7: 11.47 seconds
   - XGBoost RMSE: 0.13050360829866445
   - Lasso-Ridge RMSE: 0.1321651124398239
8. Fold 8: 18.32 seconds
   - XGBoost RMSE: 0.12837697561864264
   - Lasso-Ridge RMSE: 0.12809049699655067
9. Fold 9: 11.06 seconds
   - XGBoost RMSE: 0.12451385670886853
   - Lasso-Ridge RMSE: 0.1258758507488836
10. Fold 10: 12.35 seconds
    - XGBoost RMSE: 0.12349645866359694
    - Lasso-Ridge RMSE: 0.12542039599006338

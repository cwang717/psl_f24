# %%
# packages allowed to use:
# pandas
# scipy
# numpy
# xgboost, lightGBM, Catboost
# sklearn
# category_encoders
# feature_engine.outliers
# glmnet_python
# rpy2
# warnings

import os

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
# from sklearn.linear_model import ElasticNetCV
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBRegressor

# import matplotlib.pyplot as plt
# import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

seed = 9601
np.random.seed(seed)
# # %%
# train_dfs = []
# for i in range(1, 11):
#     fold_df = pd.read_csv(f"proj1/fold{i}/train.csv")
#     train_dfs.append(fold_df)
# train_df = pd.concat(train_dfs, ignore_index=True)

# # %%
# pd.DataFrame(train_df.isna().sum().sort_values(ascending=False)).head(20)

# train_df.fillna(
#    {'Garage_Yr_Blt': 2009, 
#     'Misc_Feature': 'NA', 
#     'Mas_Vnr_Type': 'NA'}, 
#     inplace=True)

# # %%
# def plot_data_scatterplot_for_train_df(x):
    
#    plt.figure(figsize=(5, 3))
#    sns.scatterplot(x=x, y='Sale_Price', data=train_df)
#    plt.show()

# # plot_data_scatterplot_for_train_df('Lot_Frontage')
# # plot_data_scatterplot_for_train_df('Lot_Area')
# # plot_data_scatterplot_for_train_df('Mas_Vnr_Area')
# # plot_data_scatterplot_for_train_df('BsmtFin_SF_2')
# # plot_data_scatterplot_for_train_df('Bsmt_Unf_SF')
# # plot_data_scatterplot_for_train_df('First_Flr_SF')
# # plot_data_scatterplot_for_train_df('Garage_Yr_Blt')

# # %%
# train_df_cleaned = train_df[(train_df['Lot_Frontage'] < 250)
#                             & (train_df['Lot_Area'] < 100000)
#                             & (train_df['Mas_Vnr_Area'] < 1400)
#                             & (train_df['BsmtFin_SF_2'] < 1200)
#                             & (train_df['Bsmt_Unf_SF'] < 2300)
#                             & (train_df['First_Flr_SF'] < 3500)
#                             & (train_df['Garage_Yr_Blt'] < 2010)]

# train_df_cleaned.loc[:, 'Sale_Price'] = np.log(train_df_cleaned['Sale_Price'])
# # %%
# col_to_drop = ['Street',
#                'Utilities', 
#                'Condition_2', 
#                'Roof_Matl', 
#                'Heating', 
#                'Pool_QC', 
#                'Misc_Feature', 
#                'Low_Qual_Fin_SF', 
#                'Pool_Area', 
#                'Longitude',
#                'Latitude']

# ode_cols = ['Lot_Shape', 
#             'Land_Contour', 
#             'Utilities', 
#             'Land_Slope', 
#             'Bsmt_Qual', 
#             'BsmtFin_Type_1', 
#             'Central_Air', 
#             'Functional', 
#             'Pool_QC', 
#             'Fence',
#             'Fireplace_Qu', 
#             'Garage_Finish', 
#             'Garage_Qual', 
#             'Paved_Drive', 
#             'Exter_Cond', 
#             'Kitchen_Qual', 
#             'Bsmt_Exposure', 
#             'Heating_QC', 
#             'Exter_Qual', 
#             'Bsmt_Cond']
# ode_cols = [col for col in ode_cols if col not in col_to_drop]

# ohe_cols = ['Street', 
#             'Lot_Config', 
#             'Neighborhood', 
#             'Condition_1', 
#             'Condition_2', 
#             'Bldg_Type', 
#             'House_Style', 
#             'Roof_Style', 
#             'Exterior_1st', 
#             'Exterior_2nd',
#             'Mas_Vnr_Type', 
#             'Foundation', 
#             'Electrical', 
#             'Sale_Type', 
#             'MS_Zoning', 
#             'Sale_Condition', 
#             'Heating', 
#             'Garage_Type', 
#             'Roof_Matl', 
#             'Misc_Feature', 
#             'Alley']
# ohe_cols = [col for col in ohe_cols if col not in col_to_drop]
# # # %%
# # class Winsorizer(BaseEstimator, TransformerMixin):
# #     def __init__(self, columns, quantile=0.95):
# #         self.columns = columns
# #         self.quantile = quantile
# #         self.limits = {}

# #     def fit(self, X, y=None):
# #         for column in self.columns:
# #             self.limits[column] = np.quantile(X[column], self.quantile)
# #         return self

# #     def transform(self, X):
# #         X_ = X.copy()
# #         for column in self.columns:
# #             X_[column] = np.minimum(X_[column], self.limits[column])
# #         return X_

# # # %%
# # winsorize_cols = ["Lot_Frontage", 
# #                   "Lot_Area", 
# #                   "Mas_Vnr_Area", 
# #                   "BsmtFin_SF_2", 
# #                   "Bsmt_Unf_SF", 
# #                   "Total_Bsmt_SF", 
# #                   "Second_Flr_SF", 
# #                   'First_Flr_SF', 
# #                   "Gr_Liv_Area", 
# #                   "Garage_Area", 
# #                   "Wood_Deck_SF", 
# #                   "Open_Porch_SF", 
# #                   "Enclosed_Porch", 
# #                   "Three_season_porch", 
# #                   "Screen_Porch", 
# #                   "Misc_Val",
# #                   "Garage_Yr_Blt"]

num_pipeline = Pipeline(steps=[
    # ('winsorizer', Winsorizer(columns=winsorize_cols)),
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
# ode_pipeline = Pipeline(steps=[
#     ('impute', SimpleImputer(strategy='most_frequent')),
#     ('ode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
# ])
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

# # %%
# num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
# num_cols = num_cols.drop(['Sale_Price', 'PID'])
# num_cols = [col for col in num_cols if col not in col_to_drop]

num_cols = ['Lot_Frontage', 
            'Lot_Area', 
            'Year_Built', 
            'Year_Remod_Add', 
            'Mas_Vnr_Area', 
            'BsmtFin_SF_1', 
            'BsmtFin_SF_2', 
            'Bsmt_Unf_SF', 
            'Total_Bsmt_SF', 
            'First_Flr_SF', 
            'Second_Flr_SF', 
            'Gr_Liv_Area', 
            'Bsmt_Full_Bath', 
            'Bsmt_Half_Bath', 
            'Full_Bath', 
            'Half_Bath', 
            'Bedroom_AbvGr', 
            'Kitchen_AbvGr', 
            'TotRms_AbvGrd', 
            'Fireplaces', 
            'Garage_Yr_Blt', 
            'Garage_Cars', 
            'Garage_Area', 
            'Wood_Deck_SF', 
            'Open_Porch_SF', 
            'Enclosed_Porch', 
            'Three_season_porch', 
            'Screen_Porch', 
            'Misc_Val', 
            'Mo_Sold', 
            'Year_Sold']

ohe_cols = ['Lot_Config',
            'Neighborhood',
            'Condition_1',
            'Bldg_Type',
            'House_Style',
            'Roof_Style',
            'Exterior_1st',
            'Exterior_2nd',
            'Mas_Vnr_Type',
            'Foundation',
            'Electrical',
            'Sale_Type',
            'MS_Zoning',
            'Sale_Condition',
            'Garage_Type',
            'Alley',
            'Lot_Shape',
            'Land_Contour',
            'Land_Slope',
            'Bsmt_Qual',
            'BsmtFin_Type_1',
            'Central_Air',
            'Functional',
            'Fence',
            'Fireplace_Qu',
            'Garage_Finish',
            'Garage_Qual',
            'Paved_Drive',
            'Exter_Cond',
            'Kitchen_Qual',
            'Bsmt_Exposure',
            'Heating_QC',
            'Exter_Qual',
            'Bsmt_Cond']

col_trans = ColumnTransformer(transformers=[
    ('num_p', num_pipeline, num_cols),
    # ('ode_p', ode_pipeline, ode_cols),
    ('ohe_p', ohe_pipeline, ohe_cols),
    ],
    remainder='drop', 
    n_jobs=-1)

# # Fit the column transformer on all training data
# all_train_data = pd.concat([pd.read_csv(f"proj1/fold{i}/train.csv") for i in range(1, 11)])
# col_trans.fit(all_train_data)

# %%

def load_X_y(col_trans, fold_num=None, test=False, output_y=True):
    # load data from csv files
    df = pd.DataFrame()
    folder = os.getcwd() if fold_num is None else f"proj1/fold{fold_num}"
    # print(f"Loading data from {folder}")
    if not test:
        df = pd.read_csv(f"{folder}/train.csv")
    else:
        df = pd.read_csv(f"{folder}/test.csv")
        if output_y:
            y = pd.read_csv(f"{folder}/test_y.csv")
            df = df.merge(y, on='PID')

    # # fill missing values
    # df.fillna(
    #     {'Garage_Yr_Blt': 2009, 
    #      'Misc_Feature': 'NA', 
    #      'Mas_Vnr_Type': 'NA'}, 
    #     inplace=True)
    
    # filter out outliers
    df = df[(df['Lot_Frontage'] < 250)
            & (df['Lot_Area'] < 100000)
            & (df['Mas_Vnr_Area'] < 1400)
            & (df['BsmtFin_SF_2'] < 1200)
            & (df['Bsmt_Unf_SF'] < 2300)
            & (df['First_Flr_SF'] < 3500)
            & (df['Garage_Yr_Blt'] < 2010)]

    # log transform Sale_Price
    if (not test) or (output_y):
        df.loc[:, 'Sale_Price'] = np.log(df['Sale_Price'])

    # transform data
    X = col_trans.transform(df) if test else col_trans.fit_transform(df)
    y = df['Sale_Price'] if output_y else df["PID"]

    return X, y

# %%
param_grid_XGB = {
    'learning_rate': [0.05, 0.075],
    'n_estimators': [5000, 6000],
    'max_depth': [5, 6, 7],
    'subsample': [0.7, 0.6],
}

def grid_search_10fold(param_grid):
    best_params = None
    most_good_folds = 0

    for params in tqdm(ParameterGrid(param_grid)):
        good_folds = 0
        for fold in range(1, 11):
            X_train, y_train = load_X_y(col_trans, fold)
            X_test, y_test = load_X_y(col_trans, fold, test=True)

            model = XGBRegressor(random_state=seed, **params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            threshold = 0.125 if fold < 6 else 0.135
            if rmse < threshold:
                good_folds += 1

        if good_folds > most_good_folds:
            most_good_folds = good_folds
            best_params = params
            print(f"New best params: {best_params} with {most_good_folds} good folds")

        if most_good_folds == 10:
            break

    print("Best parameters:", best_params)
    print("Most good folds:", most_good_folds)

    return best_params, most_good_folds

# %%
def validate_on_all_folds():
    for fold in tqdm(range(1, 11)):
        X_train, y_train = load_X_y(col_trans, fold)
        X_test, y_test = load_X_y(col_trans, fold, test=True)
        threshold = 0.125 if fold < 6 else 0.135

        # XGBoost model
        xgb_final = XGBRegressor(random_state=seed, 
                                learning_rate=0.05, 
                                max_depth=5, 
                                n_estimators=5000, 
                                subsample=0.7)
        xgb_final.fit(X_train, y_train)
        xgb_predictions = xgb_final.predict(X_test)
        xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)

        # # ElasticNetCV model
        # elastic_net_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
        #                               alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10],
        #                               cv=5,
        #                               random_state=seed)
        # elastic_net_cv.fit(X_train, y_train)
        # elastic_net_cv_predictions = elastic_net_cv.predict(X_test)
        # elastic_net_cv_rmse = mean_squared_error(y_test, elastic_net_cv_predictions, squared=False)

        # Lasso for feature selection
        lasso_cv = LassoCV(cv=5, random_state=seed)
        lasso_cv.fit(X_train, y_train)

        # Select features based on Lasso coefficients
        selector = SelectFromModel(lasso_cv, prefit=True, threshold='median')
        X_train_selected = selector.transform(X_train)

        # Ridge regression on selected features
        ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
        ridge_cv.fit(X_train_selected, y_train)
        X_test_selected = selector.transform(X_test)
        lasso_ridge_predictions = ridge_cv.predict(X_test_selected)
        lasso_ridge_rmse = mean_squared_error(y_test, lasso_ridge_predictions, squared=False)

        if xgb_rmse > threshold:
            print(f"Fold {fold} XGBoost RMSE:", xgb_rmse)
        # if elastic_net_cv_rmse > threshold:
        #     print(f"Fold {i} ElasticNetCV RMSE:", elastic_net_cv_rmse)
        if lasso_ridge_rmse > threshold:
            print(f"Fold {fold} Lasso-Ridge RMSE:", lasso_ridge_rmse)

# %%
# validate_on_all_folds()

# %%
def calculate_rmse(submission_file, test_y_file):
    # Read the submission file
    submission = pd.read_csv(submission_file)
    
    # Read the test_y file
    test_y = pd.read_csv(test_y_file)
    
    # Merge the dataframes on PID
    merged = pd.merge(submission, test_y, on='PID')
    
    # Apply logarithm to both predicted and actual Sale_Price values
    log_pred = np.log(merged['Sale_Price_x'])
    log_actual = np.log(merged['Sale_Price_y'])
    
    # Calculate RMSE on log-transformed values
    rmse = np.sqrt(mean_squared_error(log_actual, log_pred))
    
    return rmse

# %%
if __name__ == "__main__":
    # Load training data
    X_train, y_train = load_X_y(col_trans)

    # XGBoost model
    xgb_final = XGBRegressor(random_state=seed, 
                            learning_rate=0.05, 
                            max_depth=5, 
                            n_estimators=5000, 
                            subsample=0.7)
    xgb_final.fit(X_train, y_train)

    # Lasso for feature selection
    lasso_cv = LassoCV(cv=5, random_state=seed)
    lasso_cv.fit(X_train, y_train)

    # Select features based on Lasso coefficients
    selector = SelectFromModel(lasso_cv, prefit=True, threshold='median')
    X_train_selected = selector.transform(X_train)

    # Ridge regression on selected features
    ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    ridge_cv.fit(X_train_selected, y_train)
    
    # Load test data
    X_test, PIDs = load_X_y(col_trans, 
                            fold_num = None,
                            test=True, 
                            output_y=False)
    
    # XGBoost predictions
    xgb_predictions = xgb_final.predict(X_test)
    xgb_predicted_prices = np.exp(xgb_predictions)

    # Lasso-Ridge predictions
    X_test_selected = selector.transform(X_test)
    lasso_ridge_predictions = ridge_cv.predict(X_test_selected)
    lasso_ridge_predicted_prices = np.exp(lasso_ridge_predictions)

    # Create DataFrames with PID and predicted prices
    xgb_output = pd.DataFrame({
        'PID': PIDs,
        'Sale_Price': xgb_predicted_prices
    })

    lasso_ridge_output = pd.DataFrame({
        'PID': PIDs,
        'Sale_Price': lasso_ridge_predicted_prices
    })

    # Save predictions to CSV files
    xgb_output.to_csv('mysubmission1.txt', index=False)
    lasso_ridge_output.to_csv('mysubmission2.txt', index=False)

    print("Predictions saved to mysubmission1.txt and mysubmission2.txt")

    # # Calculate RMSE for XGBoost predictions
    # xgb_rmse = calculate_rmse('mysubmission1.txt', 'test_y.csv')
    # print(f"RMSE for XGBoost predictions (log scale): {xgb_rmse}")

    # # Calculate RMSE for Lasso-Ridge predictions
    # lasso_ridge_rmse = calculate_rmse('mysubmission2.txt', 'test_y.csv')
    # print(f"RMSE for Lasso-Ridge predictions (log scale): {lasso_ridge_rmse}")


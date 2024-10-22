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

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import ElasticNetCV
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

seed = 9601
np.random.seed(seed)
# %%
train_dfs = []
for i in range(1, 11):
    fold_df = pd.read_csv(f"proj1/fold{i}/train.csv")
    train_dfs.append(fold_df)
train_df = pd.concat(train_dfs, ignore_index=True)

# %%
pd.DataFrame(train_df.isna().sum().sort_values(ascending=False)).head(20)

train_df.fillna(
   {'Garage_Yr_Blt': 2009, 
    'Misc_Feature': 'NA', 
    'Mas_Vnr_Type': 'NA'}, 
    inplace=True)

# %%
def plot_data_scatterplot_for_train_df(x):
    
   plt.figure(figsize=(5, 3))
   sns.scatterplot(x=x, y='Sale_Price', data=train_df)
   plt.show()

plot_data_scatterplot_for_train_df('Lot_Frontage')
plot_data_scatterplot_for_train_df('Lot_Area')
plot_data_scatterplot_for_train_df('Mas_Vnr_Area')
plot_data_scatterplot_for_train_df('BsmtFin_SF_2')
plot_data_scatterplot_for_train_df('Bsmt_Unf_SF')
plot_data_scatterplot_for_train_df('First_Flr_SF')
plot_data_scatterplot_for_train_df('Garage_Yr_Blt')

# %%
train_df_cleaned = train_df[(train_df['Lot_Frontage'] < 250)
                            & (train_df['Lot_Area'] < 100000)
                            & (train_df['Mas_Vnr_Area'] < 1400)
                            & (train_df['BsmtFin_SF_2'] < 1200)
                            & (train_df['Bsmt_Unf_SF'] < 2300)
                            & (train_df['First_Flr_SF'] < 3500)
                            & (train_df['Garage_Yr_Blt'] < 2010)]

train_df_cleaned.loc[:, 'Sale_Price'] = np.log(train_df_cleaned['Sale_Price'])
# %%

ode_cols = ['Lot_Shape', 
            'Land_Contour', 
            'Utilities', 
            'Land_Slope', 
            'Bsmt_Qual', 
            'BsmtFin_Type_1', 
            'Central_Air', 
            'Functional', 
            'Pool_QC', 
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
ohe_cols = ['Street', 
            'Lot_Config', 
            'Neighborhood', 
            'Condition_1', 
            'Condition_2', 
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
            'Heating', 
            'Garage_Type', 
            'Roof_Matl', 
            'Misc_Feature', 
            'Alley']
# %%
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
ode_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# %%
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
num_cols = num_cols.drop(['Sale_Price', 'PID'])

col_trans = ColumnTransformer(transformers=[
    ('num_p', num_pipeline, num_cols),
    ('ode_p', ode_pipeline, ode_cols),
    ('ohe_p', ohe_pipeline, ohe_cols),
    ],
    remainder='drop', 
    n_jobs=-1)

print(num_cols)

# Fit the column transformer on all training data
all_train_data = pd.concat([pd.read_csv(f"proj1/fold{i}/train.csv") for i in range(1, 11)])
col_trans.fit(all_train_data)

# %%

def load_X_y(fold_num, test=False):
    if fold_num not in range(1, 11):
        raise ValueError("Fold number must be between 1 and 10")
    
    # load data from csv files
    df = pd.DataFrame()
    if not test:
        df = pd.read_csv(f"proj1/fold{fold_num}/train.csv")
    else:
        df = pd.read_csv(f"proj1/fold{fold_num}/test.csv")
        y = pd.read_csv(f"proj1/fold{fold_num}/test_y.csv")
        df = df.merge(y, on='PID')

    # fill missing values
    df.fillna(
        {'Garage_Yr_Blt': 2009, 
         'Misc_Feature': 'NA', 
         'Mas_Vnr_Type': 'NA'}, 
        inplace=True)
    
    # filter out outliers
    df = df[(df['Lot_Frontage'] < 250)
            & (df['Lot_Area'] < 100000)
            & (df['Mas_Vnr_Area'] < 1400)
            & (df['BsmtFin_SF_2'] < 1200)
            & (df['Bsmt_Unf_SF'] < 2300)
            & (df['First_Flr_SF'] < 3500)
            & (df['Garage_Yr_Blt'] < 2010)]

    # log transform Sale_Price
    df.loc[:, 'Sale_Price'] = np.log(df['Sale_Price'])

    # transform data
    X = col_trans.transform(df)
    y = df['Sale_Price']

    return X, y

# %%
param_grid_XGB = {
    'learning_rate': [0.05, 0.075, 0.1],
    'n_estimators': [5000, 6000],
    'max_depth': [5, 6, 7],
    'subsample': [0.4, 0.5, 0.6],
}

def grid_search_10fold(param_grid):
    best_params = None
    most_good_folds = 0

    for params in tqdm(ParameterGrid(param_grid)):
        good_folds = 0
        for fold in range(1, 11):
            X_train, y_train = load_X_y(fold)
            X_test, y_test = load_X_y(fold, test=True)

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

    return best_params, most_good_folds

best_params, most_good_folds = grid_search_10fold(param_grid_XGB)
print("Best parameters:", best_params)
print("Most good folds:", most_good_folds)

# %%
# Now run the cross-validation
for i in range(1, 11):
    X_train, y_train = load_X_y(i)

    # XGBoost model
    xgb_final = XGBRegressor(random_state=seed, 
                             learning_rate=0.05, 
                             max_depth=6, 
                             n_estimators=6000, 
                             subsample=0.6)
    xgb_final.fit(X_train, y_train)

    # ElasticNetCV model
    elastic_net_cv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                  alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10],
                                  cv=5,
                                  random_state=seed)
    elastic_net_cv.fit(X_train, y_train)

    X_test, y_test = load_X_y(i, test=True)

    # XGBoost predictions and RMSE
    xgb_predictions = xgb_final.predict(X_test)
    xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
    print(f"Fold {i} XGBoost RMSE:", xgb_rmse)

    # ElasticNetCV predictions and RMSE
    elastic_net_cv_predictions = elastic_net_cv.predict(X_test)
    elastic_net_cv_rmse = mean_squared_error(y_test, elastic_net_cv_predictions, squared=False)
    print(f"Fold {i} ElasticNetCV RMSE:", elastic_net_cv_rmse)

# %%

# Add these imports
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_fold(fold_num):
    X_train, y_train = load_X_y(fold_num)
    X_test, y_test = load_X_y(fold_num, test=True)

    # Load original data for reference
    original_test_data = pd.read_csv(f"proj1/fold{fold_num}/test.csv")
    original_test_data = original_test_data.merge(pd.read_csv(f"proj1/fold{fold_num}/test_y.csv"), on='PID')

    # XGBoost model
    xgb_model = XGBRegressor(random_state=seed, 
                             learning_rate=0.05, 
                             max_depth=6, 
                             n_estimators=6000, 
                             subsample=0.6)
    xgb_model.fit(X_train, y_train)

    # Predictions and RMSE
    xgb_predictions = xgb_model.predict(X_test)
    xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
    print(f"Fold {fold_num} XGBoost RMSE:", xgb_rmse)

    # Distribution comparison
    plt.figure(figsize=(12, 6))
    sns.kdeplot(y_train, label='Train')
    sns.kdeplot(y_test, label='Test')
    plt.title(f'Distribution of Sale Price (log) - Fold {fold_num}')
    plt.legend()
    plt.show()

    # Residual plot
    residuals = y_test - xgb_predictions
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=xgb_predictions, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residual Plot - Fold {fold_num}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

    # Feature importance
    feature_names = num_cols.tolist() + ode_cols + ohe_cols
    importances = permutation_importance(xgb_model, X_test, y_test, n_repeats=10, random_state=seed)

    # Get the actual feature names from the column transformer
    actual_feature_names = col_trans.get_feature_names_out()
    
    # Feature importance
    feature_names = col_trans.get_feature_names_out()
    importances = permutation_importance(xgb_model, X_test, y_test, n_repeats=10, random_state=seed)
    
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances.importances_mean})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
 
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Top 20 Feature Importances - Fold {fold_num}')
    plt.show()

    # Add this new section to show predictions with top 10 residuals and their features
    residuals = y_test - xgb_predictions
    abs_residuals = np.abs(residuals)
    top_10_indices = abs_residuals.argsort()[-10:][::-1]

    print(f"\nTop 10 residuals for Fold {fold_num}:")
    print("Actual\t\tPredicted\tResidual")
    for idx in top_10_indices:
        print(f"{y_test.iloc[idx]:.4f}\t{xgb_predictions[idx]:.4f}\t{residuals.iloc[idx]:.4f}")

    # Get feature names
    feature_names = col_trans.get_feature_names_out()

    print("\nFeature values for top 10 residuals:")
    for idx in top_10_indices:
        print(f"\nInstance with residual {residuals.iloc[idx]:.4f}:")
        for feature_name in feature_importance['feature']:
            feature_index = list(feature_names).index(feature_name)
            transformed_value = X_test[idx, feature_index]
            
            # Find original feature name and value
            original_feature = feature_name.split('__')[-1]
            if original_feature in original_test_data.columns:
                original_value = original_test_data.iloc[idx][original_feature]
                print(f"{feature_name}: {transformed_value:.4f} (Original: {original_value})")
            else:
                print(f"{feature_name}: {transformed_value:.4f} (Original feature not found)")

# Analyze fold 2 and a well-performing fold (e.g., fold 1) for comparison
analyze_fold(2)


# %%



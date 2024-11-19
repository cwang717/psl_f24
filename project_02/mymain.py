# %%
import os

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from tqdm import tqdm

seed = 9601
np.random.seed(seed)

# %%
def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])
    return data

# %%
def smooth_department_data(dept_data, n_components = 8):
    pivot_data = dept_data.pivot(
        index='Store', 
        columns='Date', 
        values='Weekly_Sales'
    ).fillna(0)
    
    store_means = pivot_data.mean(axis=1)
    centered_data = pivot_data.subtract(store_means, axis=0)
    
    U, D, Vt = np.linalg.svd(centered_data, full_matrices=False)
    
    D_reduced = np.diag(D[:n_components])
    smoothed_data = U[:, :n_components] @ D_reduced @ Vt[:n_components, :] + store_means.values.reshape(-1, 1)
    
    smoothed_df = pd.DataFrame(
        smoothed_data, 
        index=pivot_data.index, 
        columns=pivot_data.columns
    ).stack().reset_index()
    smoothed_df.columns = ['Store', 'Date', 'Smoothed_Weekly_Sales']
    
    return smoothed_df

# %%
def apply_shift(predictions):
    """Apply circular shift to Christmas period sales predictions.
    
    Args:
        predictions: DataFrame with predictions
        train_dates: Series or array of training data dates
    """
    # Determine if using one or both years based on training data
    shift_fraction = 2/7
    
    # Group by Store and Dept
    for (store, dept), group in predictions.groupby(['Store', 'Dept']):
        # Get weeks 48-52
        holiday_weeks = group[pd.to_datetime(group['Date']).dt.isocalendar().week.between(48, 52)]
        
        if len(holiday_weeks) == 5:  # Ensure we have all 5 weeks
            sales = holiday_weeks.sort_values('Date')['Weekly_Pred'].values  # Sort to ensure correct week order
            # Check if weeks 49-51 are at least 10% higher than weeks 48 and 52
            mid_weeks_avg = np.mean(sales[1:4])
            edge_weeks_avg = np.mean([sales[0], sales[4]])
            
            if mid_weeks_avg > edge_weeks_avg * 1.1:
                shifted_sales = sales.copy()
                for i in range(len(sales)):
                    next_idx = (i + 1) % 5
                    shifted_sales[next_idx] += sales[i] * shift_fraction
                    shifted_sales[i] -= sales[i] * shift_fraction
                
                # Update predictions
                predictions.loc[holiday_weeks.index, 'Weekly_Pred'] = shifted_sales
    
    return predictions

# %%
def main(eval_flag=False):
    curr_dir = os.getcwd()
    # Load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    is_fold_5 = test['Date'].between("2011-11-04", "2011-12-30").any()
    
    # Apply SVD smoothing by department
    smoothed_train = pd.DataFrame()
    for dept in train['Dept'].unique():
        dept_data = train[train['Dept'] == dept].copy()
        smoothed_dept = smooth_department_data(dept_data)
        smoothed_dept['Dept'] = dept
        smoothed_train = pd.concat([smoothed_train, smoothed_dept])
    
    train = train.merge(smoothed_train, on=['Store', 'Dept', 'Date'], how='left')
    train['Weekly_Sales'] = train['Smoothed_Weekly_Sales']
    train = train.drop(columns=['Smoothed_Weekly_Sales'])

    # pre-allocate a pd to store the predictions
    test_pred = pd.DataFrame()

    train_pairs = train[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    test_pairs = test[['Store', 'Dept']].drop_duplicates(ignore_index=True)
    unique_pairs = pd.merge(train_pairs, test_pairs, how = 'inner', on =['Store', 'Dept'])

    train_split = unique_pairs.merge(train, on=['Store', 'Dept'], how='left')
    train_split = preprocess(train_split)
    X = patsy.dmatrix('Weekly_Sales + Store + Dept + I(Yr**2) + Yr + Wk',
                    data = train_split,
                    return_type='dataframe')
    train_split = dict(tuple(X.groupby(['Store', 'Dept'])))

    test_split = unique_pairs.merge(test, on=['Store', 'Dept'], how='left')
    test_split = preprocess(test_split)
    X = patsy.dmatrix('Store + Dept + I(Yr**2) + Yr + Wk', 
                        data = test_split, 
                        return_type='dataframe')
    X['Date'] = test_split['Date']
    test_split = dict(tuple(X.groupby(['Store', 'Dept'])))

    keys = list(train_split.keys())
    for key in tqdm(keys, desc='Fitting models'):
            
        X_train = train_split[key]
        X_test = test_split[key]
    
        Y = X_train['Weekly_Sales']
        X_train = X_train.drop(['Weekly_Sales','Store', 'Dept'], axis=1)
        
        cols_to_drop = X_train.columns[(X_train == 0).all()]
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
    
        cols_to_drop = []
        for i in range(len(X_train.columns) - 1, 1, -1):
            col_name = X_train.columns[i]
            tmp_Y = X_train.iloc[:, i].values
            tmp_X = X_train.iloc[:, :i].values

            _, residuals, _, _ = np.linalg.lstsq(tmp_X, tmp_Y, rcond=None)
            if np.sum(residuals) < 1e-16:
                cols_to_drop.append(col_name)
                
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

        model = sm.OLS(Y, X_train).fit()
        mycoef = model.params.fillna(0)
        
        tmp_pred = X_test[['Store', 'Dept', 'Date']]
        X_test = X_test.drop(['Store', 'Dept', 'Date'], axis=1)
        
        tmp_pred['Weekly_Pred'] = np.dot(X_test, mycoef)
        test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)

    if is_fold_5:
        test_pred = apply_shift(test_pred)

    # Merge IsHoliday from test data before saving
    test_pred = test.merge(test_pred, on=['Store', 'Dept', 'Date'], how='left')
    test_pred['Weekly_Pred'] = test_pred['Weekly_Pred'].fillna(0)
    
    if eval_flag:
        test_with_label = pd.read_csv('../test_with_label.csv')
        test_with_actuals = test_pred.merge(
            test_with_label[['Store', 'Dept', 'Date', 'Weekly_Sales']], 
            on=['Store', 'Dept', 'Date'],
            how='left'
        )
        
        weights = test_with_actuals['IsHoliday'].apply(lambda x: 5 if x else 1)
        
        wmae = (weights * abs(test_with_actuals['Weekly_Sales'] - test_with_actuals['Weekly_Pred'])).sum() / weights.sum()
        print(f"Weighted Mean Absolute Error: {wmae:.2f}")
    
    # output the predictions
    test_pred.to_csv(os.path.join(curr_dir, 'mypred.csv'), index=False)
    

# # %%
# os.chdir(os.path.join(os.path.dirname(__file__), "Proj2_Data", "fold_5"))
# main(True)
# %%
if __name__ == "__main__":
    main()

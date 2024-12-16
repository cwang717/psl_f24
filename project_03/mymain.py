# %%
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

seed = 9601
np.random.seed(seed)

# %%
def main(eval_flag=False, save_model=False):
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    train_X, train_y = train.iloc[:, 3:], train.iloc[:, 1]
    test_X = test.iloc[:, 2:]

    alpha = 0.5
    
    # Define lambda values to try (via C = 1/lambda)
    C_values = [2.782559] #np.logspace(-4, 4, 10)  # Creates 10 points between 10^-4 and 10^4
    best_score = 0
    best_C = None
    best_model = None

    for C in tqdm(C_values, desc="Tuning"):
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=alpha,
            C=C,
            max_iter=1000,
            random_state=seed
        )
        
        scores = cross_val_score(
            model, 
            train_X, 
            train_y, 
            cv=5, 
            scoring='roc_auc'
        )
        
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_C = C
            best_model = model

    print(f"Best C (1/lambda): {best_C:.6f}")
    
    best_model.fit(train_X, train_y)
    if save_model:
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
    
    test_pred = best_model.predict_proba(test_X)[:, 1]
    test_results = pd.DataFrame({
        'id': test['id'],
        'prob': test_pred
    })
    test_results.to_csv('mysubmission.csv', index=False)
    if eval_flag:
        test_y = pd.read_csv('test_y.csv')
        test_results = test_results.merge(test_y, on='id', how='inner')
        auc_score = roc_auc_score(test_results['sentiment'], test_results['prob'])
        print(f"Test AUC Score: {auc_score:.4f}")

    return best_model

# %%
# os.chdir(os.path.join(os.path.dirname(__file__), "F24_Proj3_data", "split_1"))
# main(True)

# %%
if __name__ == "__main__":
    main()

import os
import pandas as pd

def wmae_eval():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_with_label = pd.read_csv('./Proj2_Data/test_with_label.csv')
    print(f"test_with_label.csv shape: {test_with_label.shape}")
    print()
    
    num_folds = 10
    wae = []

    for i in range(num_folds):
        file_path = f'./Proj2_Data/fold_{i+1}/test.csv'
        test = pd.read_csv(file_path)
        print(f"test.csv shape: {test.shape}")
        
        test = test.drop(columns=['IsHoliday']).merge(test_with_label, on=['Date', 'Store', 'Dept'])
        print(f"merge test and test_with_label shape: {test.shape}")

        file_path = f'./Proj2_Data/fold_{i+1}/mypred.csv'
        test_pred = pd.read_csv(file_path)
        print(f"mypred.csv shape: {test_pred.shape}")
        
        test_pred = test_pred.drop(columns=['IsHoliday'])
        new_test = test.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')
        print(f"merge test_pred and new_test shape: {new_test.shape}")

        actuals = new_test['Weekly_Sales']
        preds = new_test['Weekly_Pred']
        print(f"actuals shape: {actuals.shape}")
        print(f"preds shape: {preds.shape}")
        
        weights = new_test['IsHoliday'].apply(lambda x: 5 if x else 1)
        wae.append(sum(weights * abs(actuals - preds)) / sum(weights))
        print()

    return wae

if __name__ == "__main__":
    wae = wmae_eval()
    for folder, value in enumerate(wae, start=1):
        print(f"Folder {folder} - WMAE: {value:.3f}")
    print(f"Average over the 10 folders (5pt: less than 1580): {sum(wae) / len(wae):.3f}")
